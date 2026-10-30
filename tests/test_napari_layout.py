from __future__ import annotations

import os
import sys
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")
pytest.importorskip("napari")

from napari.components import ViewerModel
from qtpy.QtWidgets import QApplication, QLabel, QPushButton, QScrollArea, QSplitter, QTextEdit, QToolButton, QWidget

import zarrsmith.napari._convert_widget as convert_module
import zarrsmith.napari._export_widget as export_module
from zarrsmith.napari._layout import LOGO_PATH, MIN_LOG_HEIGHT, compose_dock


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def converter(app, monkeypatch):
    # The plugin redirects stdout/stderr into its log box; undo that after the test.
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    monkeypatch.setattr(convert_module, "current_viewer", lambda: ViewerModel())
    widget = convert_module.convert_widget()
    widget._log_streams = (sys.stdout, sys.stderr)  # the plugin's log-box streams
    yield widget
    widget.close()


def button_named(widget, text):
    return next(b for b in widget.findChildren(QPushButton) if b.text() == text)


def convert_button(widget):
    return button_named(widget, "Convert to zarr")


def section_of(widget):
    """The bordered section frame that contains ``widget``."""
    parent = widget.parentWidget()
    while parent is not None:
        if parent.objectName() == "zarrsmithSection":
            return parent
        parent = parent.parentWidget()
    return None


def conversion_toggle(widget):
    return next(b for b in widget.findChildren(QToolButton) if b.text().startswith("Dataset conversion"))


def show(widget, app, size=(420, 900)):
    widget.resize(*size)
    widget.show()
    app.processEvents()


# ---------------------------------------------------------------------------
# Box order and scrolling
# ---------------------------------------------------------------------------

def test_boxes_are_ordered_input_conversion_batch(converter, app):
    show(converter, app)
    boxes = [
        section_of(button_named(converter, "Visualize in napari")),
        section_of(convert_button(converter)),
        section_of(button_named(converter, "Append to batch CSV")),
    ]
    assert all(b is not None for b in boxes)
    assert len({id(b) for b in boxes}) == 3  # three different boxes
    ys = [b.mapTo(converter, b.rect().topLeft()).y() for b in boxes]
    assert ys == sorted(ys), f"expected input, conversion, batch from top to bottom, got y={ys}"


def test_one_scroll_area_holds_all_three_boxes(converter):
    scrolls = converter.findChildren(QScrollArea)
    assert len(scrolls) == 1  # no nested scroll areas
    content = scrolls[0].widget()
    for text in ("Visualize in napari", "Convert to zarr", "Append to batch CSV"):
        assert content.isAncestorOf(button_named(converter, text)), text
    assert converter.minimumWidth() <= 300  # fits a narrow side dock


# ---------------------------------------------------------------------------
# Convert button: inside the conversion box, visible when collapsed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("expanded", [False, True])
def test_convert_button_is_inside_the_conversion_box_and_visible(converter, app, expanded):
    toggle = conversion_toggle(converter)
    toggle.setChecked(expanded)
    show(converter, app)

    button = convert_button(converter)
    box = section_of(button)
    assert box is not None and box.isAncestorOf(toggle), "button and title share the conversion box"
    assert button.isVisible()  # also while the parameters are collapsed
    assert box.rect().contains(button.mapTo(box, button.rect().center()))


def test_dropdown_expands_and_collapses_only_the_parameters(converter, app):
    toggle = conversion_toggle(converter)
    button = convert_button(converter)
    show(converter, app)
    box = section_of(button)

    toggle.setChecked(False)
    app.processEvents()
    collapsed_height = box.height()
    assert toggle.isVisible() and button.isVisible()

    toggle.setChecked(True)
    app.processEvents()
    expanded_height = box.height()
    assert expanded_height > collapsed_height + 100  # the parameters open in place
    assert toggle.isVisible() and button.isVisible()

    toggle.setChecked(False)
    app.processEvents()
    assert box.height() == collapsed_height


# ---------------------------------------------------------------------------
# Log pane
# ---------------------------------------------------------------------------

def test_log_is_tall_and_resizable(converter, app):
    log = converter.findChild(QTextEdit)
    assert log.maximumHeight() > 10_000  # no fixed cap any more
    assert log.minimumHeight() >= 100

    splitter = converter.findChild(QSplitter)
    assert splitter is not None and splitter.count() == 2
    assert splitter.widget(1).minimumHeight() >= MIN_LOG_HEIGHT
    show(converter, app, (420, 760))
    assert log.height() >= 100  # several lines, not two or three


# ---------------------------------------------------------------------------
# Converting: the button is locked while running and still triggers the call
# ---------------------------------------------------------------------------

def test_convert_button_starts_disabled_until_a_dataset_is_loaded(converter):
    assert not convert_button(converter).isEnabled()


def test_button_still_runs_the_conversion_and_locks_while_running(converter, app, monkeypatch):
    calls = []

    def fake_conversion(**kwargs):
        calls.append(kwargs)
        time.sleep(0.3)  # long enough to observe the locked state
        return "converted.zarr"

    monkeypatch.setattr(convert_module, "_run_conversion", fake_conversion)

    button = convert_button(converter)
    log = converter.findChild(QTextEdit)
    # pytest swaps sys.stdout between setup and test body; point it back to the log box
    monkeypatch.setattr(sys, "stdout", converter._log_streams[0])
    monkeypatch.setattr(sys, "stderr", converter._log_streams[1])
    button.parentWidget().setEnabled(True)  # its holder; unlocked as after loading a dataset
    button.click()
    assert not button.isEnabled()  # locked at once: a second click must not start a second job
    button.click()  # ignored while disabled

    deadline = time.time() + 20
    while not button.isEnabled() and time.time() < deadline:
        app.processEvents()
        time.sleep(0.02)
    app.processEvents()
    assert button.isEnabled(), "button must be re-enabled when the conversion ends"
    assert len(calls) == 1  # exactly one conversion was started
    assert "Conversion completed: converted.zarr" in log.toPlainText()


# ---------------------------------------------------------------------------
# Output recap
# ---------------------------------------------------------------------------

def _dataset(shape=(3, 2, 10, 20, 30), axes="tczyx"):
    import numpy as np
    from zarrsmith.microscope_manager import ArrayManager

    return ArrayManager(
        np.zeros(shape, dtype="uint16"),
        {"axes": axes, "channel_names": ["a", "b"], "scales": [(2.0, 0.5, 0.5)]},
    )


def test_output_summary_follows_ranges_and_channels():
    ds = _dataset()
    full = dict(t_range=(0, 2), z_range=(0, 9), y_range=(0, 19), x_range=(0, 29), channels=("a", "b"))
    assert convert_module.output_summary(ds, **full) == (
        "3 t x 2 ch x 10 x 20 x 30 px (tczyx), 0.1 MB, voxel 2 x 0.5 x 0.5 micrometer"
    )
    cropped = dict(full, z_range=(2, 5), y_range=(0, 9), channels=("b",))
    assert convert_module.output_summary(ds, **cropped).startswith("3 t x 4 x 10 x 30 px (tzyx)")


def test_output_summary_keeps_singletons_when_asked():
    ds = _dataset()
    single = dict(t_range=(1, 1), z_range=(4, 4), y_range=(0, 19), x_range=(0, 29), channels=("a",))
    assert convert_module.output_summary(ds, **single).startswith("20 x 30 px (yx),")
    assert convert_module.output_summary(ds, **single, drop_singleton=False).startswith(
        "1 t x 1 ch x 1 x 20 x 30 px (tczyx),"
    )


def test_output_recap_sits_above_the_convert_button(converter, app):
    conversion_toggle(converter).setChecked(False)
    show(converter, app)
    label, button = converter.output_label, convert_button(converter)
    assert label.text().startswith("Load a dataset")
    assert section_of(label) is section_of(button)
    assert label.isVisible()  # also while the parameters are collapsed
    assert label.mapTo(converter, label.rect().topLeft()).y() < button.mapTo(converter, button.rect().topLeft()).y()


def test_output_recap_updates_on_load_and_crop(converter, tmp_path):
    path = tmp_path / "small.zarr"
    _dataset().to_zarr(path)
    converter.visualize_widget.file_format.value = "zarr"
    converter.visualize_widget.input_path.value = path
    assert converter.output_label.text().startswith("Output: 3 t x 2 ch x 10 x 20 x 30 px (tczyx)")
    converter.convert_widget.z_range.value = (0, 4)
    assert converter.output_label.text().startswith("Output: 3 t x 2 ch x 5 x 20 x 30 px")


# ---------------------------------------------------------------------------
# Other widgets
# ---------------------------------------------------------------------------

def test_export_widget_is_scrollable_dock(app, monkeypatch):
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    monkeypatch.setattr(export_module, "current_viewer", lambda: ViewerModel(), raising=False)
    widget = export_module.export_widget()
    assert widget.findChild(QScrollArea) is not None


def test_compose_dock_without_bottom_pane(app):
    dock = compose_dock(QWidget())
    assert dock.findChild(QScrollArea) is not None and dock.findChild(QSplitter) is None


def test_compose_dock_shows_logo(app):
    assert LOGO_PATH.is_file()
    dock = compose_dock(QWidget())
    logo = dock.findChild(QLabel, "zarrsmithLogo")
    assert logo is not None and not logo.pixmap().isNull()
