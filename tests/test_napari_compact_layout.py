from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")
pytest.importorskip("napari")

from magicgui import magicgui
from qtpy.QtWidgets import QApplication, QCheckBox, QToolButton, QVBoxLayout, QWidget

from zarrsmith.napari._convert_widget import _compact_advanced_rows, _make_section

ADVANCED = [
    "chunk_x", "chunk_y", "chunk_z", "n_levels", "downscale_z", "downscale_y",
    "downscale_x", "zarr_format", "shards", "shard_target_mb", "shard_exclude_axes",
    "drop_singleton", "num_workers",
]


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def form(app):
    @magicgui(
        chunk_z={"min": 1, "max": 999, "value": 16},
        chunk_y={"min": 1, "max": 999, "value": 512},
        chunk_x={"min": 1, "max": 999, "value": 512},
        n_levels={"min": 1, "max": 10, "value": 5},
        downscale_z={"min": 1, "value": 2},
        downscale_y={"min": 1, "value": 2},
        downscale_x={"min": 1, "value": 2},
        zarr_format={"choices": [2, 3], "value": 3},
        shards={"choices": ["none", "auto"], "value": "none"},
        shard_target_mb={"min": 1, "max": 99999, "value": 64},
        shard_exclude_axes={"choices": ["t", "c", "z"], "widget_type": "Select",
                            "allow_multiple": True, "value": ("t", "c")},
        drop_singleton={"widget_type": "CheckBox", "value": True},
        num_workers={"min": 0, "max": 256, "value": 0},
    )
    def f(chunk_x=512, chunk_y=512, chunk_z=16, n_levels=5, downscale_z=2,
          downscale_y=2, downscale_x=2, zarr_format=3, shards="none",
          shard_target_mb=64, shard_exclude_axes=("t", "c"),
          drop_singleton=True, num_workers=0):
        pass

    return f


def test_rows_are_shared_and_toggle_together(form):
    btn = QToolButton()
    form.native.layout().insertWidget(0, btn)
    rows = _compact_advanced_rows(form, btn)

    assert len(rows) == 6
    for r in rows:
        r.setVisible(False)
    form.native.show()
    QApplication.processEvents()
    assert not any(r.isVisible() for r in rows)

    for r in rows:
        r.setVisible(True)
    QApplication.processEvents()
    assert all(r.isVisible() for r in rows)
    # The panel itself must stay visible (the checkbox once hid all of it).
    assert form.native.isVisible()


def test_widgets_moved_into_rows_and_still_work(form):
    btn = QToolButton()
    form.native.layout().insertWidget(0, btn)
    rows = _compact_advanced_rows(form, btn)
    row_set = set(rows)

    for name in ADVANCED:
        native = getattr(form, name).native
        owner = native.parent()
        assert owner in row_set, f"{name} not placed in a compact row"

    form.chunk_z.value = 8
    assert form.chunk_z.value == 8
    form.num_workers.enabled = False
    assert not form.num_workers.native.isEnabled()
    assert form.drop_singleton.value is True


def test_fewer_visible_rows_than_one_per_parameter(form):
    btn = QToolButton()
    form.native.layout().insertWidget(0, btn)
    rows = _compact_advanced_rows(form, btn)
    form.native.show()
    for r in rows:
        r.setVisible(True)
    QApplication.processEvents()

    layout = form.native.layout()
    visible_children = [
        layout.itemAt(i).widget() for i in range(layout.count())
        if layout.itemAt(i).widget() is not None and layout.itemAt(i).widget().isVisible()
    ]
    # Toggle button + 6 compact rows + magicgui's call button, instead of
    # one visible row per each of the 13 parameters.
    assert set(rows) <= set(visible_children)
    assert len(visible_children) == len(rows) + 2


def test_make_section_wraps_widgets_in_a_bordered_frame(app):
    a, b = QWidget(), QWidget()
    frame = _make_section(a)
    frame.layout().addWidget(b)

    assert frame.objectName() == "zarrsmithSection"
    assert "border" in frame.styleSheet()
    assert a.parent() is frame and b.parent() is frame
    assert frame.layout().count() == 2
