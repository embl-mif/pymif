from __future__ import annotations

import os
import sys
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")
pytest.importorskip("napari")

from qtpy.QtWidgets import QApplication, QTextEdit

from pymif.napari._convert_widget import EmittingStream, make_log_appender


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def log(app):
    widget = QTextEdit()
    stream = EmittingStream()
    stream.text_written.connect(make_log_appender(widget))
    return widget, stream


def lines(widget):
    return [ln for ln in widget.toPlainText().replace(" ", "\n").split("\n") if ln.strip()]


def test_plain_output_gets_one_timestamped_line_each(log):
    widget, stream = log
    stream.write("first\n")
    stream.write("second\n")
    out = lines(widget)
    assert len(out) == 2 and out[0].endswith("first") and out[1].endswith("second")
    assert out[0].startswith("[")  # timestamp


def test_progress_redraws_replace_the_same_line(log):
    widget, stream = log
    stream.write("Writing...\n")
    stream.write("\rzarr:   0%|          | 0/10")
    for i in range(1, 11):
        stream.write(f"\rzarr: {i * 10:3d}%|{'#' * i:<10}| {i}/10")
    stream.write("\n")
    stream.write("done\n")
    out = lines(widget)
    assert len(out) == 3, out  # message, ONE bar line, message
    assert out[1].endswith("100%|##########| 10/10")
    assert "0/10" not in out[1].replace("10/10", "")  # earlier redraws are gone


def test_a_second_bar_does_not_overwrite_the_first(log):
    widget, stream = log
    for name in ("first.zarr", "second.zarr"):
        stream.write(f"\r{name}:   0%|")
        stream.write(f"\r{name}: 100%|##########|")
        stream.write("\n")
    out = lines(widget)
    assert len(out) == 2 and "first.zarr" in out[0] and "second.zarr" in out[1]


def test_other_output_interrupting_a_bar_starts_a_new_line(log):
    widget, stream = log
    stream.write("\rbar:  10%|#")
    stream.write("a warning\n")
    stream.write("\rbar:  50%|#####")
    out = lines(widget)
    assert len(out) == 3 and out[1].endswith("a warning") and out[2].endswith("50%|#####")


def test_real_tqdm_progress_is_one_line(log, monkeypatch):
    """End to end with the real tqdm callback used by ``to_zarr``."""
    from tqdm import tqdm

    widget, stream = log
    monkeypatch.setattr(sys, "stderr", stream)
    bar = tqdm(total=20, desc="viventis.zarr", mininterval=0, leave=True)
    for _ in range(20):
        bar.update(1)
        time.sleep(0.001)
    bar.close()
    out = lines(widget)
    assert len(out) == 1, out
    assert "100%" in out[0] and "20/20" in out[0]


def test_output_after_the_log_is_deleted_goes_to_the_console(app, monkeypatch):
    import io

    from qtpy.QtCore import QEvent

    widget = QTextEdit()
    stream = EmittingStream()
    stream.text_written.connect(make_log_appender(widget))
    widget.deleteLater()  # e.g. the dock holding the log was closed
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    console = io.StringIO()
    monkeypatch.setattr(sys, "__stdout__", console)
    stream.write("still here\n")  # used to raise: wrapped C/C++ object has been deleted
    assert console.getvalue() == "still here\n"
