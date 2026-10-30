"""Console-to-log helpers shared by the ZarrSmith napari widgets."""
from __future__ import annotations

import sys
from datetime import datetime

from qtpy.QtCore import QObject, Signal
from qtpy.QtGui import QTextCursor


class EmittingStream(QObject):
    """Qt-compatible text stream used to forward console output to the widget."""

    text_written = Signal(str)

    def write(self, text):
        """Emit written text to the Qt signal."""
        self.text_written.emit(str(text))

    def flush(self):
        """Provide a file-like ``flush`` method for compatibility."""
        pass

    def isatty(self):
        """Report that this stream is not an interactive terminal."""
        return False
    
def make_log_appender(log_widget):
    """Return a slot that appends timestamped console output to ``log_widget``.

    Progress bars (tqdm) redraw themselves by writing ``"\\r"`` followed by the
    new bar. Those writes replace the previous progress line instead of adding
    a new line each time; any other output starts a new line.

    The widget can be deleted while output still arrives, for example when its
    dock was closed but ``sys.stdout`` still points at this stream. That output
    then goes to the original console instead of raising.
    """
    state = {"progress_start": None}  # document position where the current bar starts

    def append_with_time(text):
        try:
            _append(text)
        except RuntimeError:  # wrapped C/C++ object of type QTextEdit has been deleted
            console = sys.__stdout__
            if console is not None:
                console.write(text)

    def _append(text):
        if not text.strip():
            if "\n" in text:  # tqdm ends a finished bar with a bare newline
                state["progress_start"] = None
            return

        is_progress = text.startswith("\r")
        ts = datetime.now().strftime("[%H:%M:%S] ")
        html = f"<span style='color:#ffffff;'>{ts}{text.strip()}</span><br>"

        cursor = QTextCursor(log_widget.document())
        cursor.movePosition(QTextCursor.End)
        if is_progress:
            if state["progress_start"] is None:
                state["progress_start"] = cursor.position()
            else:  # drop the previous redraw of this bar
                cursor.setPosition(state["progress_start"], QTextCursor.KeepAnchor)
                cursor.removeSelectedText()
        else:
            state["progress_start"] = None

        # insertHtml does NOT add an automatic newline like append()
        cursor.insertHtml(html)  # we add exactly one <br> per line

        # auto-scroll
        end = log_widget.textCursor()
        end.movePosition(QTextCursor.End)
        log_widget.setTextCursor(end)
        log_widget.ensureCursorVisible()

    return append_with_time
