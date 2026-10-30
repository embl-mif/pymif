"""Layout helpers that keep the PyMIF napari widgets usable at any dock size."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

#: Narrow enough for a side dock; the content scrolls horizontally below this.
MIN_WIDTH = 260
#: Minimum heights so that the log always shows several lines and the
#: scrollable part always shows something.
MIN_LOG_HEIGHT = 140
MIN_CONTENT_HEIGHT = 120
#: Style of the title of a section.
TITLE_STYLE = "QLabel { font-weight: bold; font-size: 13px; padding: 2px 0px; }"
#: The logo shown at the top of every widget (also the logo of the documentation).
LOGO_PATH = Path(__file__).parent / "resources" / "zarrsmith-icon.svg"
#: Height of the logo, in logical pixels (the SVG is rendered sharp at any screen scaling).
LOGO_SIZE = 40


def make_logo_header() -> QWidget:
    """A slim row with the logo, shown above the scrollable content of a dock."""
    logo = QLabel()
    logo.setObjectName("pymifLogo")
    logo.setPixmap(QIcon(str(LOGO_PATH)).pixmap(QSize(LOGO_SIZE, LOGO_SIZE)))
    header = QWidget()
    row = QHBoxLayout(header)
    row.setContentsMargins(6, 4, 6, 2)
    row.addWidget(logo)
    row.addStretch(1)
    return header


def make_title(text: str) -> QLabel:
    """The bold title of a section."""
    label = QLabel(text)
    label.setStyleSheet(TITLE_STYLE)
    return label


def make_log_pane(title: str = "PyMIF Log:"):
    """The log of a widget: a read-only text box under its title and a Clear button.

    Returns ``(section, text_box)``; the section is the ``bottom`` pane of :func:`compose_dock`.
    """
    log = QTextEdit()
    log.setReadOnly(True)
    log.setMinimumHeight(110)  # the dock splitter lets the user drag it taller
    log.setStyleSheet(
        "QTextEdit { background-color: #111; color: #ddd;"
        " font-family: Consolas, Menlo, monospace; font-size: 12px; }"
    )
    log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    clear = QPushButton("Clear")
    clear.setToolTip("Clear the log")
    clear.setMaximumWidth(60)
    clear.clicked.connect(log.clear)
    header = QWidget()
    row = QHBoxLayout(header)
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(make_title(title))
    row.addStretch(1)
    row.addWidget(clear)
    return make_section(header, log), log


def make_section(*widgets, margins=(6, 4, 6, 6), spacing=3):
    """Wrap widgets in a rounded, bordered frame to visually separate a block.

    The border uses a translucent grey so it reads on both light and dark
    napari themes. More widgets can be added later via ``section.layout()``.
    """
    frame = QFrame()
    frame.setObjectName("pymifSection")
    frame.setStyleSheet(
        "QFrame#pymifSection { border: 1px solid rgba(128, 128, 128, 0.55);"
        " border-radius: 6px; }"
    )
    inner = QVBoxLayout(frame)
    inner.setContentsMargins(*margins)
    inner.setSpacing(spacing)
    for w in widgets:
        inner.addWidget(w)
    return frame


def make_button_holder(button: QPushButton) -> QWidget:
    """A small holder for an action button.

    Disable the *holder* to lock the button: a disabled parent keeps its
    children disabled even if the button re-enables itself (magicgui does).
    """
    holder = QWidget()
    layout = QHBoxLayout(holder)
    layout.setContentsMargins(0, 2, 0, 2)
    button.setMinimumHeight(28)
    button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    layout.addWidget(button)
    return holder


def compose_dock(content: QWidget, bottom: QWidget | None = None) -> QWidget:
    """Build the dock widget: scrollable content and an optional resizable bottom pane.

    * the logo sits in a slim row above everything, always visible.
    * ``content`` (all the boxes) goes in one scroll area, so nothing is ever cut
      off, whatever the size or position of the dock (side or bottom).
    * ``bottom`` (typically the log) sits under a splitter handle, so it can be
      dragged taller or shorter.
    """
    container = QWidget()
    container.setMinimumWidth(MIN_WIDTH)
    container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    outer = QVBoxLayout(container)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)
    outer.addWidget(make_logo_header())

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    scroll.setWidget(content)
    scroll.setMinimumHeight(MIN_CONTENT_HEIGHT)

    top = QWidget()
    top_layout = QVBoxLayout(top)
    top_layout.setContentsMargins(0, 0, 0, 0)
    top_layout.setSpacing(2)
    top_layout.addWidget(scroll, 1)

    if bottom is None:
        outer.addWidget(top)
        return container

    bottom.setMinimumHeight(MIN_LOG_HEIGHT)
    splitter = QSplitter(Qt.Vertical)
    splitter.setChildrenCollapsible(False)
    splitter.addWidget(top)
    splitter.addWidget(bottom)
    splitter.setStretchFactor(0, 3)
    splitter.setStretchFactor(1, 2)
    splitter.setSizes([480, 260])
    outer.addWidget(splitter)
    container.splitter = splitter  # exposed for tests / callers
    return container


def enable_drag_and_drop(
    file_edit,
    transform: Optional[Callable[[Path], Path]] = None,
    placeholder: str = "Drop a file or folder here, or browse",
    single_only: bool = False,
) -> None:
    """Let a magicgui ``FileEdit`` accept a file or folder dropped from the file explorer.

    ``transform`` maps the dropped path to the value to set (e.g. a file to its
    folder); by default the dropped path is used as is. Paths are not resolved,
    so a mapped network drive (``Y:\\...``) keeps its drive letter.
    With ``single_only``, several files dropped at once are left to the parent
    widget (e.g. a panel that sorts them out); else the first one is used.
    """
    line = file_edit.native.findChild(QLineEdit)
    if line is None:
        raise RuntimeError("Could not find the QLineEdit inside the FileEdit.")
    line.setAcceptDrops(True)
    line.setPlaceholderText(placeholder)

    def dropped_path(event) -> Optional[Path]:
        urls = [u for u in event.mimeData().urls() if u.isLocalFile()]
        if not urls or (single_only and len(urls) > 1):
            return None
        return Path(urls[0].toLocalFile())

    def accept_paths(event):
        if dropped_path(event) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def drop(event):
        path = dropped_path(event)
        if path is None:
            event.ignore()
            return
        file_edit.value = transform(path) if transform else path
        event.acceptProposedAction()

    line.dragEnterEvent = accept_paths
    line.dragMoveEvent = accept_paths
    line.dropEvent = drop
