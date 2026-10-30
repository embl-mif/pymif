"""Mosaic readers by microscope, and detection of the microscope of an acquisition folder."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Type

from .luxendo_mosaic_manager import LuxendoMosaicManager
from .mosaic import MosaicManager
from .opera_mosaic_manager import INDEX_NAMES, OperaMosaicManager
from .zeiss_mosaic_manager import ZeissMosaicManager

#: Mosaic readers by microscope name (``pymif fuse --microscope``, napari widget).
MOSAIC_READERS: Dict[str, Type[MosaicManager]] = {
    "luxendo": LuxendoMosaicManager,
    "opera": OperaMosaicManager,
    "zeiss": ZeissMosaicManager,
}


class WrongReaderError(ValueError):
    """The acquisition was recorded by another microscope than the reader's (``detected``)."""

    def __init__(self, message: str, detected: str):
        super().__init__(message)
        self.detected = detected


def detect_microscope(path: str | Path) -> Optional[str]:
    """Microscope of the acquisition at ``path``, from its layout, or ``None`` if not recognised.

    * ``"opera"``: a Harmony export (``Images/Index.xml``), its ``Images`` folder or the index;
    * ``"luxendo"``: ``*.lux.h5`` files in ``raw/<stack>/``, in ``<stack>/`` or in the folder.

    Opera is checked first: it takes a few file checks, Luxendo lists folders (slow
    for an Opera ``Images`` folder of thousands of files on a network drive).
    """
    path = Path(path)
    index = OperaMosaicManager.find_index(path)
    if index is not None and index.name.lower() in {n.lower() for n in INDEX_NAMES}:
        return "opera"
    if path.is_dir():
        raw = path / "raw" if (path / "raw").is_dir() else path
        if any(raw.glob("*.lux.h5")) or any(raw.glob("*/*.lux.h5")):
            return "luxendo"
    return None


def check_reader(reader: Type[MosaicManager], path: str | Path) -> None:
    """Raise :class:`WrongReaderError` when ``path`` is recognised as the acquisition of
    another microscope than ``reader``'s (unrecognised folders pass)."""
    found = detect_microscope(path)
    if found is None or issubclass(reader, MOSAIC_READERS[found]):
        return
    name = next((k for k, cls in MOSAIC_READERS.items() if issubclass(reader, cls)), reader.__name__)
    raise WrongReaderError(
        f"{path} looks like {found.capitalize()} data, not {name.capitalize()}: read it with "
        f"{MOSAIC_READERS[found].__name__} (microscope '{found}').", found)
