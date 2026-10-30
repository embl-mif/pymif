"""Dataset readers by microscope, and detection of the reader of a path (CLI and napari)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Type

from .evident_manager import EvidentManager
from .luxendo_manager import LuxendoManager
from .microscope_manager import MicroscopeManager
from .opera_manager import OperaManager
from .scape_manager import ScapeManager
from .viventis_manager import ViventisManager
from .zarr_manager import ZarrManager
from .zarr_v04_manager import ZarrV04Manager
from .zeiss_manager import ZeissManager

#: Readers by microscope name (``zarrsmith convert --microscope``, napari converter).
READERS: Dict[str, Type[MicroscopeManager]] = {
    "luxendo": LuxendoManager,
    "opera": OperaManager,
    "viventis": ViventisManager,
    "zeiss": ZeissManager,
    "evident": EvidentManager,
    "zarrv04": ZarrV04Manager,
    "zarr": ZarrManager,
    "scape": ScapeManager,
}


def detect_reader(path: str | Path) -> Optional[Tuple[Type[MicroscopeManager], str]]:
    """``(reader, microscope name)`` of the dataset at ``path``, from its layout, or ``None``.

    Zarr stores of both NGFF versions are read by :class:`ZarrManager` (``zarrv04`` names a
    store without ``zarr.json``).
    """
    path = Path(path)
    if path.is_dir():
        if (path / "zarr.json").exists():
            return ZarrManager, "zarr"
        if (path / ".zgroup").exists() or (path / ".zattrs").exists():
            return ZarrManager, "zarrv04"
        if any(path.glob("*.lux.h5")) and any(path.glob("*.xml")):
            return LuxendoManager, "luxendo"
        if any(path.glob("*.ome")) and any(path.glob("*.tif")):
            return ViventisManager, "viventis"

    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".czi"):
        return ZeissManager, "zeiss"
    if suffixes.endswith(".vsi"):
        return EvidentManager, "evident"
    if suffixes.endswith((".ome.tif", ".ome.tiff")) or path.suffix.lower() in {".tif", ".tiff"}:
        metadata_dir = path.parent / "Metadata"
        if metadata_dir.exists() and any(metadata_dir.glob("*.xlif")):
            return ScapeManager, "scape"
        return OperaManager, "opera"
    return None
