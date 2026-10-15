from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, TYPE_CHECKING

import dask.array as da
import numpy as np

from .tile import Tile

if TYPE_CHECKING:
    import napari

logger = logging.getLogger("pymif")

_COLORMAPS = ("green", "magenta", "cyan", "yellow", "red", "blue")


class MosaicManager(ABC):
    """A set of separately acquired, positioned tiles (not yet one image).

    Unlike a :class:`~pymif.microscope_manager.MicroscopeManager`, which wraps one
    image, a mosaic exposes the raw tiles lazily and fuses them on request with
    :meth:`fuse`. Subclasses implement :meth:`_scan` only.

    Mosaics can be used as context managers so that open file handles are closed::

        with mosaic_manager.LuxendoMosaicManager("raw_folder") as mosaic:
            mosaic.fuse("overview.ome.zarr", ds=(4, 8, 8), projection="max")
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._open_files: list = []
        self.tiles: List[Tile] = self._scan()
        if not self.tiles:
            raise FileNotFoundError(f"No tiles found in {self.path}.")

    @abstractmethod
    def _scan(self) -> List[Tile]:
        """Discover the tiles under :attr:`path` (lazy data, positions in the sample frame)."""

    # ------------------------------------------------------------------
    # inspection
    # ------------------------------------------------------------------

    @property
    def channels(self) -> List[str]:
        """Channel names in order of first appearance."""
        return list(dict.fromkeys(t.channel for t in self.tiles))

    @property
    def groups(self) -> List[Optional[str]]:
        """Tile group labels in order of first appearance."""
        return list(dict.fromkeys(t.group for t in self.tiles))

    def select(self, channel: Optional[str] = None, group: Optional[str] = None, **attrs: Any) -> List[Tile]:
        """Tiles matching ``channel``, ``group`` and any ``attrs`` entries, e.g. ``select(stack=1)``."""
        out = []
        for t in self.tiles:
            if channel is not None and t.channel != channel:
                continue
            if group is not None and t.group != group:
                continue
            if any(t.attrs.get(k) != v for k, v in attrs.items()):
                continue
            out.append(t)
        return out

    def bounding_box(self, tiles: Optional[Sequence[Tile]] = None) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        """``(min, max)`` corners, ``(z, y, x)``, of ``tiles`` (default: all)."""
        tiles = list(self.tiles if tiles is None else tiles)
        lo = np.min([t.extent[0] for t in tiles], axis=0)
        hi = np.max([t.extent[1] for t in tiles], axis=0)
        return tuple(lo), tuple(hi)

    def tile(self, i: int):
        """Tile ``i`` as an :class:`~pymif.microscope_manager.ArrayManager` (axes ``zyx``)."""
        from ..microscope_manager.array_manager import ArrayManager

        t = self.tiles[i]
        return ArrayManager(
            self._lazy(t),
            {
                "axes": "zyx",
                "scales": [t.voxel_size],
                "units": (t.units,) * 3,
                "name": f"{t.channel} tile {i}",
            },
        )

    @staticmethod
    def _lazy(t: Tile) -> da.Array:
        z, y, x = t.shape
        return da.from_array(t.data, chunks=(min(16, z), min(1024, y), min(1024, x)))

    def visualize(
        self,
        tiles: Optional[Sequence[Tile]] = None,
        viewer: "napari.Viewer | None" = None,
        ds: Sequence[int] = (1, 1, 1),
    ):
        """Show tiles in napari, each as its own layer at its stage position.

        One colour per channel and additive blending, so overlaps and gaps are
        easy to see. Data stays lazy: only the planes napari displays are read.

        Parameters
        ----------
        tiles
            Tiles to show (default: all), e.g. ``mosaic.select(channel="cFos-647")``.
            Every displayed plane reads whole HDF5 chunks of every tile, so
            showing fewer tiles keeps napari responsive.
        viewer
            Viewer to add the layers to; a new one is created by default.
        ds
            ``(z, y, x)`` stride for a lighter preview (every n-th voxel).
        """
        import napari

        from ..utils.visualize import units_for_axes

        viewer = viewer or napari.Viewer()
        ds = tuple(int(v) for v in ds)
        units = units_for_axes({"axes": "zyx", "units": (self.tiles[0].units,) * 3}, "zyx")
        extra = {} if units is None else {"units": units}
        names = self.channels
        for t in self.tiles if tiles is None else tiles:
            i = self.tiles.index(t)
            data = self._lazy(t)[:: ds[0], :: ds[1], :: ds[2]]
            label = t.attrs.get("stack") or i
            viewer.add_image(
                data,
                name=f"{t.channel} {t.group or ''} [{label}]".replace("  ", " "),
                scale=tuple(v * d for v, d in zip(t.voxel_size, ds)),
                translate=t.origin,
                colormap=_COLORMAPS[names.index(t.channel) % len(_COLORMAPS)],
                blending="additive",
                metadata={"mosaic_tile": i},
                **extra,
            )
        return viewer

    # ------------------------------------------------------------------
    # fusion
    # ------------------------------------------------------------------

    def fuse(self, out: str | Path, **kwargs):
        """Fuse the tiles into an OME-Zarr image; see :func:`~pymif.mosaic_manager.fuse_tiles`."""
        from .fusion import fuse_tiles

        return fuse_tiles(self.tiles, out, **kwargs)

    # ------------------------------------------------------------------
    # resources
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close all open file handles."""
        for f in self._open_files:
            try:
                f.close()
            except Exception as e:  # pragma: no cover - best effort
                logger.warning("Failed to close file: %s", e)
        self._open_files = []

    def __enter__(self) -> "MosaicManager":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
