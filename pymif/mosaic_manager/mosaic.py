from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import dask.array as da
import numpy as np

from .fields import fields_of, label_fields
from .tile import Tile

if TYPE_CHECKING:
    import napari

logger = logging.getLogger("pymif")

_COLORMAPS = ("green", "magenta", "cyan", "yellow", "red", "blue")


def single_timepoint_error(found: str) -> NotImplementedError:
    """Error (also logged as a warning) for mosaics with more than one timepoint."""
    message = (
        "Timelapse mosaics are not supported yet: mosaic loading and fusion are implemented "
        f"for a single timepoint so far. Found: {found}."
    )
    logger.warning(message)
    return NotImplementedError(message)


def _centre_block(shape: Sequence[int], block: int) -> Tuple[slice, ...]:
    """Slices of a ``block``-sized cube at the centre of ``shape``, aligned to ``block``."""
    out = []
    for n in shape:
        size = min(block, n)
        start = min(n // 2 // block * block, n - size)
        out.append(slice(start, start + size))
    return tuple(out)


def _central_tiles(tiles: Sequence[Tile], n: int) -> List[Tile]:
    """Up to ``n`` tiles, taken in turn from each group, closest to their group's centre (in y/x)."""
    ranked = []
    for g in dict.fromkeys(t.group for t in tiles):
        members = [t for t in tiles if t.group == g]
        lo = np.min([t.extent[0] for t in members], axis=0)[1:]
        hi = np.max([t.extent[1] for t in members], axis=0)[1:]
        centre = (lo + hi) / 2
        dist = [np.linalg.norm((np.add(t.extent[0], t.extent[1])[1:] / 2) - centre) for t in members]
        ranked.append([members[i] for i in np.argsort(dist, kind="stable")])
    picks = []
    for k in range(max(len(r) for r in ranked)):  # round robin over groups
        picks += [r[k] for r in ranked if k < len(r)]
    return picks[:n]


def _link_layers(groups) -> None:
    """Link contrast and gamma within every group of napari layers (if napari supports it)."""
    try:
        from napari.experimental import link_layers
    except ImportError:  # pragma: no cover - older napari
        return
    for layers in groups:
        if len(layers) > 1:
            link_layers(list(layers), ("contrast_limits", "gamma"))


class MosaicManager(ABC):
    """A set of separately acquired, positioned tiles (not yet one image).

    Unlike a :class:`~pymif.microscope_manager.MicroscopeManager`, which wraps one
    image, a mosaic exposes the raw tiles lazily and fuses them on request with
    :meth:`fuse`. Subclasses implement :meth:`_scan` only.

    Mosaics can be used as context managers so that open file handles are closed::

        with mosaic_manager.LuxendoMosaicManager("raw_folder") as mosaic:
            mosaic.fuse("overview.ome.zarr", ds=(4, 8, 8), projection="max")

    When loaded, every tile is labelled with the **field** it belongs to: tiles
    whose boxes overlap or touch form one field (a connected component), named
    ``field0``, ``field1``, ... by position (see :attr:`fields`). :meth:`fuse`
    fuses the first field by default; pick another with
    ``mosaic.fuse(out, field="field1")``. Tiles of several fields are never fused
    together: that raises a :class:`~pymif.mosaic_manager.fields.MultipleFieldsError`.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._open_files: list = []
        self.tiles: List[Tile] = self._scan()
        if not self.tiles:
            raise FileNotFoundError(f"No tiles found in {self.path}.")
        self._check_single_timepoint()
        label_fields(self.tiles)

    def _check_single_timepoint(self) -> None:
        """Refuse timelapses: loading and fusion handle one timepoint so far.

        Readers already give every tile its ``timepoint``; supporting timelapses
        means removing this check and making the fusion loop over :attr:`timepoints`.
        """
        if len(self.timepoints) > 1:
            raise single_timepoint_error(f"tiles at timepoints {self.timepoints} in {self.path}")

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
    def timepoints(self) -> List[int]:
        """Timepoints of the tiles, sorted (``[0]`` for a single timepoint)."""
        return sorted({t.timepoint for t in self.tiles})

    @property
    def groups(self) -> List[Optional[str]]:
        """Tile group labels in order of first appearance."""
        return list(dict.fromkeys(t.group for t in self.tiles))

    @property
    def fields(self) -> List[str]:
        """Field names (connected groups of touching tiles), ordered by position."""
        return fields_of(self.tiles)

    def field_tiles(self, fields: "str | Sequence[str]") -> List[Tile]:
        """Tiles of one field (``"field0"``) or of several (``["field0", "field2"]``).
        Unknown fields raise ``KeyError``."""
        names = [fields] if isinstance(fields, str) else list(fields)
        unknown = [f for f in names if f not in self.fields]
        if unknown:
            raise KeyError(f"Unknown field(s) {unknown}; available: {self.fields}.")
        wanted = set(names)
        return [t for t in self.tiles if t.field in wanted]

    def select(self, channel: Optional[str] = None, group: Optional[str] = None,
               field: Optional[str] = None, **attrs: Any) -> List[Tile]:
        """Tiles matching ``channel``, ``group``, ``field`` and any ``attrs`` entries,
        e.g. ``select(field="field1", stack=1)``."""
        out = []
        for t in self.tiles:
            if channel is not None and t.channel != channel:
                continue
            if group is not None and t.group != group:
                continue
            if field is not None and t.field != field:
                continue
            if any(t.attrs.get(k) != v for k, v in attrs.items()):
                continue
            out.append(t)
        return out

    def tile_id(self, tile: Tile) -> str:
        """Stable name of a tile, ``"<channel>/<stack>"`` (e.g. ``"cFos-647/1-x00-y00"``).

        ``<stack>`` is the reader's stack name (``attrs["stack"]``) or, without
        one, the tile index. Used to skip tiles from the command line or a batch CSV.
        """
        key = tile.attrs.get("stack")
        return f"{tile.channel}/{self.tiles.index(tile) if key is None else key}"

    def tiles_by_ids(self, ids: Sequence[str]) -> List[Tile]:
        """Tiles named by ``ids``: ``"<channel>/<stack>"`` for one tile, or ``"<stack>"``
        for that stack position in every channel. Unknown ids raise ``KeyError``."""
        by_id = {self.tile_id(t): t for t in self.tiles}
        by_stack: Dict[str, List[Tile]] = {}
        for tid, t in by_id.items():
            by_stack.setdefault(tid.split("/", 1)[1], []).append(t)
        out: List[Tile] = []
        unknown = []
        for i in ids:
            found = [by_id[i]] if i in by_id else by_stack.get(i)
            if not found:
                unknown.append(i)
                continue
            out += [t for t in found if t not in out]
        if unknown:
            raise KeyError(f"Unknown tile id(s) {unknown}; ids look like {next(iter(by_id))!r} "
                           f"(channel/stack) or {next(iter(by_stack))!r} (stack, all channels).")
        return out

    def bounding_box(self, tiles: Optional[Sequence[Tile]] = None) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        """``(min, max)`` corners, ``(z, y, x)``, of ``tiles`` (default: all)."""
        tiles = list(self.tiles if tiles is None else tiles)
        lo = np.min([t.extent[0] for t in tiles], axis=0)
        hi = np.max([t.extent[1] for t in tiles], axis=0)
        return tuple(float(v) for v in lo), tuple(float(v) for v in hi)

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

    def contrast_limits(
        self,
        tiles: Optional[Sequence[Tile]] = None,
        n_tiles: int = 6,
        block: int = 64,
        percentiles: Tuple[float, float] = (0.5, 99.9),
    ) -> Dict[str, Tuple[float, float]]:
        """Display range of every channel, estimated from a small sample.

        For each channel, one ``block``-sized cube is read from the centre of up to
        ``n_tiles`` tiles (about 0.5 MB each for uint16 and ``block=64``, the HDF5
        chunk size of Luxendo tiles), and the given percentiles of those voxels are
        returned as ``(low, high)``. The sampled tiles are the ones closest to the
        centre of each tile group (e.g. each illumination side), where the sample
        usually is; edge tiles often only show background.
        """
        tiles = list(self.tiles if tiles is None else tiles)
        limits = {}
        for channel in dict.fromkeys(t.channel for t in tiles):
            picks = _central_tiles([t for t in tiles if t.channel == channel], n_tiles)
            sample = np.concatenate([np.asarray(t.data[_centre_block(t.shape, block)]).ravel() for t in picks])
            low, high = (float(v) for v in np.percentile(sample, percentiles))
            limits[channel] = (low, high if high > low else low + 1)
        return limits

    def visualize(
        self,
        tiles: Optional[Sequence[Tile]] = None,
        viewer: "napari.Viewer | None" = None,
        contrast_limits: "str | Dict[str, Tuple[float, float]] | None" = "range",
    ):
        """Show tiles in napari, each as its own layer at its stage position.

        One colour per channel and additive blending, so overlaps and gaps are
        easy to see. Data stays lazy: only the planes napari displays are read.
        All tiles of a channel share one contrast setting, and their layers are
        linked: changing the contrast or gamma of one tile changes all of them.

        Parameters
        ----------
        tiles
            Tiles to show (default: all), e.g. ``mosaic.select(channel="cFos-647")``.
            Every displayed plane reads whole HDF5 chunks of every tile, so
            showing fewer tiles keeps napari responsive.
        viewer
            Viewer to add the layers to; a new one is created by default.
        contrast_limits
            ``"range"`` (default) uses the whole range of the data type (0 to
            65535 for uint16), which reads nothing: set the contrast of one tile
            and the linked tiles follow. ``"auto"`` estimates it per channel with
            :meth:`contrast_limits` (reads a small sample, slow on network drives);
            a ``{channel: (low, high)}`` dict uses those, with the whole range for
            the channels it leaves out; ``None`` lets napari compute them per
            tile (slow on large tiles).
        """
        import napari

        from ..utils.visualize import units_for_axes

        tiles = list(self.tiles if tiles is None else tiles)
        if contrast_limits == "auto":
            contrast_limits = self.contrast_limits(tiles)
        elif contrast_limits is not None:
            given = {} if contrast_limits == "range" else dict(contrast_limits)
            contrast_limits = {
                c: given.get(c, (float(np.iinfo(t.dtype).min), float(np.iinfo(t.dtype).max))
                             if np.issubdtype(t.dtype, np.integer) else None)
                for c, t in {t.channel: t for t in tiles}.items()
            }
        viewer = viewer or napari.Viewer()
        units = units_for_axes({"axes": "zyx", "units": (self.tiles[0].units,) * 3}, "zyx")
        extra = {} if units is None else {"units": units}
        names = self.channels
        by_channel: Dict[str, list] = {}
        for t in tiles:
            i = self.tiles.index(t)
            limits = (contrast_limits or {}).get(t.channel)
            label = t.attrs.get("stack") or i
            layer = viewer.add_image(
                self._lazy(t),
                name=f"{t.channel} {t.group or ''} [{label}]".replace("  ", " ")
                + (f" {t.field}" if len(self.fields) > 1 else ""),
                scale=t.voxel_size,
                translate=t.origin,
                colormap=_COLORMAPS[names.index(t.channel) % len(_COLORMAPS)],
                blending="additive",
                contrast_limits=limits,
                metadata={"mosaic_tile": i, "mosaic_tile_id": self.tile_id(t), "mosaic_field": t.field,
                          "mosaic_path": str(self.path)},
                **extra,
            )
            if limits is not None and np.issubdtype(t.dtype, np.integer):
                # napari sets the slider range to the limits; allow the whole dtype range
                info = np.iinfo(t.dtype)
                layer.contrast_limits_range = (info.min, info.max)
                layer.contrast_limits = limits
            by_channel.setdefault(t.channel, []).append(layer)
        _link_layers(by_channel.values())
        return viewer

    # ------------------------------------------------------------------
    # fusion
    # ------------------------------------------------------------------

    def fusion_tiles(
        self,
        tiles: Optional[Sequence["Tile | str"]] = None,
        field: "str | Sequence[str] | None" = None,
    ) -> List[Tile]:
        """The tiles a fusion uses: ``tiles`` (default: all) restricted to ``field``.

        ``tiles`` are ``Tile`` objects or ids accepted by :meth:`tiles_by_ids`;
        ``field`` is one field name or a list of them (see :meth:`field_tiles`).
        Without ``tiles`` and ``field``, the tiles of the first field. The tiles keep
        the mosaic order. Tiles of several fields are returned as asked, but
        :func:`~pymif.mosaic_manager.fuse_tiles` refuses them.
        """
        if tiles is None and field is None and self.fields:
            field = self.fields[0]
            if len(self.fields) > 1:
                logger.info("Mosaic has %d fields (%s): fusing the first one, %s.",
                            len(self.fields), ", ".join(self.fields), field)
        use = self.tiles if tiles is None else self._resolve_tiles(tiles)
        if field is not None:
            wanted = {id(t) for t in self.field_tiles(field)}
            use = [t for t in use if id(t) in wanted]
            if not use:
                raise ValueError(f"None of the given tiles is in field(s) {field!r}.")
        keep = {id(t) for t in use}
        return [t for t in self.tiles if id(t) in keep]

    def fuse(
        self,
        out: str | Path,
        tiles: Optional[Sequence["Tile | str"]] = None,
        field: "str | Sequence[str] | None" = None,
        skip: Optional[Sequence["Tile | str"]] = None,
        **kwargs,
    ):
        """Fuse tiles into an OME-Zarr image; see :func:`~pymif.mosaic_manager.fuse_tiles`.

        Parameters
        ----------
        out
            Output ``.ome.zarr`` path.
        tiles
            Only fuse these tiles, as ``Tile`` objects or ids accepted by
            :meth:`tiles_by_ids` (``"cFos-647/1-x00-y00"``, or ``"1-x00-y00"`` for every
            channel). The output fits them.
        field
            Only fuse the tiles of this field (``"field1"``); combined with ``tiles``,
            only the given tiles of that field. The output fits the field. Without
            ``tiles`` and ``field``, the first field (:attr:`fields`) is fused.
        skip
            Tiles to leave blank (same forms as ``tiles``): they are not read, but the
            output keeps the size and position it has with them.

        Tiles of more than one field are not fused: a
        :class:`~pymif.mosaic_manager.fields.MultipleFieldsError` is raised.
        """
        from .fusion import fuse_tiles

        use = self.fusion_tiles(tiles, field)
        if skip:
            skip = self._resolve_tiles(skip)
        return fuse_tiles(use, out, skip=skip, **kwargs)

    def _resolve_tiles(self, tiles) -> List[Tile]:
        """``Tile`` objects of a list mixing tiles and tile ids."""
        if isinstance(tiles, (str, Tile)):
            tiles = [tiles]
        ids = [s for s in tiles if isinstance(s, str)]
        return [s for s in tiles if not isinstance(s, str)] + (self.tiles_by_ids(ids) if ids else [])


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
