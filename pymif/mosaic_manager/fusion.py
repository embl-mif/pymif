"""Fuse the tiles of a :class:`~pymif.mosaic_manager.MosaicManager` into one OME-Zarr image.

Fusion runs in two steps:

1. **Level 0** is computed region by region (one chunk footprint in ``y``/``x`` at
   a time, full ``z``) into a temporary zarr store: for each region only the tile
   voxels that fall in it are read, downsampled and merged. Memory stays bounded
   by the region size, no dask graph over all tiles is ever built, and every
   full-resolution voxel is read once. Reading and downsampling one tile inside
   one region is the unit of parallel work; merging and writing happen in the
   calling process.
2. The fused level 0 (already downsampled, so small compared with the raw tiles)
   is written to the final OME-Zarr with :meth:`MicroscopeManager.to_zarr`, so
   pyramid levels, chunks, sharding, compression, zarr format and singleton axes
   behave exactly as for any other PyMIF conversion (``pymif convert``).
"""
from __future__ import annotations

import dataclasses
import logging
import math
import shutil
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import dask.array as da
import numpy as np
import zarr

from ..utils.metadata import ChannelInfo, build_metadata
from ..utils.ngff import ZarrWriteConfig
from .tile import Tile

logger = logging.getLogger("pymif")

BLENDS = ("sigmoid", "max", "sum", "midline")
PROJECTIONS = (None, "max", "sum")

#: Default steepness of the ``sigmoid`` transition (see :func:`fuse_tiles`).
SIGMOID_STEEPNESS = 4.0

#: Number of full-resolution z planes read at once (bounds memory per worker).
_SLAB_PLANES = 64

#: Size (y, x) of the regions fused at once, in output voxels; a region spans the whole z range.
_REGION_3D = (256, 256)
_REGION_2D = (1024, 1024)

#: Percentiles of the fused values used as the display window written to the output.
DISPLAY_PERCENTILES = (0.5, 99.9)
#: Values kept from every fused region to estimate the display window (no extra reads).
_WINDOW_SAMPLES = 65536

#: ``to_zarr`` options that fusion sets itself.
_RESERVED_WRITE_OPTIONS = {"overwrite", "compute", "data_type"}


@dataclass
class _Placed:
    """A tile positioned on the output grid (integer offsets/sizes, ``z, y, x``)."""

    tile: Tile
    offset: Tuple[int, int, int]
    size: Tuple[int, int, int]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _place_tiles(tiles: Sequence[Tile], ds: Tuple[int, int, int]):
    """Place ``tiles`` on a common output grid.

    Returns ``(placed, grid_shape, out_voxel, origin)``. All tiles must share the
    voxel size and unit. A tile covers ``shape // ds`` output voxels starting at
    ``round((tile origin - global origin) / output voxel size)``.
    """
    vox = np.array(tiles[0].voxel_size)
    for t in tiles:
        if not np.allclose(t.voxel_size, vox, rtol=1e-3):
            raise ValueError(f"Tiles must share one voxel size, got {t.voxel_size} and {tuple(vox)}.")
        if t.units != tiles[0].units:
            raise ValueError(f"Tiles must share one unit, got {t.units!r} and {tiles[0].units!r}.")
    out_vox = vox * np.array(ds)
    origin = np.min([t.origin for t in tiles], axis=0)

    placed = []
    for t in tiles:
        offset = tuple(int(v) for v in np.rint((np.array(t.origin) - origin) / out_vox))
        size = tuple(int(n) // d for n, d in zip(t.shape, ds))
        if min(size) < 1:
            raise ValueError(f"Tile of shape {t.shape} is smaller than the downsampling factor {ds}.")
        placed.append(_Placed(t, offset, size))
    grid = tuple(int(v) for v in np.max([[o + s for o, s in zip(p.offset, p.size)] for p in placed], axis=0))
    return placed, grid, tuple(out_vox), tuple(float(v) for v in origin)


def _crop_bounds(crop, grid: Tuple[int, int, int]) -> List[Tuple[int, int]]:
    """``(start, stop)`` per ``z, y, x`` axis from a ``{"z": slice, ...}`` crop on the output grid."""
    bounds = []
    for axis, n in zip("zyx", grid):
        sel = (crop or {}).get(axis, (crop or {}).get(axis.upper(), slice(None)))
        if isinstance(sel, (int, np.integer)):
            sel = slice(int(sel), int(sel) + 1)
        if not isinstance(sel, slice):
            raise ValueError(f"crop[{axis!r}] must be a slice or an integer, got {sel!r}.")
        start, stop, step = sel.indices(n)
        if step != 1:
            raise ValueError(f"crop[{axis!r}] cannot have a step; use the downsampling factor instead.")
        if stop <= start:
            raise ValueError(f"crop[{axis!r}] is empty for an output of size {n} along {axis}.")
        bounds.append((start, stop))
    return bounds


def _downsample(block: np.ndarray, ds: Tuple[int, int, int]) -> np.ndarray:
    """Block-mean downsample a block whose shape is a multiple of ``ds``, to float32."""
    if ds == (1, 1, 1):
        return block.astype(np.float32)
    nz, ny, nx = block.shape
    r = block.reshape(nz // ds[0], ds[0], ny // ds[1], ds[1], nx // ds[2], ds[2])
    return r.mean(axis=(1, 3, 5), dtype=np.float32)


def _cast(arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Cast a float array to ``dtype``, rounding and clipping for integers."""
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(arr), info.min, info.max).astype(dtype)
    return arr.astype(dtype)


def _combine(acc: np.ndarray, cov: np.ndarray, blk: np.ndarray, blend: str) -> None:
    """Merge ``blk`` into the ``acc`` view in place (``cov`` marks voxels already written)."""
    if blend == "sum":
        acc += blk
    else:  # max / midline: running maximum within a group
        acc[...] = np.where(cov, np.maximum(acc, blk), blk)
    cov[...] = True


def _group_extents(placed: Sequence[_Placed], axis: int) -> Dict[Optional[str], Tuple[int, int]]:
    """``(start, stop)`` of every group along ``axis``, in output voxels."""
    extents: Dict[Optional[str], List[int]] = {}
    for p in placed:
        lo, hi = p.offset[axis], p.offset[axis] + p.size[axis]
        e = extents.setdefault(p.tile.group, [lo, hi])
        e[0], e[1] = min(e[0], lo), max(e[1], hi)
    return {g: (lo, hi) for g, (lo, hi) in extents.items()}


def _midline_ranges(placed: Sequence[_Placed], axis: int) -> Dict[Optional[str], Tuple[float, float]]:
    """Owner interval (output voxels, along ``axis``) of every group.

    Groups are ordered by the centre of their extent; consecutive groups are
    separated at the middle of their overlap (or of the gap between them).
    """
    ordered = sorted(_group_extents(placed, axis).items(), key=lambda kv: kv[1][0] + kv[1][1])
    cuts = [(a[1][1] + b[1][0]) / 2 for a, b in zip(ordered[:-1], ordered[1:])]
    bounds = [-math.inf] + cuts + [math.inf]
    return {g: (bounds[i], bounds[i + 1]) for i, (g, _) in enumerate(ordered)}


def _sigmoid_transitions(placed: Sequence[_Placed], axis: int) -> Dict[Optional[str], List[Tuple[int, int, int]]]:
    """Overlaps of every group with its neighbours along ``axis`` (groups ordered as for
    ``midline``), as ``(start, stop, direction)`` in output voxels: ``+1`` if the
    group's weight rises across the overlap, ``-1`` if it falls."""
    ordered = sorted(_group_extents(placed, axis).items(), key=lambda kv: kv[1][0] + kv[1][1])
    out: Dict[Optional[str], List[Tuple[int, int, int]]] = {g: [] for g, _ in ordered}
    for (ga, (alo, ahi)), (gb, (blo, bhi)) in zip(ordered[:-1], ordered[1:]):
        start, stop = max(alo, blo), min(ahi, bhi)
        if stop > start:  # groups separated by a gap never meet
            out[ga].append((start, stop, -1))
            out[gb].append((start, stop, +1))
    return out


# ---------------------------------------------------------------------------
# one tile inside one output region
# ---------------------------------------------------------------------------

@dataclass
class _Region:
    """One output region of one channel and the tiles that overlap it."""

    channel_index: int
    bounds: Tuple[int, int, int, int]  # y0, y1, x0, x1 in output voxels
    placed: List[_Placed]
    midline: Optional[tuple]
    transitions: Optional[tuple] = None  # sigmoid: (axis, _sigmoid_transitions(...))


def _overlap(p: _Placed, region, depth: int):
    """Output-grid ranges ``(k0, k1, j0, j1, i0, i1)`` where tile ``p`` meets ``region``, or ``None``."""
    y0, y1, x0, x1 = region
    oz, oy, ox = p.offset
    sz, sy, sx = p.size
    k0, k1 = max(0, oz), min(depth, oz + sz)
    j0, j1 = max(y0, oy), min(y1, oy + sy)
    i0, i1 = max(x0, ox), min(x1, ox + sx)
    if k1 <= k0 or j1 <= j0 or i1 <= i0:
        return None
    return k0, k1, j0, j1, i0, i1


def _target(p: _Placed, region, depth: int, flat: bool):
    """Canvas slices of ``region`` covered by tile ``p`` (``None`` if they do not overlap)."""
    ov = _overlap(p, region, depth)
    if ov is None:
        return None
    k0, k1, j0, j1, i0, i1 = ov
    yx = (slice(j0 - region[0], j1 - region[0]), slice(i0 - region[2], i1 - region[2]))
    return yx if flat else (slice(k0, k1),) + yx


def _read_block(job) -> np.ndarray:
    """Read and downsample the part of one tile inside one region (runs in workers).

    Returns float32 ``(h, w)`` for projections, ``(d, h, w)`` otherwise.
    """
    p, region, depth, ds, projection = job
    k0, k1, j0, j1, i0, i1 = _overlap(p, region, depth)
    oz, oy, ox = p.offset
    dz, dy, dx = ds
    src_y = slice((j0 - oy) * dy, (j1 - oy) * dy)
    src_x = slice((i0 - ox) * dx, (i1 - ox) * dx)
    data = p.tile.data

    if projection is not None:  # reduce over z at full resolution first, then downsample in yx
        if k0 == oz and k1 == oz + p.size[0]:
            z_start, z_stop = 0, p.tile.shape[0]  # whole tile, including trailing planes
        else:  # z crop
            z_start, z_stop = (k0 - oz) * dz, (k1 - oz) * dz
        reduced = None
        for z in range(z_start, z_stop, _SLAB_PLANES):
            raw = np.asarray(data[z : min(z + _SLAB_PLANES, z_stop), src_y, src_x])
            if projection == "max":
                part = raw.max(axis=0)
                reduced = part if reduced is None else np.maximum(reduced, part)
            else:
                part = raw.sum(axis=0, dtype=np.float32)
                reduced = part if reduced is None else reduced + part
        return _downsample(reduced[None], (1, dy, dx))[0]

    out = np.empty((k1 - k0, j1 - j0, i1 - i0), np.float32)
    step = max(1, _SLAB_PLANES // dz)
    for a in range(k0, k1, step):
        b = min(a + step, k1)
        raw = np.asarray(data[(a - oz) * dz : (b - oz) * dz, src_y, src_x])
        out[a - k0 : b - k0] = _downsample(raw, ds)
    return out


def _merge_groups(canvases, midline, region, flat: bool, shape) -> np.ndarray:
    """Merge per-group canvases: each group wins on its own side of the cut, others fill gaps."""
    if len(canvases) == 1:
        return next(iter(canvases.values()))[0]
    y0, _, x0, _ = region
    axis, ranges = midline  # axis is an index into (z, y, x)
    start = (y0, x0)[axis - 1] if axis > 0 else 0
    arr_axis = (axis - 1) if flat else axis  # canvas axis of the midline axis
    length = shape[arr_axis]
    pos = start + np.arange(length) + 0.5
    bshape = [1] * len(shape)
    bshape[arr_axis] = length

    result = np.zeros(shape, np.float32)
    filled = np.zeros(shape, bool)
    for g, (lo, hi) in ranges.items():
        if g not in canvases:
            continue
        acc, cov = canvases[g]
        m = ((pos >= lo) & (pos < hi)).reshape(bshape) & cov
        result[m] = acc[m]
        filled |= m
    for acc, cov in canvases.values():
        m = ~filled & cov
        result[m] = acc[m]
        filled |= m
    return result


# Sigmoid blending: every voxel is a weighted mean of the tiles that cover it. A
# tile's weight is d**steepness, with d the distance from the voxel to the tile's
# border (multiplied over the axes), so where two tiles overlap the share of one
# is 1 / (1 + (d_other / d_self)**steepness) = sigmoid(steepness * ln(d_self / d_other)):
# 0 at its border, 1/2 in the middle of the overlap and 1 at the other tile's
# border, whatever the overlap width. Groups are then merged the same way across
# each overlap with their neighbour along the midline axis, so the switch happens
# where ``midline`` cuts, but smoothly. Weights are kept as logarithms and summed
# with a running maximum (a streaming softmax), so tiny weights never underflow.

def _log_border_distance(start: int, size: int, lo: int, hi: int) -> np.ndarray:
    """ln of the distance from the centres of output voxels ``lo..hi`` to the nearer
    border of the interval ``[start, start + size)`` (all inside it)."""
    pos = np.arange(lo, hi, dtype=np.float64) + 0.5 - start
    return np.log(np.minimum(pos, size - pos))


def _tile_log_weight(p: _Placed, region, depth: int, flat: bool, steepness: float) -> np.ndarray:
    """Log weight of tile ``p`` on its part of ``region`` (shape of :func:`_read_block`)."""
    k0, k1, j0, j1, i0, i1 = _overlap(p, region, depth)
    ly = _log_border_distance(p.offset[1], p.size[1], j0, j1)[:, None]
    lx = _log_border_distance(p.offset[2], p.size[2], i0, i1)[None, :]
    lw = ly + lx
    if not flat:
        lw = _log_border_distance(p.offset[0], p.size[0], k0, k1)[:, None, None] + lw
    return (steepness * lw).astype(np.float32)


def _softmax_add(state, sel, values: np.ndarray, log_weight: np.ndarray) -> None:
    """Add ``values`` with weights ``exp(log_weight)`` to the running weighted sum
    ``state = (acc, wsum, log_max)`` at ``sel`` (slices or a boolean mask), in place."""
    acc, wsum, log_max = state
    old = log_max[sel]
    new = np.maximum(old, log_weight)
    rescale = np.exp(old - new)  # 0 where nothing was added yet (old = -inf)
    w = np.exp(log_weight - new)
    acc[sel] = acc[sel] * rescale + w * values
    wsum[sel] = wsum[sel] * rescale + w
    log_max[sel] = new


def _new_softmax_state(shape):
    return np.zeros(shape, np.float32), np.zeros(shape, np.float32), np.full(shape, -np.inf, np.float32)


def _group_log_weight(transitions, pos: np.ndarray, steepness: float) -> np.ndarray:
    """Log weight of a group at voxel centres ``pos`` along the midline axis: across
    each overlap with a neighbour, ``steepness * ln`` of the relative distance from the
    side where the group comes in; 0 (full weight) outside its overlaps."""
    lw = np.zeros(pos.shape, np.float64)
    for start, stop, direction in transitions:
        d = pos - start if direction > 0 else stop - pos
        lw += np.log(np.clip(d, 0.5, stop - start - 0.5) / (stop - start - 0.5))
    return steepness * lw


def _merge_sigmoid(canvases, transitions, region, flat: bool, shape, steepness: float) -> np.ndarray:
    """Weighted mean of every group, then of the groups across their overlaps."""
    means = {}
    for g, (acc, wsum, _) in canvases.items():
        cov = wsum > 0
        mean = np.zeros(shape, np.float32)
        np.divide(acc, wsum, out=mean, where=cov)
        means[g] = (mean, cov)
    if len(means) == 1:
        return next(iter(means.values()))[0]

    y0, _, x0, _ = region
    axis, per_group = transitions  # axis is an index into (z, y, x)
    start = (y0, x0)[axis - 1] if axis > 0 else 0
    arr_axis = (axis - 1) if flat else axis
    length = shape[arr_axis]
    pos = start + np.arange(length) + 0.5
    bshape = [1] * len(shape)
    bshape[arr_axis] = length
    state = _new_softmax_state(shape)
    for g, (mean, cov) in means.items():
        lw = _group_log_weight(per_group[g], pos, steepness).astype(np.float32).reshape(bshape)
        _softmax_add(state, cov, mean[cov], np.broadcast_to(lw, shape)[cov])
    acc, wsum, _ = state
    result = np.zeros(shape, np.float32)
    np.divide(acc, wsum, out=result, where=wsum > 0)
    return result


def _finish_region(r: _Region, entries, arr, depth: int, blend, dtype, flat: bool,
                   steepness: float = SIGMOID_STEEPNESS) -> np.ndarray:
    """Merge the tile blocks of region ``r`` (futures or pending jobs), write it and
    return a small sample of its non-zero values (for the display window)."""
    y0, y1, x0, x1 = r.bounds
    shape = (y1 - y0, x1 - x0) if flat else (depth, y1 - y0, x1 - x0)
    sigmoid = blend == "sigmoid"
    canvases: Dict[Optional[str], tuple] = {}
    for p, pending in entries:
        key = p.tile.group if blend in ("midline", "sigmoid") else None
        if key not in canvases:
            canvases[key] = (_new_softmax_state(shape) if sigmoid
                             else (np.zeros(shape, np.float32), np.zeros(shape, bool)))
        sl = _target(p, r.bounds, depth, flat)
        blk = pending.result() if isinstance(pending, Future) else _read_block(pending)
        if sigmoid:
            _softmax_add(canvases[key], sl, blk, _tile_log_weight(p, r.bounds, depth, flat, steepness))
        else:
            acc, cov = canvases[key]
            _combine(acc[sl], cov[sl], blk, blend)

    if sigmoid:
        merged = _merge_sigmoid(canvases, r.transitions, r.bounds, flat, shape, steepness)
    else:
        merged = _merge_groups(canvases, r.midline, r.bounds, flat, shape)
    block = _cast(merged, dtype)
    if flat:
        arr[r.channel_index, y0:y1, x0:x1] = block
    else:
        arr[r.channel_index, :, y0:y1, x0:x1] = block
    values = block.ravel()[:: max(1, block.size // _WINDOW_SAMPLES)]
    return values[values != 0]


def _fuse_level0(regions, arr, *, depth, ds, projection, blend, dtype, steepness: float,
                 workers: int, executor: str, progress: bool) -> Dict[int, np.ndarray]:
    """Fuse and write every region of level 0; return a sample of values per channel index.

    The unit of parallel work is one tile inside one region: workers only read and
    downsample, the calling process merges and writes, so the store is never
    written concurrently. At most ``2 * workers`` regions are open at a time,
    which bounds memory.
    """
    from tqdm.auto import tqdm

    flat = projection is not None
    pool = None
    if workers > 1:
        pool = (ProcessPoolExecutor if executor == "process" else ThreadPoolExecutor)(max_workers=workers)
    window = 2 * workers if pool is not None else 0
    samples: Dict[int, List[np.ndarray]] = {}

    def finish(r, entries):
        samples.setdefault(r.channel_index, []).append(
            _finish_region(r, entries, arr, depth, blend, dtype, flat, steepness))
        bar.update(1)

    try:
        with tqdm(total=len(regions), desc="fusing regions", unit="region", disable=not progress) as bar:
            open_regions = deque()
            for r in regions:
                entries = []
                for p in r.placed:
                    job = (p, r.bounds, depth, ds, projection)
                    entries.append((p, pool.submit(_read_block, job) if pool is not None else job))
                open_regions.append((r, entries))
                while len(open_regions) > window:
                    finish(*open_regions.popleft())
            while open_regions:
                finish(*open_regions.popleft())
    finally:
        if pool is not None:
            pool.shutdown(cancel_futures=True)
    return {c: np.concatenate(v) for c, v in samples.items()}


def _display_windows(samples: Dict[int, np.ndarray], n_channels: int, dtype) -> List[Tuple[float, float]]:
    """``(start, end)`` display window per channel from the fused-value samples."""
    windows = []
    for c in range(n_channels):
        values = samples.get(c)
        if values is None or values.size == 0:
            windows.append((0.0, 1.0))
            continue
        low, high = (float(v) for v in np.percentile(values, DISPLAY_PERCENTILES))
        windows.append((low, high if high > low else low + 1.0))
    return windows


def _write_display_windows(path, windows: Sequence[Tuple[float, float]]) -> None:
    """Store the display windows in the OMERO channel metadata (NGFF 0.4 and 0.5)."""
    root = zarr.open_group(str(path), mode="r+")
    attrs = root.attrs.asdict()
    nested = "ome" in attrs  # NGFF 0.5 keeps everything under "ome"
    omero = (attrs["ome"] if nested else attrs).get("omero")
    if not omero:
        return
    for channel, (start, end) in zip(omero.get("channels", []), windows):
        channel.setdefault("window", {}).update({"start": start, "end": end})
    root.attrs.update({"ome": attrs["ome"]} if nested else {"omero": omero})


def display_windows(path) -> List[Tuple[float, float]]:
    """Display window ``(start, end)`` of every channel of an OME-Zarr image, from its
    OMERO metadata (written by :func:`fuse_tiles` from the fused values)."""
    attrs = zarr.open_group(str(path), mode="r").attrs.asdict()
    omero = attrs.get("ome", attrs).get("omero") or {}
    return [(float(c["window"]["start"]), float(c["window"]["end"]))
            for c in omero.get("channels", []) if "window" in c]


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------

@dataclass
class FusionPlan:
    """Level-0 geometry of a fusion, computed without reading any pixel data."""

    tiles: List[Tile]  # tiles that are read (skipped tiles excluded)
    placed: Dict[str, List[_Placed]]  # per output channel, offsets relative to the output grid
    layout: Dict[str, List[_Placed]]  # like ``placed`` but including skipped tiles
    channels: List[str]
    ds: Tuple[int, int, int]
    blend: str
    dtype: np.dtype
    axes: str  # "czyx" or "cyx"
    grid_shape: Tuple[int, int, int]  # (z, y, x) of the 3D output grid
    shape: Tuple[int, ...]  # level 0, in ``axes`` order
    scale: Tuple[float, ...]  # level-0 voxel size of the spatial axes
    origin: Tuple[float, float, float]  # sample position of output voxel [0, 0, 0]
    n_skipped: int = 0  # tiles left blank

    @property
    def nbytes(self) -> int:
        """Size of level 0 in bytes."""
        return math.prod(self.shape) * self.dtype.itemsize

    def summary(self) -> str:
        """One line such as ``2 ch x 260 x 1100 x 1500 px (czyx), 1.71 GB, voxel 12 x 11.7 x 11.7 micrometer`` (MB below 0.1 GB)."""
        dims = " x ".join(str(v) for v in self.shape[1:])
        vox = " x ".join(f"{v:.3g}" for v in self.scale)
        size = f"{self.nbytes / 1e9:.2f} GB" if self.nbytes >= 1e8 else f"{self.nbytes / 1e6:.1f} MB"
        blank = f", {self.n_skipped} tile(s) left blank" if self.n_skipped else ""
        return f"{len(self.channels)} ch x {dims} px ({self.axes}), {size}, voxel {vox} {self.tiles[0].units}{blank}"


def plan_fusion(
    tiles: Sequence[Tile],
    *,
    ds: Sequence[int] = (1, 1, 1),
    projection: Optional[str] = None,
    blend: str = "sigmoid",
    channels: Optional[Sequence[str]] = None,
    crop: Optional[dict] = None,
    skip: Optional[Sequence[Tile]] = None,
    out_dtype=None,
    midline_axis: str = "x",
) -> FusionPlan:
    """Validate fusion arguments and compute the level-0 geometry (see :func:`fuse_tiles`)."""
    tiles = list(tiles)
    if not tiles:
        raise ValueError("No tiles to fuse.")
    if blend not in BLENDS:
        raise ValueError(f"blend must be one of {BLENDS}, got {blend!r}.")
    if projection not in PROJECTIONS:
        raise ValueError(f"projection must be one of {PROJECTIONS}, got {projection!r}.")
    ds = tuple(int(v) for v in ds)
    if len(ds) != 3 or min(ds) < 1:
        raise ValueError(f"ds must be three positive integers (z, y, x), got {ds}.")
    if midline_axis not in ("x", "y", "z"):
        raise ValueError("midline_axis must be 'x', 'y' or 'z'.")
    flat = projection is not None
    if flat and midline_axis == "z":
        raise ValueError("midline_axis='z' is not available for projections.")

    names = list(dict.fromkeys(t.channel for t in tiles))
    if channels is not None:
        missing = [c for c in channels if c not in names]
        if missing:
            raise ValueError(f"Channels {missing} not found; available: {names}.")
        names = list(channels)
        if not names:
            raise ValueError("No channel selected.")
    tiles = [t for n in names for t in tiles if t.channel == n]

    placed, grid, out_vox, origin = _place_tiles(tiles, ds)
    if crop:  # shift the grid to the crop and drop the tiles outside it
        bounds = _crop_bounds(crop, grid)
        start = tuple(b[0] for b in bounds)
        grid = tuple(b[1] - b[0] for b in bounds)
        origin = tuple(o + s * v for o, s, v in zip(origin, start, out_vox))
        placed = [_Placed(p.tile, tuple(o - s for o, s in zip(p.offset, start)), p.size) for p in placed]
        placed = [p for p in placed if _overlap(p, (0, grid[1], 0, grid[2]), grid[0]) is not None]
        if not placed:
            raise ValueError("No tile intersects the crop.")
    layout = {n: [p for p in placed if p.tile.channel == n] for n in names}
    # skipped tiles keep their place in the geometry (the output does not shrink) but are not read
    skipped = {id(t) for t in skip or ()}
    n_skipped = sum(id(p.tile) in skipped for p in placed)
    placed = [p for p in placed if id(p.tile) not in skipped]
    if not placed:
        raise ValueError("Every tile is skipped: nothing to fuse.")
    placed_by_channel = {n: [p for p in placed if p.tile.channel == n] for n in names}

    if blend == "midline" and all(t.group is None for t in tiles):
        logger.warning("blend='midline' but tiles have no group; falling back to 'max'.")
        blend = "max"

    if out_dtype is None:
        out_dtype = np.float32 if blend == "sum" else np.result_type(*{t.dtype for t in tiles})
    out_dtype = np.dtype(out_dtype)

    spatial = grid[1:] if flat else grid
    return FusionPlan(
        tiles=[p.tile for p in placed],
        placed=placed_by_channel,
        layout=layout,
        channels=names,
        ds=ds,
        blend=blend,
        dtype=out_dtype,
        axes="cyx" if flat else "czyx",
        grid_shape=grid,
        shape=(len(names),) + tuple(spatial),
        scale=tuple(out_vox[1:] if flat else out_vox),
        origin=origin,
        n_skipped=n_skipped,
    )


def _spatial_factor(value, flat: bool):
    """Pyramid downscale factor for the output axes (``(z, y, x)`` given; z dropped for projections)."""
    if value is None:
        return 2
    if isinstance(value, (list, tuple)):
        value = tuple(int(v) for v in value)
        if len(value) == 1:
            return value[0]
        if flat and len(value) == 3:
            return value[1:]
    return value


def _check_write_options(options: dict) -> None:
    fields = {f.name for f in dataclasses.fields(ZarrWriteConfig)}
    unknown = set(options) - fields
    if unknown:
        raise TypeError(f"Unknown OME-Zarr option(s) {sorted(unknown)}; valid ones are {sorted(fields - _RESERVED_WRITE_OPTIONS)}.")
    reserved = set(options) & _RESERVED_WRITE_OPTIONS
    if reserved:
        raise TypeError(f"Option(s) {sorted(reserved)} are set by fuse_tiles and cannot be passed.")


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def fuse_tiles(
    tiles: Sequence[Tile],
    out: str | Path,
    *,
    ds: Sequence[int] = (1, 1, 1),
    projection: Optional[str] = None,
    blend: str = "sigmoid",
    channels: Optional[Sequence[str]] = None,
    channel_names: Optional[Sequence[str]] = None,
    channel_colors: Optional[Sequence[str]] = None,
    crop: Optional[dict] = None,
    skip: Optional[Sequence[Tile]] = None,
    levels: int = 1,
    downscale_factor=2,
    workers: int = 1,
    executor: str = "auto",
    out_dtype=None,
    midline_axis: str = "x",
    sigmoid_steepness: float = SIGMOID_STEEPNESS,
    overwrite: bool = False,
    name: Optional[str] = None,
    temp_dir: str | Path | None = None,
    progress: bool = True,
    **write_options,
):
    """Fuse ``tiles`` into an OME-Zarr image and return it as a ``ZarrManager``.

    Parameters
    ----------
    tiles
        Tiles to fuse (all must share voxel size and unit).
    out
        Output ``.ome.zarr`` path.
    ds
        ``(z, y, x)`` integer downsampling factors applied while reading the tiles
        (each output voxel is the mean of its block).
    projection
        ``None`` for a 3D volume, ``"max"``/``"sum"`` for a 2D projection along z.
    blend
        How tiles that overlap are merged:

        * ``"sigmoid"`` (default) -- smooth transition: across an overlap, the
          share of each tile follows a sigmoid of the log ratio of the distances
          to the two tile borders, from 0 at a tile's border through 1/2 in the
          middle of the overlap to 1 at the other tile's border, so no seam is
          visible. The tiles of each ``group`` are fused separately, then the
          groups are merged the same way across their overlap along
          ``midline_axis`` (the switch is centred where ``"midline"`` cuts);
        * ``"max"`` -- keep the brighter value;
        * ``"sum"`` -- add the values, so overlaps show up as brighter bands;
        * ``"midline"`` -- ``max`` within each tile ``group``; where groups
          overlap, the image switches sharply from one group to the next at the
          middle of their overlap, along ``midline_axis``.
    channels
        Tile channels to fuse, in output order. Default: all, in tile order.
    channel_names, channel_colors
        Names and colours (hex or matplotlib names) written for the output
        channels. Default: the tile channel names and the PyMIF palette.
    crop
        Part of the output to write, in output voxels (after ``ds``), e.g.
        ``{"y": slice(100, 900), "z": slice(0, 50)}``. For projections, a ``z``
        crop limits the planes that are projected.
    skip
        Tiles to leave blank: they are not read, but the output keeps the size and
        position it has with them, so their area stays 0 (where no other tile
        covers it). With ``blend="midline"``/``"sigmoid"`` they still count for
        the position of the switch between groups.
    levels
        Number of pyramid levels.
    downscale_factor
        Factor between pyramid levels: one integer or ``(z, y, x)``.
    workers
        Number of tiles read in parallel.
    executor
        ``"process"``, ``"thread"`` or ``"auto"`` (processes when ``workers > 1`` and
        every tile's data is picklable, e.g. :class:`~pymif.mosaic_manager.H5Array`;
        threads otherwise). h5py serialises reads inside one process, so processes
        are what actually speeds up reading from HDF5 tiles.
    out_dtype
        Output dtype. Default: input dtype for ``sigmoid``/``max``/``midline``,
        ``float32`` for ``sum``.
    midline_axis
        ``"x"``, ``"y"`` or ``"z"``: axis along which groups are separated
        (``midline`` and ``sigmoid``).
    sigmoid_steepness
        Steepness of the ``sigmoid`` transition (> 0). The share of a tile
        goes from 10% to 90% over the central ``1 - 2 / (1 + 9 ** (1 / s))``
        of the overlap: about 27% for the default 4, 14% for 8, 50% for 2.
    overwrite
        Replace ``out`` if it exists.
    name
        Image name stored in the metadata.
    temp_dir
        Folder for the temporary fused level 0 (deleted at the end). Default:
        next to ``out``. A local disk is faster when ``out`` is on a network share.
    progress
        Show progress bars.
    **write_options
        Any other :meth:`~pymif.microscope_manager.MicroscopeManager.to_zarr`
        option, as for ``pymif convert``: ``chunks``, ``shards``, ``shard_target_mb``,
        ``shard_exclude_axes``, ``zarr_format``, ``ngff_version``, ``compressor``,
        ``compressor_level``, ``drop_singleton``, ``num_workers``, ``storage_options``.
        ``chunks`` is given in the output axis order (``c, z, y, x``, or ``c, y, x``
        for projections).

    The display window of every channel (OMERO ``window`` metadata) is set to the
    0.5th-99.9th percentile of the fused values, ignoring blank areas, so viewers
    open the result with a usable contrast; see :func:`display_windows`.

    Use :func:`plan_fusion` with the same arguments to get the output shape and
    size without reading or writing anything.
    """
    from ..microscope_manager.array_manager import ArrayManager
    from ..microscope_manager.zarr_manager import ZarrManager

    if executor not in ("auto", "process", "thread"):
        raise ValueError("executor must be 'auto', 'process' or 'thread'.")
    if not sigmoid_steepness > 0:
        raise ValueError(f"sigmoid_steepness must be positive, got {sigmoid_steepness!r}.")
    _check_write_options(write_options)
    plan = plan_fusion(tiles, ds=ds, projection=projection, blend=blend, channels=channels,
                       crop=crop, skip=skip, out_dtype=out_dtype, midline_axis=midline_axis)
    blend, out_dtype, names, ds = plan.blend, plan.dtype, plan.channels, plan.ds
    flat = projection is not None
    n_ch = len(names)
    for label, values in (("channel_names", channel_names), ("channel_colors", channel_colors)):
        if values is not None and len(values) != n_ch:
            raise ValueError(f"{label} has {len(values)} entries but {n_ch} channels are fused ({names}).")

    chunks = write_options.get("chunks")
    if chunks is None:
        chunks = (1, 512, 512) if flat else (1, 64, 256, 256)
    chunks = tuple(int(c) for c in chunks)
    if len(chunks) != len(plan.axes):
        raise ValueError(f"chunks must have {len(plan.axes)} values for axes {plan.axes!r}, got {chunks}.")
    chunks = tuple(max(1, min(c, s)) for c, s in zip(chunks, plan.shape))
    logger.info("Fusing %d tiles: %s, blend=%s, projection=%s", len(plan.tiles), plan.summary(), blend, projection)

    out = Path(out)
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"{out} exists; pass overwrite=True to replace it.")
        shutil.rmtree(out)
    tmp = Path(temp_dir or out.parent) / f".{out.name}.level0.zarr"
    if tmp.exists():
        shutil.rmtree(tmp)

    try:
        # ----- 1. level 0, region by region, into a plain temporary store -----
        cy, cx = (min(r, n) for r, n in zip(_REGION_2D if flat else _REGION_3D, plan.shape[-2:]))
        level0 = zarr.open_array(str(tmp), mode="w", shape=plan.shape, dtype=out_dtype,
                                 chunks=(1,) + (plan.shape[1:2] if not flat else ()) + (cy, cx),
                                 fill_value=0)
        depth = plan.grid_shape[0]
        axis = "zyx".index(midline_axis)
        regions = []
        for ci, n in enumerate(names):
            placed = plan.placed[n]
            midline = (axis, _midline_ranges(plan.layout[n], axis)) if blend == "midline" else None
            transitions = (axis, _sigmoid_transitions(plan.layout[n], axis)) if blend == "sigmoid" else None
            for y0 in range(0, plan.shape[-2], cy):
                for x0 in range(0, plan.shape[-1], cx):
                    bounds = (y0, min(y0 + cy, plan.shape[-2]), x0, min(x0 + cx, plan.shape[-1]))
                    inside = [p for p in placed if _overlap(p, bounds, depth) is not None]
                    if inside:  # regions without tiles keep the fill value (0)
                        regions.append(_Region(ci, bounds, inside, midline, transitions))

        auto = executor == "auto"
        if auto:
            picklable = all(getattr(t.data, "picklable", False) for t in plan.tiles)
            executor = "process" if workers > 1 and picklable else "thread"
        options = dict(depth=depth, ds=ds, projection=projection, blend=blend, dtype=out_dtype,
                       steepness=float(sigmoid_steepness), workers=workers, progress=progress)
        try:
            samples = _fuse_level0(regions, level0, executor=executor, **options)
        except (BrokenProcessPool, RuntimeError) as e:
            if not (auto and executor == "process"):
                raise
            # e.g. worker processes cannot be spawned from this interpreter (some GUI
            # launchers); every region is rewritten as a whole, so redoing them is safe.
            logger.warning("Worker processes failed (%s); retrying with threads.", e)
            samples = _fuse_level0(regions, level0, executor="thread", **options)
        windows = _display_windows(samples, n_ch, out_dtype)

        # ----- 2. pyramid + OME-Zarr through the standard PyMIF writer -----
        metadata = build_metadata(
            size=[plan.shape],
            scales=[plan.scale],
            units=(plan.tiles[0].units,) * len(plan.scale),
            channels=[ChannelInfo(
                (channel_names or names)[i],
                None if channel_colors is None else channel_colors[i],
            ) for i in range(n_ch)],
            dtype=str(out_dtype),
            axes=plan.axes,
            chunksize=[chunks],
            name=name or "fused mosaic",
            data_type="intensity",
        )
        dataset = ArrayManager(da.from_zarr(str(tmp)).rechunk(chunks), metadata)
        dataset.build_pyramid(num_levels=max(1, int(levels)),
                              downscale_factor=_spatial_factor(downscale_factor, flat))
        dataset.to_zarr(str(out), overwrite=True, progress=progress, **write_options)
        # Sensible display range for viewers (the writer's default is the whole dtype range).
        _write_display_windows(out, windows)
        logger.info("Display window per channel: %s", windows)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return ZarrManager(str(out), mode="r")
