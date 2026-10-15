"""Fuse the tiles of a :class:`~pymif.mosaic_manager.Mosaic` into one OME-Zarr image.

The output is computed region by region (one output chunk footprint in ``y``/``x``
at a time, full ``z``): for each region only the tile voxels that fall in it are
read, downsampled, merged and written. Memory therefore stays bounded by the
region size and no dask graph over all tiles is ever built; every full-resolution
voxel is read once.
"""
from __future__ import annotations

import logging
import math
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import dask.array as da
import numpy as np
import zarr

from ..utils.metadata import ChannelInfo, build_metadata
from .tile import Tile

logger = logging.getLogger("pymif")

BLENDS = ("max", "sum", "count", "midline")
PROJECTIONS = (None, "max", "sum")
DOWNSAMPLINGS = ("mean", "stride")

#: Number of full-resolution z planes read at once (bounds memory per worker).
_SLAB_PLANES = 32


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

    Returns ``(placed, out_shape, out_voxel, origin)``. All tiles must share the
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
    out_shape = tuple(int(v) for v in np.max([[o + s for o, s in zip(p.offset, p.size)] for p in placed], axis=0))
    return placed, out_shape, tuple(out_vox), tuple(origin)


def _downsample(block: np.ndarray, ds: Tuple[int, int, int], how: str) -> np.ndarray:
    """Downsample a block whose shape is a multiple of ``ds`` to float32."""
    if how == "stride":
        return block[:: ds[0], :: ds[1], :: ds[2]].astype(np.float32)
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


def _midline_ranges(placed: Sequence[_Placed], axis: int) -> Dict[Optional[str], Tuple[float, float]]:
    """Owner interval (output voxels, along ``axis``) of every group.

    Groups are ordered by the centre of their extent; consecutive groups are
    separated at the middle of their overlap (or of the gap between them).
    """
    extents: Dict[Optional[str], List[int]] = {}
    for p in placed:
        lo, hi = p.offset[axis], p.offset[axis] + p.size[axis]
        e = extents.setdefault(p.tile.group, [lo, hi])
        e[0], e[1] = min(e[0], lo), max(e[1], hi)
    ordered = sorted(extents.items(), key=lambda kv: kv[1][0] + kv[1][1])
    cuts = [(a[1][1] + b[1][0]) / 2 for a, b in zip(ordered[:-1], ordered[1:])]
    bounds = [-math.inf] + cuts + [math.inf]
    return {g: (bounds[i], bounds[i + 1]) for i, (g, _) in enumerate(ordered)}


# ---------------------------------------------------------------------------
# one output region
# ---------------------------------------------------------------------------

def _add_tile(canvas, cov, p: _Placed, region, out_shape, ds, projection, blend, downsample):
    """Read the part of tile ``p`` inside ``region`` and merge it into ``canvas``."""
    y0, y1, x0, x1 = region
    oz, oy, ox = p.offset
    sz, sy, sx = p.size
    j0, j1 = max(y0, oy), min(y1, oy + sy)
    i0, i1 = max(x0, ox), min(x1, ox + sx)
    if j1 <= j0 or i1 <= i0:
        return
    dz, dy, dx = ds
    flat = projection is not None
    cy, cx = slice(j0 - y0, j1 - y0), slice(i0 - x0, i1 - x0)

    if blend == "count":  # coverage only: no I/O
        if flat:
            canvas[cy, cx] += 1
        else:
            canvas[oz : oz + sz, cy, cx] += 1
        return

    src_y = slice((j0 - oy) * dy, (j1 - oy) * dy)
    src_x = slice((i0 - ox) * dx, (i1 - ox) * dx)
    data = p.tile.data

    if flat:
        reduced = None
        nplanes = p.tile.shape[0]
        for k0 in range(0, nplanes, _SLAB_PLANES):
            raw = np.asarray(data[k0 : min(k0 + _SLAB_PLANES, nplanes), src_y, src_x])
            blk = _downsample(raw, (1, dy, dx), downsample)
            part = blk.max(axis=0) if projection == "max" else blk.sum(axis=0)
            if reduced is None:
                reduced = part
            elif projection == "max":
                np.maximum(reduced, part, out=reduced)
            else:
                reduced += part
        _combine(canvas[cy, cx], cov[cy, cx], reduced, blend)
        return

    step = max(1, _SLAB_PLANES // dz)
    for k0 in range(0, sz, step):
        k1 = min(k0 + step, sz)
        raw = np.asarray(data[k0 * dz : k1 * dz, src_y, src_x])
        blk = _downsample(raw, ds, downsample)
        zs = slice(oz + k0, oz + k1)
        _combine(canvas[zs, cy, cx], cov[zs, cy, cx], blk, blend)


def _fuse_region(placed, region, out_shape, ds, projection, blend, downsample, dtype, midline):
    """Fuse all ``placed`` tiles (of one channel) inside ``region = (y0, y1, x0, x1)``."""
    y0, y1, x0, x1 = region
    flat = projection is not None
    shape = (y1 - y0, x1 - x0) if flat else (out_shape[0], y1 - y0, x1 - x0)
    acc_dtype = np.uint32 if blend == "count" else np.float32

    canvases: Dict[Optional[str], Tuple[np.ndarray, np.ndarray]] = {}
    for p in placed:
        key = p.tile.group if blend == "midline" else None
        if key not in canvases:
            canvases[key] = (np.zeros(shape, acc_dtype), np.zeros(shape, bool))
        acc, cov = canvases[key]
        _add_tile(acc, cov, p, region, out_shape, ds, projection, blend, downsample)

    if len(canvases) == 1:
        result = next(iter(canvases.values()))[0]
    else:  # midline: each group wins on its own side of the cut, others fill the gaps
        axis, ranges = midline  # axis is an index into (z, y, x)
        start = (y0, x0)[axis - 1] if axis > 0 else 0
        length = shape[-1] if axis == 2 else shape[-2] if axis == 1 else shape[0]
        pos = start + np.arange(length) + 0.5
        arr_axis = (axis - 1) if flat else axis  # canvas axis of the midline axis
        bshape = [1] * len(shape)
        bshape[arr_axis] = length

        result = np.zeros(shape, np.float32)
        filled = np.zeros(shape, bool)
        for g, (lo, hi) in ranges.items():
            if g not in canvases:
                continue
            acc, cov = canvases[g]
            owner = ((pos >= lo) & (pos < hi)).reshape(bshape)
            m = owner & cov
            result[m] = acc[m]
            filled |= m
        for acc, cov in canvases.values():
            m = ~filled & cov
            result[m] = acc[m]
            filled |= m

    return _cast(result, dtype)


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def fuse_tiles(
    tiles: Sequence[Tile],
    out: str | Path,
    *,
    ds: Sequence[int] = (1, 1, 1),
    projection: Optional[str] = None,
    blend: str = "max",
    downsample: str = "mean",
    channels: Optional[Sequence[str]] = None,
    workers: int = 1,
    levels: int = 1,
    chunks: Optional[Sequence[int]] = None,
    out_dtype=None,
    midline_axis: str = "x",
    overwrite: bool = False,
    name: Optional[str] = None,
):
    """Fuse ``tiles`` into an OME-Zarr image and return it as a ``ZarrManager``.

    Parameters
    ----------
    tiles
        Tiles to fuse (all must share voxel size and unit).
    out
        Output ``.ome.zarr`` path.
    ds
        ``(z, y, x)`` integer downsampling factors applied while reading.
    projection
        ``None`` for a 3D volume, ``"max"``/``"sum"`` for a 2D projection along z.
    blend
        How tiles that overlap are merged:

        * ``"max"`` -- keep the brighter value;
        * ``"sum"`` -- add the values, so overlaps show up as brighter bands;
        * ``"count"`` -- number of tiles covering each voxel (no pixel data is
          read); shows exactly where tiles start and end;
        * ``"midline"`` -- ``max`` within each tile ``group``; where groups
          overlap, every group is used on its own side of the middle of the
          overlap, along ``midline_axis``.
    downsample
        ``"mean"`` (block average) or ``"stride"`` (every n-th voxel, faster).
    channels
        Channel names to fuse, in output order. Default: all, in tile order.
    workers
        Threads fusing regions in parallel.
    levels
        Number of pyramid levels to write (each halves every spatial axis).
    chunks
        Output chunk shape in output axis order (without singleton handling),
        ``(1, 64, 256, 256)`` for 3D and ``(1, 512, 512)`` for projections by default.
    out_dtype
        Output dtype. Default: input dtype for ``max``/``midline``, ``float32``
        for ``sum`` and ``uint16`` for ``count``.
    midline_axis
        ``"x"``, ``"y"`` or ``"z"``: axis along which groups are separated.
    overwrite
        Replace ``out`` if it exists.
    name
        Image name stored in the metadata.
    """
    from ..microscope_manager.zarr_manager import ZarrManager

    tiles = list(tiles)
    if not tiles:
        raise ValueError("No tiles to fuse.")
    if blend not in BLENDS:
        raise ValueError(f"blend must be one of {BLENDS}, got {blend!r}.")
    if projection not in PROJECTIONS:
        raise ValueError(f"projection must be one of {PROJECTIONS}, got {projection!r}.")
    if downsample not in DOWNSAMPLINGS:
        raise ValueError(f"downsample must be one of {DOWNSAMPLINGS}, got {downsample!r}.")
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
    by_channel = {n: [t for t in tiles if t.channel == n] for n in names}

    placed, out_shape, out_vox, _ = _place_tiles([t for n in names for t in by_channel[n]], ds)
    placed_by_channel = {n: [p for p in placed if p.tile.channel == n] for n in names}

    if blend == "midline" and all(t.group is None for t in tiles):
        logger.warning("blend='midline' but tiles have no group; falling back to 'max'.")
        blend = "max"
    axis = "zyx".index(midline_axis)

    if out_dtype is None:
        out_dtype = {"sum": np.float32, "count": np.uint16}.get(blend)
        if out_dtype is None:
            out_dtype = np.result_type(*[t.dtype for t in tiles])
    out_dtype = np.dtype(out_dtype)

    # ----- output geometry -----
    spatial = out_shape[1:] if flat else out_shape
    scales0 = out_vox[1:] if flat else out_vox
    axes = "cyx" if flat else "czyx"
    nsp = len(spatial)
    if chunks is None:
        chunks = (1, 512, 512) if flat else (1, 64, 256, 256)
    chunks = tuple(int(c) for c in chunks)
    if len(chunks) != len(axes):
        raise ValueError(f"chunks must have {len(axes)} values for axes {axes!r}, got {chunks}.")

    sizes, scales, chunksizes = [], [], []
    sp, sc = tuple(spatial), tuple(scales0)
    for lvl in range(max(1, int(levels))):
        sizes.append((len(names),) + sp)
        scales.append(sc)
        chunksizes.append((1,) + tuple(max(1, min(c, s)) for c, s in zip(chunks[1:], sp)))
        factors = tuple(2 if s >= 2 else 1 for s in sp)
        sp = tuple(max(1, s // f) for s, f in zip(sp, factors))
        sc = tuple(v * f for v, f in zip(sc, factors))

    gb = math.prod(sizes[0]) * out_dtype.itemsize / 1e9
    logger.info("Fusing %d tiles into %s (%.2f GB, level 0), blend=%s, projection=%s",
                len(tiles), sizes[0], gb, blend, projection)

    metadata = build_metadata(
        size=sizes,
        scales=scales,
        units=(tiles[0].units,) * nsp,
        channels=[ChannelInfo(n) for n in names],
        dtype=str(out_dtype),
        axes=axes,
        chunksize=chunksizes,
        name=name or "fused mosaic",
    )

    out = Path(out)
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"{out} exists; pass overwrite=True to replace it.")
        shutil.rmtree(out)
    ZarrManager(str(out), mode="w", metadata=metadata)  # creates the empty pyramid
    root = zarr.open_group(str(out), mode="r+")
    arr0 = root["0"]

    # ----- level 0: region by region -----
    cy, cx = chunksizes[0][-2:]
    tasks = []
    for ci, n in enumerate(names):
        midline = (axis, _midline_ranges(placed_by_channel[n], axis)) if blend == "midline" else None
        for y0 in range(0, spatial[-2], cy):
            for x0 in range(0, spatial[-1], cx):
                region = (y0, min(y0 + cy, spatial[-2]), x0, min(x0 + cx, spatial[-1]))
                tasks.append((ci, n, region, midline))

    def run(task):
        ci, n, region, midline = task
        block = _fuse_region(placed_by_channel[n], region, out_shape, ds, projection, blend,
                             downsample, out_dtype, midline)
        y0, y1, x0, x1 = region
        if flat:
            arr0[ci, y0:y1, x0:x1] = block
        else:
            arr0[ci, :, y0:y1, x0:x1] = block

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(run, tasks))
    else:
        for task in tasks:
            run(task)

    # ----- coarser levels: block mean of the previous one -----
    for lvl in range(1, len(sizes)):
        prev, cur = sizes[lvl - 1], sizes[lvl]
        axes_f = {1 + k: max(1, prev[1 + k] // cur[1 + k]) for k in range(nsp)}
        src = da.from_zarr(root[str(lvl - 1)])
        coarse = da.coarsen(np.mean, src, {0: 1, **axes_f}, trim_excess=True)
        coarse = coarse.map_blocks(_cast, out_dtype, dtype=out_dtype)
        coarse = coarse.rechunk(chunksizes[lvl])
        da.store(coarse, root[str(lvl)], lock=False)

    return ZarrManager(str(out), mode="r")
