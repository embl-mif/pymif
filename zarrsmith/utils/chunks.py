"""Chunk / shape helpers shared by the pyramid builder, the Zarr writer, the CLI and napari."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import dask.array as da
import numpy as np

from .axes import metadata_axes, spatial_values_for_axes

#: The default number of levels stops when no spatial axis of the coarsest level is above this.
MAX_COARSEST_SIZE = 2048


def shape_tuple(value: Any, ndim: int) -> tuple[int, ...] | None:
    """Return a positive shape tuple of length ``ndim`` from ``value``, or ``None``."""
    if value is None or isinstance(value, (str, bytes)):
        return None
    try:
        candidate = tuple(int(v) for v in value)
    except TypeError:
        return None
    if len(candidate) != ndim or any(v <= 0 for v in candidate):
        return None
    return candidate


def get_chunks(arr: da.Array) -> tuple[int, ...]:
    """Return one normalized chunk tuple for a dask array."""
    if hasattr(arr, "chunksize") and arr.chunksize is not None:
        return tuple(int(x) for x in arr.chunksize)
    return tuple(int(c[0]) for c in arr.chunks)


def rechunk_to_shape(arr: da.Array, chunks: Sequence[int]) -> da.Array:
    """Rechunk ``arr`` to ``chunks``, clipped so no axis exceeds the array's extent."""
    normalized = tuple(max(1, min(int(c), int(s))) for c, s in zip(chunks, arr.shape))
    if normalized == get_chunks(arr):
        return arr
    return arr.rechunk(normalized)


def default_chunk_size(metadata: dict[str, Any], max_size_mb: float = 100):
    """Default chunk shape of a dataset (``metadata['axes']``, ``['size']`` and ``['dtype']``).

    T and C chunks are 1. Z is split in 2, then Z, Y and X are all split twice more until a chunk
    is at most ``max_size_mb``. Returns ``(chunks, chunk size in MB, number of chunks along z, y, x)``.
    """
    axes = metadata_axes(metadata)
    size_map = dict(zip(axes, metadata["size"][0]))
    itemsize = np.dtype(metadata.get("dtype", "uint16")).itemsize
    n_chunks = {"z": 2, "y": 1, "x": 1}
    while True:
        chunk_map = {"t": 1, "c": 1, **{ax: int(size_map.get(ax, 1) / n + 1) for ax, n in n_chunks.items()}}
        size_mb = itemsize * chunk_map["z"] * chunk_map["y"] * chunk_map["x"] / 1024 / 1024
        if size_mb <= max_size_mb:
            break
        n_chunks = {ax: n * 2 for ax, n in n_chunks.items()}
    chunks = tuple(int(max(1, min(size_map.get(ax, 1), chunk_map[ax]))) for ax in axes)
    return chunks, size_mb, n_chunks


def default_num_levels(metadata: dict[str, Any], downscale_factor: Any = 2) -> int:
    """Default number of pyramid levels: until no spatial axis of the coarsest level is above
    MAX_COARSEST_SIZE voxels. ``downscale_factor``: one value, or one per spatial axis (or Z, Y, X)."""
    axes = metadata_axes(metadata)
    size_map = dict(zip(axes, metadata["size"][0]))
    shape = [int(size_map[ax]) for ax in axes if ax in "zyx"]
    if not shape:
        return 1
    factors = spatial_values_for_axes(downscale_factor, axes, name="downscale_factor", allow_float=False)
    n = 1
    while any(s > MAX_COARSEST_SIZE and f > 1 for s, f in zip(shape, factors)):  # factor 1: never smaller
        n += 1
        shape = [max(1, s // int(f)) for s, f in zip(shape, factors)]
    return n
