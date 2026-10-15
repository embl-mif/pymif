"""Chunk / shape helpers shared by the pyramid builder and the Zarr writer."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import dask.array as da


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
