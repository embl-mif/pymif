from __future__ import annotations

from collections.abc import Sequence

import dask.array as da
import numpy as np
import scipy

from .downsampling import SpatialFactor, normalize_spatial_factor


def _normalize_zoom_scale(arr_ndim: int, scale: SpatialFactor, spatial_axes: Sequence[int] | None):
    if spatial_axes is None:
        if arr_ndim < 3:
            raise ValueError("Array must have at least 3 dimensions when spatial_axes is omitted.")
        spatial_axes = tuple(range(arr_ndim - 3, arr_ndim))
        spatial_scale = normalize_spatial_factor(scale, name="scale", allow_float=True)
    else:
        spatial_axes = tuple(int(ax) for ax in spatial_axes)
        if isinstance(scale, (int, float)):
            spatial_scale = tuple(float(scale) for _ in spatial_axes)
        else:
            spatial_scale = tuple(float(v) for v in scale)
            if len(spatial_scale) == 3 and len(spatial_axes) != 3:
                # Legacy ZYX factors have already been reduced by most callers;
                # fall back to the trailing values for defensive compatibility.
                spatial_scale = spatial_scale[-len(spatial_axes):]
        if len(spatial_scale) != len(spatial_axes):
            raise ValueError("scale length must match spatial_axes length.")

    factors = [1.0] * arr_ndim
    for axis, factor in zip(spatial_axes, spatial_scale):
        factors[axis] = float(factor)
    return factors


def _zoom_numpy(arr: np.ndarray, scale: SpatialFactor, spatial_axes: Sequence[int] | None = None) -> np.ndarray:
    """Nearest-neighbor zoom/shrink on selected spatial axes."""
    from scipy.ndimage import zoom

    factors = _normalize_zoom_scale(arr.ndim, scale, spatial_axes)
    for axis, factor in enumerate(factors):
        if arr.shape[axis] * factor < 1:
            factors[axis] = 1.0
    return zoom(arr, zoom=factors, order=0)


def _zoom_dask(arr: da.Array, scale: SpatialFactor, spatial_axes: Sequence[int] | None = None) -> da.Array:
    """Nearest-neighbor zoom/shrink on selected spatial axes for Dask arrays."""
    if not isinstance(arr, da.Array):
        arr = da.from_array(arr, chunks="auto")

    factors = _normalize_zoom_scale(arr.ndim, scale, spatial_axes)
    for axis, factor in enumerate(factors):
        if arr.shape[axis] * factor < 1:
            factors[axis] = 1.0

    def _zoom_block(block, zoom):
        return scipy.ndimage.zoom(block, zoom=zoom, order=0)

    out_chunks = tuple(
        tuple(max(1, int(round(chunk * factor))) for chunk in axis_chunks)
        for axis_chunks, factor in zip(arr.chunks, factors)
    )

    return arr.map_blocks(
        _zoom_block,
        zoom=factors,
        dtype=arr.dtype,
        chunks=out_chunks,
    )
