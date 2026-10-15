from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Union
import warnings

import dask.array as da
import numpy as np
import zarr

from .axes import normalize_axes, spatial_axis_indices, spatial_values_for_axes
from .downsampling import (
    SpatialFactor,
    axis_names_from_multiscales,
    downsample_nearest,
    level_scale_ratios_from_multiscales,
    relative_level_factors_for_axes,
)
from .ngff import _get_group_multiscales, _infer_data_type_from_group
from .zoom import _zoom_dask, _zoom_numpy


def write_image_region(
    root: zarr.Group,
    mode: str,
    data: Union[np.ndarray, da.Array, list[Union[np.ndarray, da.Array]]],
    t: int | slice = slice(None),
    c: int | slice = slice(None),
    z: int | slice = slice(None),
    y: int | slice = slice(None),
    x: int | slice = slice(None),
    level: int = 0,
    group_name: Optional[str] = None,
    downscale_factor: SpatialFactor | None = None,
):
    """Write image data into an existing OME-Zarr pyramid region.

    The target array axes are read from the group's multiscales metadata.  Axes
    may be any subset of ``tczyx``; selectors for missing axes are ignored.
    """
    return _write_region(
        root=root,
        mode=mode,
        data=data,
        selectors={"t": t, "c": c, "z": z, "y": y, "x": x},
        level=level,
        group_name=group_name,
        downscale_factor=downscale_factor,
        expected_data_type="intensity",
    )


def _write_region(
    *,
    root: zarr.Group,
    mode: str,
    data: Union[np.ndarray, da.Array, list[Union[np.ndarray, da.Array]]],
    selectors: dict[str, int | slice],
    level: int,
    group_name: Optional[str],
    downscale_factor: SpatialFactor | None,
    expected_data_type: str | None = None,
):
    if mode not in ("r+", "a", "w"):
        raise PermissionError(
            f"Dataset opened in read-only mode ({mode!r}). Reopen with mode='r+' to allow modifications."
        )

    group = _get_nested_group(root, group_name)
    if group is None:
        available = list(root.group_keys())
        warnings.warn(f"Group {group_name!r} not found. Available root groups: {available}", UserWarning)
        return

    if expected_data_type is not None:
        actual_data_type = _infer_data_type_from_group(group)
        if actual_data_type != expected_data_type:
            raise ValueError(
                f"write_{expected_data_type}_region cannot write to a "
                f"{actual_data_type!r} dataset ({group_name or '/'})."
            )

    multiscales_all = _get_group_multiscales(group)
    if not multiscales_all:
        warnings.warn(
            f"No 'multiscales' attribute found in group {group_name or '/'!r}. Nothing to write.",
            UserWarning,
        )
        return

    multiscales = multiscales_all[0]
    datasets = multiscales.get("datasets", [])
    n_levels = len(datasets)
    if n_levels == 0:
        warnings.warn(f"Group {group_name or '/'!r} contains no pyramid datasets.", UserWarning)
        return
    if not 0 <= level < n_levels:
        raise ValueError(f"level must be in [0, {n_levels}), got {level}.")

    axes = normalize_axes(axis_names_from_multiscales(multiscales), ndim=group[datasets[level]["path"]].ndim)

    if downscale_factor is None:
        level_scale_ratios = level_scale_ratios_from_multiscales(multiscales, datasets, level)
        if level_scale_ratios is None:
            level_scale_ratios = relative_level_factors_for_axes(n_levels, level, axes, downscale_factor=2)
    else:
        level_scale_ratios = relative_level_factors_for_axes(n_levels, level, axes, downscale_factor=downscale_factor)

    if isinstance(data, (np.ndarray, da.Array)):
        data_list = _generate_pyramid(
            data,
            total_levels=n_levels,
            ref_level=level,
            axes=axes,
            level_scale_ratios=level_scale_ratios,
        )
    elif isinstance(data, list):
        data_list = data
        if len(data_list) != n_levels:
            warnings.warn(
                f"Provided pyramid has {len(data_list)} levels, but dataset expects {n_levels}.",
                UserWarning,
            )
    else:
        raise TypeError("data must be a NumPy array, Dask array, or list of such arrays.")

    for i, subdata in enumerate(data_list):
        if i >= n_levels:
            break
        arr_path = datasets[i]["path"]
        if arr_path not in group:
            warnings.warn(f"Dataset path {arr_path!r} not found in group {group_name or '/'!r}.", UserWarning)
            continue

        zarr_array = group[arr_path]
        if subdata.ndim != len(axes):
            warnings.warn(
                f"Region write for axes {''.join(axes)!r} expects {len(axes)}D data, "
                f"got shape {subdata.shape}. Skipping level {i}.",
                UserWarning,
            )
            continue

        index = _scale_index(
            (selectors.get("t", slice(None)), selectors.get("c", slice(None)), selectors.get("z", slice(None)), selectors.get("y", slice(None)), selectors.get("x", slice(None))),
            subdata.shape,
            level_scale_ratios[i],
            axes=axes,
        )

        if isinstance(subdata, da.Array):
            subdata = subdata.compute()

        expected_shape = zarr_array[index].shape
        if subdata.shape != expected_shape:
            warnings.warn(
                f"Shape mismatch for level {i}: data={subdata.shape}, expected={expected_shape}. Skipping level.",
                UserWarning,
            )
            continue
        zarr_array[index] = subdata

    store = getattr(root, "store", None)
    if store is not None and hasattr(store, "flush"):
        store.flush()


def _get_nested_group(root: zarr.Group, group_name: str | None) -> zarr.Group | None:
    """Resolve a potentially nested group path such as ``labels/nuclei``."""
    if group_name in (None, "", "/"):
        return root
    parts = [p for p in str(group_name).split("/") if p]
    group = root
    for part in parts:
        if part not in group:
            return None
        group = group[part]
    return group


def _is_reciprocal_integer_shrink(scale: Sequence[float]) -> tuple[bool, tuple[int, ...]]:
    """Return integer downsampling factors for shrink scales such as 1, 1/2, 1/4."""
    factors = []
    for value in scale:
        value = float(value)
        if value <= 0 or value > 1:
            return False, ()
        factor = 1.0 / value
        rounded = int(round(factor))
        if rounded < 1 or abs(factor - rounded) > 1e-6:
            return False, ()
        factors.append(rounded)
    return True, tuple(factors)


def _generate_pyramid(
    ref_data,
    total_levels: int,
    ref_level: int = 0,
    *,
    axes: str | Sequence[str] = "tczyx",
    downscale_factor: SpatialFactor = 2,
    level_scale_ratios: list[tuple[float, ...]] | None = None,
):
    """Generate a full pyramid from a reference level using axis-aware zoom."""
    axes = normalize_axes(axes, ndim=ref_data.ndim)
    if level_scale_ratios is None:
        level_scale_ratios = relative_level_factors_for_axes(total_levels, ref_level, axes, downscale_factor)
    if len(level_scale_ratios) != total_levels:
        raise ValueError("level_scale_ratios must contain one entry per pyramid level.")

    zoom_fn = _zoom_dask if isinstance(ref_data, da.Array) else _zoom_numpy
    spatial_axes = spatial_axis_indices(axes)
    pyramid = [None] * total_levels
    pyramid[ref_level] = ref_data

    if not spatial_axes:
        for i in range(total_levels):
            if pyramid[i] is None:
                pyramid[i] = ref_data
        return pyramid

    for i in range(ref_level - 1, -1, -1):
        scale = tuple(
            level_scale_ratios[i + 1][axis] / level_scale_ratios[i][axis]
            for axis in range(len(spatial_axes))
        )
        pyramid[i] = zoom_fn(pyramid[i + 1], scale=scale, spatial_axes=spatial_axes)

    for i in range(ref_level + 1, total_levels):
        scale = tuple(
            level_scale_ratios[i - 1][axis] / level_scale_ratios[i][axis]
            for axis in range(len(spatial_axes))
        )
        is_integer_shrink, integer_factors = _is_reciprocal_integer_shrink(scale)
        if is_integer_shrink:
            pyramid[i] = downsample_nearest(pyramid[i - 1], integer_factors, spatial_axes)
        else:
            pyramid[i] = zoom_fn(pyramid[i - 1], scale=scale, spatial_axes=spatial_axes)

    return pyramid


def _scale_index(
    index_tuple,
    shape,
    scale_factor: SpatialFactor,
    axes: str | Sequence[str] = "tczyx",
):
    """Scale spatial selectors for an array with arbitrary ``tczyx`` axes."""
    axes = normalize_axes(axes, ndim=len(shape))
    if len(index_tuple) == 5:
        selectors = dict(zip(("t", "c", "z", "y", "x"), index_tuple))
    elif len(index_tuple) == 4:
        selectors = dict(zip(("t", "z", "y", "x"), index_tuple))
        selectors.setdefault("c", slice(None))
    else:
        raise ValueError("index_tuple must be ordered as (t,c,z,y,x) or (t,z,y,x).")

    spatial_factors = spatial_values_for_axes(scale_factor, axes, name="scale_factor", allow_float=True)
    spatial_iter = iter(spatial_factors)

    def scale_spatial(sel, size, factor):
        if isinstance(sel, (int, np.integer)):
            return int(sel / factor)
        if isinstance(sel, slice):
            start = None if sel.start is None else int(sel.start / factor)
            stop = size if start is None else start + size
            return slice(start, stop, None)
        if isinstance(sel, Sequence) and not isinstance(sel, (str, bytes)):
            return [int(v / factor) for v in sel]
        return sel

    out = []
    for axis_number, ax in enumerate(axes):
        sel = selectors.get(ax, slice(None))
        if ax in {"z", "y", "x"}:
            out.append(scale_spatial(sel, shape[axis_number], next(spatial_iter)))
        else:
            out.append(sel)
    return tuple(out)
