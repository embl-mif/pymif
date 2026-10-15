from __future__ import annotations

from collections.abc import Sequence
from numbers import Real
from typing import Any, TypeAlias

import dask.array as da
import numpy as np

from .axes import SPATIAL_AXIS_SET, normalize_axes, spatial_values_for_axes

SpatialFactor: TypeAlias = int | float | Sequence[int | float]
SpatialScale: TypeAlias = tuple[float, ...]
SpatialIntFactor: TypeAlias = tuple[int, ...]
SPATIAL_AXES = SPATIAL_AXIS_SET


def normalize_spatial_factor(
    value: SpatialFactor,
    *,
    name: str = "downscale_factor",
    allow_float: bool = True,
) -> SpatialScale | SpatialIntFactor:
    """Normalize scalar or ZYX factor to a 3-tuple.

    This helper keeps the old public behavior.  Axis-aware code should use
    :func:`normalize_spatial_factor_for_axes`.

    Scalar 2      -> (2, 2, 2)
    Tuple (1,2,2) -> (1, 2, 2)
    """
    if isinstance(value, Real) and not isinstance(value, bool):
        values = (float(value),) * 3
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 3:
            raise ValueError(
                f"{name} must be a scalar or a ZYX sequence of length 3, got {value!r}."
            )
        values = tuple(float(v) for v in value)
    else:
        raise TypeError(
            f"{name} must be a scalar or a ZYX sequence of length 3, "
            f"got {type(value).__name__}."
        )

    if any(v <= 0 for v in values):
        raise ValueError(f"{name} values must be > 0, got {values!r}.")

    if not allow_float:
        ints = tuple(int(v) for v in values)
        if any(float(i) != v for i, v in zip(ints, values)):
            raise ValueError(f"{name} values must be integers, got {values!r}.")
        return ints
    return values



def normalize_spatial_factor_for_axes(
    value: SpatialFactor,
    axes: str | Sequence[str],
    *,
    name: str = "downscale_factor",
    allow_float: bool = True,
) -> SpatialScale | SpatialIntFactor:
    """Normalize a factor to the spatial axes present in ``axes``."""
    return spatial_values_for_axes(value, axes, name=name, allow_float=allow_float)


def spatial_factor_power(factor: SpatialFactor, exponent: int) -> SpatialScale:
    values = normalize_spatial_factor(factor, allow_float=True)
    if exponent >= 0:
        return tuple(v**exponent for v in values)
    return tuple(1.0 / (v ** (-exponent)) for v in values)


def spatial_factor_power_for_axes(
    factor: SpatialFactor,
    axes: str | Sequence[str],
    exponent: int,
) -> SpatialScale:
    values = normalize_spatial_factor_for_axes(factor, axes, allow_float=True)
    if exponent >= 0:
        return tuple(v**exponent for v in values)
    return tuple(1.0 / (v ** (-exponent)) for v in values)


def multiply_spatial_scales(
    scales: Sequence[int | float],
    factors: SpatialFactor,
) -> tuple[float, float, float]:
    values = normalize_spatial_factor(factors, allow_float=True)
    if len(scales) != len(values):
        raise ValueError(
            f"scales must contain one value for each spatial factor, got {scales!r}."
        )
    return tuple(float(s) * f for s, f in zip(scales, values))


def relative_level_factors(
    total_levels: int,
    ref_level: int,
    downscale_factor: SpatialFactor = 2,
) -> list[SpatialScale]:
    """Return level scale ratios relative to `ref_level`.

    Example with downscale_factor=(1,2,2), ref_level=0:

        level 0 -> (1, 1, 1)
        level 1 -> (1, 2, 2)
        level 2 -> (1, 4, 4)

    Example with ref_level=1:

        level 0 -> (1, 0.5, 0.5)
        level 1 -> (1, 1, 1)
        level 2 -> (1, 2, 2)
    """
    if not 0 <= ref_level < total_levels:
        raise ValueError(f"ref_level must be in [0, {total_levels}), got {ref_level}.")
    return [spatial_factor_power(downscale_factor, level - ref_level) for level in range(total_levels)]

def relative_level_factors_for_axes(
    total_levels: int,
    ref_level: int,
    axes: str | Sequence[str],
    downscale_factor: SpatialFactor = 2,
) -> list[SpatialScale]:
    """Return axis-aware level scale ratios relative to ``ref_level``."""
    if not 0 <= ref_level < total_levels:
        raise ValueError(f"ref_level must be in [0, {total_levels}), got {ref_level}.")
    return [
        spatial_factor_power_for_axes(downscale_factor, axes, level - ref_level)
        for level in range(total_levels)
    ]

def axis_names_from_multiscales(multiscales: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for axis in multiscales.get("axes", []) or []:
        if isinstance(axis, dict):
            name = axis.get("name")
        else:
            name = axis
        if name is not None:
            names.append(str(name).lower())
    return names


def _dataset_scale_vector(dataset: dict[str, Any]) -> Sequence[int | float] | None:
    for transform in dataset.get("coordinateTransformations", []) or []:
        if isinstance(transform, dict) and transform.get("type") == "scale":
            return transform.get("scale")

    return None


def level_scale_ratios_from_multiscales(
    multiscales: dict[str, Any],
    datasets: Sequence[dict[str, Any]],
    ref_level: int,
) -> list[SpatialScale] | None:
    """Infer relative spatial level factors from NGFF multiscales metadata.

    Returned tuples are ordered like the spatial axes in the metadata, not
    necessarily legacy ZYX.
    """
    if not datasets or not 0 <= ref_level < len(datasets):
        return None

    axis_names = axis_names_from_multiscales(multiscales)
    if not axis_names:
        return None
    axes = normalize_axes(axis_names)
    spatial_positions = [i for i, name in enumerate(axes) if name in SPATIAL_AXIS_SET]

    level_scales: list[SpatialScale] = []
    for dataset in datasets:
        scale_vector = _dataset_scale_vector(dataset)
        if scale_vector is None:
            return None
        if len(scale_vector) != len(axes):
            return None
        spatial = tuple(float(scale_vector[i]) for i in spatial_positions)
        level_scales.append(spatial)

    ref_scale = level_scales[ref_level]
    return [
        tuple(
            level_scale[i] / ref_scale[i]
            for i in range(len(ref_scale))
        )
        for level_scale in level_scales
    ]


def downsample_nearest(arr, factors: Sequence[int], spatial_axes: Sequence[int]):
    """Nearest-neighbour downsampling by striding the spatial axes.

    Works on NumPy and Dask arrays. Axes whose length is not divisible by the
    factor are edge-padded first, so the output length is ``ceil(n / factor)``.
    A factor of 1 leaves an axis untouched.
    """
    if len(factors) != len(spatial_axes):
        raise ValueError("factors must match spatial_axes length.")
    xp = da if isinstance(arr, da.Array) else np
    pad_width = [(0, 0)] * arr.ndim
    slicing = [slice(None)] * arr.ndim
    for axis, factor in zip(spatial_axes, factors):
        factor = int(factor)
        if factor <= 1:
            continue
        remainder = int(arr.shape[axis]) % factor
        if remainder:
            pad_width[axis] = (0, factor - remainder)
        slicing[axis] = slice(0, None, factor)
    if any(width != (0, 0) for width in pad_width):
        arr = xp.pad(arr, pad_width, mode="edge")
    return arr[tuple(slicing)]
