"""Shared construction of the normalized ZarrSmith metadata dictionary."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .ngff import parse_channel_color
from .units import normalize_unit


@dataclass(frozen=True)
class ChannelInfo:
    """Name and display color of a single channel.

    ``color`` may be anything understood by
    :func:`~zarrsmith.utils.ngff.parse_channel_color`; it is
    normalized to ``RRGGBB`` by :func:`build_metadata`.
    """

    name: str
    color: Any = None


def build_metadata(
    *,
    size: Sequence[Sequence[int]],
    scales: Sequence[Sequence[float]],
    units: Sequence[str | None],
    channels: Iterable[ChannelInfo],
    dtype: str,
    axes: str = "tczyx",
    time_increment: float = 1.0,
    time_increment_unit: str | None = "s",
    **extra: Any,
) -> dict[str, Any]:
    """Assemble the metadata dictionary every manager exposes as ``.metadata``.

    Centralizing this guarantees that all managers use the same keys, that
    units are spelled the NGFF way (``micrometer``, ``second`` ...) and that
    channel colors are always ``RRGGBB`` strings.

    Parameters
    ----------
    size : list of tuple
        Array shape of every pyramid level.
    scales : list of tuple
        Spatial voxel size of every pyramid level (``z, y, x`` order).
    units : sequence of str
        Unit of every spatial axis.
    channels : iterable of ChannelInfo
        Channel names and colors, in channel order.
    dtype : str
        Pixel data type.
    axes : str
        Axis order of the data, ``"tczyx"`` for the vendor readers.
    time_increment, time_increment_unit
        Spacing between timepoints and its unit.
    **extra
        Additional manager-specific keys stored verbatim (e.g. ``plane_files``).
    """
    channels = list(channels)
    return {
        "size": [tuple(int(v) for v in s) for s in size],
        "scales": [tuple(float(v) for v in s) for s in scales],
        "units": tuple(normalize_unit(u) for u in units),
        "time_increment": float(time_increment),
        "time_increment_unit": normalize_unit(time_increment_unit),
        "channel_names": [c.name for c in channels],
        "channel_colors": [parse_channel_color(c.color, i) for i, c in enumerate(channels)],
        "dtype": str(dtype),
        "axes": axes,
        **extra,
    }


def scale_for_level(
    base_scale: Sequence[float],
    base_shape: Sequence[int],
    level_shape: Sequence[int],
) -> tuple[float, ...]:
    """Voxel size of a pyramid level, from the base voxel size and the shapes.

    ``scale_level = base_scale * base_shape / level_shape`` for every spatial
    axis, so the physical extent of the volume is preserved across levels.
    All three arguments are given in the same (spatial) axis order.
    """
    return tuple(
        float(s) * b / l for s, b, l in zip(base_scale, base_shape, level_shape)
    )
