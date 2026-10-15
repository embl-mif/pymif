from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral, Real
from typing import Any

ALLOWED_AXES: tuple[str, ...] = tuple("tczyx")
ALLOWED_AXIS_SET: set[str] = set(ALLOWED_AXES)
SPATIAL_AXES: tuple[str, ...] = tuple("zyx")
SPATIAL_AXIS_SET: set[str] = set(SPATIAL_AXES)
DATA_TYPES: tuple[str, ...] = ("intensity", "label")


def infer_axes_from_ndim(ndim: int) -> tuple[str, ...]:
    """Infer a legacy-friendly axes tuple from an array dimensionality."""
    mapping = {
        5: tuple("tczyx"),
        4: tuple("tzyx"),
        3: tuple("zyx"),
        2: tuple("yx"),
        1: tuple("x"),
    }
    try:
        return mapping[int(ndim)]
    except (KeyError, ValueError):
        raise ValueError(
            "metadata['axes'] is required for arrays whose dimensionality is not 1-5."
        ) from None


def normalize_axes(axes: str | Sequence[str] | None, ndim: int | None = None) -> tuple[str, ...]:
    """Return validated, lowercase axis labels.

    PyMIF Zarr datasets intentionally support only image-style axes made from
    the labels ``t``, ``c``, ``z``, ``y`` and ``x``.  Any subset and any order of
    those labels is accepted, but labels must be unique and must match the array
    dimensionality when ``ndim`` is supplied.
    """
    if axes is None:
        if ndim is None:
            raise ValueError("metadata['axes'] is required.")
        labels = infer_axes_from_ndim(ndim)
    elif isinstance(axes, str):
        labels = tuple(axes.lower())
    elif isinstance(axes, Sequence):
        labels = tuple(str(ax).lower() for ax in axes)
    else:
        raise TypeError("axes must be a string or a sequence of axis labels.")

    if not labels:
        raise ValueError("axes cannot be empty.")

    invalid = [ax for ax in labels if ax not in ALLOWED_AXIS_SET]
    if invalid:
        raise ValueError(
            "Invalid axis label(s) "
            f"{invalid!r}. Only labels from 'tczyx' are supported."
        )

    duplicates = sorted({ax for ax in labels if labels.count(ax) > 1})
    if duplicates:
        raise ValueError(f"Duplicate axis label(s) are not allowed: {duplicates!r}.")

    if ndim is not None and len(labels) != int(ndim):
        raise ValueError(f"axes has length {len(labels)} but array ndim is {ndim}.")

    return labels


def axes_to_string(axes: str | Sequence[str] | None, ndim: int | None = None) -> str:
    """Normalize axes and return the compact string representation."""
    return "".join(normalize_axes(axes, ndim=ndim))


def spatial_axes_in_order(axes: str | Sequence[str]) -> tuple[str, ...]:
    """Return the spatial axis labels present in ``axes``, preserving order."""
    labels = normalize_axes(axes)
    return tuple(ax for ax in labels if ax in SPATIAL_AXIS_SET)


def spatial_axis_indices(axes: str | Sequence[str]) -> tuple[int, ...]:
    """Return the integer positions of spatial axes in ``axes``."""
    labels = normalize_axes(axes)
    return tuple(i for i, ax in enumerate(labels) if ax in SPATIAL_AXIS_SET)


def normalize_data_type(data_type: str | None = None, *, is_label: bool | None = None) -> str:
    """Normalize PyMIF's image-vs-label flag to ``intensity`` or ``label``."""
    if data_type is None:
        return "label" if is_label else "intensity"

    value = str(data_type).strip().lower().replace("-", "_")
    aliases = {
        "image": "intensity",
        "intensity_image": "intensity",
        "intensity": "intensity",
        "raw": "intensity",
        "labels": "label",
        "label_image": "label",
        "label": "label",
        "segmentation": "label",
        "mask": "label",
    }

    if value not in aliases:
        raise ValueError("data_type must be either 'intensity' or 'label'.")

    normalized = aliases[value]
    if is_label is not None:
        expected = "label" if is_label else "intensity"
        if normalized != expected:
            raise ValueError(
                f"data_type={normalized!r} conflicts with is_label={is_label!r}."
            )
    return normalized


def spatial_values_for_axes(
    value: Real | Sequence[Real] | None,
    axes: str | Sequence[str],
    *,
    name: str,
    default: Real = 1,
    allow_float: bool = True,
) -> tuple[float, ...] | tuple[int, ...]:
    """Normalize a scalar/spatial sequence to the spatial axes present.

    Sequences may be either length ``n_spatial`` in the order used by ``axes`` or
    length 3 in legacy ZYX order.  The latter keeps old calls like
    ``downscale_factor=(1, 2, 2)`` working for datasets that contain only YX.
    """
    labels = normalize_axes(axes)
    present_spatial = spatial_axes_in_order(labels)
    n_spatial = len(present_spatial)

    if value is None:
        values = (float(default),) * n_spatial
    elif isinstance(value, Real) and not isinstance(value, bool):
        values = (float(value),) * n_spatial
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw = tuple(float(v) for v in value)
        if len(raw) == n_spatial:
            values = raw
        elif len(raw) == 3:
            zyx_map = dict(zip(SPATIAL_AXES, raw))
            values = tuple(zyx_map[ax] for ax in present_spatial)
        else:
            raise ValueError(
                f"{name} must be a scalar, a sequence matching the spatial axes "
                f"{present_spatial!r}, or a legacy ZYX sequence of length 3. "
                f"Got {value!r}."
            )
    else:
        raise TypeError(
            f"{name} must be a scalar or sequence, got {type(value).__name__}."
        )

    if any(v <= 0 for v in values):
        raise ValueError(f"{name} values must be > 0, got {values!r}.")

    if allow_float:
        return values

    ints = tuple(int(v) for v in values)
    if any(float(i) != v for i, v in zip(ints, values)):
        raise ValueError(f"{name} values must be integers, got {values!r}.")
    return ints


def index_list_from_selection(selection: Any, axis_size: int) -> list[int]:
    """Return explicit indices represented by an int, slice or index sequence."""
    if selection is None:
        return list(range(axis_size))
    if isinstance(selection, Integral):
        idx = int(selection)
        if idx < 0:
            idx += axis_size
        return [idx]
    if isinstance(selection, slice):
        return list(range(*selection.indices(axis_size)))
    if isinstance(selection, Sequence) and not isinstance(selection, (str, bytes)):
        out = []
        for item in selection:
            idx = int(item)
            if idx < 0:
                idx += axis_size
            out.append(idx)
        return out
    raise TypeError(
        "Selections must be None, int, slice, or a sequence of integer indices."
    )


def selection_length_and_spacing(selection: Any, axis_size: int, axis_name: str) -> tuple[int, int]:
    """Return selected length and uniform spacing for metadata updates."""
    indices = index_list_from_selection(selection, axis_size)
    if len(indices) == 0:
        return 0, 1

    if min(indices) < 0 or max(indices) >= axis_size:
        raise ValueError(
            f"Index for axis {axis_name!r} out of range. Axis size is {axis_size}."
        )

    if len(indices) < 2:
        return len(indices), 1

    diffs = [b - a for a, b in zip(indices[:-1], indices[1:])]
    if len(set(diffs)) != 1:
        raise ValueError(f"Non-uniform spacing in axis {axis_name!r}: {indices!r}")

    if diffs[0] <= 0:
        raise ValueError(f"Indices for axis {axis_name!r} must be strictly increasing.")

    return len(indices), int(diffs[0])


def to_tczyx(arr: Any, axes: str | Sequence[str], target: str = "tczyx") -> Any:
    """Reorder ``arr`` to ``target`` axes, inserting singleton dimensions.

    Works for any array supporting ``reshape``/``transpose`` semantics through
    dask (a dask array is expected). Axes of ``arr`` that are not in ``target``
    raise ``ValueError``; target axes missing from ``arr`` are added with size 1.

    Parameters
    ----------
    arr : dask.array.Array
        Input array.
    axes : str or sequence of str
        Axis labels of ``arr`` (case-insensitive), e.g. ``"zyx"`` or ``"CYX"``.
    target : str, optional
        Desired axis order, ``"tczyx"`` by default.
    """
    import dask.array as da

    current = "".join(str(a).lower() for a in axes)
    target = target.lower()
    if len(current) != arr.ndim:
        raise ValueError(f"axes {current!r} do not match array ndim={arr.ndim}.")
    extra = [a for a in current if a not in target]
    if extra:
        raise ValueError(f"Cannot map axes {extra!r} onto target {target!r}.")

    for ax in target:
        if ax not in current:
            arr = da.expand_dims(arr, axis=0)
            current = ax + current
    return da.transpose(arr, axes=[current.index(ax) for ax in target])
