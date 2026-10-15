from __future__ import annotations

from typing import Optional, Sequence, Union

import dask.array as da

from .axes import index_list_from_selection, normalize_axes, selection_length_and_spacing, spatial_axes_in_order

Selection = Optional[Union[int, slice, Sequence[int]]]


def subset_dask_array(
    arr: da.Array,
    axes: str,
    T: Selection = None,
    C: Selection = None,
    Z: Selection = None,
    Y: Selection = None,
    X: Selection = None,
) -> da.Array:
    """Subset a Dask array using axis labels rather than fixed TCZYX positions."""
    axes_tuple = normalize_axes(axes, ndim=arr.ndim)
    axis_indices = {"t": T, "c": C, "z": Z, "y": Y, "x": X}
    for ax, indices in axis_indices.items():
        if indices is None or ax not in axes_tuple:
            continue
        axis = axes_tuple.index(ax)
        slicer = [slice(None)] * arr.ndim
        # Preserve dimensionality when a single integer is requested; PyMIF
        # metadata keeps the axis with length 1 after subsetting.
        slicer[axis] = [indices] if isinstance(indices, int) else indices
        arr = arr[tuple(slicer)]
    return arr


def validate_uniform_spacing(indices: Sequence[int], name: str):
    """Ensure spacing between integer indices is uniform."""
    if len(indices) < 2:
        return 1
    diffs = [b - a for a, b in zip(indices[:-1], indices[1:])]
    if len(set(diffs)) != 1:
        raise ValueError(f"Non-uniform spacing in {name} axis: {indices}")
    return diffs[0]


def _subset_list(values, selection, axis_size):
    if values is None:
        return values
    if selection is None:
        return values
    idx = index_list_from_selection(selection, axis_size)
    return [values[i] for i in idx]


def subset_metadata(
    metadata: dict,
    T: Selection = None,
    C: Selection = None,
    Z: Selection = None,
    Y: Selection = None,
    X: Selection = None,
) -> dict:
    """Update metadata after subsetting any available subset of image axes."""
    new_metadata = metadata.copy()
    axes = normalize_axes(metadata["axes"])
    shape = list(metadata["size"][0])
    axis_map = {"t": T, "c": C, "z": Z, "y": Y, "x": X}

    new_size = list(shape)
    spacing_by_axis = {}

    for i, ax in enumerate(axes):
        selection = axis_map[ax]
        if selection is None:
            continue

        if ax == "c":
            if isinstance(selection, slice):
                indices = list(range(*selection.indices(shape[i])))
            elif isinstance(selection, int):
                indices = [selection]
            else:
                indices = list(selection)

            if len(set(indices)) != len(indices):
                raise ValueError(f"Duplicate indices are not allowed for axis {ax!r}: {indices!r}")

            new_size[i] = len(indices)
            continue

        selected_len, spacing = selection_length_and_spacing(selection, shape[i], ax)
        new_size[i] = selected_len
        spacing_by_axis[ax] = spacing

    new_metadata["size"] = [tuple(new_size)]

    if T is not None and "t" in axes and metadata.get("time_increment") is not None:
        new_metadata["time_increment"] = spacing_by_axis.get("t", 1) * metadata["time_increment"]

    spatial_axes = spatial_axes_in_order(axes)
    old_scales = list(metadata.get("scales", [tuple(1.0 for _ in spatial_axes)])[0])
    new_scales = list(old_scales)
    for scale_index, ax in enumerate(spatial_axes):
        if axis_map[ax] is not None:
            new_scales[scale_index] *= spacing_by_axis.get(ax, 1)
    new_metadata["scales"] = [tuple(new_scales)]

    if C is not None and "c" in axes:
        c_axis = axes.index("c")
        if "channel_names" in metadata:
            new_metadata["channel_names"] = _subset_list(metadata.get("channel_names"), C, shape[c_axis])
        if "channel_colors" in metadata:
            new_metadata["channel_colors"] = _subset_list(metadata.get("channel_colors"), C, shape[c_axis])

    return new_metadata
