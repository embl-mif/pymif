"""Dataset helpers shared by the napari widgets (no Qt dependency, so unit-testable)."""
from __future__ import annotations

from zarrsmith.utils.axes import metadata_axes, metadata_axis_size
from zarrsmith.utils.visualize import units_for_axes


def dataset_axes(dataset) -> str:
    """Return normalized dataset axes, defaulting to legacy TCZYX metadata."""
    return metadata_axes(dataset.metadata)


def axis_index(dataset, axis):
    """Position of ``axis`` in the dataset axes, or ``None`` if absent."""
    axes = dataset_axes(dataset)
    return axes.index(axis) if axis in axes else None


def axis_size(dataset, axis, default=1):
    """Size of ``axis`` at the base resolution level (``default`` if absent)."""
    return metadata_axis_size(dataset.metadata, axis, default)


def scale_for_axes(dataset, requested_axes):
    """Return a napari scale tuple for the requested spatial axis labels."""
    axes = dataset_axes(dataset)
    spatial_axes = [ax for ax in axes if ax in "zyx"]
    scale_map = dict(zip(spatial_axes, dataset.metadata.get("scales", [(1,) * len(spatial_axes)])[0]))
    return tuple(scale_map.get(ax, 1) for ax in requested_axes if ax in axes)


def units_kwargs(dataset, requested_axes):
    """``{"units": ...}`` for napari layers over ``requested_axes`` (empty if unavailable).

    The ROI / Zrange / CropBox helper layers are scaled in physical units, so
    they need the same units as the image layers; otherwise napari warns
    "Inconsistent units across layers" and stops using units for rendering.
    """
    axes = dataset_axes(dataset)
    units = units_for_axes(dataset.metadata, [ax for ax in requested_axes if ax in axes])
    return {} if units is None else {"units": units}


def points_projection_kwargs():
    """Keyword arguments that make a napari ``Points`` layer show out-of-slice points.

    napari 0.9 deprecated ``out_of_slice_display`` in favour of
    ``projection_mode``; ``rescale_linear`` is what the old flag mapped to.
    Older napari versions only know the old flag (napari 0.7 already has a
    ``projection_mode`` argument, but only with ``none``/``all``).
    """
    from napari.layers import Points

    modes = getattr(Points, "_projectionclass", None)
    if modes is not None and "rescale_linear" in {str(m) for m in modes}:
        return {"projection_mode": "rescale_linear"}
    return {"out_of_slice_display": True}
