from __future__ import annotations

import warnings
from typing import Any, Dict, Iterable, List, TYPE_CHECKING, Union

import dask.array as da

from .axes import normalize_axes, spatial_axes_in_order
from .colors import hex_to_rgb, parse_channel_color
from .units import normalize_unit

if TYPE_CHECKING:
    import napari


def _parse_color(color: Union[int, str]) -> tuple[float, float, float]:
    """Convert an OME int, hex string or color name to an RGB float tuple for napari."""
    return hex_to_rgb(parse_channel_color(color))


def _axis_scale(metadata: Dict[str, Any], axes: tuple[str, ...], level: int, *, drop_channel: bool) -> tuple[float, ...]:
    spatial_axes = spatial_axes_in_order(axes)
    spatial_scale = metadata.get("scales", [tuple(1.0 for _ in spatial_axes)])[level]
    spatial_map = dict(zip(spatial_axes, spatial_scale))
    scale = []
    for ax in axes:
        if drop_channel and ax == "c":
            continue
        if ax == "t":
            scale.append(float(metadata.get("time_increment") or 1.0))
        elif ax in spatial_map:
            scale.append(float(spatial_map[ax]))
        else:
            scale.append(1.0)
    return tuple(scale)


def units_for_axes(metadata: Dict[str, Any], requested: "Iterable[str]"):
    """Per-axis units for napari layers covering the axes ``requested``, or ``None``.

    Layers added without units default to ``pixel``. napari then disables units
    for rendering ("Inconsistent units across layers") as soon as it also holds
    layers with physical units. Every layer that PyMIF adds, image or helper
    (ROI, Z range...), therefore takes its units from here so they all agree
    with each other and with a converted zarr opened by ``napari-ome-zarr``.

    ``None`` is returned (so the caller omits ``units``) when napari has no
    layer units or a unit cannot be parsed.
    """
    try:
        from napari.utils.transforms._units import get_units_from_name
    except ImportError:  # napari without layer units
        return None

    axes = normalize_axes(metadata.get("axes"))
    spatial = dict(zip(spatial_axes_in_order(axes), metadata.get("units") or ()))
    units = []
    for ax in requested:
        if ax == "t":
            units.append(normalize_unit(metadata.get("time_increment_unit")))
        else:
            units.append(normalize_unit(spatial.get(ax)))
    try:
        get_units_from_name(units)  # validate every entry
    except Exception:
        return None
    return tuple(units)


def _axis_units(metadata: Dict[str, Any], axes: tuple[str, ...], *, drop_channel: bool):
    """Units of the axes shown by an image/label layer (channel axis optionally removed)."""
    return units_for_axes(metadata, [ax for ax in axes if not (drop_channel and ax == "c")])


def _set_axis_labels(viewer, axes: tuple[str, ...], *, drop_channel: bool) -> None:
    labels = tuple(ax.upper() for ax in axes if not (drop_channel and ax == "c"))
    try:
        if len(labels) == len(viewer.dims.axis_labels):
            viewer.dims.axis_labels = labels
    except Exception:
        pass

def visualize(
    data_levels: List[da.Array],
    metadata: Dict[str, Any],
    start_level: int = 0,
    stop_level: int = -1,
    in_memory: bool = False,
    viewer: "napari.Viewer | None" = None,
) -> "napari.Viewer | None":
    """Visualize an axis-aware multiscale dataset with napari.

    Datasets without a ``t`` axis are passed to napari without an artificial time
    dimension, so napari will not expose an active T slider.  Datasets without a
    ``c`` axis are displayed as a single image/label layer rather than channel
    layers.
    """
    try:
        import napari
    except ImportError:
        warnings.warn(
            "napari is not installed. Install with `pip install pymif[napari]` "
            "to use visualization.",
            stacklevel=2,
        )
        return None

    if not data_levels:
        raise ValueError("No data levels supplied for visualization.")
    if not 0 <= start_level < len(data_levels):
        raise ValueError(f"start_level={start_level} is out of bounds for {len(data_levels)} levels.")
    if stop_level > 0 and stop_level > len(data_levels):
        raise ValueError(f"stop_level={stop_level} is out of bounds for {len(data_levels)} levels.")
    if stop_level > 0 and start_level >= stop_level:
        raise ValueError(f"start_level={start_level} must be lower than stop_level={stop_level}.")

    if viewer is None:
        viewer = napari.Viewer()

    axes = normalize_axes(metadata.get("axes"), ndim=data_levels[0].ndim)
    pyramid = data_levels[start_level:] if stop_level == -1 else data_levels[start_level:stop_level]
    if in_memory:
        try:
            pyramid = [p.compute() for p in pyramid]
        except Exception as exc:
            raise RuntimeError(f"Failed to load data into memory: {exc}") from exc

    data_type = str(metadata.get("data_type", "intensity")).lower()
    scale = _axis_scale(metadata, axes, start_level, drop_channel=("c" in axes and data_type != "label"))

    if data_type == "label":
        label_kwargs = {}
        label_units = _axis_units(metadata, axes, drop_channel=False)
        if label_units is not None:
            label_kwargs["units"] = label_units
        viewer.add_labels(
            pyramid,
            name=metadata.get("name", "labels"),
            scale=scale,
            metadata=metadata,
            multiscale=True,
            **label_kwargs,
        )
        _set_axis_labels(viewer, axes, drop_channel=False)
        return viewer

    add_kwargs = {
        "scale": scale,
        "metadata": metadata,
        "multiscale": True,
    }
    image_units = _axis_units(metadata, axes, drop_channel=("c" in axes))
    if image_units is not None:
        add_kwargs["units"] = image_units

    try:
        # Python ints: ``2 * np.uint16(40000)`` would silently wrap around to 14464.
        max_val = int(da.max(data_levels[-1]).compute())
        min_val = int(da.min(data_levels[-1]).compute())
        add_kwargs["contrast_limits"] = [max(0, min_val), max(1, 2 * max_val)]
    except Exception:
        pass

    if "c" in axes:
        c_axis = axes.index("c")
        num_channels = int(data_levels[0].shape[c_axis])
        channel_names = metadata.get("channel_names") or [f"ch_{i}" for i in range(num_channels)]
        channel_colors = metadata.get("channel_colors") or []
        if channel_colors:
            try:
                add_kwargs["colormap"] = [_parse_color(c) for c in channel_colors]
            except Exception:
                add_kwargs["colormap"] = ["gray"] * num_channels
        else:
            add_kwargs["colormap"] = ["gray"] * num_channels
        viewer.add_image(
            pyramid,
            name=channel_names,
            channel_axis=c_axis,
            **add_kwargs,
        )
        _set_axis_labels(viewer, axes, drop_channel=True)
    else:
        viewer.add_image(
            pyramid,
            name=metadata.get("name", "image"),
            colormap="gray",
            **add_kwargs,
        )
        _set_axis_labels(viewer, axes, drop_channel=False)

    return viewer
