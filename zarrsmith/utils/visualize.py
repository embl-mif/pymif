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
    layers with physical units. Every layer that ZarrSmith adds, image or helper
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
            "napari is not installed. Install with `pip install zarrsmith[napari]` "
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

    pyramid = data_levels[start_level:] if stop_level == -1 else data_levels[start_level:stop_level]
    if in_memory:
        try:
            pyramid = [p.compute() for p in pyramid]
        except Exception as exc:
            raise RuntimeError(f"Failed to load data into memory: {exc}") from exc

    add_layers(viewer, pyramid, metadata, level=start_level)
    return viewer


def channel_names(metadata: Dict[str, Any], n: int) -> list[str]:
    """Names of the ``n`` channels of a dataset (``ch_0``, ``ch_1``, ... when not in its metadata)."""
    names = metadata.get("channel_names") or []
    return [str(name) for name in names] if len(names) == n else [f"ch_{i}" for i in range(n)]


def layer_geometry(metadata: Dict[str, Any], axes, level: int = 0, *, drop_channel: bool) -> Dict[str, Any]:
    """``scale`` (and ``units``, when known) of a napari layer showing ``level`` of a dataset."""
    axes = normalize_axes(axes)
    geometry = {"scale": _axis_scale(metadata, axes, level, drop_channel=drop_channel)}
    units = _axis_units(metadata, axes, drop_channel=drop_channel)
    if units is not None:
        geometry["units"] = units
    return geometry


def add_layers(viewer, data_levels: List[Any], metadata: Dict[str, Any], *, level: int = 0,
               name: str | None = None, **kwargs) -> list:
    """Add a dataset to ``viewer`` and return its layers: one labels layer for label data, else one
    image layer per channel (named ``"<name>: <channel>"`` when ``name`` is given).

    ``data_levels`` are the levels shown (``level``: the first one's index in the dataset).
    The colours are the channel colours of the metadata (gray without them); ``kwargs`` go to napari
    and replace any of these settings (e.g. ``contrast_limits``, ``blending``, ``colormap``).
    """
    axes = normalize_axes(metadata.get("axes"), ndim=data_levels[0].ndim)
    data = list(data_levels)
    if str(metadata.get("data_type", "intensity")).lower() == "label":
        settings = dict(layer_geometry(metadata, axes, level, drop_channel=False), metadata=metadata,
                        multiscale=True, name=name or metadata.get("name", "labels"))
        layers = [viewer.add_labels(data, **{**settings, **kwargs})]
        _set_axis_labels(viewer, axes, drop_channel=False)
        return layers

    settings = dict(layer_geometry(metadata, axes, level, drop_channel="c" in axes),
                    metadata=metadata, multiscale=True)
    if "c" not in axes:
        settings.update(name=name or metadata.get("name", "image"), colormap="gray")
        layers = [viewer.add_image(data, **{**settings, **kwargs})]
        _set_axis_labels(viewer, axes, drop_channel=False)
        return layers

    c_axis = axes.index("c")
    n = int(data_levels[0].shape[c_axis])
    names = channel_names(metadata, n)
    colours = metadata.get("channel_colors") or []
    try:
        colormap = [_parse_color(c) for c in colours] if len(colours) == n else ["gray"] * n
    except Exception:
        colormap = ["gray"] * n
    settings.update(name=[f"{name}: {ch}" for ch in names] if name else names, colormap=colormap,
                    channel_axis=c_axis)
    layers = viewer.add_image(data, **{**settings, **kwargs})
    _set_axis_labels(viewer, axes, drop_channel=True)
    return list(layers)
