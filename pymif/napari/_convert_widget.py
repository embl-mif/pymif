from napari import current_viewer
from datetime import datetime
from napari.qt.threading import thread_worker
import numpy as np
import pandas as pd
from magicgui import magicgui
import pymif.microscope_manager as mm
from ._batch_csv import append_batch_csv
from ._layout import compose_dock, enable_drag_and_drop, make_button_holder
from ._layout import make_section as _make_section
from ._dataset_helpers import axis_index as _axis_index
from ._dataset_helpers import axis_size as _axis_size
from ._dataset_helpers import dataset_axes as _dataset_axes
from ._dataset_helpers import scale_for_axes as _scale_for_axes
from ._dataset_helpers import points_projection_kwargs
from ._dataset_helpers import units_kwargs as _units_kwargs
from magicgui.widgets import FileEdit
import sys
from qtpy.QtWidgets import QTextEdit, QLineEdit
from qtpy.QtCore import QObject, Qt, Signal, QUrl
from qtpy.QtGui import QTextCursor
from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QToolButton, QFrame, QSizePolicy
from pathlib import Path
from matplotlib import rc
rc('font', size=12)
rc('font', family='Arial')
# rc('font', serif='Times')
rc('pdf', fonttype=42)
# rc('text', usetex=True)

import numpy as np
from napari.layers.utils import layer_utils

def conservative_compute_multiscale_level(
    requested_shape,
    shape_threshold,
    downsample_factors,
):
    """Computed desired level of the multiscale given requested field of view.

    The level of the multiscale should be the lowest resolution such that
    the requested shape is above the shape threshold. By passing a shape
    threshold corresponding to the shape of the canvas on the screen this
    ensures that we have at least one data pixel per screen pixel, but no
    more than we need.

    Parameters
    ----------
    requested_shape : tuple
        Requested shape of field of view in data coordinates
    shape_threshold : tuple
        Maximum size of a displayed tile in pixels.
    downsample_factors : list of tuple
        Downsampling factors for each level of the multiscale. Must be increasing
        for each level of the multiscale.

    Returns
    -------
    level : int
        Level of the multiscale to be viewing.
    """
    # Scale shape by downsample factors
    threshold_factor = 0.1  # Adjust this factor to control the conservativeness
    scaled_shape = requested_shape / downsample_factors

    # Find the highest level (lowest resolution) allowed
    locations = np.argwhere(np.any(scaled_shape > (np.asarray(shape_threshold) * threshold_factor), axis=1))
    level = locations[-1][0] if len(locations) > 0 else 0
    print('Level in viewer:', level)
    return level


layer_utils.compute_multiscale_level = conservative_compute_multiscale_level

# -------------------------
# MagicGUI controls
# -------------------------

from ._log import EmittingStream, make_log_appender  # noqa: F401  (re-exported)


def dataset_reader(microscope):
    """Return the PyMIF manager class matching the selected microscope key."""
    if microscope == "zeiss":
        reader = mm.ZeissManager
    elif microscope == "viventis":
        reader = mm.ViventisManager
    elif microscope == "opera":
        reader = mm.OperaManager
    elif microscope == "luxendo":
        reader = mm.LuxendoManager
    elif microscope == "zarr":
        reader = mm.ZarrManager
    elif microscope == "scape":
        reader = mm.ScapeManager
    else:
        raise ValueError(f"Unsupported microscope: {microscope}")
    
    return reader


def _chunk_shape_for_dataset(dataset, chunk_z=16, chunk_y=512, chunk_x=512):
    """Map widget Z/Y/X chunk values onto whatever axes the dataset has."""
    chunk_map = {"t": 1, "c": 1, "z": chunk_z, "y": chunk_y, "x": chunk_x}
    axes = _dataset_axes(dataset)
    return tuple(int(max(1, min(_axis_size(dataset, ax), chunk_map[ax]))) for ax in axes)


def _present_range(dataset, axis, requested_range):
    """Return a clipped inclusive range tuple for a present axis, else None."""
    if _axis_index(dataset, axis) is None:
        return None
    max_idx = _axis_size(dataset, axis) - 1
    start, end = requested_range
    return int(max(0, min(start, max_idx))), int(max(0, min(end, max_idx)))


def _set_range_widget(widget, size, enabled=True):
    max_idx = max(0, int(size) - 1)
    widget.max = max_idx
    widget.value = (0, max_idx)
    widget.enabled = bool(enabled and size > 1)

def get_chunk_size(dataset_size, max_size_mb=100, axes="tczyx"):
    """Estimate a chunk shape for any dataset axis subset.

    The returned tuple follows ``axes``.  Legacy callers that pass a five-value
    TCZYX size still receive a five-value chunk tuple.
    """
    axes = str(axes).lower()
    size_map = dict(zip(axes, dataset_size))
    z = max(1, int(size_map.get("z", 1)))
    y = max(1, int(size_map.get("y", 1)))
    x = max(1, int(size_map.get("x", 1)))

    n_chunks = {"z": 4, "y": 1, "x": 1}
    chunk_map = {"t": 1, "c": 1}

    while True:
        chunk_map.update(
            {
                "z": int(z / n_chunks["z"] + 1),
                "y": int(y / n_chunks["y"] + 1),
                "x": int(x / n_chunks["x"] + 1),
            }
        )
        size_mb = 2 * chunk_map["z"] * chunk_map["y"] * chunk_map["x"] / 1024 / 1024
        if size_mb <= max_size_mb:
            break
        n_chunks = {k: v * 2 for k, v in n_chunks.items()}

    return tuple(int(max(1, min(size_map.get(ax, 1), chunk_map[ax]))) for ax in axes)


def get_n_levels(dataset_size, axes="tczyx"):
    """Estimate a default number of pyramid levels from present spatial axes."""
    axes = str(axes).lower()
    size_map = dict(zip(axes, dataset_size))
    shape = [max(1, int(size_map[ax])) for ax in axes if ax in "zyx"]
    if not shape:
        return 1

    n = 1
    while any(s > 2048 for s in shape):
        n += 1
        shape = [max(1, s // 2) for s in shape]

    return max(3, n)



BATCH_CSV_COLUMNS = [
    "input",
    "microscope",
    "output",
    "chunk_size",
    "max_size(MB)",
    "scene_index",
    "zarr_format",
    "downscale_factor",
    "subset",
    "channel_colors",
    "channel_names",
    "num_levels",
    "shards",
    "shard_target_mb",
    "shard_exclude_axes",
    "drop_singleton",
    "num_workers",
]


def _indices_to_expr(indices):
    indices = [int(i) for i in indices]
    if not indices:
        return ""
    if len(indices) == 1:
        return str(indices[0])
    step = indices[1] - indices[0]
    if step > 0 and all((b - a) == step for a, b in zip(indices[:-1], indices[1:])):
        start = indices[0]
        stop = indices[-1]+1
        return f"{start}:{stop}:{step}"
    return ",".join(str(i) for i in indices)


def _range_to_indices(range_value):
    start, end = (int(range_value[0]), int(range_value[1]))
    if end < start:
        start, end = end, start
    return list(range(start, end + 1))


def _subset_string_from_widget(dataset, t_range, z_range, y_range, x_range, channels):
    axes = _dataset_axes(dataset)
    parts = []
    if "t" in axes:
        parts.append(f"t={_indices_to_expr(_range_to_indices(t_range))}")
    if "c" in axes and channels:
        channel_names = list(dataset.metadata.get("channel_names", []))
        indices = [channel_names.index(name) for name in channels if name in channel_names]
        if indices:
            parts.append(f"c={_indices_to_expr(indices)}")
    if "z" in axes:
        parts.append(f"z={_indices_to_expr(_range_to_indices(z_range))}")
    if "y" in axes:
        parts.append(f"y={_indices_to_expr(_range_to_indices(y_range))}")
    if "x" in axes:
        parts.append(f"x={_indices_to_expr(_range_to_indices(x_range))}")
    return ";".join(parts)


def _channel_colors_from_metadata(dataset, selected_channel_names):
    channel_names = list(dataset.metadata.get("channel_names", []))
    colors = list(dataset.metadata.get("channel_colors", []))
    if not channel_names or not colors or len(channel_names) != len(colors):
        return []
    lookup = dict(zip(channel_names, colors))
    return [str(lookup[name]) for name in selected_channel_names if name in lookup]


def output_summary(dataset, *, t_range, z_range, y_range, x_range, channels, drop_singleton=True):
    """Level-0 recap of the converted dataset, in the format of the mosaic fusion plan.

    One line such as ``3 t x 2 ch x 120 x 1024 x 1024 px (tczyx), 1.51 GB,
    voxel 2 x 0.65 x 0.65 micrometer``, following the selected ranges and
    channels and, with ``drop_singleton``, the t/c/z axes dropped on write.
    """
    axes = _dataset_axes(dataset)
    ranges = {"t": t_range, "z": z_range, "y": y_range, "x": x_range}
    sizes = {}
    for ax in axes:
        if ax == "c":
            names = dataset.metadata.get("channel_names", [])
            selected = [ch for ch in channels if ch in names]
            sizes[ax] = len(selected) if selected else _axis_size(dataset, "c")
        else:
            start, end = _present_range(dataset, ax, ranges[ax])
            sizes[ax] = abs(end - start) + 1
    out_axes = "".join(ax for ax in axes if not (drop_singleton and ax in "tcz" and sizes[ax] == 1))

    labels = {"t": "t", "c": "ch"}
    parts = [f"{sizes[ax]} {labels[ax]}" for ax in out_axes if ax in labels]
    spatial = [ax for ax in out_axes if ax in "zyx"]
    parts.append(" x ".join(str(sizes[ax]) for ax in spatial) + " px")
    dims = " x ".join(parts)

    nbytes = int(np.prod([sizes[ax] for ax in out_axes])) * np.dtype(dataset.data[0].dtype).itemsize
    size = f"{nbytes / 1e9:.2f} GB" if nbytes >= 1e8 else f"{nbytes / 1e6:.1f} MB"

    vox = " x ".join(f"{v:.3g}" for v in _scale_for_axes(dataset, spatial))
    units = dict(zip([ax for ax in axes if ax in "zyx"], dataset.metadata.get("units") or ()))
    unit = " ".join(dict.fromkeys(str(units[ax]) for ax in spatial if units.get(ax)))
    return f"{dims} ({out_axes}), {size}, voxel {vox} {unit}".rstrip()


def _append_batch_csv(csv_path, row):
    return append_batch_csv(csv_path, row, BATCH_CSV_COLUMNS)

def _compact_advanced_rows(widget, insert_after):
    """Regroup the "Additional parameters" of the converter into compact rows.

    Related controls (chunks, downscale, ...) share one row instead of one row
    each. The magicgui widgets are moved, not copied, so ``widget.chunk_x`` and
    friends keep working (values, ``enabled``, signals). Their original
    one-per-row wrappers are hidden for good.

    Returns the new row widgets; show or hide them together as one group.
    """
    layout = widget.native.layout()

    def row(label, *items):
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)
        title = QLabel(label)
        title.setMinimumWidth(78)
        row_layout.addWidget(title)
        for prefix, name in items:
            mg = getattr(widget, name)
            wrapper = mg.native.parent()
            if prefix:
                row_layout.addWidget(QLabel(prefix))
            row_layout.addWidget(mg.native, 1)
            # A checkbox has no row wrapper (its parent is the whole panel).
            if wrapper is not widget.native:
                wrapper.setVisible(False)
        return row_widget

    rows = [
        row("Chunks", ("Z", "chunk_z"), ("Y", "chunk_y"), ("X", "chunk_x")),
        row("Downscale", ("Z", "downscale_z"), ("Y", "downscale_y"), ("X", "downscale_x")),
        row("Output", ("Levels", "n_levels"), ("Zarr", "zarr_format")),
        row("Sharding", (None, "shards"), ("MB", "shard_target_mb")),
        row("Never merge", (None, "shard_exclude_axes")),
        row("Other", (None, "drop_singleton"), ("Threads", "num_workers")),
    ]
    widget.shard_exclude_axes.native.setMaximumHeight(60)

    index = layout.indexOf(insert_after) + 1
    for offset, row_widget in enumerate(rows):
        layout.insertWidget(index + offset, row_widget)
    return rows


def _run_conversion(
    reader,
    path,
    scene_index,
    chunks,
    t_range,
    z_range,
    y_range,
    x_range,
    channels,
    n_levels,
    downscale_factor,
    output_path,
    file_format,
    zarr_format,
    shards="auto",
    shard_target_mb=None,
    shard_exclude_axes=None,
    drop_singleton=True,
    num_workers=None,
    ):
    """Run the conversion pipeline used by the napari worker thread."""
    print("Starting conversion in background thread...")

    if file_format == "zeiss":
        dataset = reader(path, scene_index=scene_index, chunks=chunks)
    else:
        dataset = reader(path, chunks=chunks)

    axes = _dataset_axes(dataset)
    print("Dataset axes:", axes)

    if len(channels) == 0 or "c" not in axes:
        channels_index = None
    else:
        channels_index = [
            dataset.metadata["channel_names"].index(ch)
            for ch in channels
            if ch in dataset.metadata.get("channel_names", [])
        ]

    subset_kwargs = {}
    tr = _present_range(dataset, "t", t_range)
    zr = _present_range(dataset, "z", z_range)
    yr = _present_range(dataset, "y", y_range)
    xr = _present_range(dataset, "x", x_range)

    if tr is not None and _axis_size(dataset, "t") > 1:
        subset_kwargs["T"] = list(range(tr[0], tr[1] + 1))
    if channels_index is not None and _axis_size(dataset, "c") > 1:
        subset_kwargs["C"] = channels_index
    if zr is not None and _axis_size(dataset, "z") > 1:
        subset_kwargs["Z"] = list(range(zr[0], zr[1] + 1))
    if yr is not None:
        subset_kwargs["Y"] = list(range(yr[0], yr[1] + 1))
    if xr is not None:
        subset_kwargs["X"] = list(range(xr[0], xr[1] + 1))

    if subset_kwargs:
        dataset.subset_dataset(**subset_kwargs, rebuild_pyramid=False)

    chunks = _chunk_shape_for_dataset(
        dataset,
        chunk_z=chunks[2] if len(chunks) > 2 else 1,
        chunk_y=chunks[3] if len(chunks) > 3 else 512,
        chunk_x=chunks[4] if len(chunks) > 4 else 512,
    )

    print("Requested input chunks:", chunks)
    print("Chunks before pyramid:", [arr.chunksize for arr in dataset.data])

    dataset.build_pyramid(num_levels=n_levels, downscale_factor=downscale_factor)

    dataset.data = [
        arr.rechunk(chunks) if arr.ndim == len(chunks) else arr
        for arr in dataset.data
    ]
    dataset.metadata["chunksize"] = [tuple(arr.chunksize) for arr in dataset.data]

    print("Chunks after pyramid:", [arr.chunksize for arr in dataset.data])

    ngff_version = "0.4" if zarr_format == 2 else "0.5"
    to_zarr_kwargs = {
        "zarr_format": zarr_format,
        "ngff_version": ngff_version,
        "drop_singleton": bool(drop_singleton),
        "shards": shards,
    }
    if num_workers is not None:
        to_zarr_kwargs["num_workers"] = int(num_workers)
    if shards is not None:
        if shard_target_mb is not None:
            to_zarr_kwargs["shard_target_mb"] = shard_target_mb
        if shard_exclude_axes is not None:
            to_zarr_kwargs["shard_exclude_axes"] = shard_exclude_axes

    dataset.to_zarr(output_path, **to_zarr_kwargs)

    return output_path

@thread_worker
def convert_worker(**kwargs):
    """Background worker wrapper used by the napari conversion widget."""
    return _run_conversion(**kwargs)


def convert_widget():
    """Create the main PyMIF napari conversion widget."""
    viewer = current_viewer()
    
    def lock_roi_in_3d(event=None):
        if "ROI" not in viewer.layers:
            return
        if viewer.dims.ndisplay == 3:
            viewer.layers["ROI"].editable = False
            viewer.layers["ROI"].selectable = False
            make_convert_widget.y_range.enabled = False
            make_convert_widget.x_range.enabled = False
        else:
            viewer.layers["ROI"].editable = True
            viewer.layers["ROI"].selectable = True
            dataset = _state.get("dataset")
            make_convert_widget.y_range.enabled = bool(dataset and _axis_index(dataset, "y") is not None)
            make_convert_widget.x_range.enabled = bool(dataset and _axis_index(dataset, "x") is not None)

    def ensure_crop_layers(dataset):
        axes = _dataset_axes(dataset)
        if "ROI" not in viewer.layers and {"y", "x"}.issubset(axes):
            ymax = _axis_size(dataset, "y") - 1
            xmax = _axis_size(dataset, "x") - 1
            rect = np.array([
                [0, 0],
                [0, xmax],
                [ymax, xmax],
                [ymax, 0],
            ])
            layer = viewer.add_shapes(
                data=rect,
                shape_type="rectangle",
                name="ROI",
                edge_color="orange",
                face_color="transparent",
                edge_width=10,
                ndim=2,
                scale=_scale_for_axes(dataset, "yx"),
                **_units_kwargs(dataset, "yx"),
            )
            layer.mode = "add_rectangle"

        if "Zrange" not in viewer.layers and {"z", "y", "x"}.issubset(axes):
            points = np.array([
                [0, _axis_size(dataset, "y") // 2, _axis_size(dataset, "x") // 2],
                [_axis_size(dataset, "z") - 1, _axis_size(dataset, "y") // 2, _axis_size(dataset, "x") // 2],
            ])
            layer = viewer.add_points(
                data=points,
                name="Zrange",
                face_color="lime",
                size=20,
                ndim=3,
                **points_projection_kwargs(),
                scale=_scale_for_axes(dataset, "zyx"),
                **_units_kwargs(dataset, "zyx"),
            )
            layer.mode = "add"

        if "CropBox" not in viewer.layers and {"z", "y", "x"}.issubset(axes):
            layer = viewer.add_shapes(
                data=np.empty((0, 2, 3)),
                shape_type="polygon",
                name="CropBox",
                edge_color="cyan",
                face_color="transparent",
                ndim=3,
                scale=_scale_for_axes(dataset, "zyx"),
                **_units_kwargs(dataset, "zyx"),
            )
            layer.editable = False          # prevents adding/moving/editing shapes
            layer.selectable = False        # prevents selecting the shape
            layer.mode = "pan_zoom"         # ensures no shape tool is active
            layer.opacity = 0.7
            layer.edge_width = 2
            layer.blending = "additive"

    def update_3d_box():
        if any(name not in viewer.layers for name in ("ROI", "Zrange", "CropBox")):
            return
        roi_layer = viewer.layers["ROI"]
        zrange_layer = viewer.layers["Zrange"]
        if len(roi_layer.data) == 0 or len(zrange_layer.data) < 2:
            return
        zvals = [d[0] for d in zrange_layer.data]

        zrange = list(zvals[:2])

        roi = [list(r) for r in roi_layer.data[0]]

        polygon = [
            [[zrange[0]] + roi[0], [zrange[0]] + roi[1], [zrange[0]] + roi[2], [zrange[0]] + roi[3]],
            [[zrange[1]] + roi[0], [zrange[1]] + roi[1], [zrange[1]] + roi[2], [zrange[1]] + roi[3]],
            [[zrange[0]] + roi[0], [zrange[1]] + roi[0], [zrange[1]] + roi[1], [zrange[0]] + roi[1]],
            [[zrange[0]] + roi[0], [zrange[1]] + roi[0], [zrange[1]] + roi[3], [zrange[0]] + roi[3]],
            [[zrange[0]] + roi[1], [zrange[1]] + roi[1], [zrange[1]] + roi[2], [zrange[0]] + roi[2]],
            [[zrange[0]] + roi[3], [zrange[1]] + roi[3], [zrange[1]] + roi[2], [zrange[0]] + roi[2]],
        ]

        viewer.layers["CropBox"].data = polygon
        viewer.layers["CropBox"].refresh()

    def clamp(v, vmin, vmax):
        return int(max(vmin, min(v, vmax)))


    def roi_to_widget(event=None):
        if _syncing["roi"]:
            return
        _syncing["roi"] = True
    
        try:
            if "ROI" not in viewer.layers:
                return
            layer = viewer.layers["ROI"]
            if layer is None or len(layer.data) == 0:
                return

            (y0, x0), (_, _), (y1, x1), (_, _) = layer.data[0]

            y0, y1 = sorted([int(y0), int(y1)])
            x0, x1 = sorted([int(x0), int(x1)])

            dataset = _state["dataset"]
            b = {
                "y_max": _axis_size(dataset, "y") - 1,
                "x_max": _axis_size(dataset, "x") - 1,
            }

            y0 = clamp(y0, 0, b["y_max"])
            y1 = clamp(y1, 0, b["y_max"])
            x0 = clamp(x0, 0, b["x_max"])
            x1 = clamp(x1, 0, b["x_max"])

            layer.data = [np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]])]

            make_convert_widget.y_range.value = (y0, y1)
            make_convert_widget.x_range.value = (x0, x1)

            update_3d_box()
        finally:
            _syncing["roi"] = False

    def zpoints_to_widget(event=None):
        if _syncing["z"]:
            return
        _syncing["z"] = True
        
        try:
            if "Zrange" not in viewer.layers:
                return
            layer = viewer.layers["Zrange"]
            if layer is None or len(layer.data) < 2:
                return

            zs = sorted(int(p[0]) for p in layer.data[:2])

            make_convert_widget.z_range.value = (zs[0], zs[1])

            update_3d_box()
        finally:
            _syncing["z"] = False

    def widget_to_roi(event=None):
        if _syncing["roi"]:
            return
        _syncing["roi"] = True

        try:    
            if "ROI" not in viewer.layers:
                return
            layer = viewer.layers["ROI"]

            dataset = _state["dataset"]
            b = {
                "y_max": _axis_size(dataset, "y") - 1,
                "x_max": _axis_size(dataset, "x") - 1,
            }

            y0 = clamp(make_convert_widget.y_range.value[0], 0, b["y_max"])
            y1 = clamp(make_convert_widget.y_range.value[1], 0, b["y_max"])
            x0 = clamp(make_convert_widget.x_range.value[0], 0, b["x_max"])
            x1 = clamp(make_convert_widget.x_range.value[1], 0, b["x_max"])

            y0, y1 = sorted([y0, y1])
            x0, x1 = sorted([x0, x1])

            rect = np.array([
                [y0, x0],
                [y0, x1],
                [y1, x1],
                [y1, x0],
            ])

            layer.data = [rect]

            update_3d_box()
        finally:
            _syncing["roi"] = False
    
    def widget_to_zpoints(event=None):
        if _syncing["z"]:
            return
        _syncing["z"] = True
        
        try:
            if "Zrange" not in viewer.layers:
                return
            layer = viewer.layers["Zrange"]

            z0 = make_convert_widget.z_range.value[0]
            z1 = make_convert_widget.z_range.value[1]

            prev_points = layer.data

            points = np.array([
                [z0, prev_points[0,1], prev_points[0,2]],
                [z1, prev_points[1,1], prev_points[1,2]],
            ])

            layer.data = points

            update_3d_box()
        finally:
            _syncing["z"] = False
        
    def keep_single_roi(event=None):
        if _syncing["roi"]:
            return
        _syncing["roi"] = True

        try:    
            if "ROI" not in viewer.layers:
                return
            layer = viewer.layers["ROI"]
            if len(layer.data) > 1:
                layer.data = [layer.data[-1]]

        finally:
            _syncing["roi"] = False

    def keep_two_zpoints(event=None):
        if _syncing["z"]:
            return
        _syncing["z"] = True
        
        try:
            if "Zrange" not in viewer.layers:
                return
            layer = viewer.layers["Zrange"]
            if len(layer.data) > 2:
                layer.data = layer.data[-2:]
        finally:
            _syncing["z"] = False

    _state = {"dataset": None}
    _syncing = {"roi": False, "z": False}


    @magicgui(
        call_button="Visualize in napari",
        input_path={"widget_type": "FileEdit", "mode": "d"},
        scene_index={"widget_type": "SpinBox", "min": 0, "max": 100, "step": 1},
        file_format={"choices": ["viventis", "opera", "luxendo", "zarr", "zeiss","scape"]},
    )
    def make_visualize_widget(
        file_format="zarr",
        scene_index=0,
        input_path: FileEdit = None,
    ):
        dataset = _state["dataset"]
        if dataset is None:
            return

        viewer.layers.clear()

        if make_visualize_widget.file_format.value in ["zarr"]:
            viewer.open(make_visualize_widget.input_path.value, plugin='napari-ome-zarr')
        else:
            dataset.visualize(viewer=viewer)


        # initialize ROI to full image when spatial axes are available
        ensure_crop_layers(dataset)

        widget_to_roi()
        widget_to_zpoints()
        update_3d_box()

        if "ROI" in viewer.layers:
            viewer.layers["ROI"].mode = "add_rectangle"
            viewer.layers["ROI"].events.data.connect(keep_single_roi)
            viewer.layers["ROI"].events.data.connect(roi_to_widget)
            make_convert_widget.y_range.changed.connect(widget_to_roi)
            make_convert_widget.x_range.changed.connect(widget_to_roi)

        if "Zrange" in viewer.layers:
            viewer.layers["Zrange"].events.data.connect(keep_two_zpoints)
            viewer.layers["Zrange"].events.data.connect(zpoints_to_widget)
            make_convert_widget.z_range.changed.connect(widget_to_zpoints)

    enable_drag_and_drop(make_visualize_widget.input_path)
    # ---

    @magicgui(
        call_button="Convert to zarr",

        chunk_z={"label": "Chunk Z", "min": 1, "max": 2**16, "step": 1, "value": 16},
        chunk_y={"label": "Chunk Y", "min": 8, "max": 2**16, "step": 1, "value": 512},
        chunk_x={"label": "Chunk X", "min": 8, "max": 2**16, "step": 1, "value": 512},
        
        n_levels={"label": "Resolution levels", "min": 1, "max": 10, "value": 5},
        downscale_z={"label": "Downscale Z", "min": 1, "max": 64, "step": 1, "value": 2},
        downscale_y={"label": "Downscale Y", "min": 1, "max": 64, "step": 1, "value": 2},
        downscale_x={"label": "Downscale X", "min": 1, "max": 64, "step": 1, "value": 2},
        zarr_format={"label": "Zarr format", "choices": [2, 3], "value": 3},

        shards={"label": "Sharding (zarr v3 only)", "choices": ["none", "auto"], "value": "auto"},
        shard_target_mb={"label": "Shard target (MB)", "min": 1, "max": 100000, "step": 1, "value": 1024},
        shard_exclude_axes={
            "label": "Never merge axes",
            "choices": ["t", "c", "z", "y", "x"],
            "widget_type": "Select",
            "allow_multiple": True,
            "value": ("t", "c"),
        },
        drop_singleton={"label": "Drop singleton axes", "widget_type": "CheckBox", "value": True},
        num_workers={"label": "Write threads (0 = auto)", "min": 0, "max": 256, "step": 1, "value": 0},

        t_range={"label": "T range", "widget_type": "RangeSlider", "min": 0, "max": 2**16, "step": 1, "value": (0, 2**16)},
        single_t = {"label": "Single T frame", "widget_type": "CheckBox", "value": False},
        z_range={"label": "Z range", "widget_type": "RangeSlider", "min": 0, "max": 2**16, "step": 1, "value": (0, 2**16)},
        single_z = {"label": "Single Z plane", "widget_type": "CheckBox", "value": False},
        y_range={"label": "Y range", "widget_type": "RangeSlider", "min": 0, "max": 2**16, "step": 1, "value": (0, 2**16)},
        x_range={"label": "X range", "widget_type": "RangeSlider", "min": 0, "max": 2**16, "step": 1, "value": (0, 2**16)},

        channels={
            "label": "Channels",
            "choices": ["ch0", "ch1", "ch2"],
            "widget_type": "Select",
            "allow_multiple": True,
        },

        output_path={"widget_type": "FileEdit", "mode": "d"},
    )
    def make_convert_widget(
        t_range=(0,1000),
        single_t=False,
        z_range=(0,1000),
        single_z=False,
        y_range=(0,1000),
        x_range=(0,1000),
        channels=(),
        chunk_x=512,
        chunk_y=512,
        chunk_z=16,
        n_levels=5,
        downscale_z=2,
        downscale_y=2,
        downscale_x=2,
        zarr_format=3,
        shards="auto",
        shard_target_mb=1024,
        shard_exclude_axes=("t", "c"),
        drop_singleton=True,
        num_workers=0,
        output_path: FileEdit = None,
    ):

        reader = dataset_reader(make_visualize_widget.file_format.value)
        path = make_visualize_widget.input_path.value
        scene_index = make_visualize_widget.scene_index.value
        file_format = make_visualize_widget.file_format.value

        chunks = (1, 1, chunk_z, chunk_y, chunk_x)
        downscale_factor = (downscale_z, downscale_y, downscale_x)

        shards_value = None if shards == "none" else "auto"

        worker = convert_worker(
                reader=reader,
                path=path,
                scene_index=scene_index,
                chunks=chunks,
                t_range=t_range,
                z_range=z_range,
                y_range=y_range,
                x_range=x_range,
                channels=channels,
                n_levels=n_levels,
                downscale_factor=downscale_factor,
                output_path=output_path,
                file_format=file_format,
                zarr_format=zarr_format,
                shards=shards_value,
                shard_target_mb=float(shard_target_mb) if shards_value is not None else None,
                shard_exclude_axes=tuple(shard_exclude_axes) if shards_value is not None else None,
                drop_singleton=bool(drop_singleton),
                num_workers=int(num_workers) if int(num_workers) > 0 else None,
            )
        
        _set_convert_enabled(False)
        make_batch_csv_widget.enabled = False
        make_visualize_widget.enabled = False

        @worker.returned.connect
        def _on_done(result_path):
            print(f"Conversion completed: {result_path}")
            _set_convert_enabled(True)
            make_visualize_widget.enabled = True

        @worker.errored.connect
        def _on_error(err):
            print("Conversion failed:", err)
            _set_convert_enabled(True)
            make_visualize_widget.enabled = True

        worker.start()

    ui = {}  # widgets created later, when the dock is composed

    output_label = QLabel("Load a dataset to see the output size.")
    output_label.setWordWrap(True)
    output_label.setStyleSheet("QLabel { color: #aaaaaa; }")

    def _update_output_summary(*_):
        dataset = _state["dataset"]
        if dataset is None:
            return
        w = make_convert_widget
        try:
            text = output_summary(
                dataset,
                t_range=w.t_range.value,
                z_range=w.z_range.value,
                y_range=w.y_range.value,
                x_range=w.x_range.value,
                channels=w.channels.value,
                drop_singleton=w.drop_singleton.value,
            )
        except Exception as e:  # the recap must never break the widget
            text = f"cannot estimate ({e})"
        output_label.setText(f"Output: {text}")

    for name in ("t_range", "z_range", "y_range", "x_range", "channels", "drop_singleton"):
        getattr(make_convert_widget, name).changed.connect(_update_output_summary)

    def _set_convert_enabled(enabled):
        """Enable or disable the conversion panel *and* its Convert button.

        The button sits below the (collapsible) panel, outside it, so that it stays
        visible when the panel is collapsed; disabling the panel alone would
        therefore not lock it, and a
        second click could start a second conversion. Its holder is disabled as
        a whole because magicgui re-enables its own call button after every
        call, which would undo disabling just the button.
        """
        make_convert_widget.enabled = enabled
        if "button_holder" in ui:
            ui["button_holder"].setEnabled(enabled)

    @make_convert_widget.zarr_format.changed.connect
    def _update_sharding_availability(zarr_format):
        # Sharding is a zarr v3 / NGFF 0.5 feature only.
        sharding_available = int(zarr_format) == 3
        make_convert_widget.shards.enabled = sharding_available
        make_convert_widget.shard_target_mb.enabled = sharding_available
        make_convert_widget.shard_exclude_axes.enabled = sharding_available
        if not sharding_available:
            make_convert_widget.shards.value = "none"

    @magicgui(
        call_button="Append to batch CSV",
        batch_csv_path={"widget_type": "FileEdit", "mode": "w"},
    )
    def make_batch_csv_widget(batch_csv_path: FileEdit = None):
        dataset = _state["dataset"]
        if dataset is None:
            return

        csv_path = Path(batch_csv_path) if batch_csv_path else None
        if csv_path is None:
            csv_path = Path(make_visualize_widget.input_path.value).parent / "batch.csv"

        selected_channel_names = list(make_convert_widget.channels.value) if "c" in _dataset_axes(dataset) else []
        normalized_channel_names = [c.replace(" ", "_") for c in selected_channel_names]
        subset = _subset_string_from_widget(
            dataset,
            make_convert_widget.t_range.value,
            make_convert_widget.z_range.value,
            make_convert_widget.y_range.value,
            make_convert_widget.x_range.value,
            selected_channel_names,
        )

        selected_colors = _channel_colors_from_metadata(dataset, selected_channel_names)
        shards_selected = make_convert_widget.shards.value
        row = {
            "input": str(Path(make_visualize_widget.input_path.value).resolve()),
            "microscope": str(make_visualize_widget.file_format.value),
            "output": str(Path(make_convert_widget.output_path.value)),
            "chunk_size": " ".join(str(v) for v in (1, 1, make_convert_widget.chunk_z.value, make_convert_widget.chunk_y.value, make_convert_widget.chunk_x.value)),
            "max_size(MB)": "",
            "scene_index": str(make_visualize_widget.scene_index.value),
            "zarr_format": str(make_convert_widget.zarr_format.value),
            "downscale_factor": " ".join(str(v) for v in (make_convert_widget.downscale_z.value, make_convert_widget.downscale_y.value, make_convert_widget.downscale_x.value)),
            "subset": subset,
            "channel_colors": " ".join(selected_colors),
            "channel_names": " ".join(normalized_channel_names),
            "num_levels": str(make_convert_widget.n_levels.value),
            # A blank cell means "auto" in the CLI, so disabling must be explicit.
            "shards": "none" if shards_selected == "none" else str(shards_selected),
            "shard_target_mb": "" if shards_selected == "none" else str(make_convert_widget.shard_target_mb.value),
            "shard_exclude_axes": (
                "" if shards_selected == "none"
                else (" ".join(make_convert_widget.shard_exclude_axes.value) or "none")
            ),
            "drop_singleton": str(bool(make_convert_widget.drop_singleton.value)).lower(),
            "num_workers": "" if int(make_convert_widget.num_workers.value) <= 0 else str(int(make_convert_widget.num_workers.value)),
        }
        csv_path = _append_batch_csv(csv_path, row)
        print(f"Appended batch row to {csv_path}. Run it with:\n  pymif convert --batch \"{csv_path}\"")

    make_batch_csv_widget.batch_csv_path.value = str((Path(make_visualize_widget.input_path.value).parent if make_visualize_widget.input_path.value else Path.cwd()) / "batch.csv")
    make_batch_csv_widget.enabled = False


    # -------------------------
    # Input path callback
    # -------------------------

    @make_visualize_widget.file_format.changed.connect
    def _update_mode(file_format):
        if file_format == "zeiss": 
            make_visualize_widget.scene_index.enabled = True 
        else: 
            make_visualize_widget.scene_index.enabled = False
        
        if file_format in ["zarr", "viventis", "luxendo"]:
            make_visualize_widget.input_path.mode = "d" 
        else: 
            make_visualize_widget.input_path.mode = "r"

    @make_visualize_widget.input_path.changed.connect
    def _on_input_path_change(path):
        if not path:
            return
        
        reader = dataset_reader(make_visualize_widget.file_format.value)

        if make_visualize_widget.file_format.value == "zeiss":
            dataset = reader(path, scene_index=make_visualize_widget.scene_index.value)
        else:
            dataset = reader(path)

        for i in dataset.metadata:
            print(f"{i.upper()}: {dataset.metadata[i]}")

        _state["dataset"] = dataset

        suffix = ""
        if path.suffix == ".zarr":
            suffix = "_1"

        if path.is_file() or path.suffix==".zarr":
            name = path.stem          # removes extension
        else:
            name = path.name          # directory name as-is

        default_output = path.parent / f"{name}{suffix}.zarr"
        make_convert_widget.output_path.value = default_output
        make_batch_csv_widget.batch_csv_path.value = str(default_output.parent / "batch.csv")
        make_batch_csv_widget.enabled = True


        _set_convert_enabled(True)

        axes = _dataset_axes(dataset)
        chunk_size = get_chunk_size(dataset.metadata["size"][0], max_size_mb=100, axes=axes)
        chunk_map = dict(zip(axes, chunk_size))
        num_levels = get_n_levels(dataset.metadata["size"][0], axes=axes)

        make_convert_widget.chunk_x.max = _axis_size(dataset, "x") + 1
        make_convert_widget.chunk_x.value = chunk_map.get("x", 1)
        make_convert_widget.chunk_x.enabled = "x" in axes

        make_convert_widget.chunk_y.max = _axis_size(dataset, "y") + 1
        make_convert_widget.chunk_y.value = chunk_map.get("y", 1)
        make_convert_widget.chunk_y.enabled = "y" in axes

        make_convert_widget.chunk_z.max = _axis_size(dataset, "z") + 1
        make_convert_widget.chunk_z.value = chunk_map.get("z", 1)
        make_convert_widget.chunk_z.enabled = "z" in axes

        make_convert_widget.n_levels.value = num_levels
        make_convert_widget.downscale_z.enabled = "z" in axes
        make_convert_widget.downscale_y.enabled = "y" in axes
        make_convert_widget.downscale_x.enabled = "x" in axes

        _set_range_widget(make_convert_widget.t_range, _axis_size(dataset, "t"), enabled="t" in axes)
        _set_range_widget(make_convert_widget.z_range, _axis_size(dataset, "z"), enabled="z" in axes)
        _set_range_widget(make_convert_widget.y_range, _axis_size(dataset, "y"), enabled="y" in axes)
        _set_range_widget(make_convert_widget.x_range, _axis_size(dataset, "x"), enabled="x" in axes)
        make_convert_widget.single_t.enabled = "t" in axes
        make_convert_widget.single_z.enabled = "z" in axes

        channel_names = dataset.metadata.get("channel_names", []) if "c" in axes else []
        n_channels = len(channel_names)
        row_height = 20  # approx height per item in pixels
        channels_widget = make_convert_widget.channels.native
        channels_widget.setMaximumHeight(row_height * max(1, n_channels))
        make_convert_widget.channels.choices = channel_names
        make_convert_widget.channels.value = tuple(channel_names)
        make_convert_widget.channels.enabled = n_channels > 0

        make_convert_widget.output_path.value = default_output
        _update_output_summary()

        viewer.layers.clear()

    make_visualize_widget.scene_index.enabled = False
    _set_convert_enabled(False)
    viewer.dims.events.ndisplay.connect(lock_roi_in_3d)
    # lock_roi_in_3d()

    def sync_single_z(checked):
        dataset = _state.get("dataset")
        has_z = bool(dataset and _axis_index(dataset, "z") is not None and _axis_size(dataset, "z") > 1)
        if checked and has_z:
            z_val = make_convert_widget.z_range.value
            # clamp to single plane
            make_convert_widget.z_range.value = (z_val[0], z_val[0])
            make_convert_widget.z_range.enabled = False
        else:
            make_convert_widget.z_range.enabled = has_z

    make_convert_widget.single_z.changed.connect(sync_single_z)

    def sync_single_t(checked):
        dataset = _state.get("dataset")
        has_t = bool(dataset and _axis_index(dataset, "t") is not None and _axis_size(dataset, "t") > 1)
        if checked and has_t:
            t_val = make_convert_widget.t_range.value
            # clamp to single frame
            make_convert_widget.t_range.value = (t_val[0], t_val[0])
            make_convert_widget.t_range.enabled = False
        else:
            make_convert_widget.t_range.enabled = has_t

    make_convert_widget.single_t.changed.connect(sync_single_t)
    # -------------------------
    # Compose single dock widget
    # -------------------------

    container = QWidget()
    layout = QVBoxLayout(container)
    
    title1_label = QLabel("Dataset loading:")
    title1_label.setStyleSheet("""
        QLabel {
            font-weight: bold;
            font-size: 13px;
            padding: 2px 0px;
        }
    """)

    # Collapsible section header, same pattern as "Additional parameters".
    title2_btn = QToolButton()
    title2_btn.setCheckable(True)
    title2_btn.setChecked(False)
    title2_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

    def _toggle_conversion(checked):
        # Only the parameters collapse: the title and the Convert button of the
        # box stay visible.
        make_convert_widget.native.setVisible(checked)
        title2_btn.setText("Dataset conversion: ▾" if checked else "Dataset conversion: ▸")
        title2_btn.setStyleSheet("""
            QToolButton {
                font-weight: bold;
                font-size: 13px;
                padding: 2px 0px;
                color: %s;
            }
            QToolButton:hover {
                color: white;
            }
        """ % ("#ffffff" if checked else "#aaaaaa"))

    title2_btn.toggled.connect(_toggle_conversion)

    section_loading = _make_section(title1_label, make_visualize_widget.native)
    section_conversion = _make_section(title2_btn)
    layout.setSpacing(6)
    layout.setContentsMargins(4, 4, 4, 4)
    layout.addWidget(section_loading)
    layout.addWidget(section_conversion)

    advanced_btn = QToolButton()
    advanced_btn.setText("Additional parameters ▸")
    advanced_btn.setStyleSheet("""
        QToolButton {
            font-size: 13px;
            font-weight: 600;
            color: #cccccc;
            padding: 4px 0px;
        }
        QToolButton:hover {
            color: white;
        }
        """)
    advanced_btn.setCheckable(True)
    advanced_btn.setChecked(False)
    advanced_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

    def _toggle_advanced(checked):
        for w in advanced_widgets:
            w.setVisible(checked)
        advanced_btn.setText("Additional parameters ▾" if checked else "Additional parameters ▸")
        advanced_btn.setStyleSheet("""
            QToolButton {
                font-size: 13px;
                font-weight: 600;
                color: %s;
            }
        """ % ("#ffffff" if checked else "#aaaaaa"))
        
    advanced_btn.toggled.connect(_toggle_advanced)

    make_convert_widget.native.layout().insertWidget(7, advanced_btn)

    advanced_widgets = _compact_advanced_rows(make_convert_widget, advanced_btn)
    for w in advanced_widgets:
        w.setVisible(False)

    # Tighter vertical spacing and a shorter channel list keep the panel small.
    make_convert_widget.native.layout().setSpacing(2)
    make_convert_widget.native.layout().setContentsMargins(0, 0, 0, 0)
    make_convert_widget.channels.native.setMaximumHeight(64)

    section_conversion.layout().addWidget(make_convert_widget.native)
    _toggle_conversion(title2_btn.isChecked())

    title_csv_label = QLabel("Batch CSV export:")
    title_csv_label.setStyleSheet("""
        QLabel {
            font-weight: bold;
            font-size: 13px;
            padding: 2px 0px;
        }
    """)
    layout.addWidget(_make_section(title_csv_label, make_batch_csv_widget.native))
    # layout.addWidget(reset_roi_widget.native)

    # make_visualize_widget.input_path.tooltip = (
    #     "Select a folder containing OME-Zarr data (.zarr)"
    # )

    log_widget = QTextEdit()
    log_widget.setReadOnly(True)
    log_widget.setMinimumHeight(110)  # the dock splitter lets the user drag it taller
    log_widget.setStyleSheet("""
        QTextEdit {
            background-color: #111;
            color: #ddd;
            font-family: Consolas, Menlo, monospace;
            font-size: 12px;
        }
    """)

    stdout_stream = EmittingStream()
    stderr_stream = EmittingStream()

    previous_streams = (sys.stdout, sys.stderr)
    sys.stdout = stdout_stream
    sys.stderr = stderr_stream

    def _restore_console(*_):
        # Closing the widget deletes its log: output must not keep going to it.
        if sys.stdout is stdout_stream:
            sys.stdout = previous_streams[0]
        if sys.stderr is stderr_stream:
            sys.stderr = previous_streams[1]

    log_widget.destroyed.connect(_restore_console)

    append_with_time = make_log_appender(log_widget)

    stdout_stream.text_written.connect(append_with_time)
    stderr_stream.text_written.connect(append_with_time)

    title3_label = QLabel("PyMIF Log:")
    title3_label.setStyleSheet("""
        QLabel {
            font-weight: bold;
            font-size: 13px;
            padding: 2px 0px;
        }
    """)

    clear_btn = QPushButton("Clear")
    clear_btn.setToolTip("Clear the log")
    clear_btn.setMaximumWidth(60)
    clear_btn.clicked.connect(log_widget.clear)

    log_header = QWidget()
    log_header_layout = QHBoxLayout(log_header)
    log_header_layout.setContentsMargins(0, 0, 0, 0)
    log_header_layout.addWidget(title3_label)
    log_header_layout.addStretch(1)
    log_header_layout.addWidget(clear_btn)
    log_section = _make_section(log_header, log_widget)
    log_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    # The Convert button stays inside the "Dataset conversion" box, below the
    # (collapsible) parameters, so it is visible even when they are collapsed.
    convert_button = make_convert_widget._call_button.native
    make_convert_widget.native.layout().removeWidget(convert_button)
    button_holder = make_button_holder(convert_button)
    ui["button_holder"] = button_holder  # locked as a whole while converting (see _set_convert_enabled)
    button_holder.setEnabled(make_convert_widget.native.isEnabled())
    section_conversion.layout().addWidget(output_label)  # visible with the parameters collapsed
    section_conversion.layout().addWidget(button_holder)

    # All three boxes (input, conversion, batch) scroll together; the log is a
    # resizable pane below.
    layout.addStretch(1)
    dock = compose_dock(container, bottom=log_section)
    # exposed for tests
    dock.visualize_widget, dock.convert_widget = make_visualize_widget, make_convert_widget
    dock.output_label = output_label
    return dock
