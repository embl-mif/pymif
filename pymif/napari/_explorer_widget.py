"""napari widget to explore the features of segmented objects, with napari-clusters-plotter.

Three things are loaded (dropped on their field, all at once anywhere on the panel, or browsed):

* an intensity OME-Zarr (all channels and resolution levels, read lazily),
* the labels OME-Zarr (optional: else the first group in ``<intensity>.ome.zarr/labels/``),
* a CSV with one row per object: a label column and any features.

The features are plotted by napari-clusters-plotter, under the panel, with everything it can do
(x / y features, colour of the plot and the labels by any feature, selections to make clusters,
histogram, ...). The plot is linked with the viewer:

* click a point: the viewer goes to that object (its plane, centred, zoomed in) and selects its label,
* hover a point: a thumbnail of the object (XY and XZ through its centre, the intensity with the
  object outlined in magenta) appears next to the mouse,
* pick a label in napari: its point is highlighted in the plot.

Where the objects are comes from centroid columns of the CSV: ``centroid-0/1/2`` (skimage
``regionprops_table``), ``centroid_z/y/x`` or ``z/y/x``, in voxels of the labels (full resolution) or
in µm. Without them they are computed from a low resolution level of the labels (slow for big
images, and small objects can be missed). The size of the thumbnail comes from ``bbox-*`` columns
(regionprops) or a size column (``area``, ``volume``, ...).
"""
from __future__ import annotations

import traceback
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from napari import current_viewer
from napari.qt.threading import create_worker
from napari.utils import Colormap
from napari.utils.notifications import show_error, show_info, show_warning
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QCursor, QGuiApplication, QImage, QPixmap
from qtpy.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QVBoxLayout, QWidget,
)
from scipy.spatial import cKDTree
from skimage.segmentation import find_boundaries

import pymif.microscope_manager as mm
from ._layout import compose_dock, make_button_holder, make_section

try:
    import napari_clusters_plotter._utilities as ncp_utilities
    from biaplotter.artists import Scatter
    from napari_clusters_plotter import PlotterWidget
except ImportError as error:  # the widget then says how to install it
    ncp_utilities = Scatter = PlotterWidget = None
    _PLOTTER_IMPORT_ERROR = error
else:
    _PLOTTER_IMPORT_ERROR = None

#: Guessed label column of the table.
LABEL_COLUMNS = ("label", "label_id", "labels", "id", "object_id", "cell_id")
#: Columns with the size of an object (voxels).
SIZE_COLUMNS = ("area", "volume", "num_pixels", "n_voxels", "voxels", "size")
#: Columns with the time point of an object.
TIME_COLUMNS = ("frame", "t", "timepoint", "time")
#: Suffixes of a features table.
TABLE_SUFFIXES = (".csv", ".tsv", ".txt")
#: Key of the labels layer metadata with the label of every row of its features (0 first).
LABEL_IDS_KEY = "pymif_label_ids"

#: Size of one thumbnail view on the screen (pixels).
THUMBNAIL_PX = 170
#: Half width of a thumbnail, in object radii.
THUMBNAIL_CONTEXT = 2.5
#: Thumbnails are read from the finest level with at most this many voxels per axis.
MAX_BOX_PX = 128
#: A point is hovered when the mouse is this close (screen pixels).
HOVER_RADIUS_PX = 7
#: Wait this long on a point before reading its thumbnail (ms).
HOVER_DELAY_MS = 40
#: Number of thumbnails kept in memory.
MAX_THUMBNAILS = 500
#: After a click the viewer shows this many times the thumbnail field of view.
JUMP_CONTEXT = 1.5
#: Smallest size of the plot points (matplotlib points², the area of a marker).
MIN_POINT_SIZE = 10
#: Outline of the object, and of the other objects, in the thumbnail.
OBJECT_RGB = (255, 0, 255)
OTHER_RGB = (0, 200, 255)

_TITLE_STYLE = "QLabel { font-weight: bold; font-size: 13px; padding: 2px 0px; }"


# ---------------------------------------------------------------------------
# napari-clusters-plotter
# ---------------------------------------------------------------------------

def _patch_plotter():
    """Two changes to napari-clusters-plotter, for the layers and the plot of this widget only.

    * It colours a labels layer by ``np.unique`` of the whole label image on every redraw (minutes
      for a big image) and gives the colour of row i of the features to the i-th label. The rows
      here are the labels of the table, sorted, so they are given in the layer metadata.
    * biaplotter makes the points smaller the more there are (8000 / n): with tens of thousands of
      objects they can't be seen any more. A scatter plot with ``pymif_min_size`` keeps that size.
    """
    if getattr(ncp_utilities, "_pymif_patched", False):
        return
    unique_values = ncp_utilities._get_unique_values
    default_size = Scatter.default_size.fget

    def labels_of_features(layer):
        ids = layer.metadata.get(LABEL_IDS_KEY)
        return ids if ids is not None else unique_values(layer)

    ncp_utilities._get_unique_values = labels_of_features
    Scatter.default_size = property(lambda self: max(getattr(self, "pymif_min_size", 0), default_size(self)))
    ncp_utilities._pymif_patched = True


# ---------------------------------------------------------------------------
# helpers (no Qt, unit tested)
# ---------------------------------------------------------------------------

class Pyramid:
    """All resolution levels of an OME-Zarr image (lazy dask arrays) with their axes and voxel sizes (µm)."""

    def __init__(self, levels, metadata):
        self.levels, self.metadata = list(levels), metadata
        self.axes = metadata["axes"].lower()
        self.scales = [np.asarray(s, float) for s in metadata["scales"]]

    @property
    def spatial(self):
        return [a for a in self.axes if a in "zyx"]

    def drop_channel(self):
        """Labels: drop a channel axis (of size 1)."""
        if "c" in self.axes:
            c = self.axes.index("c")
            self.levels = [level[(slice(None),) * c + (0,)] for level in self.levels]
            self.axes = self.axes.replace("c", "")

    def layer_scale(self):
        """napari scale of the layer (the channel axis, if any, becomes separate layers)."""
        spatial = iter(self.scales[0])
        return [next(spatial) if a in "zyx" else 1.0 for a in self.axes if a != "c"]

    def box(self, centre_um, half_um, channel=0, t=0):
        """
        Cube of side 2 x half_um around centre_um ({axis: µm}) from the finest level that has at most
        MAX_BOX_PX voxels per axis, zero padded at the image borders.
        Returns (array of the spatial axes, index of its first voxel, voxel size, offset of the level (µm)).
        """
        level = next((i for i, s in enumerate(self.scales) if np.all(2 * half_um / s <= MAX_BOX_PX)),
                     len(self.levels) - 1)
        array, size = self.levels[level], self.scales[level]
        offset = (size - self.scales[0]) / 2  # voxel centres of a coarser level are shifted
        index, inside, start, shape = [], [], [], []
        for axis, n in zip(self.axes, array.shape):
            if axis == "t":
                index.append(min(t, n - 1))
            elif axis == "c":
                index.append(min(channel, n - 1))
            else:
                k = len(start)
                half = max(1, int(round(half_um / size[k])))
                centre = int(round((centre_um.get(axis, 0) - offset[k]) / size[k]))
                low, high = centre - half, centre + half + 1
                index.append(slice(min(max(low, 0), n), max(min(high, n), 0)))
                inside.append(slice(index[-1].start - low, index[-1].stop - low))
                start.append(low)
                shape.append(high - low)
        out = np.zeros(shape, dtype=array.dtype)
        if all(s.stop > s.start for s in inside):
            out[tuple(inside)] = np.asarray(array[tuple(index)])
        return out, np.array(start), size, offset


def resample(labels, start, size, offset, shape, to_start, to_size, to_offset):
    """Labels of one box (nearest voxel) at the voxel centres of another box."""
    index, inside = [], np.ones(shape, bool)
    for k, n in enumerate(shape):
        um = (to_start[k] + np.arange(n)) * to_size[k] + to_offset[k]
        i = np.round((um - offset[k]) / size[k]).astype(int) - start[k]
        valid = (i >= 0) & (i < labels.shape[k])
        inside &= valid.reshape([-1 if j == k else 1 for j in range(len(shape))])
        index.append(np.clip(i, 0, labels.shape[k] - 1))
    out = labels[np.ix_(*index)]
    out[~inside] = 0
    return out


def thumbnail_views(image, labels, label):
    """RGB views through the centre of a box: XY (and XZ in 3D), the object outlined in magenta."""
    low, high = np.percentile(image, (1, 99.8))
    gray = np.clip((image - low) / max(high - low, 1e-9), 0, 1)
    if image.ndim == 3:
        z, y = image.shape[0] // 2, image.shape[1] // 2
        views = [(gray[z], labels[z])] + ([(gray[:, y], labels[:, y])] if image.shape[0] > 1 else [])
    else:
        views = [(gray, labels)]
    out = []
    for g, lab in views:
        rgb = np.repeat((g * 255).astype(np.uint8)[..., None], 3, axis=2)
        edges = find_boundaries(lab, mode="inner")
        other = edges & (lab > 0) & (lab != label)
        rgb[other] = (0.4 * rgb[other] + 0.6 * np.array(OTHER_RGB)).astype(np.uint8)
        rgb[edges & (lab == label)] = OBJECT_RGB
        out.append(np.ascontiguousarray(rgb))
    return out


def read_table(path, **kwargs):
    """CSV / TSV, the separator guessed from the header."""
    with open(path, newline="") as f:
        header = f.readline()
    return pd.read_csv(path, sep=max([",", ";", "\t"], key=header.count), **kwargs)


def clean_table(table, label_column, notes=None):
    """
    One row per label (> 0), sorted by label; text columns become categories (1, 2, ...; 0 = empty);
    inf becomes NaN (not plotted: the plot can't set its axis limits to inf).
    What was changed is appended to ``notes``.
    """
    notes = [] if notes is None else notes
    if label_column not in table.columns:
        raise ValueError(f"no column '{label_column}' in the table")
    table = table.copy()
    table[label_column] = pd.to_numeric(table[label_column], errors="coerce")
    table = table[table[label_column] > 0].copy()
    table[label_column] = table[label_column].astype(np.int64)
    table = table.drop_duplicates(label_column).sort_values(label_column).reset_index(drop=True)
    numbers = table.select_dtypes("number").columns
    infinite = np.isinf(table[numbers].to_numpy(float)).sum(axis=0)
    if infinite.any():
        notes.append("inf -> NaN: " + ", ".join(f"{c} ({n} rows)" for c, n in zip(numbers, infinite) if n))
        table[numbers] = table[numbers].replace([np.inf, -np.inf], np.nan)
    for column in table.columns:
        if not pd.api.types.is_numeric_dtype(table[column]):
            categories = pd.Categorical(table[column])
            table[column] = pd.Categorical(categories.codes.astype(int) + 1)  # int: biaplotter's categories
            names = ", ".join(f"{i + 1} = {c}" for i, c in enumerate(categories.categories[:20]))
            notes.append(f"{column} (categories): {names}{', ...' if len(categories.categories) > 20 else ''}")
    return table


def guess_label_column(columns):
    """The label column among ``columns`` (by name), else the first one."""
    return next((c for c in columns if c.lower() in LABEL_COLUMNS), columns[0] if len(columns) else "")


def centroid_columns(table, axes):
    """Names of the centroid columns (in the order of the axes), or None."""
    lower = {c.lower(): c for c in table.columns}
    n = len(axes)
    for columns in ([f"centroid-{i}" for i in range(n)], [f"centroid_{a}" for a in axes],
                    [f"centroid-{a}" for a in axes], [f"centroid_{i}" for i in range(n)], list(axes)):
        if all(c in lower for c in columns):
            return [lower[c] for c in columns]
    return None


def object_radius_um(table, n_axes, voxel_um):
    """Radius of every object (µm) from regionprops bbox columns or a size column (voxels), else NaN."""
    lower = {c.lower(): c for c in table.columns}
    bbox = [f"bbox-{i}" for i in range(2 * n_axes)]
    if all(b in lower for b in bbox):
        b = table[[lower[c] for c in bbox]].to_numpy(float)
        return np.max((b[:, n_axes:] - b[:, :n_axes]) * voxel_um, axis=1) / 2
    for name in SIZE_COLUMNS:
        if name in lower and pd.api.types.is_numeric_dtype(table[lower[name]]):
            voxels = table[lower[name]].to_numpy(float)
            radius = (3 * voxels / (4 * np.pi)) ** (1 / 3) if n_axes == 3 else np.sqrt(voxels / np.pi)
            return radius * np.exp(np.mean(np.log(voxel_um)))
    return np.full(len(table), np.nan)


def centroids_from_labels(labels, ids, max_voxels=3e7, notes=None):
    """(centroid µm, radius µm) of every label in ids, from the finest level with at most max_voxels voxels."""
    notes = [] if notes is None else notes
    t = labels.axes.find("t")
    if t >= 0 and labels.levels[0].shape[t] > 1:
        raise ValueError("time-lapse labels: add centroid columns to the table "
                         "(centroid-0/1/2, centroid_z/y/x or z/y/x)")
    level = next((i for i, a in enumerate(labels.levels) if np.prod(a.shape) <= max_voxels),
                 len(labels.levels) - 1)
    data = labels.levels[level]
    data = np.asarray(data[(slice(None),) * t + (0,)] if t >= 0 else data)  # a single time point
    where = np.flatnonzero(data)
    found, inverse, counts = np.unique(data.ravel()[where], return_inverse=True, return_counts=True)
    coords = np.unravel_index(where, data.shape)
    sums = np.stack([np.bincount(inverse, weights=c, minlength=len(found)) for c in coords], axis=1)
    size = labels.scales[level]
    centre = sums / counts[:, None] * size + (size - labels.scales[0]) / 2
    volume = counts * np.prod(size)
    radius = (3 * volume / (4 * np.pi)) ** (1 / 3) if data.ndim == 3 else np.sqrt(volume / np.pi)
    rows = pd.Index(found).get_indexer(ids)
    out_centre, out_radius = np.full((len(ids), data.ndim), np.nan), np.full(len(ids), np.nan)
    out_centre[rows >= 0], out_radius[rows >= 0] = centre[rows[rows >= 0]], radius[rows[rows >= 0]]
    notes.append(f"centroids computed on level {level} of the labels")
    if (rows < 0).any():
        notes.append(f"{(rows < 0).sum()} labels of the table are not in level {level}: no position for them")
    return out_centre, out_radius


def contrast_limits(array):
    """1 - 99.9 percentiles of an image, from at most ~4 million of its voxels."""
    step = max(1, int(np.ceil((array.size / 4e6) ** (1 / array.ndim))))
    sample = np.asarray(array[(slice(None, None, step),) * array.ndim])
    low, high = np.percentile(sample, (1, 99.9))
    return [float(low), float(max(high, low + 1))]


def image_contrast_limits(image):
    """Contrast limits of every channel (a single one without a channel axis), from the lowest level."""
    lowest = image.levels[-1]
    if "t" in image.axes:
        lowest = lowest[(slice(None),) * image.axes.index("t") + (0,)]
    if "c" not in image.axes:
        return [contrast_limits(lowest)]
    c = image.axes.replace("t", "").index("c")
    return [contrast_limits(lowest[(slice(None),) * c + (i,)]) for i in range(lowest.shape[c])]


def channel_names(image):
    """Name of every channel of an image (``intensity`` without a channel axis)."""
    if "c" not in image.axes:
        return ["intensity"]
    n = image.levels[0].shape[image.axes.index("c")]
    names = image.metadata.get("channel_names") or []
    return [str(c) for c in names] if len(names) == n else [f"channel {i}" for i in range(n)]


def hex_colormap(value):
    rgb = to_rgb("#" + str(value).lstrip("#")[:6])
    return Colormap([[0, 0, 0, 1], [*rgb, 1]], name="#" + str(value).lstrip("#")[:6].lower())


def add_image_layers(viewer, image, name, limits):
    """The intensity image, one layer per channel (all levels, read when needed)."""
    multiscale = len(image.levels) > 1
    data = image.levels if multiscale else image.levels[0]
    if "c" not in image.axes:
        return [viewer.add_image(data, multiscale=multiscale, scale=image.layer_scale(), name=name,
                                 contrast_limits=limits[0])]
    channels = channel_names(image)
    colours = image.metadata.get("channel_colors") or []
    colormaps = None  # napari's default colours, unless the file has its own
    if len(colours) == len(channels) and any(str(h).lstrip("#").upper()[:6] != "FFFFFF" for h in colours):
        colormaps = [hex_colormap(h) for h in colours]
    return viewer.add_image(data, channel_axis=image.axes.index("c"), multiscale=multiscale,
                            scale=image.layer_scale(), name=[f"{name}: {ch}" for ch in channels],
                            colormap=colormaps, contrast_limits=limits, blending="additive")


def zarr_path(text):
    """A dropped path as the OME-Zarr folder (a metadata file inside it -> the folder)."""
    path = Path(str(text).strip().strip('"')).expanduser()
    if path.is_file() and path.name in ("zarr.json", ".zattrs", ".zgroup"):
        path = path.parent
    if path.name == "labels" and path.is_dir():  # the labels group of an image: its first label image
        path = next((p for p in sorted(path.iterdir()) if p.is_dir()), path)
    return path


def short_name(path):
    return Path(path).name.replace(".ome.zarr", "").replace(".zarr", "")


def sort_dropped(paths):
    """Files dropped together, by kind: {"table" | "labels" | "image": path}.
    A table by its suffix, labels by their name, anything else is the intensity image."""
    out = {}
    for path in paths:
        name = Path(path).name.lower()
        if Path(path).suffix.lower() in TABLE_SUFFIXES:
            out["table"] = str(path)
        elif "label" in name:
            out["labels"] = str(path)
        else:
            out["image"] = str(path)
    return out


def open_labels(image_zarr, labels_path, notes):
    """The labels OME-Zarr at ``labels_path``, else the first label group of the intensity OME-Zarr."""
    if labels_path:
        path = zarr_path(labels_path)
        labels_zarr = mm.ZarrManager(str(path))
        return Pyramid(labels_zarr.data, labels_zarr.metadata), short_name(path)
    groups = getattr(image_zarr, "labels", None) or {}
    if not groups:
        raise ValueError("no labels given, and none in <intensity>.ome.zarr/labels/")
    name = next(iter(groups))
    notes.append(f"labels: '{name}' of the intensity OME-Zarr")
    return Pyramid(groups[name].data, groups[name].metadata), name


def read_dataset(image_path, labels_path, table_path, label_column, centroids_um=False):
    """
    Read everything the widget shows (runs in a background thread): a generator that yields progress
    messages and returns a dict with the image and labels pyramids, the cleaned table and where every
    object is (centroid and thumbnail half width, µm).
    """
    notes = []
    image_path = zarr_path(image_path)
    yield f"Reading {image_path.name} ..."
    image_zarr = mm.ZarrManager(str(image_path))
    image = Pyramid(image_zarr.data, image_zarr.metadata)
    labels, labels_name = open_labels(image_zarr, labels_path, notes)
    labels.drop_channel()
    if (np.allclose(labels.scales[0], 1) and not np.allclose(image.scales[0], 1)
            and labels.levels[0].shape[-3:] == image.levels[0].shape[-3:]):
        # labels saved without voxel size, on the grid of the image
        labels.scales = [s * image.scales[0] for s in labels.scales]

    yield f"Reading {Path(table_path).name} ..."
    table = clean_table(read_table(table_path), label_column, notes)
    ids = table[label_column].to_numpy()
    spatial = labels.spatial
    columns = centroid_columns(table, spatial)
    if columns is not None:
        centroid = table[columns].to_numpy(float)
        if not centroids_um:
            centroid = centroid * labels.scales[0]
        radius = object_radius_um(table, len(spatial), labels.scales[0])
        where = f"centroids from {', '.join(columns)}"
    else:
        yield "Computing the centroids from the labels ..."
        centroid, radius = centroids_from_labels(labels, ids, notes=notes)
        where = "centroids computed from the labels"
    voxel = labels.scales[0].min()
    half = np.where(np.isfinite(radius), np.maximum(THUMBNAIL_CONTEXT * radius, 6 * voxel), 12 * voxel)

    yield "Contrast limits ..."
    return {
        "image": image, "image_name": short_name(image_path), "limits": image_contrast_limits(image),
        "labels": labels, "labels_name": labels_name, "table": table, "label_column": label_column,
        "centroid_columns": columns or [], "centroid_um": centroid, "half_um": half,
        "where": where, "notes": notes,
    }


class Session:
    """What is loaded: the layers, the table (one row per label, sorted) and where every object is."""

    def __init__(self, image, labels, layer, image_layers, table, label_column, centroid_um, half_um):
        self.image, self.labels, self.layer, self.image_layers = image, labels, layer, image_layers
        self.table, self.label_column = table, label_column
        self.rows = pd.Index(table[label_column].to_numpy())
        self.centroid_um, self.half_um = centroid_um, half_um
        self.time_column = next((c for c in TIME_COLUMNS if c in table.columns), None)

    def row(self, label):
        return int(self.rows.get_indexer([label])[0])

    def thumbnail(self, label, channel, field_of_view_um=0):
        """RGB views of an object (runs in a background thread)."""
        row = self.row(label)
        if row < 0 or not np.all(np.isfinite(self.centroid_um[row])):
            return []
        centre = dict(zip(self.labels.spatial, self.centroid_um[row]))
        half = field_of_view_um / 2 if field_of_view_um > 0 else self.half_um[row]
        t = int(self.table[self.time_column].iloc[row]) if self.time_column else 0
        image, *image_grid = self.image.box(centre, half, channel, t)
        labels, *labels_grid = self.labels.box(centre, half, 0, t)
        if labels.ndim != image.ndim:
            return []
        return thumbnail_views(image.astype(np.float32), resample(labels, *labels_grid, image.shape, *image_grid),
                               label)


# ---------------------------------------------------------------------------
# widgets
# ---------------------------------------------------------------------------

class PathField(QWidget):
    """A path box that takes a file or folder dropped on it (or picked with the … button)."""

    def __init__(self, placeholder, folder=True):
        super().__init__()
        self.folder = folder
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        self.edit.setAcceptDrops(False)  # the drop goes to this widget
        self.setAcceptDrops(True)
        browse = QPushButton("…")
        browse.setFixedWidth(28)
        browse.clicked.connect(self._browse)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit)
        layout.addWidget(browse)

    def dragEnterEvent(self, event):
        # several files: the panel sorts them out
        if len(event.mimeData().urls()) == 1:
            event.acceptProposedAction()
        else:
            event.ignore()

    dragMoveEvent = dragEnterEvent

    def dropEvent(self, event):
        self.edit.setText(event.mimeData().urls()[0].toLocalFile())
        event.acceptProposedAction()

    def _browse(self):
        if self.folder:
            path = QFileDialog.getExistingDirectory(self, "OME-Zarr folder")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Features table", "", "Tables (*.csv *.tsv *.txt);;All (*)")
        if path:
            self.edit.setText(path)

    @property
    def path(self):
        return self.edit.text().strip().strip('"')


class ThumbnailPopup(QFrame):
    """Small window next to the mouse: the views of an object and a few of its features."""

    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowTransparentForInput)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setStyleSheet("QFrame {background-color: #262930; border: 1px solid #5c6370;}"
                           "QLabel {color: #e0e0e0; border: none;}")
        self.views = [QLabel(), QLabel()]
        self.text = QLabel()
        row = QHBoxLayout()
        for view in self.views:
            row.addWidget(view)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addLayout(row)
        layout.addWidget(self.text)

    def show_views(self, text, views):
        self.text.setText(text)
        for i, widget in enumerate(self.views):
            if i < len(views):
                h, w = views[i].shape[:2]
                image = QImage(views[i].tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)
                widget.setPixmap(QPixmap.fromImage(image).scaled(
                    THUMBNAIL_PX, THUMBNAIL_PX, Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.FastTransformation))
            widget.setVisible(i < len(views))
        self.adjustSize()
        self.move_near_mouse()
        self.show()

    def move_near_mouse(self):
        mouse = QCursor.pos()
        screen = QGuiApplication.screenAt(mouse) or QGuiApplication.primaryScreen()
        area = screen.availableGeometry()
        x, y = mouse.x() + 16, mouse.y() + 16
        if x + self.width() > area.right():
            x = mouse.x() - 16 - self.width()
        if y + self.height() > area.bottom():
            y = mouse.y() - 16 - self.height()
        self.move(x, y)


def _title(text):
    label = QLabel(text)
    label.setStyleSheet(_TITLE_STYLE)
    return label


def _form(*rows):
    """A widget with a form layout; a row is (label, widget) or a widget spanning both columns."""
    widget = QWidget()
    form = QFormLayout(widget)
    form.setContentsMargins(0, 0, 0, 0)
    for row in rows:
        if isinstance(row, tuple):
            form.addRow(*row)
        else:
            form.addRow(row)
    return widget


class ObjectsExplorer(QWidget):
    """The panel: loads the data and links the plot of napari-clusters-plotter with the viewer."""

    def __init__(self, viewer, plotter):
        super().__init__()
        self.viewer, self.plotter = viewer, plotter
        self.session = None
        self._loader = None
        self._syncing = False  # True while the plot and the viewer update each other
        self._points_data = self._point_labels = self._tree = None
        self._hover_label = None
        self._thumbnails = OrderedDict()  # (label, channel, field of view) -> views
        self._worker = self._worker_key = None
        self.popup = ThumbnailPopup(self)
        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.setInterval(HOVER_DELAY_MS)
        self._hover_timer.timeout.connect(self._request_thumbnail)
        self._build_ui()
        self._connect_plotter()

    # ----------------------------------------------------------------- panel
    def _build_ui(self):
        self.image_field = PathField("intensity .ome.zarr")
        self.labels_field = PathField("labels .ome.zarr (empty: <intensity>/labels/)")
        self.table_field = PathField("features .csv", folder=False)
        self.label_column = QComboBox()
        self.label_column.setEditable(True)
        self.centroids_um = QCheckBox("Centroids of the table are in µm (not voxels)")
        self.load_button = QPushButton("Load")
        self.load_button.clicked.connect(self.load)
        self.load_holder = make_button_holder(self.load_button)
        self.table_field.edit.textChanged.connect(self._table_changed)

        self.zoom = QCheckBox("Zoom in on the clicked object")
        self.zoom.setChecked(True)
        self.isolate = QCheckBox("Show only the clicked object")
        self.isolate.toggled.connect(self._isolate_toggled)
        self.hover = QCheckBox("Thumbnail when hovering a point")
        self.hover.setChecked(True)
        self.channel = QComboBox()
        self.field_of_view = QDoubleSpinBox()
        self.field_of_view.setRange(0, 100000)
        self.field_of_view.setDecimals(1)
        self.field_of_view.setSuffix(" µm")
        self.field_of_view.setSpecialValueText("auto (object size)")
        for setting in (self.channel.currentIndexChanged, self.field_of_view.valueChanged, self.hover.toggled):
            setting.connect(self._hide_popup)

        self.info = QLabel("Drop the intensity, labels and features files, then Load.")
        self.info.setWordWrap(True)

        data = _form(("Intensity", self.image_field), ("Labels", self.labels_field),
                     ("Features", self.table_field), ("Label column", self.label_column), self.centroids_um)
        links = _form(self.zoom, self.isolate, self.hover, ("Thumbnail channel", self.channel),
                      ("Thumbnail width", self.field_of_view))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(make_section(_title("Data (drop the files here):"), data, self.load_holder, self.info))
        layout.addWidget(make_section(_title("Plot and viewer:"), links))
        layout.addStretch(1)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        """Several files dropped on the panel: sorted by kind (table, labels, intensity)."""
        fields = {"table": self.table_field, "labels": self.labels_field, "image": self.image_field}
        for kind, path in sort_dropped(url.toLocalFile() for url in event.mimeData().urls()).items():
            fields[kind].edit.setText(path)
        event.acceptProposedAction()

    def _table_changed(self, text):
        path = Path(text.strip().strip('"'))
        if not path.is_file():
            return
        try:
            columns = list(read_table(path, nrows=0).columns)
        except Exception:
            return
        current = self.label_column.currentText()
        self.label_column.clear()
        self.label_column.addItems(columns)
        self.label_column.setCurrentText(current if current in columns else guess_label_column(columns))

    def _isolate_toggled(self, on):
        if self.session is not None and not on:
            self.session.layer.show_selected_label = False

    # ----------------------------------------------------------------- loading
    def load(self):
        """Read the data in the background, then show it (the data loaded before stays until then)."""
        if not self.image_field.path or not self.table_field.path:
            show_warning("Give at least the intensity OME-Zarr and the features table")
            return
        if self._loader is not None:
            return
        self.load_holder.setEnabled(False)
        self.info.setText("Loading ...")
        self._loader = create_worker(
            read_dataset, self.image_field.path, self.labels_field.path, self.table_field.path,
            self.label_column.currentText(), self.centroids_um.isChecked(),
            _connect={"yielded": self.info.setText, "returned": self._show, "errored": self._load_failed,
                      "finished": self._load_done})

    def _load_failed(self, error):
        traceback.print_exception(error)
        self.info.setText(f"Could not load the data: {error}")
        show_error(f"Could not load the data: {error}")

    def _load_done(self):
        self._loader = None
        self.load_holder.setEnabled(True)

    def _show(self, data):
        try:
            self._show_dataset(data)
        except Exception as error:
            self._load_failed(error)

    def _show_dataset(self, data):
        image, labels, table, label_column = data["image"], data["labels"], data["table"], data["label_column"]
        for note in data["notes"]:
            print(note)
        self._clear()
        image_layers = add_image_layers(self.viewer, image, data["image_name"], data["limits"])
        features = table.rename(columns={"index": "index (table)"}) if label_column != "index" else table.copy()
        ids = table[label_column].to_numpy()
        features["index"] = ids  # napari: row <-> label
        multiscale = len(labels.levels) > 1
        layer = self.viewer.add_labels(labels.levels if multiscale else labels.levels[0], multiscale=multiscale,
                                       scale=labels.layer_scale(), name=data["labels_name"], features=features,
                                       opacity=0.7, metadata={LABEL_IDS_KEY: np.r_[0, ids]})
        self.session = Session(image, labels, layer, list(image_layers), table, label_column,
                               data["centroid_um"], data["half_um"])

        self.channel.blockSignals(True)
        self.channel.clear()
        self.channel.addItems(channel_names(image))
        self.channel.blockSignals(False)
        self.viewer.layers.selection.select_only(layer)  # -> the plotter takes its features
        bbox = {c for c in table.columns if c.lower().startswith("bbox")}
        self._choose_plot_features(table, {label_column, "index", *data["centroid_columns"], *bbox})
        self.info.setText(f"{len(table)} objects, {data['where']}")

    def _choose_plot_features(self, table, skip):
        """First features of the table on x, y and as colour."""
        numeric = [c for c in table.columns if c not in skip and pd.api.types.is_numeric_dtype(table[c])]
        if not numeric:
            return
        boxes = self.plotter.control_widget
        hue = numeric[2] if len(numeric) > 2 else numeric[-1]
        for box, column in ((boxes.x_axis_box, numeric[0]), (boxes.y_axis_box, numeric[min(1, len(numeric) - 1)])):
            box.blockSignals(True)
            box.setCurrentText(column)
            box.blockSignals(False)
        if boxes.hue_box.currentText() == hue:
            self.plotter.plot_needs_update.emit()
        else:
            boxes.hue_box.setCurrentText(hue)

    def _clear(self):
        self._hide_popup()
        self._thumbnails.clear()
        self._points_data = self._point_labels = self._tree = None
        if self.session is not None:
            for layer in [self.session.layer, *self.session.image_layers]:
                if layer in self.viewer.layers:
                    self.viewer.layers.remove(layer)
        self.session = None

    # ----------------------------------------------------------------- plot <-> viewer
    def _connect_plotter(self):
        canvas = self.plotter.plotting_widget
        canvas.artists["SCATTER"].pymif_min_size = MIN_POINT_SIZE
        # click on a point: our jump (the plotter's own takes the row number as label and reads the whole image)
        canvas.artists["SCATTER"].highlighted_changed_signal.disconnect(self.plotter._on_highlighted_changed)
        for artist in canvas.artists.values():
            artist.highlighted_changed_signal.connect(self._on_highlighted)
        # label picked in napari: highlight its point (same row number problem in the plotter's own)
        self.plotter._update_selected_object_feature = self._on_label_selected
        canvas.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        for event in ("axes_leave_event", "figure_leave_event", "button_press_event"):
            canvas.canvas.mpl_connect(event, self._hide_popup)

    def _plot_points(self):
        """Label of every point of the plot (-1: another layer), updated when the plot data change."""
        data = self.plotter.plotting_widget.active_artist.data
        if self.session is None or data is None:
            return None
        if data is not self._points_data:
            features = self.plotter._get_features()
            column = self.session.label_column
            if len(features) != len(data) or column not in features.columns:
                return None
            ours = features["layer"].astype(str).to_numpy() == self.session.layer.name
            labels = pd.to_numeric(features[column], errors="coerce").fillna(-1).to_numpy(np.int64)
            self._point_labels = np.where(ours, labels, -1)
            self._points_data, self._tree = data, None
        return self._point_labels

    def _point_under(self, event):
        """Index of the plot point within HOVER_RADIUS_PX of the mouse, or None."""
        if self._plot_points() is None:
            return None
        data = self._points_data
        if self._tree is None:  # in data coordinates scaled to 0-1
            finite = np.flatnonzero(np.isfinite(data).all(axis=1))
            if len(finite) == 0:
                return None
            low, span = data[finite].min(axis=0), np.ptp(data[finite], axis=0)
            span[span == 0] = 1
            self._tree = cKDTree((data[finite] - low) / span), finite, low, span
        tree, finite, low, span = self._tree
        axes = event.inaxes
        (x0, x1), (y0, y1) = axes.get_xlim(), axes.get_ylim()
        radius = HOVER_RADIUS_PX * max(abs(x1 - x0) / axes.bbox.width / span[0],
                                       abs(y1 - y0) / axes.bbox.height / span[1])
        near = tree.query_ball_point((np.array([event.xdata, event.ydata]) - low) / span, radius)
        if not near:
            return None
        near = finite[near]
        distance = np.hypot(*(axes.transData.transform(data[near]) - [event.x, event.y]).T)
        best = np.argmin(distance)
        return near[best] if distance[best] <= HOVER_RADIUS_PX else None

    def _on_highlighted(self, highlighted):
        """A point clicked in the plot: go to its object."""
        if self._syncing or highlighted is None or self._plot_points() is None:
            return
        points = np.flatnonzero(highlighted)
        if len(points) == 1 and self._point_labels[points[0]] > 0:
            self.go_to(int(self._point_labels[points[0]]))

    def go_to(self, label):
        """Move the viewer to an object: its plane, centred and zoomed in, its label selected."""
        s = self.session
        row = s.row(label)
        if row < 0 or not np.all(np.isfinite(s.centroid_um[row])):
            show_warning(f"label {label}: position unknown")
            return
        centre = s.centroid_um[row]
        first = max(0, self.viewer.dims.ndim - len(centre))
        for axis, value in enumerate(centre, start=first):
            self.viewer.dims.set_point(axis, value)
        if s.time_column is not None and first > 0:
            self.viewer.dims.set_point(first - 1, float(s.table[s.time_column].iloc[row]))
        scene = getattr(self.viewer, "scene", self.viewer)  # napari >= 0.9: viewer.scene.camera
        scene.camera.center = tuple(centre[-3:])
        if self.zoom.isChecked():
            canvas = self._canvas_size()
            if canvas is None:
                show_warning("Could not read the size of the canvas: no zoom")
            else:
                scene.camera.zoom = 0.9 * min(canvas) / (2 * JUMP_CONTEXT * s.half_um[row])
        self._syncing = True
        try:
            s.layer.selected_label = label
            s.layer.show_selected_label = self.isolate.isChecked()
        finally:
            self._syncing = False

    def _canvas_size(self):
        """(height, width) of the canvas in screen pixels, or None."""
        for get in (lambda: self.viewer.canvas.size,  # napari >= 0.9
                    lambda: self.viewer._canvas_size,
                    lambda: self.viewer.window._qt_viewer.canvas.size):
            try:
                size = tuple(get())
            except (AttributeError, TypeError):
                continue
            if len(size) == 2 and min(size) > 0:
                return size
        return None

    def _on_label_selected(self, event=None):
        """A label picked in napari: highlight its point in the plot."""
        if self._syncing or self.session is None or getattr(event, "source", self.session.layer) is not self.session.layer:
            return
        points = self._plot_points()
        artist = self.plotter.plotting_widget.active_artist
        if points is None or not isinstance(artist, Scatter):
            return
        highlighted = points == self.session.layer.selected_label
        if not highlighted.any() or (artist.highlighted is not None and np.array_equal(artist.highlighted, highlighted)):
            return
        self._syncing = True
        try:
            artist.highlighted = highlighted
        finally:
            self._syncing = False

    # ----------------------------------------------------------------- hover thumbnails
    def _on_mouse_move(self, event):
        canvas = self.plotter.plotting_widget
        index = None
        if (self.session is not None and self.hover.isChecked() and event.inaxes is canvas.axes
                and event.button is None and isinstance(canvas.active_artist, Scatter)
                and canvas.active_selector is None):
            index = self._point_under(event)
        label = -1 if index is None else int(self._point_labels[index])
        if label <= 0:
            self._hide_popup()
            return
        if label != self._hover_label:
            self._hover_label = label
            self._hover_timer.start()
        elif self.popup.isVisible():
            self.popup.move_near_mouse()

    def _hide_popup(self, *_):
        self._hover_label = None
        self._hover_timer.stop()
        self.popup.hide()

    def _thumbnail_key(self):
        return self._hover_label, self.channel.currentIndex(), self.field_of_view.value()

    def _hover_text(self):
        features, row = self.session.layer.features, self.session.row(self._hover_label)
        lines = [f"label {self._hover_label}"]
        for name in dict.fromkeys([self.plotter.x_axis, self.plotter.y_axis, self.plotter.hue_axis]):
            if name in features.columns and row >= 0:
                value = features[name].iloc[row]
                lines.append(f"{name}: {value:.4g}" if isinstance(value, (float, np.floating)) else f"{name}: {value}")
        return "\n".join(lines)

    def _request_thumbnail(self):
        if self._hover_label is None or self.session is None:
            return
        key = self._thumbnail_key()
        if key in self._thumbnails:
            self._thumbnails.move_to_end(key)
            self.popup.show_views(self._hover_text(), self._thumbnails[key])
            return
        self.popup.show_views(self._hover_text() + "\nloading ...", [])
        if self._worker is None:  # else: read when the one being read is done
            self._worker_key = key
            self._worker = create_worker(self.session.thumbnail, *key, _connect={
                "returned": self._thumbnail_read, "errored": self._thumbnail_failed,
                "finished": self._worker_done})

    def _thumbnail_read(self, views):
        self._thumbnails[self._worker_key] = views
        while len(self._thumbnails) > MAX_THUMBNAILS:
            self._thumbnails.popitem(last=False)
        if self._worker_key == self._thumbnail_key():
            self.popup.show_views(self._hover_text(), views)

    def _thumbnail_failed(self, error):
        print(f"thumbnail of label {self._worker_key[0]}: {error!r}")
        self._thumbnails[self._worker_key] = []

    def _worker_done(self):
        self._worker = None
        if self._hover_label is not None and self._thumbnail_key() not in self._thumbnails:
            self._request_thumbnail()


def explorer_widget():
    """The Objects Explorer dock: the panel, and under it the plot of napari-clusters-plotter.

    The dock exposes ``dock.explorer`` (the panel, :class:`ObjectsExplorer`) and ``dock.plotter``.
    """
    if _PLOTTER_IMPORT_ERROR is not None:
        message = QLabel("The Objects Explorer needs napari-clusters-plotter:\n"
                         "pip install napari-clusters-plotter\n\n"
                         f"({_PLOTTER_IMPORT_ERROR})")
        message.setWordWrap(True)
        return compose_dock(make_section(message))
    _patch_plotter()
    viewer = current_viewer()
    plotter = PlotterWidget(viewer)
    explorer = ObjectsExplorer(viewer, plotter)
    dock = compose_dock(explorer, bottom=make_section(_title("Plot and select features:"), plotter))
    dock.splitter.setSizes([340, 660])
    dock.explorer, dock.plotter = explorer, plotter
    return dock
