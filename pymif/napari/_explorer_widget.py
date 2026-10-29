"""napari widget to explore the features of segmented objects, with napari-clusters-plotter.

Three things are loaded (dropped on their field, all at once anywhere on the panel, or browsed):

* an intensity OME-Zarr (all channels and resolution levels, read lazily),
* the labels OME-Zarr (optional: else the first group in ``<intensity>.ome.zarr/labels/``),
* a CSV with one row per object: a label column and any features.

A table from several images has two more columns, ``image_path`` and ``labels_path``: the intensity and
labels OME-Zarr of the image of every row, relative to the folder of the table. The intensity and labels
fields are then not used: the panel lists the images of the table, the plot shows the objects of the
ticked ones, and the viewer shows one image at a time (the one of the last clicked object).

The features are plotted by napari-clusters-plotter, under the panel, with everything it can do
(x / y features, colour of the plot and the labels by any feature, selections to make clusters,
histogram, ...). The plot is linked with the viewer:

* click a point: the viewer goes to that object (its image, its plane, centred, zoomed in) and selects
  its label,
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

import threading
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
from qtpy.QtGui import QCursor, QFont, QGuiApplication, QImage, QPixmap
from qtpy.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPushButton, QVBoxLayout, QWidget,
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
#: Columns of a table from several images: the intensity and labels OME-Zarr of every row,
#: relative to the folder of the table (or absolute).
IMAGE_PATH, LABELS_PATH = "image_path", "labels_path"
PATH_COLUMNS = (IMAGE_PATH, LABELS_PATH)
#: Feature added for a table from several images: the image of every object (1, 2, ... as in the panel).
IMAGE_INDEX = "image_index"
#: Name of the labels layer for a table from several images.
LABELS_LAYER = "labels"
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
      for a big image) and gives the colour of row i of the features to the i-th label. The label of
      every row is given in the layer metadata instead (rows of an image that isn't shown get ids
      that aren't in the label image).
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

class RepeatedLabelsError(ValueError):
    """The table has several rows for the same object (the user is asked to fix the table)."""


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


def has_path_columns(columns):
    """True for a table from several images (it has the image_path and labels_path columns)."""
    return all(c in columns for c in PATH_COLUMNS)


def path_text(value):
    """A path of the table as text: '' when empty, '\\' read as '/' (a table written on Windows)."""
    return "" if pd.isna(value) else str(value).strip().strip('"').replace("\\", "/")


def resolve_path(value, folder):
    """A path of the table, relative to ``folder`` (the folder of the table) unless absolute; None if empty."""
    text = path_text(value)
    if not text:
        return None
    path = Path(text).expanduser()
    return path if path.is_absolute() else Path(folder) / path


def repeated_labels_message(rows, label_column, image_column=None):
    """What to tell the user about the repeated objects in ``rows`` (the repeated rows of the table)."""
    key = label_column if image_column is None else [image_column, label_column]
    counts = rows.groupby(key).size().sort_values(ascending=False)
    others = len(counts) - 1
    others = f" ({others} other label{' is' if others == 1 else 's are'} repeated too)" if others else ""
    if image_column is None:
        return (f"Label {counts.index[0]} appears {counts.iloc[0]} times in the table{others}.\n\n"
                "The table seems to come from several images. To explore all of them, add the columns "
                f"{IMAGE_PATH} and {LABELS_PATH}: the intensity and labels OME-Zarr of the image of every "
                "row, relative to the folder of the table. Or keep only the rows of one image.")
    image, label = counts.index[0]
    return (f"Label {label} of {image} appears {counts.iloc[0]} times in the table{others}.\n\n"
            "Each object of an image needs a single row.")


def clean_table(table, label_column, notes=None, image_column=None):
    """
    One row per label (> 0), sorted by label (by image, then label, with an ``image_column``);
    text columns become categories (1, 2, ...; 0 = empty), except the path columns;
    inf becomes NaN (not plotted: the plot can't set its axis limits to inf).
    A label repeated (in an image) raises a RepeatedLabelsError. What was changed is appended to ``notes``.
    """
    notes = [] if notes is None else notes
    if label_column not in table.columns:
        raise ValueError(f"no column '{label_column}' in the table")
    table = table.copy()
    table[label_column] = pd.to_numeric(table[label_column], errors="coerce")
    table = table[table[label_column] > 0].copy()
    table[label_column] = table[label_column].astype(np.int64)
    key = [label_column]
    if image_column is not None:
        table[image_column] = table[image_column].map(path_text)
        empty = table[image_column] == ""
        if empty.any():
            notes.append(f"{empty.sum()} rows without {image_column}: left out")
            table = table[~empty].copy()
        key = [image_column, label_column]
    repeated = table.duplicated(key, keep=False)
    if repeated.any():
        raise RepeatedLabelsError(repeated_labels_message(table[repeated], label_column, image_column))
    table = table.sort_values(key).reset_index(drop=True)
    numbers = table.select_dtypes("number").columns
    infinite = np.isinf(table[numbers].to_numpy(float)).sum(axis=0)
    if infinite.any():
        notes.append("inf -> NaN: " + ", ".join(f"{c} ({n} rows)" for c, n in zip(numbers, infinite) if n))
        table[numbers] = table[numbers].replace([np.inf, -np.inf], np.nan)
    for column in table.columns:
        if column in PATH_COLUMNS or pd.api.types.is_numeric_dtype(table[column]):
            continue
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


def read_image(image_path, labels_path, rows, label_column, centroids_um=False):
    """
    Read one image (in a background thread): its pyramids, where its objects (``rows``, its rows of the
    table) are, as centroid and thumbnail half width (µm), and the contrast limits of its channels.
    """
    notes = []
    image_path = zarr_path(image_path)
    image_zarr = mm.ZarrManager(str(image_path))
    image = Pyramid(image_zarr.data, image_zarr.metadata)
    labels, labels_name = open_labels(image_zarr, labels_path, notes)
    labels.drop_channel()
    if (np.allclose(labels.scales[0], 1) and not np.allclose(image.scales[0], 1)
            and labels.levels[0].shape[-3:] == image.levels[0].shape[-3:]):
        # labels saved without voxel size, on the grid of the image
        labels.scales = [s * image.scales[0] for s in labels.scales]

    spatial = labels.spatial
    columns = centroid_columns(rows, spatial)
    if columns is not None:
        centroid = rows[columns].to_numpy(float)
        if not centroids_um:
            centroid = centroid * labels.scales[0]
        radius = object_radius_um(rows, len(spatial), labels.scales[0])
        where = f"centroids from {', '.join(columns)}"
    else:
        centroid, radius = centroids_from_labels(labels, rows[label_column].to_numpy(), notes=notes)
        where = "centroids computed from the labels"
    voxel = labels.scales[0].min()
    half = np.where(np.isfinite(radius), np.maximum(THUMBNAIL_CONTEXT * radius, 6 * voxel), 12 * voxel)
    return {
        "image": image, "image_name": short_name(image_path), "limits": image_contrast_limits(image),
        "labels": labels, "labels_name": labels_name, "centroid_columns": columns or [],
        "centroid_um": centroid, "half_um": half, "where": where, "notes": notes,
    }


class ImageSource:
    """One image of the table: its files, its rows of the table and, read when first needed, its data."""

    def __init__(self, key, image_path, labels_path, rows, problem=None):
        self.key, self.image_path, self.labels_path = key, image_path, labels_path
        self.rows = np.asarray(rows, int)  # sorted by label
        self.problem = problem  # why the image can't be used, else None
        self._data, self._lock = None, threading.Lock()

    @property
    def name(self):
        return short_name(self.image_path)

    @property
    def is_read(self):
        return self._data is not None

    def read(self, table, label_column, centroids_um=False):
        """See read_image; read once (from any thread)."""
        with self._lock:
            if self._data is None:
                self._data = read_image(self.image_path, self.labels_path, table.iloc[self.rows],
                                        label_column, centroids_um)
        return self._data


def image_sources(table, table_path):
    """The images of a table from several images, in the order of the table, with what is wrong with
    their files (``problem``) when they can't be used."""
    folder = Path(table_path).parent
    sources = []
    for key, rows in table.groupby(IMAGE_PATH, sort=False).indices.items():
        image = resolve_path(key, folder)
        labels = {path_text(v) for v in table[LABELS_PATH].iloc[rows]}
        labels_path = resolve_path(next(iter(labels)), folder) if len(labels) == 1 else None
        if len(labels) > 1:
            problem = f"its rows have {len(labels)} different {LABELS_PATH}"
        elif labels_path is None:
            problem = f"no {LABELS_PATH}"
        elif not image.exists():
            problem = f"not found: {image}"
        elif not labels_path.exists():
            problem = f"labels not found: {labels_path}"
        else:
            problem = None
        sources.append(ImageSource(key, image, labels_path, rows, problem))
    return sources


def read_dataset(image_path, labels_path, table_path, label_column, centroids_um=False):
    """
    Read the table and its first usable image (runs in a background thread): a generator that yields
    progress messages and returns a dict with the cleaned table and its images (ImageSource).
    A table with image_path and labels_path columns is from several images: ``image_path`` and
    ``labels_path`` are then not used.
    """
    notes = []
    yield f"Reading {Path(table_path).name} ..."
    table = read_table(table_path)
    multi = has_path_columns(table.columns)
    table = clean_table(table, label_column, notes, IMAGE_PATH if multi else None)
    if multi:
        sources = image_sources(table, table_path)
        if all(s.problem is not None for s in sources):
            raise ValueError("none of the images of the table can be read: "
                             + "; ".join(f"{s.key}: {s.problem}" for s in sources[:3]))
    elif not image_path:
        raise ValueError(f"give the intensity OME-Zarr (or {IMAGE_PATH} and {LABELS_PATH} columns in the table)")
    else:
        sources = [ImageSource(None, zarr_path(image_path), labels_path or None, np.arange(len(table)))]
    active = next(i for i, s in enumerate(sources) if s.problem is None)
    yield f"Reading {sources[active].name} ..."
    sources[active].read(table, label_column, centroids_um)
    return {"table": table, "label_column": label_column, "sources": sources, "active": active,
            "multi": multi, "centroids_um": centroids_um, "notes": notes}


class Session:
    """What is loaded: the table (one row per object, sorted), its images and the layers of the shown one."""

    def __init__(self, table, label_column, sources, multi=False, centroids_um=False):
        self.table, self.label_column, self.sources = table, label_column, sources
        self.multi, self.centroids_um = multi, centroids_um
        self.labels = table[label_column].to_numpy()
        self.source_of_row = np.full(len(table), -1)
        for i, source in enumerate(sources):
            self.source_of_row[source.rows] = i
        self.time_column = next((c for c in TIME_COLUMNS if c in table.columns), None)
        self.active = None  # index of the shown image
        self.layer, self.image_layers = None, []
        self.feature_rows = np.zeros(0, int)  # table row of every row of the layer features
        self.extra = {}  # columns added to the features by the plotter (clusters, ...): name -> (values, categorical)

    def data(self, index):
        """What was read from an image (read now if needed: from a background thread)."""
        return self.sources[index].read(self.table, self.label_column, self.centroids_um)

    def position(self, row):
        """(centroid µm, thumbnail half width µm) of the object of a table row."""
        source = self.sources[self.source_of_row[row]]
        data = self.data(self.source_of_row[row])
        k = int(np.searchsorted(source.rows, row))
        return data["centroid_um"][k], data["half_um"][k]

    def row_of(self, label, index=None):
        """Table row of a label of an image (the shown one by default), or -1."""
        rows = self.sources[self.active if index is None else index].rows
        k = int(np.searchsorted(self.labels[rows], label))
        return int(rows[k]) if k < len(rows) and self.labels[rows[k]] == label else -1

    def features(self, rows):
        """Features of the labels layer for these table rows, and the label of every row (0 first) for
        the plotter: the label in the shown image, else an id that isn't in its label image."""
        features = self.table.iloc[rows].drop(columns=list(PATH_COLUMNS) if self.multi else [])
        features = features.rename(columns={"index": "index (table)"}).reset_index(drop=True)
        if self.multi:
            features[IMAGE_INDEX] = pd.Categorical(self.source_of_row[rows] + 1)
        ids = np.where(self.source_of_row[rows] == self.active, self.labels[rows], -1 - np.asarray(rows))
        features["index"] = ids  # napari: label -> row
        for column, (values, categorical) in self.extra.items():
            features[column] = pd.Categorical(values[rows]) if categorical else values[rows]
        return features, np.r_[0, ids]

    def keep_plotter_columns(self):
        """Columns that the plotter added to the layer features (clusters, ...), kept by table row."""
        if self.layer is None or len(self.layer.features) != len(self.feature_rows):
            return
        ours = set(self.features(self.feature_rows[:0])[0].columns) - set(self.extra)
        for column, values in self.layer.features.items():
            categorical = isinstance(values.dtype, pd.CategoricalDtype)
            if column in ours or not (categorical or pd.api.types.is_numeric_dtype(values)):
                continue
            if column not in self.extra:
                default = np.zeros(len(self.table), object) if categorical else np.full(len(self.table), np.nan)
                self.extra[column] = (default, categorical)
            self.extra[column][0][self.feature_rows] = values.to_numpy() if categorical else values.to_numpy(float)

    def thumbnail(self, row, channel, field_of_view_um=0):
        """RGB views of the object of a table row (runs in a background thread)."""
        data = self.data(self.source_of_row[row])
        centroid, half = self.position(row)
        if not np.all(np.isfinite(centroid)):
            return []
        labels_pyramid = data["labels"]
        centre = dict(zip(labels_pyramid.spatial, centroid))
        half = field_of_view_um / 2 if field_of_view_um > 0 else half
        t = int(self.table[self.time_column].iloc[row]) if self.time_column else 0
        image, *image_grid = data["image"].box(centre, half, channel, t)
        labels, *labels_grid = labels_pyramid.box(centre, half, 0, t)
        if labels.ndim != image.ndim:
            return []
        return thumbnail_views(image.astype(np.float32), resample(labels, *labels_grid, image.shape, *image_grid),
                               self.labels[row])


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
        self._loader = self._opening = self._pending = None
        self._multi_table = False  # the table in the field has image_path and labels_path columns
        self._ticked = set()  # images whose objects are plotted
        self._syncing = False  # True while the plot and the viewer update each other
        self._points_data = self._point_rows = self._tree = None
        self._hover_row = None
        self._thumbnails = OrderedDict()  # (table row, channel, field of view) -> views
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

        self.images = QListWidget()
        self.images.setMaximumHeight(170)
        self.images.itemChanged.connect(self._ticks_changed)
        self.images.itemDoubleClicked.connect(
            lambda item: self.show_image(item.data(Qt.ItemDataRole.UserRole)))
        all_images, shown_image = QPushButton("All"), QPushButton("Only the shown one")
        all_images.clicked.connect(lambda: self._tick(lambda i: True))
        shown_image.clicked.connect(lambda: self._tick(lambda i: i == self.session.active))
        buttons = QWidget()
        row = QHBoxLayout(buttons)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(all_images)
        row.addWidget(shown_image)
        hint = QLabel("Tick the images to plot. Double-click one to show it (or click one of its points).")
        hint.setWordWrap(True)

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
        self.images_section = make_section(_title("Images of the table:"), hint, self.images, buttons)
        self.images_section.setVisible(False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(make_section(_title("Data (drop the files here):"), data, self.load_holder, self.info))
        layout.addWidget(self.images_section)
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
        try:
            columns = list(read_table(path, nrows=0).columns) if path.is_file() else None
        except Exception:
            columns = None
        self._multi_table = columns is not None and has_path_columns(columns)
        tip = f"Not used: the table has {IMAGE_PATH} and {LABELS_PATH} columns" if self._multi_table else ""
        for field in (self.image_field, self.labels_field):
            field.setEnabled(not self._multi_table)
            field.setToolTip(tip)
        if not columns:
            return
        if self._multi_table and self.session is None:
            self.info.setText(f"A table from several images: they are read from its {IMAGE_PATH} and "
                              f"{LABELS_PATH} columns.")
        current = self.label_column.currentText()
        self.label_column.clear()
        self.label_column.addItems(columns)
        self.label_column.setCurrentText(current if current in columns else guess_label_column(columns))

    def _isolate_toggled(self, on):
        if self.session is not None and self.session.layer is not None and not on:
            self.session.layer.show_selected_label = False

    # ----------------------------------------------------------------- loading
    def load(self):
        """Read the data in the background, then show it (the data loaded before stays until then)."""
        if not self.table_field.path:
            show_warning("Give the features table")
            return
        if not self._multi_table and not self.image_field.path:
            show_warning(f"Give the intensity OME-Zarr (or {IMAGE_PATH} and {LABELS_PATH} columns in the table)")
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
        if isinstance(error, RepeatedLabelsError):
            self.info.setText("Could not load the table: repeated labels.")
            QMessageBox.warning(self, "Repeated labels", str(error))
            return
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
        for note in data["notes"]:
            print(note)
        self._clear()
        s = Session(data["table"], data["label_column"], data["sources"], data["multi"], data["centroids_um"])
        self.session = s
        self._ticked = {i for i, source in enumerate(s.sources) if source.problem is None}
        self._fill_image_list()
        self.images_section.setVisible(s.multi)
        self._show_image(data["active"])

    def _fill_image_list(self):
        s = self.session
        self.images.blockSignals(True)
        self.images.clear()
        for i, source in enumerate(s.sources):
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, i)
            item.setToolTip(f"{source.image_path}\n{source.labels_path}")
            if source.problem is None:
                item.setText(f"{i + 1}. {source.name} ({len(source.rows)} objects)")
                item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked if i in self._ticked else Qt.CheckState.Unchecked)
            else:
                item.setText(f"{i + 1}. {source.key}: {source.problem}")
                item.setFlags(Qt.ItemFlag.NoItemFlags)  # greyed out
            self.images.addItem(item)
        self.images.blockSignals(False)

    def _mark_shown_image(self):
        for i in range(self.images.count()):
            item = self.images.item(i)
            font = QFont(item.font())
            font.setBold(item.data(Qt.ItemDataRole.UserRole) == self.session.active)
            item.setFont(font)

    def _ticks_changed(self, item=None):
        """Images ticked or unticked in the list: the plot shows the objects of the ticked ones."""
        ticked = {self.images.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.images.count())
                  if self.images.item(i).checkState() == Qt.CheckState.Checked}
        if not ticked:  # an empty plot: keep the last image
            show_info("At least one image stays ticked")
            self._fill_image_list()
            self._mark_shown_image()
            return
        self._ticked = ticked
        self._set_features()

    def _tick(self, which):
        """Tick the usable images for which ``which(index)`` is true, untick the others."""
        if self.session is None:
            return
        self.images.blockSignals(True)
        for i in range(self.images.count()):
            item = self.images.item(i)
            if item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                checked = which(item.data(Qt.ItemDataRole.UserRole))
                item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        self.images.blockSignals(False)
        self._ticks_changed()

    def show_image(self, index, row=None):
        """Show an image of the table (read in the background if needed), then go to a row's object."""
        s = self.session
        if s is None or s.sources[index].problem is not None:
            return
        if index == s.active:
            if row is not None:
                self.go_to(row)
            return
        if self._opening is not None:  # after the image being read
            self._pending = (index, row)
            return
        if not s.sources[index].is_read:
            self.info.setText(f"Reading {s.sources[index].name} ...")
        self._opening = create_worker(s.data, index, _connect={
            "returned": lambda _: self._image_read(s, index, row), "errored": self._load_failed,
            "finished": self._opening_done})

    def _image_read(self, session, index, row):
        if session is not self.session:  # loaded again meanwhile
            return
        try:
            self._show_image(index)
            if row is not None:
                self.go_to(row)
        except Exception as error:
            self._load_failed(error)

    def _opening_done(self):
        self._opening = None
        if self._pending is not None:
            pending, self._pending = self._pending, None
            self.show_image(*pending)

    def _show_image(self, index):
        """Show an image already read: its channels in new layers, its labels in the labels layer."""
        s = self.session
        data = s.data(index)
        first = s.layer is None
        axes = None if first else (self.plotter.x_axis, self.plotter.y_axis, self.plotter.hue_axis)
        self._hide_popup()
        for layer in s.image_layers:
            if layer in self.viewer.layers:
                self.viewer.layers.remove(layer)
        s.image_layers = list(add_image_layers(self.viewer, data["image"], data["image_name"], data["limits"]))
        s.active = index
        labels = data["labels"]
        multiscale = len(labels.levels) > 1 or s.multi  # several images: the layer takes any of them
        levels = labels.levels if multiscale else labels.levels[0]
        if first:
            s.layer = self.viewer.add_labels(levels, multiscale=multiscale, scale=labels.layer_scale(),
                                             name=LABELS_LAYER if s.multi else data["labels_name"], opacity=0.7)
        else:
            s.layer.data = levels
            s.layer.scale = labels.layer_scale()
            self.viewer.layers.move(self.viewer.layers.index(s.layer), len(self.viewer.layers))
        self._set_features()
        self.viewer.layers.selection.select_only(s.layer)  # -> the plotter takes its features

        channel = self.channel.currentIndex()
        self.channel.blockSignals(True)
        self.channel.clear()
        self.channel.addItems(channel_names(data["image"]))
        self.channel.setCurrentIndex(min(max(channel, 0), self.channel.count() - 1))
        self.channel.blockSignals(False)
        if first:
            bbox = {c for c in s.table.columns if c.lower().startswith("bbox")}
            self._choose_plot_features({s.label_column, "index", IMAGE_INDEX, *data["centroid_columns"], *bbox})
        else:
            self._set_plot_features(*axes)
        self._mark_shown_image()
        if s.multi:
            usable = [source for source in s.sources if source.problem is None]
            unusable = len(s.sources) - len(usable)
            self.info.setText(f"{sum(len(source.rows) for source in usable)} objects in {len(usable)} images"
                              + (f" ({unusable} can't be read)" if unusable else "")
                              + f". Shown: {s.sources[index].name}, {data['where']}")
        else:
            self.info.setText(f"{len(s.table)} objects, {data['where']}")

    def _set_features(self):
        """Features of the labels layer: the objects of the ticked images (and the plotter's columns)."""
        s = self.session
        s.keep_plotter_columns()
        ticked = sorted(self._ticked)
        rows = np.concatenate([s.sources[i].rows for i in ticked]) if ticked else np.zeros(0, int)
        features, ids = s.features(rows)
        s.feature_rows = rows
        self._points_data = self._point_rows = self._tree = None
        s.layer.metadata[LABEL_IDS_KEY] = ids  # before the features: the plotter recolours the labels
        s.layer.features = features

    def _choose_plot_features(self, skip):
        """First features of the table on x, y and as colour."""
        table = self.session.table
        numeric = [c for c in table.columns if c not in skip and pd.api.types.is_numeric_dtype(table[c])]
        if numeric:
            hue = numeric[2] if len(numeric) > 2 else numeric[-1]
            self._set_plot_features(numeric[0], numeric[min(1, len(numeric) - 1)], hue)

    def _set_plot_features(self, x, y, hue):
        boxes = self.plotter.control_widget
        for box, column in ((boxes.x_axis_box, x), (boxes.y_axis_box, y)):
            if box.findText(column) >= 0:
                box.blockSignals(True)
                box.setCurrentText(column)
                box.blockSignals(False)
        if boxes.hue_box.currentText() == hue or boxes.hue_box.findText(hue) < 0:
            self.plotter.plot_needs_update.emit()
        else:
            boxes.hue_box.setCurrentText(hue)

    def _clear(self):
        self._hide_popup()
        self._thumbnails.clear()
        self._points_data = self._point_rows = self._tree = None
        self._pending = None
        if self.session is not None:
            for layer in [self.session.layer, *self.session.image_layers]:
                if layer is not None and layer in self.viewer.layers:
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
        """Table row of every point of the plot (-1: another layer), updated when the plot data change."""
        data = self.plotter.plotting_widget.active_artist.data
        s = self.session
        if s is None or s.layer is None or data is None:
            return None
        if data is not self._points_data:
            features = self.plotter._get_features()
            if len(features) != len(data) or "layer" not in features.columns:
                return None
            ours = features["layer"].astype(str).to_numpy() == s.layer.name
            if ours.sum() != len(s.feature_rows):
                return None
            rows = np.full(len(data), -1)
            rows[ours] = s.feature_rows
            self._point_rows = rows
            self._points_data, self._tree = data, None
        return self._point_rows

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
        """A point clicked in the plot: go to its object (showing its image first)."""
        if self._syncing or highlighted is None or self._plot_points() is None:
            return
        points = np.flatnonzero(highlighted)
        if len(points) == 1 and self._point_rows[points[0]] >= 0:
            row = int(self._point_rows[points[0]])
            self.show_image(int(self.session.source_of_row[row]), row)

    def go_to(self, row):
        """Move the viewer to the object of a table row of the shown image: its plane, centred and
        zoomed in, its label selected."""
        s = self.session
        centre, half = s.position(row)
        if not np.all(np.isfinite(centre)):
            show_warning(f"label {s.labels[row]}: position unknown")
            return
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
                scene.camera.zoom = 0.9 * min(canvas) / (2 * JUMP_CONTEXT * half)
        self._syncing = True
        try:
            s.layer.selected_label = int(s.labels[row])
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
        s = self.session
        if self._syncing or s is None or s.layer is None or getattr(event, "source", s.layer) is not s.layer:
            return
        rows = self._plot_points()
        artist = self.plotter.plotting_widget.active_artist
        if rows is None or not isinstance(artist, Scatter):
            return
        ours = rows >= 0
        highlighted = np.zeros(len(rows), bool)
        highlighted[ours] = ((s.source_of_row[rows[ours]] == s.active)
                             & (s.labels[rows[ours]] == s.layer.selected_label))
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
        row = -1 if index is None else int(self._point_rows[index])
        if row < 0:
            self._hide_popup()
            return
        if row != self._hover_row:
            self._hover_row = row
            self._hover_timer.start()
        elif self.popup.isVisible():
            self.popup.move_near_mouse()

    def _hide_popup(self, *_):
        self._hover_row = None
        self._hover_timer.stop()
        self.popup.hide()

    def _thumbnail_key(self):
        return self._hover_row, self.channel.currentIndex(), self.field_of_view.value()

    def _hover_text(self):
        s, row = self.session, self._hover_row
        lines = [f"label {s.labels[row]}"]
        if s.multi:
            lines[0] += f" of {s.sources[s.source_of_row[row]].name}"
        features = s.layer.features
        position = int(np.searchsorted(s.feature_rows, row))
        if position < len(features):
            for name in dict.fromkeys([self.plotter.x_axis, self.plotter.y_axis, self.plotter.hue_axis]):
                if name in features.columns:
                    value = features[name].iloc[position]
                    lines.append(f"{name}: {value:.4g}" if isinstance(value, (float, np.floating)) else f"{name}: {value}")
        return "\n".join(lines)

    def _request_thumbnail(self):
        if self._hover_row is None or self.session is None:
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
        print(f"thumbnail of table row {self._worker_key[0]}: {error!r}")
        self._thumbnails[self._worker_key] = []

    def _worker_done(self):
        self._worker = None
        if self._hover_row is not None and self._thumbnail_key() not in self._thumbnails:
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
