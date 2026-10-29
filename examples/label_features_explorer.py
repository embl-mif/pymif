"""
Explore the features of segmented objects in napari, with napari-clusters-plotter.

Drop (or browse) three things on the "Label features" panel, then press Load:
  - an intensity OME-Zarr (all channels and resolution levels, read lazily with pymif)
  - the labels OME-Zarr (optional: else the first group in <intensity>.ome.zarr/labels/)
  - a CSV with one row per object: a label column + any features
The files can be dropped one by one on their field, or all at once anywhere on the panel.

The scatter plot of napari-clusters-plotter opens with the labels selected, with
everything it can do: x / y features, colour the plot and the labels by any feature
(hue), draw selections (lasso, ellipse, rectangle) to make clusters, histogram, ...
On top of that:
  - click a point: the viewer goes to that object (its plane, centred, zoomed in)
    and selects its label
  - hover a point: a thumbnail of the object (XY and XZ through its centre, the
    intensity with the object outlined in magenta) appears next to the mouse
  - pick a label in napari: its point is highlighted in the plot

Where the objects are comes from centroid columns of the CSV: centroid-0/1/2 (skimage
regionprops_table), centroid_z/y/x or z/y/x, in voxels of the labels (full resolution),
or in µm (tick the box). Without them they are computed from a low resolution level of
the labels (slow for big images, and small objects can be missed). The size of the
thumbnail comes from bbox-* columns (regionprops) or a size column (area, volume, ...).

Run:
    python label_features_explorer.py
    python label_features_explorer.py image.ome.zarr labels.ome.zarr features.csv
    python label_features_explorer.py image.ome.zarr "" features.csv --label-column label

Needs napari, pymif and napari-clusters-plotter (pip install napari-clusters-plotter).
"""

import argparse
import traceback
from collections import OrderedDict
from pathlib import Path

import napari
import napari_clusters_plotter._utilities as ncp_utilities
import numpy as np
import pandas as pd
import pymif.microscope_manager as mm
from biaplotter.artists import Scatter
from matplotlib.colors import to_rgb
from napari.qt.threading import create_worker
from napari.utils import Colormap
from napari.utils.notifications import show_error, show_info, show_warning
from napari_clusters_plotter import PlotterWidget
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QCursor, QGuiApplication, QImage, QPixmap
from qtpy.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget,
)
from scipy.spatial import cKDTree
from skimage.segmentation import find_boundaries

# ---------------------------------------------------------------------------
# parameters
LABEL_COLUMNS = ("label", "label_id", "labels", "id", "object_id", "cell_id")  # guessed label column
SIZE_COLUMNS = ("area", "volume", "num_pixels", "n_voxels", "voxels", "size")    # object size (voxels)
THUMBNAIL_PX = 170         # size of one thumbnail view on the screen
THUMBNAIL_CONTEXT = 2.5    # thumbnail half width = 2.5 x object radius
MAX_BOX_PX = 128           # thumbnails are read from the finest level with at most this many voxels per axis
HOVER_RADIUS_PX = 7        # a point is hovered when the mouse is this close (screen pixels)
HOVER_DELAY_MS = 40        # wait this long on a point before reading its thumbnail
JUMP_CONTEXT = 1.5         # after a click the viewer shows 1.5 x the thumbnail field of view
OBJECT_RGB = (255, 0, 255)  # outline of the object in the thumbnail
OTHER_RGB = (0, 200, 255)   # outlines of the other objects
# ---------------------------------------------------------------------------


# The plotter colours a labels layer by np.unique(the whole label image) on every redraw (minutes for
# a big image) and gives the colour of row i of the features to the i-th label. The rows here are the
# labels of the table, sorted, so they are given directly.
_unique_values = ncp_utilities._get_unique_values


def _labels_of_features(layer):
    ids = layer.metadata.get("feature_label_ids")
    return ids if ids is not None else _unique_values(layer)


ncp_utilities._get_unique_values = _labels_of_features


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


def clean_table(table, label_column):
    """One row per label (> 0), sorted by label; text columns become categories (1, 2, ...; 0 = empty)."""
    if label_column not in table.columns:
        raise ValueError(f"no column '{label_column}' in the table")
    table = table.copy()
    table[label_column] = pd.to_numeric(table[label_column], errors="coerce")
    table = table[table[label_column] > 0].copy()
    table[label_column] = table[label_column].astype(np.int64)
    table = table.drop_duplicates(label_column).sort_values(label_column).reset_index(drop=True)
    for column in table.columns:
        if not pd.api.types.is_numeric_dtype(table[column]):
            categories = pd.Categorical(table[column])
            table[column] = pd.Categorical(categories.codes.astype(int) + 1)  # int: biaplotter's categories
            names = ", ".join(f"{i + 1} = {c}" for i, c in enumerate(categories.categories[:20]))
            print(f"{column} (categories): {names}{', ...' if len(categories.categories) > 20 else ''}")
    return table


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


def centroids_from_labels(labels, ids, max_voxels=3e7):
    """(centroid µm, radius µm) of every label in ids, from the finest level with at most max_voxels voxels."""
    if "t" in labels.axes:
        raise ValueError("time-lapse labels: add centroid columns to the table")
    level = next((i for i, a in enumerate(labels.levels) if np.prod(a.shape) <= max_voxels),
                 len(labels.levels) - 1)
    show_info(f"Computing the centroids on level {level} of the labels ...")
    data = np.asarray(labels.levels[level])
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
    if (rows < 0).any():
        show_warning(f"{(rows < 0).sum()} labels of the table are not in level {level}: no position for them")
    return out_centre, out_radius


def contrast_limits(array):
    """1 - 99.9 percentiles of an image, from at most ~4 million of its voxels."""
    step = max(1, int(np.ceil((array.size / 4e6) ** (1 / array.ndim))))
    sample = np.asarray(array[(slice(None, None, step),) * array.ndim])
    low, high = np.percentile(sample, (1, 99.9))
    return [float(low), float(max(high, low + 1))]


def hex_colormap(value):
    rgb = to_rgb("#" + str(value).lstrip("#")[:6])
    return Colormap([[0, 0, 0, 1], [*rgb, 1]], name="#" + str(value).lstrip("#")[:6].lower())


def add_image_layers(viewer, image, name):
    """The intensity image, one layer per channel (all levels, read when needed)."""
    multiscale = len(image.levels) > 1
    data = image.levels if multiscale else image.levels[0]
    lowest = image.levels[-1]
    if "t" in image.axes:
        lowest = lowest[(slice(None),) * image.axes.index("t") + (0,)]
    if "c" not in image.axes:
        return [viewer.add_image(data, multiscale=multiscale, scale=image.layer_scale(), name=name,
                                 contrast_limits=contrast_limits(lowest))]
    c = image.axes.replace("t", "").index("c")
    n = lowest.shape[c]
    channels = image.metadata.get("channel_names") or [f"channel {i}" for i in range(n)]
    colours = image.metadata.get("channel_colors") or []
    colormaps = None  # napari's default colours, unless the file has its own
    if len(colours) == n and any(str(h).lstrip("#").upper()[:6] != "FFFFFF" for h in colours):
        colormaps = [hex_colormap(h) for h in colours]
    limits = [contrast_limits(lowest[(slice(None),) * c + (i,)]) for i in range(n)]
    return viewer.add_image(data, channel_axis=image.axes.index("c"), multiscale=multiscale,
                            scale=image.layer_scale(), name=[f"{name}: {ch}" for ch in channels],
                            colormap=colormaps, contrast_limits=limits, blending="additive")


def zarr_path(text):
    """A dropped path as the OME-Zarr folder (a metadata file inside it -> the folder)."""
    path = Path(text.strip().strip('"')).expanduser()
    if path.is_file() and path.name in ("zarr.json", ".zattrs", ".zgroup"):
        path = path.parent
    if path.name == "labels" and path.is_dir():  # the labels group of an image: its first label image
        path = next((p for p in sorted(path.iterdir()) if p.is_dir()), path)
    return path


def short_name(path):
    return Path(path).name.replace(".ome.zarr", "").replace(".zarr", "")


class Session:
    """What is loaded: the layers, the table (one row per label, sorted) and where every object is."""

    def __init__(self, image, labels, layer, image_layers, table, label_column, centroid_um, half_um):
        self.image, self.labels, self.layer, self.image_layers = image, labels, layer, image_layers
        self.table, self.label_column = table, label_column
        self.rows = pd.Index(table[label_column].to_numpy())
        self.centroid_um, self.half_um = centroid_um, half_um
        self.time_column = next((c for c in ("frame", "t", "timepoint", "time") if c in table.columns), None)

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

    def __init__(self):
        super().__init__(None, Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint
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


class FeatureExplorer(QWidget):
    """Loads the data and links the plot of napari-clusters-plotter with the viewer."""

    def __init__(self, viewer, plotter):
        super().__init__()
        self.viewer, self.plotter = viewer, plotter
        self.session = None
        self._syncing = False  # True while the plot and the viewer update each other
        self._points_data = self._point_labels = self._tree = None
        self._hover_label = None
        self._thumbnails = OrderedDict()  # (label, channel, field of view) -> views
        self._worker = self._worker_key = None
        self.popup = ThumbnailPopup()
        self._hover_timer = QTimer()
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
        load = QPushButton("Load")
        load.clicked.connect(self.load)
        self.table_field.edit.textChanged.connect(self._table_changed)

        data = QGroupBox("Data (drop the files here)")
        form = QFormLayout(data)
        form.addRow("Intensity", self.image_field)
        form.addRow("Labels", self.labels_field)
        form.addRow("Features", self.table_field)
        form.addRow("Label column", self.label_column)
        form.addRow(self.centroids_um)
        form.addRow(load)

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

        links = QGroupBox("Plot and viewer")
        form = QFormLayout(links)
        form.addRow(self.zoom)
        form.addRow(self.isolate)
        form.addRow(self.hover)
        form.addRow("Thumbnail channel", self.channel)
        form.addRow("Thumbnail width", self.field_of_view)

        self.info = QLabel("")
        self.info.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(data)
        layout.addWidget(links)
        layout.addWidget(self.info)
        layout.addStretch()
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        """Several files dropped on the panel: sorted by kind (table, labels, intensity)."""
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if Path(path).suffix.lower() in (".csv", ".tsv", ".txt"):
                self.table_field.edit.setText(path)
            elif "label" in Path(path).name.lower():
                self.labels_field.edit.setText(path)
            else:
                self.image_field.edit.setText(path)
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
        guess = next((c for c in columns if c.lower() in LABEL_COLUMNS), columns[0] if columns else "")
        self.label_column.setCurrentText(current if current in columns else guess)

    def _isolate_toggled(self, on):
        if self.session is not None and not on:
            self.session.layer.show_selected_label = False

    # ----------------------------------------------------------------- loading
    def load(self):
        try:
            self._load()
        except Exception as error:
            traceback.print_exc()
            show_error(f"Could not load the data: {error}")

    def _load(self):
        if not self.image_field.path or not self.table_field.path:
            show_warning("Give at least the intensity OME-Zarr and the features table")
            return
        image_path = zarr_path(self.image_field.path)
        image_zarr = mm.ZarrManager(str(image_path))
        image = Pyramid(image_zarr.data, image_zarr.metadata)
        labels, labels_name = self._open_labels(image_zarr)
        labels.drop_channel()
        if (np.allclose(labels.scales[0], 1) and not np.allclose(image.scales[0], 1)
                and labels.levels[0].shape[-3:] == image.levels[0].shape[-3:]):
            # labels saved without voxel size, on the grid of the image
            labels.scales = [s * image.scales[0] for s in labels.scales]

        label_column = self.label_column.currentText()
        table = clean_table(read_table(self.table_field.path), label_column)
        ids = table[label_column].to_numpy()
        spatial = labels.spatial
        columns = centroid_columns(table, spatial)
        if columns is not None:
            centroid = table[columns].to_numpy(float)
            if not self.centroids_um.isChecked():
                centroid = centroid * labels.scales[0]
            radius = object_radius_um(table, len(spatial), labels.scales[0])
            where = f"centroids from {', '.join(columns)}"
        else:
            centroid, radius = centroids_from_labels(labels, ids)
            where = "centroids computed from the labels"
        voxel = labels.scales[0].min()
        half = np.where(np.isfinite(radius), np.maximum(THUMBNAIL_CONTEXT * radius, 6 * voxel), 12 * voxel)

        self._clear()  # the data loaded before (only now: it stays if the new data can't be read)
        image_layers = add_image_layers(self.viewer, image, short_name(image_path))
        features = table.rename(columns={"index": "index (table)"}) if label_column != "index" else table.copy()
        features["index"] = ids  # napari: row <-> label
        multiscale = len(labels.levels) > 1
        layer = self.viewer.add_labels(labels.levels if multiscale else labels.levels[0], multiscale=multiscale,
                                       scale=labels.layer_scale(), name=labels_name, features=features,
                                       opacity=0.7, metadata={"feature_label_ids": np.r_[0, ids]})
        self.session = Session(image, labels, layer, list(image_layers), table, label_column, centroid, half)

        self.channel.blockSignals(True)
        self.channel.clear()
        self.channel.addItems(image.metadata.get("channel_names") or ["channel 0"] if "c" in image.axes
                              else ["intensity"])
        self.channel.blockSignals(False)
        self.viewer.layers.selection.select_only(layer)  # -> the plotter takes its features
        bbox = {c for c in table.columns if c.lower().startswith("bbox")}
        self._choose_plot_features(table, {label_column, "index", *(columns or []), *bbox})
        self.info.setText(f"{len(table)} objects, {where}")

    def _open_labels(self, image_zarr):
        if self.labels_field.path:
            path = zarr_path(self.labels_field.path)
            labels_zarr = mm.ZarrManager(str(path))
            return Pyramid(labels_zarr.data, labels_zarr.metadata), short_name(path)
        groups = getattr(image_zarr, "labels", None) or {}
        if not groups:
            raise ValueError("no labels given, and none in <intensity>.ome.zarr/labels/")
        name = next(iter(groups))
        show_info(f"Labels: '{name}' of the intensity OME-Zarr")
        return Pyramid(groups[name].data, groups[name].metadata), name

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
        self.viewer.camera.center = tuple(centre[-3:])
        if self.zoom.isChecked():
            try:
                canvas = self.viewer.window._qt_viewer.canvas.size  # (height, width) in screen pixels
                self.viewer.camera.zoom = 0.9 * min(canvas) / (2 * JUMP_CONTEXT * s.half_um[row])
            except AttributeError:
                pass
        self._syncing = True
        try:
            s.layer.selected_label = label
            s.layer.show_selected_label = self.isolate.isChecked()
        finally:
            self._syncing = False

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
        while len(self._thumbnails) > 500:
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


def main():
    parser = argparse.ArgumentParser(description="Explore the features of segmented objects in napari.")
    parser.add_argument("image", nargs="?", default="", help="intensity .ome.zarr")
    parser.add_argument("labels", nargs="?", default="", help="labels .ome.zarr ('' = <image>/labels/)")
    parser.add_argument("table", nargs="?", default="", help="features .csv")
    parser.add_argument("--label-column", help="column of the table with the label ids")
    parser.add_argument("--centroids-um", action="store_true", help="centroids of the table are in µm")
    args = parser.parse_args()

    viewer = napari.Viewer(title="Label features explorer")
    plotter = PlotterWidget(viewer)
    explorer = FeatureExplorer(viewer, plotter)
    viewer.window.add_dock_widget(explorer, name="Label features", area="right")
    viewer.window.add_dock_widget(plotter, name="Plot & select features", area="right")

    explorer.image_field.edit.setText(args.image)
    explorer.labels_field.edit.setText(args.labels)
    explorer.table_field.edit.setText(args.table)
    if args.label_column:
        explorer.label_column.setCurrentText(args.label_column)
    explorer.centroids_um.setChecked(args.centroids_um)
    if args.image and args.table:
        QTimer.singleShot(0, explorer.load)
    napari.run()


if __name__ == "__main__":
    main()
