from __future__ import annotations

import os
import time

import dask.array as da
import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")
pytest.importorskip("napari")

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication

import pymif.microscope_manager as mm
from pymif.microscope_manager.zarr_manager import ZarrDataset
from pymif.napari import _explorer_widget as ew

#: Two cubes in a 1 x 8 x 32 x 32 label image: label -> (z, y, x) slices.
OBJECTS = {3: (slice(2, 6), slice(4, 12), slice(4, 12)), 5: (slice(1, 3), slice(20, 28), slice(16, 30))}
#: The objects of a second image (label 3 again, elsewhere).
OTHER_OBJECTS = {3: (slice(4, 8), slice(20, 30), slice(2, 10)), 8: (slice(0, 2), slice(2, 6), slice(24, 30))}
SCALES = ((2.0, 0.5, 0.5), (4.0, 1.0, 1.0))


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _metadata(channels=("A", "B")):
    n = len(channels)
    return {
        "size": [(1, n, 8, 32, 32), (1, n, 4, 16, 16)],
        "chunksize": [(1, 1, 4, 16, 16), (1, 1, 4, 8, 8)],
        "scales": list(SCALES),
        "units": ("micrometer",) * 3,
        "time_increment": 1.0,
        "time_increment_unit": "second",
        "channel_names": list(channels),
        "channel_colors": ["FF0000", "00FF00"][:n] if n > 1 else ["FFFFFF"],
        "dtype": "uint16",
        "plane_files": None,
        "axes": "tczyx",
        "name": "cells",
    }


def _labels(objects=OBJECTS):
    labels = np.zeros((1, 8, 32, 32), np.uint16)
    for label, box in objects.items():
        labels[(0, *box)] = label
    return labels


def _write(path, array, channels=("A", "B")):
    """An OME-Zarr of two levels from a (t, c, z, y, x) array."""
    pyramid = [da.from_array(array), da.from_array(array[:, :, ::2, ::2, ::2])]
    mm.ArrayManager(pyramid, _metadata(channels)).to_zarr(str(path), progress=False)


def _image(seed=0):
    return np.random.default_rng(seed).integers(0, 1000, (1, 2, 8, 32, 32), dtype=np.uint16)


@pytest.fixture
def dataset(tmp_path):
    """An OME-Zarr with two channels and the labels in labels/cells, and a table without centroids."""
    path = tmp_path / "cells.ome.zarr"
    _write(path, _image())
    zarr = mm.ZarrManager(str(path), mode="a")
    zarr.create_empty_group("cells", _metadata(), is_label=True)
    zarr.write_label_region(_labels(), level=0, group="labels/cells")
    table = tmp_path / "features.csv"
    pd.DataFrame({"label": [5, 3, 0], "area": [224, 256, 1], "solidity": [np.inf, 0.9, 1.0],
                  "kind": ["b", "a", "a"]}).to_csv(table, index=False)
    return path, table


@pytest.fixture
def multi_dataset(tmp_path):
    """Two images with their labels as separate files, and a table of both (plus a missing image),
    with paths relative to the table (one written with Windows separators)."""
    for name, objects, seed in (("a", OBJECTS, 0), ("b", OTHER_OBJECTS, 1)):
        _write(tmp_path / "images" / f"{name}.ome.zarr", _image(seed))
        _write(tmp_path / "labels" / f"{name}_labels.ome.zarr", _labels(objects)[:, None], channels=("labels",))
    table = tmp_path / "features.csv"
    pd.DataFrame({
        "label": [5, 3, 3, 8, 1],
        "image_path": ["images/a.ome.zarr", "images/a.ome.zarr", "images\\b.ome.zarr", "images\\b.ome.zarr",
                       "images/missing.ome.zarr"],
        "labels_path": ["labels/a_labels.ome.zarr", "labels/a_labels.ome.zarr", "labels\\b_labels.ome.zarr",
                        "labels\\b_labels.ome.zarr", "labels/missing_labels.ome.zarr"],
        "area": [224, 256, 320, 48, 10], "solidity": [0.8, 0.9, 0.7, 0.6, 0.5],
    }).to_csv(table, index=False)
    return table


def _read(*args, **kwargs):
    """Run ``read_dataset`` to the end: (messages, returned dict)."""
    steps = ew.read_dataset(*args, **kwargs)
    messages = []
    while True:
        try:
            messages.append(next(steps))
        except StopIteration as stop:
            return messages, stop.value


def _wait(app, done, seconds=5):
    """Process Qt events until ``done()`` (background workers report through them)."""
    for _ in range(int(seconds / 0.01)):
        if done():
            return
        app.processEvents()
        time.sleep(0.01)
    raise AssertionError("timed out")


def _explorer(monkeypatch):
    pytest.importorskip("napari_clusters_plotter")
    from napari.components import ViewerModel

    viewer = ViewerModel()
    monkeypatch.setattr(ew, "current_viewer", lambda: viewer)
    dock = ew.explorer_widget()
    monkeypatch.setattr(ew, "_test_dock", dock, raising=False)  # kept alive: Qt deletes the panel with it
    return viewer, dock.explorer, dock.plotter


def _click(plotter, point):
    highlighted = np.zeros(len(plotter.plotting_widget.active_artist.data), bool)
    highlighted[point] = True
    plotter.plotting_widget.active_artist.highlighted = highlighted


def test_clean_table_sorts_labels_and_fixes_values():
    notes = []
    table = pd.DataFrame({"id": [7, "x", 2, 0], "solidity": [0.5, 1.0, np.inf, 1.0],
                          "kind": ["b", "a", "a", "a"]})
    out = ew.clean_table(table, "id", notes)
    assert out["id"].tolist() == [2, 7]  # no text or 0 labels
    assert np.isnan(out["solidity"][0]) and out["solidity"][1] == 0.5
    assert out["kind"].tolist() == [1, 2]
    assert notes[0] == "inf -> NaN: solidity (1 rows)"
    with pytest.raises(ValueError, match="no column"):
        ew.clean_table(table, "label")


def test_repeated_labels_are_an_error():
    table = pd.DataFrame({"label": [7, 2, 7, 9, 2, 7], "area": range(6)})
    with pytest.raises(ew.RepeatedLabelsError) as error:
        ew.clean_table(table, "label")
    message = str(error.value)
    assert message.startswith("Label 7 appears 3 times in the table (1 other label is repeated too).")
    assert "image_path and labels_path" in message

    table["image_path"] = ["b", "a", "c", "c", "b", "a"]
    out = ew.clean_table(table, "label", image_column="image_path")  # a label once per image: fine
    assert out[["image_path", "label"]].values.tolist() == [["a", 2], ["a", 7], ["b", 2], ["b", 7],
                                                             ["c", 7], ["c", 9]]
    table.loc[2, "image_path"] = "a"
    with pytest.raises(ew.RepeatedLabelsError, match="Label 7 of a appears 2 times"):
        ew.clean_table(table, "label", image_column="image_path")


def test_paths_of_the_table(tmp_path):
    assert ew.resolve_path(" images\\a.ome.zarr ", tmp_path) == tmp_path / "images" / "a.ome.zarr"
    assert ew.resolve_path(tmp_path / "x.zarr", "/elsewhere") == tmp_path / "x.zarr"  # absolute: as is
    assert ew.resolve_path(np.nan, tmp_path) is None and ew.resolve_path("", tmp_path) is None
    assert ew.has_path_columns(["label", "image_path", "labels_path"])
    assert not ew.has_path_columns(["label", "image_path"])


def test_image_sources_report_problems(tmp_path):
    (tmp_path / "a.zarr").mkdir()
    (tmp_path / "a_labels.zarr").mkdir()
    (tmp_path / "c.zarr").mkdir()
    table = pd.DataFrame({"label": [1, 2, 1, 1, 2, 1],
                          "image_path": ["a.zarr", "a.zarr", "b.zarr", "c.zarr", "c.zarr", "d.zarr"],
                          "labels_path": ["a_labels.zarr", "a_labels.zarr", "b_labels.zarr", "c1.zarr", "c2.zarr", ""]})
    sources = ew.image_sources(table, tmp_path / "features.csv")
    assert [s.key for s in sources] == ["a.zarr", "b.zarr", "c.zarr", "d.zarr"]
    assert sources[0].problem is None and sources[0].rows.tolist() == [0, 1]
    assert sources[0].labels_path == tmp_path / "a_labels.zarr"
    assert sources[1].problem == f"not found: {tmp_path / 'b.zarr'}"
    assert sources[2].problem == "its rows have 2 different labels_path"
    assert sources[3].problem == "no labels_path"


def test_columns_found_by_name():
    table = pd.DataFrame({"Label": [1], "Centroid_Z": [1.0], "centroid_y": [2.0], "centroid_x": [3.0],
                          "bbox-0": [0], "bbox-1": [0], "bbox-2": [0], "bbox-3": [4], "bbox-4": [2], "bbox-5": [2]})
    assert ew.guess_label_column(list(table.columns)) == "Label"
    assert ew.centroid_columns(table, "zyx") == ["Centroid_Z", "centroid_y", "centroid_x"]
    assert ew.object_radius_um(table, 3, np.array([1.0, 0.5, 0.5])) == pytest.approx([2.0])
    assert np.isnan(ew.object_radius_um(table[["Label"]], 3, np.ones(3))).all()


def test_sort_dropped_and_zarr_path(tmp_path):
    assert ew.sort_dropped(["/d/img.ome.zarr", "/d/seg_labels.zarr", "/d/feat.CSV"]) == {
        "image": "/d/img.ome.zarr", "labels": "/d/seg_labels.zarr", "table": "/d/feat.CSV"}
    (tmp_path / "img.zarr" / "labels" / "nuclei").mkdir(parents=True)
    (tmp_path / "img.zarr" / "zarr.json").write_text("{}")
    assert ew.zarr_path(tmp_path / "img.zarr" / "zarr.json") == tmp_path / "img.zarr"
    assert ew.zarr_path(f'"{tmp_path / "img.zarr" / "labels"}"') == tmp_path / "img.zarr" / "labels" / "nuclei"


def _dataset(array, axes, scales=SCALES[:1]):
    return ZarrDataset(data=[da.from_array(array)], zarr_data=None, metadata={"axes": axes, "scales": list(scales)})


def test_as_labels_drops_the_channel_and_fixes_the_voxel_size():
    image = _dataset(_image(), "tczyx")
    labels = ew.as_labels(_dataset(_labels()[:, None], "tczyx", [(1, 1, 1)]), image)
    assert ew.dataset_axes(labels) == "tzyx" and labels.data[0].shape == (1, 8, 32, 32)
    assert ew.layer_geometry(labels.metadata, "tzyx", drop_channel=False)["scale"] == (1.0, *SCALES[0])
    same = _dataset(_labels(), "tzyx")
    assert ew.as_labels(same, image) is same  # nothing to change


def test_centroids_from_labels_with_a_single_time_point():
    labels = _dataset(_labels(), "tzyx")
    centre, radius = ew.centroids_from_labels(labels, np.array([5, 3, 9]))
    assert centre[1] == pytest.approx([3.5 * 2, 7.5 * 0.5, 7.5 * 0.5])
    assert centre[0] == pytest.approx([1.5 * 2, 23.5 * 0.5, 22.5 * 0.5])
    assert np.isnan(centre[2]).all() and np.isnan(radius[2])

    two_times = _dataset(np.concatenate([_labels()] * 2), "tzyx")
    with pytest.raises(ValueError, match="time-lapse"):
        ew.centroids_from_labels(two_times, np.array([3]))


def test_thumbnail_outlines_the_object():
    labels = _dataset(_labels()[0], "zyx")
    box, start, size, offset = ew.box(labels, {"z": 7.0, "y": 3.75, "x": 3.75}, 4.0)
    assert box.shape == (5, 17, 17) and (box == 3).any()
    views = ew.thumbnail_views(box.astype(np.float32), box, 3)
    assert len(views) == 2 and views[0].dtype == np.uint8
    assert (views[0] == ew.OBJECT_RGB).all(axis=-1).any()


def test_read_dataset(dataset):
    image_path, table = dataset
    messages, data = _read(str(image_path), "", str(table), "label")
    assert messages == ["Reading features.csv ...", "Reading cells ..."]
    assert not data["multi"] and data["table"]["label"].tolist() == [3, 5]
    assert np.isnan(data["table"]["solidity"][1])
    image = data["sources"][0].read(data["table"], "label")  # read already: the same data
    assert image["labels_name"] == "cells" and len(image["limits"]) == 2
    assert image["centroid_um"][0] == pytest.approx([3.5 * 2, 7.5 * 0.5, 7.5 * 0.5])
    assert "labels: 'cells' of the intensity OME-Zarr" in image["notes"]


def test_read_dataset_of_several_images(multi_dataset):
    messages, data = _read("", "", str(multi_dataset), "label")  # the intensity and labels fields: not used
    assert data["multi"] and [s.name for s in data["sources"]] == ["a", "b", "missing"]
    assert data["sources"][2].problem.startswith("not found")
    assert data["sources"][0].is_read and not data["sources"][1].is_read  # the others: when needed
    assert data["table"]["image_path"].tolist()[2:4] == ["images/b.ome.zarr"] * 2


def test_widget_loads_and_goes_to_an_object(app, dataset, monkeypatch):
    viewer, explorer, plotter = _explorer(monkeypatch)
    image_path, table = dataset
    explorer.table_field.value = table
    assert explorer.label_column.currentText() == "label"

    explorer.image_field.value = image_path
    explorer.load()  # in a background thread, the Load button locked meanwhile
    assert not explorer.load_holder.isEnabled()
    _wait(app, lambda: explorer._loader is None)
    assert explorer.load_holder.isEnabled()
    assert explorer.info.text() == "2 objects, centroids computed from the labels"
    assert [layer.name for layer in viewer.layers] == ["cells: A", "cells: B", "cells"]
    assert explorer.session.layer.metadata[ew.LABEL_IDS_KEY].tolist() == [0, 3, 5]
    assert not explorer.images_section.isVisibleTo(explorer)
    assert (plotter.x_axis, plotter.y_axis) == ("area", "solidity")
    assert [explorer.channel.itemText(i) for i in range(explorer.channel.count())] == ["A", "B"]

    # a click on the point of label 5 (row 1) moves the viewer to it
    _click(plotter, 1)
    assert explorer.session.layer.selected_label == 5
    assert viewer.dims.point[-3:] == pytest.approx(explorer.session.position(1)[0], abs=2)

    views = explorer.session.thumbnail(0, channel=1)
    assert len(views) == 2

    # loading again replaces the layers
    explorer._show(_read(str(image_path), "", str(table), "label")[1])
    assert len(viewer.layers) == 3


def test_drops_on_a_field_and_on_the_panel(app, monkeypatch, tmp_path):
    from qtpy.QtCore import QMimeData, QPoint, QPointF, QUrl
    from qtpy.QtGui import QDragEnterEvent, QDropEvent
    from qtpy.QtWidgets import QLineEdit

    _, explorer, _ = _explorer(monkeypatch)
    image, labels, table = tmp_path / "a.ome.zarr", tmp_path / "a_labels.ome.zarr", tmp_path / "f.csv"

    def mime(*paths):
        data = QMimeData()
        data.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
        return data

    def drag_enter(widget, data):
        event = QDragEnterEvent(QPoint(1, 1), Qt.DropAction.CopyAction, data, Qt.MouseButton.NoButton,
                                Qt.KeyboardModifier.NoModifier)
        event.accept()
        widget.dragEnterEvent(event)
        return event.isAccepted()

    line = explorer.labels_field.native.findChild(QLineEdit)
    assert drag_enter(line, mime(labels))  # one file: the field takes it
    assert not drag_enter(line, mime(image, labels, table))  # several: left to the panel
    several = mime(image, labels, table)
    explorer.dropEvent(QDropEvent(QPointF(1, 1), Qt.DropAction.CopyAction, several, Qt.MouseButton.NoButton,
                                  Qt.KeyboardModifier.NoModifier))
    assert (explorer.image_field.value, explorer.labels_field.value, explorer.table_field.value) == (
        image, labels, table)
    assert ew.field_path(ew.path_field("d", "empty")) == ""


def test_repeated_labels_prompt_the_user(app, dataset, monkeypatch):
    viewer, explorer, _ = _explorer(monkeypatch)
    image_path, table = dataset
    pd.DataFrame({"label": [5, 3, 5], "area": [1, 2, 3]}).to_csv(table, index=False)
    prompts = []
    monkeypatch.setattr(ew.QMessageBox, "warning", lambda parent, title, text: prompts.append((title, text)))
    explorer.image_field.value = image_path
    explorer.table_field.value = table
    explorer.load()
    _wait(app, lambda: explorer._loader is None)
    assert prompts and prompts[0][0] == "Repeated labels"
    assert prompts[0][1].startswith("Label 5 appears 2 times in the table.")
    assert explorer.session is None and len(viewer.layers) == 0


def test_widget_with_several_images(app, multi_dataset, monkeypatch):
    viewer, explorer, plotter = _explorer(monkeypatch)
    explorer.table_field.value = multi_dataset
    assert not explorer.image_field.enabled and not explorer.labels_field.enabled
    explorer.load()
    _wait(app, lambda: explorer._loader is None)
    s = explorer.session
    assert explorer.images_section.isVisibleTo(explorer)
    assert [explorer.images.item(i).text() for i in range(3)][:2] == ["1. a (2 objects)", "2. b (2 objects)"]
    assert not explorer.images.item(2).flags() & Qt.ItemFlag.ItemIsEnabled  # missing: greyed out
    assert explorer.info.text().startswith("4 objects in 2 images (1 can't be read). Shown: a")
    assert [layer.name for layer in viewer.layers] == ["a: A", "a: B", "labels"]

    # all the objects of the two images are plotted; only those of the shown image keep their label id
    assert s.feature_rows.tolist() == [0, 1, 2, 3]
    assert s.layer.features[ew.IMAGE_INDEX].tolist() == [1, 1, 2, 2]
    assert s.layer.metadata[ew.LABEL_IDS_KEY].tolist() == [0, 3, 5, -3, -4]
    assert len(plotter.plotting_widget.active_artist.data) == 4
    x_axis = plotter.x_axis
    colours = s.layer.colormap.color_dict
    colour_of_row_2 = np.asarray(colours[-3])  # label 3 of image b, not shown
    assert not np.allclose(np.asarray(colours[3]), colour_of_row_2)  # label 3 of image a: its own colour

    # a click on label 3 of image b (row 2): its image is read and shown, then the viewer goes to it
    _click(plotter, 2)
    _wait(app, lambda: s.active == 1 and explorer._opening is None)
    assert [layer.name for layer in viewer.layers] == ["b: A", "b: B", "labels"]  # the labels on top
    assert s.layer.selected_label == 3
    assert viewer.dims.point[-3:] == pytest.approx(s.position(2)[0], abs=2)
    assert s.layer.metadata[ew.LABEL_IDS_KEY].tolist() == [0, -1, -2, 3, 8]
    assert np.allclose(np.asarray(s.layer.colormap.color_dict[3]), colour_of_row_2)  # label 3: now the one of b
    assert plotter.x_axis == x_axis  # the plot keeps its features
    assert explorer.images.item(1).font().bold() and not explorer.images.item(0).font().bold()

    # picking label 8 in napari highlights its point (of image b, not a label of image a)
    s.layer.selected_label = 8
    explorer._on_label_selected()
    assert np.flatnonzero(plotter.plotting_widget.active_artist.highlighted).tolist() == [3]

    # thumbnails of any image, the clusters drawn in the plot kept when images are unticked
    assert len(s.thumbnail(1, 0)) == 2
    features = s.layer.features.copy()
    features["MANUAL_CLUSTER_ID"] = pd.Categorical([0, 1, 2, 2])
    s.layer.features = features
    explorer.images.item(0).setCheckState(Qt.CheckState.Unchecked)
    assert s.feature_rows.tolist() == [2, 3]
    assert len(plotter.plotting_widget.active_artist.data) == 2
    explorer.images.item(0).setCheckState(Qt.CheckState.Checked)
    assert s.layer.features["MANUAL_CLUSTER_ID"].tolist() == [0, 1, 2, 2]

    # the last ticked image can't be unticked
    explorer._tick(lambda i: i == 1)
    explorer.images.item(1).setCheckState(Qt.CheckState.Unchecked)
    assert explorer._ticked == {1}

    # double-click on an image: shown
    explorer.show_image(0)
    _wait(app, lambda: s.active == 0 and explorer._opening is None)
    assert [layer.name for layer in viewer.layers] == ["a: A", "a: B", "labels"]
