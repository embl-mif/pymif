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

from qtpy.QtWidgets import QApplication

import pymif.microscope_manager as mm
from pymif.napari import _explorer_widget as ew

#: Two cubes in a 1 x 8 x 32 x 32 label image: label -> (z, y, x) slices.
OBJECTS = {3: (slice(2, 6), slice(4, 12), slice(4, 12)), 5: (slice(1, 3), slice(20, 28), slice(16, 30))}
SCALES = ((2.0, 0.5, 0.5), (4.0, 1.0, 1.0))


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _metadata():
    return {
        "size": [(1, 2, 8, 32, 32), (1, 2, 4, 16, 16)],
        "chunksize": [(1, 1, 4, 16, 16), (1, 1, 4, 8, 8)],
        "scales": list(SCALES),
        "units": ("micrometer",) * 3,
        "time_increment": 1.0,
        "time_increment_unit": "second",
        "channel_names": ["A", "B"],
        "channel_colors": ["FF0000", "00FF00"],
        "dtype": "uint16",
        "plane_files": None,
        "axes": "tczyx",
        "name": "cells",
    }


def _labels():
    labels = np.zeros((1, 8, 32, 32), np.uint16)
    for label, box in OBJECTS.items():
        labels[(0, *box)] = label
    return labels


@pytest.fixture
def dataset(tmp_path):
    """An OME-Zarr with two channels and the labels in labels/cells, and a table without centroids."""
    image = np.random.default_rng(0).integers(0, 1000, (1, 2, 8, 32, 32), dtype=np.uint16)
    pyramid = [da.from_array(image), da.from_array(image[:, :, ::2, ::2, ::2])]
    path = tmp_path / "cells.ome.zarr"
    mm.ArrayManager(pyramid, _metadata()).to_zarr(str(path), progress=False)
    zarr = mm.ZarrManager(str(path), mode="a")
    zarr.create_empty_group("cells", _metadata(), is_label=True)
    zarr.write_label_region(_labels(), level=0, group="labels/cells")
    table = tmp_path / "features.csv"
    pd.DataFrame({"label": [5, 3, 0], "area": [224, 256, 1], "solidity": [np.inf, 0.9, 1.0],
                  "kind": ["b", "a", "a"]}).to_csv(table, index=False)
    return path, table


def _read(*args, **kwargs):
    """Run ``read_dataset`` to the end: (messages, returned dict)."""
    steps = ew.read_dataset(*args, **kwargs)
    messages = []
    while True:
        try:
            messages.append(next(steps))
        except StopIteration as stop:
            return messages, stop.value


def test_clean_table_sorts_labels_and_fixes_values():
    notes = []
    table = pd.DataFrame({"id": [7, "x", 2, 7, 0], "solidity": [0.5, 1.0, np.inf, 0.2, 1.0],
                          "kind": ["b", "a", "a", "c", "a"]})
    out = ew.clean_table(table, "id", notes)
    assert out["id"].tolist() == [2, 7]  # no text, 0 or duplicate labels
    assert np.isnan(out["solidity"][0]) and out["solidity"][1] == 0.5
    assert out["kind"].tolist() == [1, 2]
    assert notes[0] == "inf -> NaN: solidity (1 rows)"
    with pytest.raises(ValueError, match="no column"):
        ew.clean_table(table, "label")


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


def test_centroids_from_labels_with_a_single_time_point():
    labels = ew.Pyramid([da.from_array(_labels())], {"axes": "tzyx", "scales": [SCALES[0]]})
    centre, radius = ew.centroids_from_labels(labels, np.array([5, 3, 9]))
    assert centre[1] == pytest.approx([3.5 * 2, 7.5 * 0.5, 7.5 * 0.5])
    assert centre[0] == pytest.approx([1.5 * 2, 23.5 * 0.5, 22.5 * 0.5])
    assert np.isnan(centre[2]).all() and np.isnan(radius[2])

    two_times = ew.Pyramid([da.from_array(np.concatenate([_labels()] * 2))], {"axes": "tzyx", "scales": [SCALES[0]]})
    with pytest.raises(ValueError, match="time-lapse"):
        ew.centroids_from_labels(two_times, np.array([3]))


def test_thumbnail_outlines_the_object():
    labels = ew.Pyramid([da.from_array(_labels()[0])], {"axes": "zyx", "scales": [SCALES[0]]})
    box, start, size, offset = labels.box({"z": 7.0, "y": 3.75, "x": 3.75}, 4.0)
    assert box.shape == (5, 17, 17) and (box == 3).any()
    views = ew.thumbnail_views(box.astype(np.float32), box, 3)
    assert len(views) == 2 and views[0].dtype == np.uint8
    assert (views[0] == ew.OBJECT_RGB).all(axis=-1).any()


def test_read_dataset(dataset):
    image_path, table = dataset
    messages, data = _read(str(image_path), "", str(table), "label")
    assert "Computing the centroids from the labels ..." in messages
    assert data["labels_name"] == "cells" and data["table"]["label"].tolist() == [3, 5]
    assert data["centroid_um"][0] == pytest.approx([3.5 * 2, 7.5 * 0.5, 7.5 * 0.5])
    assert np.isnan(data["table"]["solidity"][1])
    assert len(data["limits"]) == 2
    assert "labels: 'cells' of the intensity OME-Zarr" in data["notes"]


def test_widget_loads_and_goes_to_an_object(app, dataset, monkeypatch):
    pytest.importorskip("napari_clusters_plotter")
    from napari.components import ViewerModel

    viewer = ViewerModel()
    monkeypatch.setattr(ew, "current_viewer", lambda: viewer)
    dock = ew.explorer_widget()
    explorer, plotter = dock.explorer, dock.plotter
    image_path, table = dataset
    explorer.table_field.edit.setText(str(table))
    assert explorer.label_column.currentText() == "label"

    explorer.image_field.edit.setText(str(image_path))
    explorer.load()  # in a background thread, the Load button locked meanwhile
    assert not explorer.load_holder.isEnabled()
    for _ in range(500):
        if explorer._loader is None:
            break
        app.processEvents()
        time.sleep(0.01)
    assert explorer.load_holder.isEnabled()
    assert explorer.info.text() == "2 objects, centroids computed from the labels"
    assert [layer.name for layer in viewer.layers] == ["cells: A", "cells: B", "cells"]
    assert explorer.session.layer.metadata[ew.LABEL_IDS_KEY].tolist() == [0, 3, 5]
    assert (plotter.x_axis, plotter.y_axis) == ("area", "solidity")
    assert [explorer.channel.itemText(i) for i in range(explorer.channel.count())] == ["A", "B"]

    # a click on the point of label 5 (row 1) moves the viewer to it
    highlighted = np.zeros(2, bool)
    highlighted[1] = True
    plotter.plotting_widget.active_artist.highlighted = highlighted
    assert explorer.session.layer.selected_label == 5
    assert viewer.dims.point[-3:] == pytest.approx(explorer.session.centroid_um[1], abs=2)

    views = explorer.session.thumbnail(3, channel=1)
    assert len(views) == 2

    # loading again replaces the layers
    explorer._show(_read(str(image_path), "", str(table), "label")[1])
    assert len(viewer.layers) == 3
