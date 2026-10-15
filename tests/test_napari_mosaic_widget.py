from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")
pytest.importorskip("napari")

from qtpy.QtWidgets import QApplication

from pymif.mosaic_manager import MosaicManager, Tile
from pymif.napari import _mosaic_widget as mw


class FakeMosaic(MosaicManager):
    def _scan(self):
        def t(v, x, ch, g):
            return Tile(np.full((8, 16, 16), v, np.uint16), (0, 0, x), (1.0, 1.0, 1.0), ch, g)

        return [t(1, 0, "A", "L"), t(2, 8, "A", "R"), t(3, 0, "B", "L"), t(4, 8, "B", "R")]


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_select_tiles_and_groups():
    m = FakeMosaic(".")
    assert mw.group_labels(m) == ["L", "R"]
    assert len(mw.select_tiles(m, ["A"], ["L", "R"])) == 2
    assert [t.channel for t in mw.select_tiles(m, ["A", "B"], ["R"])] == ["A", "B"]
    assert mw.select_tiles(m, [], ["L"]) == []


def test_default_output_names(tmp_path):
    assert mw.default_output(tmp_path, None, (4, 8, 8)).name == f"{tmp_path.name}_3D_ds4-8-8.ome.zarr"
    assert mw.default_output(tmp_path, "max", (4, 8, 8)).name == f"{tmp_path.name}_MIP_ds8-8.ome.zarr"


def test_fuse_kwargs():
    kw = mw.fuse_kwargs(channels=("A",), projection="none (3D)", blend="count", ds_z=2, ds_y=4,
                        ds_x=4, downsample="stride", levels=2, workers=3, overwrite=True)
    assert kw == {"channels": ["A"], "projection": None, "blend": "count", "ds": (2, 4, 4),
                  "downsample": "stride", "levels": 2, "workers": 3, "overwrite": True}


def test_widget_populates_and_plans(app, tmp_path):
    dock = mw.mosaic_widget()
    assert not dock.fuse_widget.enabled  # nothing loaded yet

    dock.state["mosaic"] = FakeMosaic(".")
    dock.populate(dock.state["mosaic"], tmp_path)
    fw = dock.fuse_widget
    assert list(fw.channels.value) == ["A", "B"]
    assert fw.blend.value == "midline"  # two illumination groups
    assert list(dock.show_widget.channels.value) == ["A"]

    fw.ds_z.value, fw.ds_y.value, fw.ds_x.value = 1, 2, 2
    assert "2 ch x 8 x 8 x 12 px" in dock.plan_label.text()
    assert Path(fw.output_path.value).name.endswith("_3D_ds1-2-2.ome.zarr")

    fw.projection.value = "max"
    assert "x 8 x 12 px (cyx)" in dock.plan_label.text()
    assert Path(fw.output_path.value).name.endswith("_MIP_ds2-2.ome.zarr")

    custom = tmp_path / "mine.ome.zarr"
    fw.output_path.value = custom
    fw.ds_y.value = 4  # a user-chosen path is not replaced
    assert Path(fw.output_path.value) == custom


def test_visualize_places_tiles_with_shared_contrast():
    from napari.components import ViewerModel

    m = FakeMosaic(".")
    v = ViewerModel()
    m.visualize(tiles=m.select(channel="A"), viewer=v)
    assert len(v.layers) == 2
    assert tuple(v.layers[1].translate) == (0, 0, 8)
    assert tuple(v.layers[1].scale) == (1, 1, 1)
    assert v.layers[1].data.shape == (8, 16, 16)
    # one contrast for the channel, linked across its tiles
    assert tuple(v.layers[0].contrast_limits) == tuple(v.layers[1].contrast_limits)
    assert tuple(v.layers[0].contrast_limits_range) == (0, 65535)
    v.layers[0].contrast_limits = (0, 500)
    assert tuple(v.layers[1].contrast_limits) == (0, 500)


def test_contrast_limits_sample_tile_centres():
    from pymif.mosaic_manager.mosaic import _centre_block

    assert _centre_block((2079, 1024, 1024), 64) == (slice(1024, 1088), slice(512, 576), slice(512, 576))
    assert _centre_block((8, 16, 16), 64) == (slice(0, 8), slice(0, 16), slice(0, 16))

    rng = np.random.default_rng(0)

    class Noisy(MosaicManager):
        def _scan(self):
            return [Tile(rng.integers(100, 200, (8, 16, 16)).astype(np.uint16), (0, 0, 16 * i),
                         (1.0, 1.0, 1.0), "A") for i in range(10)]

    low, high = Noisy(".").contrast_limits()["A"]
    assert 100 <= low < 110 and 190 < high <= 199


def test_drop_helpers(tmp_path):
    f = tmp_path / "a.json"
    f.write_text("{}")
    assert mw.folder_of(f) == tmp_path
    assert mw.folder_of(tmp_path) == tmp_path
    assert mw.output_for_drop(tmp_path, tmp_path / "x" / "acq_3D_ds4-8-8.ome.zarr") == tmp_path / "acq_3D_ds4-8-8.ome.zarr"
    assert mw.output_for_drop(tmp_path / "y.ome.zarr", Path()) == tmp_path / "y.ome.zarr"
    assert mw.output_for_drop(tmp_path, Path()) == tmp_path / "fused.ome.zarr"


def _drop(file_edit, path):
    from qtpy.QtCore import QMimeData, QPointF, Qt, QUrl
    from qtpy.QtGui import QDropEvent
    from qtpy.QtWidgets import QLineEdit

    data = QMimeData()
    data.setUrls([QUrl.fromLocalFile(str(path))])
    event = QDropEvent(QPointF(2, 2), Qt.DropAction.CopyAction, data,
                       Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
    file_edit.native.findChild(QLineEdit).dropEvent(event)
    assert event.isAccepted()


def test_drag_and_drop_on_folder_fields(app, tmp_path):
    dock = mw.mosaic_widget()
    tile_json = tmp_path / "raw" / "stack_1" / "Cam_bottom_00000.json"
    tile_json.parent.mkdir(parents=True)
    tile_json.write_text("{}")

    _drop(dock.load_widget.input_path, tmp_path)
    assert Path(dock.load_widget.input_path.value) == tmp_path
    _drop(dock.load_widget.input_path, tile_json)  # a file stands for its folder
    assert Path(dock.load_widget.input_path.value) == tile_json.parent

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _drop(dock.fuse_widget.output_path, out_dir)
    assert Path(dock.fuse_widget.output_path.value) == out_dir / "fused.ome.zarr"


def test_contrast_uses_central_tiles_of_each_group():
    from pymif.mosaic_manager.mosaic import _central_tiles

    tiles = [
        Tile(np.zeros((2, 4, 4), np.uint16), (0, 4 * iy, 4 * ix + (100 if g == "R" else 0)), (1.0, 1.0, 1.0), "A", g)
        for g in ("L", "R") for iy in range(3) for ix in range(3)
    ]
    picks = _central_tiles(tiles, 2)
    assert [(t.group, t.origin[1:]) for t in picks] == [("L", (4.0, 4.0)), ("R", (4.0, 104.0))]
    assert len(_central_tiles(tiles, 6)) == 6
