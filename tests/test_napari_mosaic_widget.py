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


def test_visualize_places_tiles():
    from napari.components import ViewerModel

    m = FakeMosaic(".")
    v = ViewerModel()
    m.visualize(tiles=m.select(channel="A"), viewer=v, ds=(1, 2, 2))
    assert len(v.layers) == 2
    assert tuple(v.layers[1].translate) == (0, 0, 8)
    assert tuple(v.layers[1].scale) == (1, 2, 2)
    assert v.layers[1].data.shape == (8, 8, 8)
