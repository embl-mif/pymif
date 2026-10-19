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
    kw = mw.fuse_kwargs(channels=("A",), projection="none (3D)", blend="sum", ds_z=2, ds_y=4,
                        ds_x=4, levels=2, workers=3, overwrite=True)
    assert kw == {"channels": ["A"], "projection": None, "blend": "sum", "ds": (2, 4, 4),
                  "levels": 2, "workers": 3, "overwrite": True}


def test_widget_populates_and_plans(app, tmp_path):
    dock = mw.mosaic_widget()
    assert not dock.fuse_widget.enabled  # nothing loaded yet

    dock.state["mosaic"] = FakeMosaic(".")
    dock.populate(dock.state["mosaic"], tmp_path)
    fw = dock.fuse_widget
    assert list(fw.channels.value) == ["A", "B"]
    assert fw.blend.value == "sigmoid"
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


def test_skipped_tile_ids_from_layers():
    from napari.components import ViewerModel

    m = FakeMosaic(".")
    for i, t in enumerate(m.tiles):
        t.attrs["stack"] = f"s{i % 2}"
    v = ViewerModel()
    m.visualize(viewer=v, contrast_limits={"A": (0, 10), "B": (0, 10)})
    layers = mw.tile_layers(v.layers, m.path)
    assert len(layers) == 4
    assert mw.skipped_tile_ids(v.layers, v.layers.selection, m.path, None) == []

    layers[1].visible = False                       # A/s1 hidden
    assert mw.skipped_tile_ids(v.layers, v.layers.selection, m.path, "visible") == ["A/s1"]
    assert mw.skipped_tile_ids(v.layers, v.layers.selection, m.path, "visible", all_channels=True) == ["s1"]
    assert m.tiles_by_ids(["s1"]) == [m.tiles[1], m.tiles[3]]

    v.layers.selection = {layers[0], layers[2]}     # only A/s0 and B/s0 selected
    assert mw.skipped_tile_ids(v.layers, v.layers.selection, m.path, "selected") == ["A/s1", "B/s1"]
    # layers of another mosaic are ignored
    assert mw.skipped_tile_ids(v.layers, v.layers.selection, "elsewhere", "visible") == []


def test_batch_csv_export_runs_with_the_cli(app, tmp_path, monkeypatch):
    import sys

    import pandas as pd

    import pymif.microscope_manager as mm
    from pymif.cli import pymif as cli
    from pymif.mosaic_manager import LuxendoMosaicManager

    from .test_luxendo_mosaic import _write_tile

    acq = tmp_path / "acq"  # one dye, left and right side
    for stack, side, tx in (("1-x00-y00", "L", 0.0), ("2-x00-y00", "R", 20.0)):
        _write_tile(acq / "raw" / f"stack_{stack}_GFP-{side}", "Cam_bottom_00000", f"GFP-{side}", stack,
                    tx, 0.0, 30.0, shape=(8, 16, 16), value=5)

    dock = mw.mosaic_widget()
    mosaic = LuxendoMosaicManager(acq, progress=False)
    dock.state["mosaic"] = mosaic
    dock.populate(mosaic, acq)
    fw = dock.fuse_widget
    fw.projection.value = "max"
    fw.ds_z.value, fw.ds_y.value, fw.ds_x.value = 1, 2, 2
    out = tmp_path / "fused.ome.zarr"
    fw.output_path.value = out
    csv = tmp_path / "batch.csv"
    dock.batch_widget.batch_csv_path.value = csv
    dock.batch_widget()
    dock.batch_widget()                              # appends a second row

    table = pd.read_csv(csv, dtype=str).fillna("")
    assert len(table) == 2
    row = table.iloc[0]
    assert row["microscope"] == "luxendo" and row["projection"] == "max"
    assert row["downsample"] == "1 2 2" and row["blend"] == "sigmoid" and row["groups"] == ""

    pd.read_csv(csv).iloc[:1].to_csv(csv, index=False)
    monkeypatch.setattr(sys, "argv", ["pymif", "fuse", "--batch", str(csv)])
    cli.main()
    assert mm.ZarrManager(str(out)).metadata["axes"] == "yx"


def test_batch_row_groups_and_skips():
    values = {"channels": ["A", "B"], "groups": ["L"], "projection": "none (3D)", "blend": "max",
              "ds_z": 4, "ds_y": 8, "ds_x": 8, "levels": 3, "workers": 4,
              "overwrite": True}
    row = mw.batch_row(input_path="acq", microscope="luxendo", output="out.zarr", values=values,
                       all_groups=["L", "R"], skip_ids=["A/s1", "s2"])
    assert row["groups"] == "L" and row["skip_tiles"] == "A/s1 s2" and row["projection"] == "none"
    assert row["channels"] == "A B" and row["overwrite"] == "true" and row["downsample"] == "4 8 8"


def test_widget_leaves_hidden_tiles_blank(app, monkeypatch):
    from napari.components import ViewerModel

    viewer = ViewerModel()  # same layer list and selection as a viewer, without OpenGL
    monkeypatch.setattr(mw, "current_viewer", lambda: viewer)
    dock = mw.mosaic_widget()
    m = FakeMosaic(".")
    for t in m.tiles:  # stack names, as real readers give, so positions match across channels
        t.attrs["stack"] = f"pos-{t.group}"
    dock.state["mosaic"] = m
    dock.populate(m, Path("."))
    m.visualize(viewer=viewer, contrast_limits={"A": (0, 10), "B": (0, 10)})
    fw = dock.fuse_widget
    fw.ds_z.value, fw.ds_y.value, fw.ds_x.value = 1, 1, 1
    assert "left blank" not in dock.plan_label.text()

    fw.tiles_to_fuse.value = "visible layers only"
    mw.tile_layers(viewer.layers, m.path)[0].visible = False
    fw.blend.value = "max"  # any change refreshes the estimate
    assert "1 tile(s) left blank" in dock.plan_label.text()
    fw.skip_all_channels.value = True               # same position in channel B too
    assert "2 tile(s) left blank" in dock.plan_label.text()
    assert "x 8 x 16 x 24 px" in dock.plan_label.text()  # output keeps its full size

    fw.skip_all_channels.value = False
    fw.tiles_to_fuse.value = "selected layers only"
    viewer.layers.selection = set(mw.tile_layers(viewer.layers, m.path)[:3])  # 3 of the 4 tiles
    assert "1 tile(s) left blank" in dock.plan_label.text()


def test_apply_display_windows():
    from napari.components import ViewerModel

    v = ViewerModel()
    v.add_image(np.zeros((4, 4), np.uint16))
    v.add_image(np.zeros((4, 4), np.uint16))
    mw.apply_display_windows(v.layers, [(100.0, 1200.0), (5.0, 50.0)], "uint16")
    assert tuple(v.layers[0].contrast_limits) == (100.0, 1200.0)
    assert tuple(v.layers[1].contrast_limits) == (5.0, 50.0)
    assert tuple(v.layers[0].contrast_limits_range) == (0, 65535)


def _stacked_mosaic():
    """FakeMosaic with stack names: s0 at x 0..16, s1 at x 8..24, in channels A and B."""
    m = FakeMosaic(".")
    for i, t in enumerate(m.tiles):
        t.attrs["stack"] = f"s{i % 2}"
    return m


def test_tile_map_picks_stack_positions():
    from napari.components import ViewerModel

    m = _stacked_mosaic()
    fp = mw.tile_footprints(m)
    assert list(fp) == ["s0", "s1"]  # one rectangle per position, shared by both channels
    assert fp["s1"].tolist() == [[0, 8], [16, 24]]

    v = ViewerModel()
    m.visualize(viewer=v, contrast_limits={"A": (0, 10), "B": (0, 10)})
    layer = mw.add_tile_map(v, m)
    assert v.layers.extent.units is not None  # same units as the tile layers (no napari warning)
    assert mw.find_tile_map(v.layers, m.path) is layer and mw.find_tile_map(v.layers, "elsewhere") is None
    assert mw.picked_stack_ids(layer) == ["s0", "s1"]
    assert mw.picked_stack_ids(None) is None

    assert mw.toggle_tile_at(layer, (0, 4, 20)) == "s1"      # 3D cursor position, 2D map
    assert mw.picked_stack_ids(layer) == ["s0"]
    assert mw.picked_tiles(m, m.tiles, ["s0"]) == [m.tiles[0], m.tiles[2]]
    assert mw.picked_tiles(m, m.tiles, None) == m.tiles
    assert not np.allclose(layer.face_color[0], layer.face_color[1])  # unpicked looks different
    assert mw.toggle_tile_at(layer, (0, 100, 100)) is None   # no tile there
    assert mw.toggle_tile_at(layer, (0, 4, 20)) == "s1"
    assert mw.picked_stack_ids(layer) == ["s0", "s1"]

    # the map cannot be edited, also after switching to 3D and back
    assert not layer.editable
    v.dims.ndisplay = 3
    v.dims.ndisplay = 2
    assert not layer.editable

    # loading again replaces the map
    assert mw.add_tile_map(v, m) is not layer and sum(lay.name.startswith(mw.TILE_MAP) for lay in v.layers) == 1


class _Click:
    """Minimal napari mouse event for the viewer drag callbacks."""

    def __init__(self, position, pos=(10, 10), modifiers=()):
        self.position, self.pos, self.button, self.type = position, np.array(pos), 1, "mouse_press"
        self.modifiers = list(modifiers)


def _press(viewer, event, moves=(), to=None, before_release=None):
    """Press, move through canvas positions ``moves`` (ending at world position ``to``), release.

    ``before_release()`` runs after the moves, while the button is still down.
    """
    gens = [cb(viewer, event) for cb in viewer.mouse_drag_callbacks]
    gens = [g for g in gens if g is not None and next(g, StopIteration) is not StopIteration]
    for pos in moves:
        event.type, event.pos = "mouse_move", np.array(pos)
        if to is not None:
            event.position = to
        for g in gens:
            next(g, None)
    if before_release is not None:
        before_release()
    event.type = "mouse_release"
    for g in gens:
        next(g, None)


def test_widget_fuses_and_exports_picked_tiles_only(app, monkeypatch):
    from napari.components import ViewerModel

    viewer = ViewerModel()
    monkeypatch.setattr(mw, "current_viewer", lambda: viewer)
    dock = mw.mosaic_widget()
    m = _stacked_mosaic()
    dock.state["mosaic"] = m
    dock.populate(m, Path("."))
    layer = mw.find_tile_map(viewer.layers, m.path)
    assert layer is not None
    fw = dock.fuse_widget
    fw.ds_z.value, fw.ds_y.value, fw.ds_x.value = 1, 1, 1
    assert "x 8 x 16 x 24 px" in dock.plan_label.text()

    _press(viewer, _Click((0, 4, 20)), moves=[(30, 30)])  # a drag pans: nothing picked
    assert mw.picked_stack_ids(layer) == ["s0", "s1"]
    _press(viewer, _Click((0, 4, 20)), moves=[(11, 11)])  # a click with a little jitter
    assert mw.picked_stack_ids(layer) == ["s0"]
    assert "x 8 x 16 x 16 px" in dock.plan_label.text()   # the output fits the picked tiles

    viewer.dims.ndisplay = 3                               # no picking in 3D
    _press(viewer, _Click((0, 4, 20)))
    assert mw.picked_stack_ids(layer) == ["s0"]
    viewer.dims.ndisplay = 2

    values = {name: getattr(fw, name).value for name in
              ("channels", "groups", "projection", "blend", "ds_z", "ds_y", "ds_x", "levels", "workers", "overwrite")}
    row = mw.batch_row(input_path="acq", microscope="luxendo", output="out.zarr", values=values,
                       all_groups=["L", "R"], skip_ids=[], picked=["s0"], all_stacks={"s0", "s1"})
    assert row["tiles"] == "s0"
    row = mw.batch_row(input_path="acq", microscope="luxendo", output="out.zarr", values=values,
                       all_groups=["L", "R"], skip_ids=[], picked=["s0", "s1"], all_stacks={"s0", "s1"})
    assert row["tiles"] == ""                             # every tile: the default


def test_tile_map_box_selection(app, monkeypatch):
    from napari.components import ViewerModel

    class Grid(MosaicManager):  # 3 x 3 tiles of 10 um, overlapping by 2 um
        def _scan(self):
            return [Tile(np.zeros((2, 10, 10), np.uint16), (0, 8 * iy, 8 * ix), (1.0, 1.0, 1.0), "A",
                         attrs={"stack": f"y{iy}x{ix}"}) for iy in range(3) for ix in range(3)]

    viewer = ViewerModel()
    monkeypatch.setattr(mw, "current_viewer", lambda: viewer)
    dock = mw.mosaic_widget()
    m = Grid(".")
    dock.state["mosaic"] = m
    dock.populate(m, Path("."))
    layer = mw.find_tile_map(viewer.layers, m.path)
    fw = dock.fuse_widget
    fw.ds_z.value, fw.ds_y.value, fw.ds_x.value = 1, 1, 1
    # a box inside the first tile also touches the 2 um overlaps of its neighbours (from 8 um)
    assert mw.tiles_in_box(layer, (0, 0, 0), (0, 9, 9)) == [0, 1, 3, 4]
    assert mw.tiles_in_box(layer, (0, 1, 1), (0, 7, 7)) == [0]

    def box_layers():
        return [lay for lay in viewer.layers if lay.metadata.get("mosaic_drag_box")]

    viewer.layers.selection = {viewer.layers[0]}
    seen = {}
    _press(viewer, _Click((0, 0, 0), modifiers=["Control"]), moves=[(30, 30), (60, 60)], to=(0, 9, 9),
           before_release=lambda: seen.update(boxes=[lay.data[0].copy() for lay in box_layers()],
                                              selection=set(viewer.layers.selection),
                                              picked=mw.picked_stack_ids(layer)))
    assert len(seen["boxes"]) == 1                                 # one box, drawn while dragging
    assert seen["boxes"][0].min(axis=0).tolist() == [0, 0] and seen["boxes"][0].max(axis=0).tolist() == [9, 9]
    assert seen["selection"] == {viewer.layers[0]}                 # the layer selection is unchanged
    assert len(seen["picked"]) == 9                                # applied on release only
    assert box_layers() == []                                      # and removed on release
    assert mw.picked_stack_ids(layer) == ["y0x2", "y1x2", "y2x0", "y2x1", "y2x2"]
    assert "x 2 x 26 x 26 px" in dock.plan_label.text()     # still the full extent (corner tiles)
    _press(viewer, _Click((0, 0, 0), modifiers=["Shift"]), moves=[(60, 60)], to=(0, 6, 22))
    assert mw.picked_stack_ids(layer) == ["y0x0", "y0x1", "y0x2", "y1x2", "y2x0", "y2x1", "y2x2"]
    _press(viewer, _Click((0, 15, 15), modifiers=["Shift"]))  # Shift + click picks one tile
    assert "y1x1" in mw.picked_stack_ids(layer) and len(mw.picked_stack_ids(layer)) == 8
    _press(viewer, _Click((0, 15, 15), modifiers=["Shift"]))  # ... and does not toggle it
    assert len(mw.picked_stack_ids(layer)) == 8
    _press(viewer, _Click((0, 15, 15), modifiers=["Control"]))  # Ctrl + click unpicks it
    assert "y1x1" not in mw.picked_stack_ids(layer)
    _press(viewer, _Click((0, 0, 0), modifiers=["Alt"]), moves=[(60, 60)], to=(0, 26, 26))  # Alt: zoom
    assert len(mw.picked_stack_ids(layer)) == 7


def test_show_tiles_follows_picks_and_keeps_contrast(app, monkeypatch):
    from napari.components import ViewerModel

    viewer = ViewerModel()
    monkeypatch.setattr(mw, "current_viewer", lambda: viewer)
    dock = mw.mosaic_widget()
    m = _stacked_mosaic()
    monkeypatch.setattr(m, "contrast_limits", lambda *a, **k: pytest.fail("no contrast estimation"))
    dock.state["mosaic"] = m
    dock.populate(m, Path("."))
    dock.show_widget.channels.value = ["A", "B"]
    dock.show_widget()
    shown = mw.tile_layers(viewer.layers, m.path)
    assert len(shown) == 4 and viewer.layers[-1].name == mw.TILE_MAP  # the map stays on top
    assert tuple(shown[0].contrast_limits) == (0, 65535)              # full range: nothing read
    shown[0].contrast_limits = (0, 500)                               # linked: every A tile

    layer = mw.find_tile_map(viewer.layers, m.path)
    mw.toggle_tile_at(layer, (0, 4, 20))                              # unpick s1
    dock.show_widget()
    shown = mw.tile_layers(viewer.layers, m.path)
    assert [lay.metadata["mosaic_tile_id"] for lay in shown] == ["A/s0", "B/s0"]
    assert tuple(shown[0].contrast_limits) == (0, 500) and tuple(shown[1].contrast_limits) == (0, 65535)
    assert mw.find_tile_map(viewer.layers, m.path) is layer


class TwoFieldMosaic(MosaicManager):
    """One channel, two separate fields: stacks s0/s1 (x 0..24) and s2 (x 100..116)."""

    def _scan(self):
        def t(x, stack):
            return Tile(np.full((8, 16, 16), 7, np.uint16), (0, 0, x), (1.0, 1.0, 1.0), "A", None,
                        attrs={"stack": stack})

        return [t(0, "s0"), t(8, "s1"), t(100, "s2")]


def test_select_tiles_by_field():
    m = TwoFieldMosaic(".")
    assert m.fields == ["field0", "field1"]
    assert [t.attrs["stack"] for t in mw.select_tiles(m, ["A"], [mw.NO_GROUP], ["field1"])] == ["s2"]
    assert len(mw.select_tiles(m, ["A"], [mw.NO_GROUP])) == 3  # no field filter
    assert "field1: 1 stack position(s): s2" in mw.field_summary(m)
    assert mw.field_summary(FakeMosaic(".")) == ""
    assert mw.stack_fields(m) == {"s0": "field0", "s1": "field0", "s2": "field1"}


def test_widget_selects_first_field_and_refuses_several(app, tmp_path):
    dock = mw.mosaic_widget()
    m = TwoFieldMosaic(".")
    dock.state["mosaic"] = m
    dock.populate(m, tmp_path)
    fw = dock.fuse_widget
    assert list(fw.fields.choices) == ["field0", "field1"] and list(fw.fields.value) == ["field0"]
    assert list(dock.show_widget.fields.value) == ["field0"]
    fw.ds_z.value = fw.ds_y.value = fw.ds_x.value = 1
    assert "separate fields" not in dock.plan_label.text()
    assert "x 16 x 24 px" in dock.plan_label.text()  # the output fits field0
    assert Path(fw.output_path.value).name.endswith("_field0.ome.zarr")

    fw.fields.value = ["field1"]
    assert "x 16 x 16 px" in dock.plan_label.text()  # the output fits field1
    assert Path(fw.output_path.value).name.endswith("_field1.ome.zarr")

    # several fields: shown as an error, neither fused nor appended to the batch CSV
    fw.fields.value = ["field0", "field1"]
    assert "2 separate fields (field0, field1), cannot be fused" in dock.plan_label.text()
    fw.output_path.value = tmp_path / "both.ome.zarr"
    fw()
    assert not (tmp_path / "both.ome.zarr").exists()
    dock.batch_widget.batch_csv_path.value = tmp_path / "batch.csv"
    dock.batch_widget()
    assert not (tmp_path / "batch.csv").exists()


def test_batch_row_fields():
    values = {"channels": ["A"], "groups": ["L"], "fields": ["field1"], "projection": "max", "blend": "max",
              "ds_z": 1, "ds_y": 2, "ds_x": 2, "levels": 1, "workers": 1, "overwrite": False}
    row = mw.batch_row(input_path="acq", microscope="luxendo", output="o.zarr", values=values,
                       all_groups=["L"], skip_ids=[], all_fields=["field0", "field1"])
    assert row["fields"] == "field1"
    # an empty cell means the first field, so it is only left empty for a single-field mosaic
    values["fields"] = ["field0"]
    row = mw.batch_row(input_path="acq", microscope="luxendo", output="o.zarr", values=values,
                       all_groups=["L"], skip_ids=[], all_fields=["field0", "field1"])
    assert row["fields"] == "field0"
    row = mw.batch_row(input_path="acq", microscope="luxendo", output="o.zarr", values=values,
                       all_groups=["L"], skip_ids=[], all_fields=["field0"])
    assert row["fields"] == ""
    assert "fields" in mw.MOSAIC_BATCH_COLUMNS


def test_batch_row_flatfield():
    values = {"channels": ["A"], "groups": ["L"], "projection": "max", "blend": "max",
              "ds_z": 1, "ds_y": 2, "ds_x": 2, "levels": 1, "workers": 1, "overwrite": False}
    common = dict(input_path="acq", output="o.zarr", values=values, all_groups=["L"], skip_ids=[])
    assert mw.batch_row(microscope="luxendo", **common)["flatfield"] == ""  # reader without the option
    assert mw.batch_row(microscope="opera", flatfield=False, **common)["flatfield"] == "false"
    assert mw.batch_row(microscope="opera", flatfield=True, **common)["flatfield"] == "true"
    assert "flatfield" in mw.MOSAIC_BATCH_COLUMNS
