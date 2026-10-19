from __future__ import annotations

import numpy as np
import pytest

from pymif.mosaic_manager import (MosaicManager, MultipleFieldsError, Tile, connected_components, fuse_tiles,
                                  plan_fusion)


def _tile(value, origin, channel="A", stack=None, shape=(4, 8, 8)):
    return Tile(np.full(shape, value, np.uint16), origin, (1.0, 1.0, 1.0), channel,
                attrs={} if stack is None else {"stack": stack})


class TwoFields(MosaicManager):
    """Two channels; field0 = stacks s0, s1 (touching, x 0..14), field1 = stack s2 (x 100..108),
    plus stack s3 below s2 (y 8..16), touching it exactly at its border."""

    def _scan(self):
        out = []
        for ch, v in (("A", 10), ("B", 20)):
            out += [
                _tile(v, (0, 0, 100), ch, "s2"),
                _tile(v, (0, 0, 0), ch, "s0"),
                _tile(v, (0, 0, 6), ch, "s1"),
                _tile(v, (0, 8, 100), ch, "s3"),
            ]
        return out


def test_connected_components_numbered_by_position():
    tiles = [_tile(1, (0, 0, 100)), _tile(1, (0, 0, 0)), _tile(1, (0, 0, 6)), _tile(1, (0, 50, 0))]
    # by y then x: (0, 0..14) -> 0, (0, 100) -> 1, (50, 0) -> 2
    assert connected_components(tiles) == [1, 0, 0, 2]
    # a gap of 2 voxels is bridged by a tolerance of 3
    assert connected_components([_tile(1, (0, 0, 0)), _tile(1, (0, 0, 10))]) == [0, 1]
    assert connected_components([_tile(1, (0, 0, 0)), _tile(1, (0, 0, 10))], tolerance=(3, 3, 3)) == [0, 0]


def test_mosaic_labels_fields():
    m = TwoFields(".")
    assert m.fields == ["field0", "field1"]
    assert {m.tile_id(t) for t in m.field_tiles("field0")} == {"A/s0", "A/s1", "B/s0", "B/s1"}
    assert {t.attrs["stack"] for t in m.select(field="field1", channel="A")} == {"s2", "s3"}
    with pytest.raises(KeyError):
        m.field_tiles("field7")


def test_reader_fields_are_kept():
    class Given(TwoFields):
        def _scan(self):
            tiles = super()._scan()
            for t in tiles:
                t.field = "well-" + t.attrs["stack"]
            return tiles

    assert set(Given(".").fields) == {"well-s0", "well-s1", "well-s2", "well-s3"}


def test_fuse_one_field_fits_it(tmp_path):
    m = TwoFields(".")
    zm = m.fuse(tmp_path / "f0.ome.zarr", field="field0", channels=["A"], projection="max")
    img = zm.data[0].compute().squeeze()
    assert img.shape == (8, 14) and (img == 10).all()


def test_fuse_defaults_to_first_field(tmp_path):
    m = TwoFields(".")
    assert m.fusion_tiles() == m.field_tiles("field0")
    zm = m.fuse(tmp_path / "default.ome.zarr", channels=["A"], projection="max")
    img = zm.data[0].compute().squeeze()
    assert img.shape == (8, 14) and (img == 10).all()


def test_fuse_field_by_tile_ids_and_skip(tmp_path):
    m = TwoFields(".")
    # field1 only, skipping s3: s3 keeps its place (the output does not shrink) but stays blank
    zm = m.fuse(tmp_path / "f1.ome.zarr", field="field1", skip=["s3"], channels=["B"], projection="max")
    img = zm.data[0].compute().squeeze()
    assert img.shape == (16, 8)
    assert (img[:8] == 20).all() and (img[8:] == 0).all()
    # the same tiles given as ids
    assert m.fusion_tiles(["s2", "s3"]) == m.field_tiles("field1")
    # tiles of another field are dropped by field=
    assert {t.attrs["stack"] for t in m.fusion_tiles(["s0", "s2"], field="field1")} == {"s2"}
    with pytest.raises(ValueError):
        m.fusion_tiles(["s0"], field="field1")


def test_fusing_several_fields_is_refused(tmp_path):
    m = TwoFields(".")
    plan = plan_fusion(m.tiles, projection="max")
    assert plan.fields == ["field0", "field1"] and "will not be done" in plan.error
    assert "2 separate fields" in plan.summary() and "cannot be fused" in plan.summary()
    assert plan_fusion(m.field_tiles("field1")).error is None
    with pytest.raises(MultipleFieldsError, match="will not be done"):
        m.fuse(tmp_path / "both.ome.zarr", field=["field0", "field1"], channels=["A"], projection="max")
    with pytest.raises(MultipleFieldsError):
        m.fuse(tmp_path / "ids.ome.zarr", tiles=["s0", "s2"], channels=["A"], projection="max")
    with pytest.raises(MultipleFieldsError):
        fuse_tiles(m.tiles, tmp_path / "all.ome.zarr", channels=["A"], projection="max")
    # skipped tiles still define the geometry, so they count
    with pytest.raises(MultipleFieldsError):
        m.fuse(tmp_path / "skip.ome.zarr", tiles=["s0", "s1", "s2", "s3"], skip=["s2", "s3"],
               channels=["A"], projection="max")
    assert not list(tmp_path.iterdir())  # refused before anything is written
