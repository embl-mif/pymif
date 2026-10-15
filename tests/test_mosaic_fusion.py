from __future__ import annotations

import numpy as np
import pytest

from pymif.mosaic_manager import MosaicManager, Tile, fuse_tiles


def _tile(value, origin, shape=(4, 8, 8), channel="A", group=None, vox=(1.0, 1.0, 1.0)):
    return Tile(np.full(shape, value, np.uint16), origin, vox, channel, group)


def _read(zm, level=0):
    return zm.data[level].compute()


def test_max_blend_places_and_merges(tmp_path):
    # Two 8-wide tiles overlapping in x on columns 4..8; the second is brighter.
    tiles = [_tile(10, (0, 0, 0)), _tile(30, (0, 0, 4))]
    zm = fuse_tiles(tiles, tmp_path / "o.zarr", chunks=(1, 4, 8, 8))
    img = _read(zm)  # (c, z, y, x) -> singleton dims may be dropped by the reader
    img = img.reshape(4, 8, 12)
    assert img[..., :4].max() == 10
    assert (img[..., 4:8] == 30).all()
    assert (img[..., 8:] == 30).all()


def test_sum_blend_shows_overlap(tmp_path):
    tiles = [_tile(10, (0, 0, 0)), _tile(30, (0, 0, 4))]
    zm = fuse_tiles(tiles, tmp_path / "o.zarr", blend="sum", chunks=(1, 4, 8, 8))
    img = _read(zm).reshape(4, 8, 12)
    assert (img[..., :4] == 10).all()
    assert (img[..., 4:8] == 40).all()
    assert (img[..., 8:] == 30).all()
    assert zm.metadata["dtype"] == "float32"


def test_count_blend_marks_tile_extents(tmp_path):
    tiles = [_tile(10, (0, 0, 0)), _tile(30, (0, 0, 4))]
    zm = fuse_tiles(tiles, tmp_path / "o.zarr", blend="count", chunks=(1, 4, 8, 8))
    img = _read(zm).reshape(4, 8, 12)
    assert (img[..., :4] == 1).all()
    assert (img[..., 4:8] == 2).all()
    assert (img[..., 8:] == 1).all()


def test_midline_cuts_between_groups(tmp_path):
    # Left covers x 0..8, right covers x 4..12: overlap 4..8, cut at x=6.
    tiles = [_tile(10, (0, 0, 0), group="L"), _tile(200, (0, 0, 4), group="R")]
    zm = fuse_tiles(tiles, tmp_path / "o.zarr", blend="midline", chunks=(1, 4, 8, 8))
    img = _read(zm).reshape(4, 8, 12)
    assert (img[..., :6] == 10).all()   # left side owns up to the cut ...
    assert (img[..., 6:] == 200).all()  # ... right side owns the rest


def test_projection_max_is_2d(tmp_path):
    t = Tile(np.zeros((4, 8, 8), np.uint16), (0, 0, 0), (2.0, 1.0, 1.0), "A")
    t.data[2, 3, 3] = 99
    zm = fuse_tiles([t], tmp_path / "o.zarr", projection="max", chunks=(1, 8, 8))
    assert zm.metadata["axes"].endswith("yx") and "z" not in zm.metadata["axes"]
    img = _read(zm).reshape(8, 8)
    assert img[3, 3] == 99 and img.sum() == 99


def test_downsample_mean_and_stride(tmp_path):
    data = np.arange(4 * 8 * 8, dtype=np.uint16).reshape(4, 8, 8)
    t = Tile(data, (0, 0, 0), (1.0, 1.0, 1.0), "A")
    zm = fuse_tiles([t], tmp_path / "m.zarr", ds=(2, 2, 2), chunks=(1, 2, 4, 4))
    expected = data.reshape(2, 2, 4, 2, 4, 2).mean(axis=(1, 3, 5))
    assert np.allclose(_read(zm).reshape(2, 4, 4), np.rint(expected))
    assert zm.metadata["scales"][0] == (2.0, 2.0, 2.0)

    zs = fuse_tiles([t], tmp_path / "s.zarr", ds=(2, 2, 2), downsample="stride", chunks=(1, 2, 4, 4))
    assert np.array_equal(_read(zs).reshape(2, 4, 4), data[::2, ::2, ::2])


def test_channels_and_levels(tmp_path):
    tiles = [_tile(1, (0, 0, 0), channel="A"), _tile(2, (0, 0, 0), channel="B")]
    zm = fuse_tiles(tiles, tmp_path / "o.zarr", levels=2, chunks=(1, 4, 8, 8))
    assert zm.metadata["channel_names"] == ["A", "B"]
    assert len(zm.data) == 2
    assert zm.data[1].shape[-1] == 4
    assert (_read(zm)[1] == 2).all()


def test_parallel_matches_serial(tmp_path):
    rng = np.random.default_rng(0)
    tiles = [
        Tile(rng.integers(0, 1000, (4, 16, 16), dtype=np.uint16), (0, 0, 8 * i), (1.0, 1.0, 1.0), "A")
        for i in range(3)
    ]
    a = fuse_tiles(tiles, tmp_path / "a.zarr", chunks=(1, 4, 8, 8), workers=1)
    b = fuse_tiles(tiles, tmp_path / "b.zarr", chunks=(1, 4, 8, 8), workers=4)
    assert np.array_equal(_read(a), _read(b))


def test_validation(tmp_path):
    t = _tile(1, (0, 0, 0))
    with pytest.raises(ValueError):
        fuse_tiles([t], tmp_path / "x.zarr", blend="bogus")
    fuse_tiles([t], tmp_path / "y.zarr")
    with pytest.raises(FileExistsError):
        fuse_tiles([t], tmp_path / "y.zarr")
    fuse_tiles([t], tmp_path / "y.zarr", overwrite=True)


def test_mosaic_base_inspection():
    class M(MosaicManager):
        def _scan(self):
            return [_tile(1, (0, 0, 0), channel="A", group="L"), _tile(2, (0, 0, 20), channel="B", group="R")]

    m = M(".")
    assert m.channels == ["A", "B"] and m.groups == ["L", "R"]
    assert len(m.select(channel="B")) == 1
    lo, hi = m.bounding_box()
    assert lo == (0, 0, 0) and hi == (4, 8, 28)
    assert m.tile(0).data[0].shape == (4, 8, 8)


def test_h5_tiles_with_worker_processes(tmp_path):
    import h5py

    from pymif.mosaic_manager import H5Array

    rng = np.random.default_rng(1)
    tiles = []
    for i in range(3):
        f = tmp_path / f"t{i}.h5"
        with h5py.File(f, "w") as h:
            h["Data"] = rng.integers(0, 1000, (8, 16, 16), dtype=np.uint16)
        tiles.append(Tile(H5Array(f), (0, 0, 8 * i), (1.0, 1.0, 1.0), "A", "L" if i < 2 else "R"))

    kw = dict(ds=(2, 2, 2), blend="midline", chunks=(1, 4, 8, 8))
    serial = fuse_tiles(tiles, tmp_path / "s.zarr", workers=1, **kw)
    procs = fuse_tiles(tiles, tmp_path / "p.zarr", workers=2, executor="process", **kw)
    assert np.array_equal(_read(serial), _read(procs))

    mip = fuse_tiles(tiles, tmp_path / "m.zarr", projection="max", workers=2, ds=(1, 1, 1), chunks=(1, 8, 8))
    expected = np.zeros((16, 32), np.uint16)
    for i, t in enumerate(tiles):
        np.maximum(expected[:, 8 * i : 8 * i + 16], t.data[:].max(axis=0), out=expected[:, 8 * i : 8 * i + 16])
    assert np.array_equal(_read(mip).reshape(16, 32), expected)


def test_crop_on_output_grid(tmp_path):
    data = np.arange(4 * 8 * 8, dtype=np.uint16).reshape(4, 8, 8)
    tiles = [Tile(data, (0, 0, 0), (1.0, 1.0, 1.0), "A"), Tile(data + 1000, (0, 0, 8), (1.0, 1.0, 1.0), "A")]
    zm = fuse_tiles(tiles, tmp_path / "c.zarr", crop={"z": slice(1, 3), "y": slice(2, 6), "x": slice(6, 10)},
                    chunks=(1, 2, 4, 4), progress=False)
    img = _read(zm)
    assert img.shape == (2, 4, 4)
    assert np.array_equal(img[..., :2], data[1:3, 2:6, 6:8])
    assert np.array_equal(img[..., 2:], data[1:3, 2:6, 0:2] + 1000)
    assert zm.metadata["scales"][0] == (1.0, 1.0, 1.0)

    # z crop of a projection only projects those planes
    mip = fuse_tiles(tiles[:1], tmp_path / "m.zarr", projection="max", crop={"z": slice(0, 2)}, progress=False)
    assert np.array_equal(_read(mip), data[:2].max(axis=0))

    with pytest.raises(ValueError):
        fuse_tiles(tiles, tmp_path / "e.zarr", crop={"x": slice(0, 10, 2)})


def test_write_options_are_forwarded(tmp_path):
    import json

    rng = np.random.default_rng(2)
    tiles = [Tile(rng.integers(0, 100, (8, 32, 32), dtype=np.uint16), (0, 0, 24 * i), (1.0, 1.0, 1.0), ch)
             for ch in ("A", "B") for i in range(2)]
    zm = fuse_tiles(
        tiles, tmp_path / "o.ome.zarr",
        channel_names=["dapi", "gfp"], channel_colors=["blue", "00FF00"],
        levels=3, downscale_factor=(1, 2, 2), chunks=(1, 4, 16, 16),
        shards=(1, 8, 32, 32), zarr_format=3, compressor="blosc", progress=False,
    )
    md = zm.metadata
    assert md["channel_names"] == ["dapi", "gfp"]
    assert md["channel_colors"] == ["0000FF", "00FF00"]
    assert [s[-3:] for s in md["size"]] == [(8, 32, 56), (8, 16, 28), (8, 8, 14)]
    meta = json.loads((tmp_path / "o.ome.zarr" / "0" / "zarr.json").read_text())
    assert meta["chunk_grid"]["configuration"]["chunk_shape"] == [1, 8, 32, 32]  # shard
    assert any(c["name"] == "sharding_indexed" for c in meta["codecs"])

    v2 = fuse_tiles(tiles, tmp_path / "v2.zarr", zarr_format=2, progress=False)
    assert (tmp_path / "v2.zarr" / ".zattrs").exists() and v2.metadata["channel_names"] == ["A", "B"]

    with pytest.raises(TypeError):
        fuse_tiles(tiles, tmp_path / "x.zarr", not_an_option=1)
    with pytest.raises(ValueError):
        fuse_tiles(tiles, tmp_path / "x.zarr", channel_names=["only one"])
    assert not any(p.name.endswith(".level0.zarr") for p in tmp_path.iterdir())  # temp store removed


def test_skipped_tiles_are_left_blank(tmp_path):
    # three 8-px tiles at x = 0, 6, 12 (2-px overlaps); skipping the last one must not shrink the output
    tiles = [_tile(10 * (i + 1), (0, 0, 6 * i)) for i in range(3)]
    full = _read(fuse_tiles(tiles, tmp_path / "full.zarr", progress=False))
    zm = fuse_tiles(tiles, tmp_path / "skip.zarr", skip=[tiles[2]], progress=False)
    img = _read(zm)
    assert img.shape == full.shape == (4, 8, 20)
    assert (img[..., 14:] == 0).all()             # only the skipped tile covered this
    assert (img[..., 12:14] == 20).all()          # overlap now only from tile 1
    assert np.array_equal(img[..., :12], full[..., :12])

    count = _read(fuse_tiles(tiles, tmp_path / "count.zarr", blend="count", skip=[tiles[1]], progress=False))
    assert (count[..., :8] == 1).all() and (count[..., 8:12] == 0).all() and (count[..., 12:] == 1).all()

    from pymif.mosaic_manager import plan_fusion
    assert "1 tile(s) left blank" in plan_fusion(tiles, skip=[tiles[0]]).summary()
    with pytest.raises(ValueError):
        plan_fusion(tiles, skip=tiles)


def test_tile_ids_and_skip_by_id(tmp_path):
    class M(MosaicManager):
        def _scan(self):
            out = []
            for ch in ("A", "B"):
                for i in range(2):
                    t = _tile(5, (0, 0, 8 * i), channel=ch)
                    t.attrs["stack"] = f"1-x0{i}-y00"
                    out.append(t)
            return out

    m = M(".")
    assert m.tile_id(m.tiles[0]) == "A/1-x00-y00"
    assert m.tiles_by_ids(["B/1-x01-y00"]) == [m.tiles[3]]
    assert m.tiles_by_ids(["1-x01-y00"]) == [m.tiles[1], m.tiles[3]]  # every channel
    with pytest.raises(KeyError):
        m.tiles_by_ids(["nope"])
    img = _read(m.fuse(tmp_path / "o.zarr", skip=["1-x01-y00"], progress=False))
    assert img.shape == (2, 4, 8, 16) and (img[..., 8:] == 0).all()
