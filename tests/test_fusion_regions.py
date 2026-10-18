"""Region planning and the parallel / overlap-only fusion paths."""
from __future__ import annotations

import numpy as np
import pytest
import zarr

from pymif.mosaic_manager import Tile, fuse_tiles
from pymif.mosaic_manager import fusion


def _noise_tiles(tmp_path, shape=(20, 40, 40), step=(32, 32), groups=True, zarr_chunks=None):
    """2x2 overlapping tiles of random data, as numpy or plane-chunked zarr arrays."""
    rng = np.random.default_rng(1)
    tiles = []
    for iy in range(2):
        for ix in range(2):
            data = rng.integers(0, 4000, size=shape, dtype=np.uint16)
            if zarr_chunks is not None:
                z = zarr.create_array(str(tmp_path / f"src{iy}{ix}.zarr"), shape=shape,
                                      chunks=zarr_chunks, dtype="u2")
                z[:] = data
                data = zarr.open_array(str(tmp_path / f"src{iy}{ix}.zarr"), mode="r")
            group = ("L" if ix == 0 else "R") if groups else None
            tiles.append(Tile(data, (0, iy * step[0], ix * step[1]), (1, 1, 1), "A", group))
    return tiles


def _fused(tmp_path, name, tiles, **kw):
    zm = fuse_tiles(tiles, tmp_path / f"{name}.zarr", progress=False, **kw)
    return zm.data[0].compute()


@pytest.mark.parametrize("blend,axis", [("sigmoid", "x"), ("sigmoid", "z"), ("max", "x"),
                                        ("sum", "x"), ("midline", "z")])
def test_z_slabs_match_whole_depth(tmp_path, monkeypatch, blend, axis):
    tiles = _noise_tiles(tmp_path)
    if axis == "z":  # groups stacked along z instead of x
        tiles = [Tile(t.data, (t.origin[2] / 2, t.origin[1], 0.0), (1, 1, 1), "A", t.group) for t in tiles]
    kw = dict(blend=blend, midline_axis=axis, chunks=(1, 8, 16, 16))
    monkeypatch.setattr(fusion, "_REGION_3D", (16, 16))
    monkeypatch.setattr(fusion, "_REGION_VOXELS", 10**9)  # one slab: whole depth
    whole = _fused(tmp_path, "whole", tiles, **kw)
    monkeypatch.setattr(fusion, "_REGION_VOXELS", 16 * 16 * 3)  # slabs of 3 planes
    slabs = _fused(tmp_path, "slabs", tiles, **kw)
    np.testing.assert_array_equal(whole, slabs)


def test_sigmoid_overlap_only_matches_full_update():
    """The overlap-only path gives exactly what the plain weighted update gives."""
    rng = np.random.default_rng(3)
    shape = (6, 30, 30)
    boxes = [(slice(0, 6), slice(0, 20), slice(0, 20)), (slice(0, 6), slice(0, 20), slice(10, 30)),
             (slice(0, 6), slice(12, 30), slice(0, 20)), (slice(1, 5), slice(8, 25), slice(5, 28))]
    fast, ref = fusion._new_softmax_state(shape), fusion._new_softmax_state(shape)
    previous = []
    for sl in boxes:
        sub = tuple(s.stop - s.start for s in sl)
        blk = rng.random(sub, dtype=np.float32) * 1000
        lw = (rng.random(sub, dtype=np.float32) * 10 - 5)
        fusion._sigmoid_add(fast, sl, blk, lw, previous)
        old_style_add(ref, sl, blk, lw)
    for a, b in zip(fast, ref):
        np.testing.assert_array_equal(a, b)


def old_style_add(state, sel, values, log_weight):
    """Reference: the weighted update as written before the optimisation."""
    acc, wsum, log_max = state
    old = log_max[sel]
    new = np.maximum(old, log_weight)
    rescale = np.exp(old - new)
    w = np.exp(log_weight - new)
    acc[sel] = acc[sel] * rescale + w * values
    wsum[sel] = wsum[sel] * rescale + w
    log_max[sel] = new


def test_region_shape_follows_source_chunks(monkeypatch):
    monkeypatch.setattr(fusion, "_REGION_VOXELS", 2**23)
    plane = Tile(zarr.zeros((200, 1024, 1024), chunks=(1, 1024, 1024), dtype="u2"), (0, 0, 0), (1, 1, 1), "A")
    # plane chunks: regions cover 2 chunks in y and x (capped), z cut to fit the budget
    assert fusion._region_shape([plane], (200, 4000, 4000), (1, 1, 1), False) == (2, 2048, 2048)
    # downsampling shrinks the chunks on the output grid
    assert fusion._region_shape([plane], (100, 2000, 2000), (2, 4, 4), False) == (32, 512, 512)
    blocky = Tile(zarr.zeros((200, 512, 512), chunks=(64, 128, 128), dtype="u2"), (0, 0, 0), (1, 1, 1), "A")
    rz, ry, rx = fusion._region_shape([blocky], (200, 4000, 4000), (1, 1, 1), False)
    assert (ry, rx) == (256, 256) and rz % 64 == 0  # whole source chunks along z
    unknown = Tile(np.zeros((10, 64, 64), np.uint16), (0, 0, 0), (1, 1, 1), "A")
    assert fusion._region_shape([unknown], (10, 600, 600), (1, 1, 1), False)[1:] == (256, 256)
    assert fusion._region_shape([plane], (200, 4000, 4000), (1, 1, 1), True) == (200, 2048, 2048)


@pytest.mark.parametrize("executor", ["thread", "process"])
def test_parallel_regions_match_serial(tmp_path, monkeypatch, executor):
    tiles = _noise_tiles(tmp_path, zarr_chunks=(1, 40, 40))  # zarr arrays pickle to processes
    monkeypatch.setattr(fusion, "_REGION_3D", (16, 16))
    monkeypatch.setattr(fusion, "_REGION_MAX", 16)
    monkeypatch.setattr(fusion, "_REGION_VOXELS", 16 * 16 * 4)
    kw = dict(chunks=(1, 8, 16, 16))
    serial = _fused(tmp_path, "serial", tiles, workers=1, **kw)
    parallel = _fused(tmp_path, "par", tiles, workers=3, executor=executor, **kw)
    np.testing.assert_array_equal(serial, parallel)


def test_uncompressed_temp_gives_same_result(tmp_path):
    tiles = _noise_tiles(tmp_path)
    a = _fused(tmp_path, "a", tiles, chunks=(1, 8, 16, 16), temp_compression=True)
    b = _fused(tmp_path, "b", tiles, chunks=(1, 8, 16, 16), temp_compression=False)
    np.testing.assert_array_equal(a, b)
