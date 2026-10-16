from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest

import pymif.microscope_manager as mm
from pymif.cli import pymif as cli
from pymif.mosaic_manager import fusion

from .test_luxendo_mosaic import _write_tile


@pytest.fixture
def acquisition(tmp_path):
    """Raw Luxendo-like acquisition: 2 x 1 tiles per side, two dyes, L/R sides overlapping in x."""
    raw = tmp_path / "acq" / "raw"
    rng = np.random.default_rng(0)
    for stack, side, tx in (("1-x00-y00", "L", 0.0), ("1-x01-y00", "L", 28.0),
                            ("2-x00-y00", "R", 50.0), ("2-x01-y00", "R", 78.0)):
        for dye in ("GFP", "RFP"):
            _write_tile(raw / f"stack_{stack}_{dye}-{side}", "Cam_bottom_00000", f"{dye}-{side}", stack,
                        tx, 0.0, 30.0, shape=(8, 16, 16), value=int(rng.integers(1, 1000)))
    return tmp_path / "acq"


@pytest.fixture(autouse=True)
def small_regions(monkeypatch):
    """Several fusion regions even for tiny test data."""
    monkeypatch.setattr(fusion, "_REGION_3D", (8, 8))
    monkeypatch.setattr(fusion, "_REGION_2D", (8, 8))


def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["pymif", *argv])
    cli.main()


def test_fuse_3d_with_convert_options(acquisition, tmp_path, monkeypatch):
    out = tmp_path / "out_3D.ome.zarr"
    _run(monkeypatch, "fuse", "-i", str(acquisition), "-z", str(out),
         "-rds", "2", "2", "2", "-b", "midline", "-nl", "2", "-df", "1", "2", "2",
         "-cs", "1", "2", "4", "4", "-sh", "none", "-cn", "gfp", "rfp", "-cc", "lime", "magenta",
         "-rw", "2", "-nw", "2")
    z = mm.ZarrManager(str(out))
    md = z.metadata
    assert md["axes"] == "czyx"
    assert md["channel_names"] == ["gfp", "rfp"]
    assert md["channel_colors"] == ["00FF00", "FF00FF"]
    assert len(z.data) == 2 and z.data[1].shape[1] == z.data[0].shape[1]  # z not downscaled (-df 1 2 2)
    assert md["scales"][0] == (6.0, 4.0, 4.0)  # voxel 3 x 2 x 2 um, read downsample 2
    meta = json.loads((out / "0" / "zarr.json").read_text())
    assert meta["chunk_grid"]["configuration"]["chunk_shape"] == [1, 2, 4, 4]
    assert not any(c["name"] == "sharding_indexed" for c in meta["codecs"])

    # same command again fails without --overwrite, works with it
    with pytest.raises(FileExistsError):
        _run(monkeypatch, "fuse", "-i", str(acquisition), "-z", str(out), "-rds", "2")
    _run(monkeypatch, "fuse", "-i", str(acquisition), "-z", str(out), "--downsample", "2", "--overwrite")


def test_fuse_mip_subset_and_groups(acquisition, tmp_path, monkeypatch):
    out = tmp_path / "mip.ome.zarr"
    _run(monkeypatch, "fuse", "-i", str(acquisition), "-z", str(out), "-m", "luxendo",
         "-p", "max", "-s", "c=1;y=2:6", "-g", "L", "-ms", "1", "-zf", "2")
    z = mm.ZarrManager(str(out))
    assert z.metadata["axes"] == "yx"  # single channel dropped (drop_singleton default)
    assert z.data[0].shape[0] == 4
    # only the L tiles (x from -15 to 45 um, 2 um voxels) are fused
    assert z.data[0].shape[1] == (45 - -15) // 2
    assert (out / ".zattrs").exists()  # zarr v2


def test_fuse_skip_tiles(acquisition, tmp_path, monkeypatch):
    full, skipped = tmp_path / "full.ome.zarr", tmp_path / "skipped.ome.zarr"
    _run(monkeypatch, "fuse", "-i", str(acquisition), "-z", str(full), "-p", "max", "-sh", "none")
    _run(monkeypatch, "fuse", "-i", str(acquisition), "-z", str(skipped), "-p", "max", "-sh", "none",
         "-st", "2-x01-y00", "GFP/1-x00-y00")
    a = mm.ZarrManager(str(full)).data[0].compute()
    b = mm.ZarrManager(str(skipped)).data[0].compute()
    assert a.shape == b.shape                     # the output keeps its size
    # the last R tile (x 63..95 um -> px 39..55) is gone in both channels
    assert (b[:, :, 42:] == 0).all() and (a[:, :, 42:] > 0).all()
    # the first L tile only in GFP (channel 0); px 0..14 is covered by it alone
    assert (b[0, :, :14] == 0).all() and (b[1, :, :14] > 0).all()
    with pytest.raises(KeyError):
        _run(monkeypatch, "fuse", "-i", str(acquisition), "-z", str(tmp_path / "x.zarr"), "-st", "nope")


def test_fuse_sigmoid_steepness(acquisition, tmp_path, monkeypatch):
    soft, sharp = tmp_path / "soft.ome.zarr", tmp_path / "sharp.ome.zarr"
    _run(monkeypatch, "fuse", "-i", str(acquisition), "-z", str(soft), "-p", "max", "-sh", "none")
    _run(monkeypatch, "fuse", "-i", str(acquisition), "-z", str(sharp), "-p", "max", "-sh", "none",
         "-b", "sigmoid", "-ss", "50")
    a = mm.ZarrManager(str(soft)).data[0].compute()
    b = mm.ZarrManager(str(sharp)).data[0].compute()
    assert a.shape == b.shape and not np.array_equal(a, b)
    with pytest.raises(ValueError):
        _run(monkeypatch, "fuse", "-i", str(acquisition), "-z", str(tmp_path / "x.zarr"), "-ss", "-1")


def test_mosaic_convert_matches_tile_layout(acquisition, tmp_path):
    out = tmp_path / "layout.ome.zarr"
    cli.mosaic_convert(acquisition, out, blend="max", projection="max", channels=["GFP"],
                       num_levels=1, shards=None, drop_singleton=False)
    img = mm.ZarrManager(str(out)).data[0].compute()[0]
    # tiles are 32 um wide (16 px x 2 um) at x = -15, 13, 35, 63 um: no gaps
    assert img.min() > 0
    assert img.shape[-1] == (63 - -15 + 32) // 2


def test_fuse_batch(acquisition, tmp_path, monkeypatch):
    csv = tmp_path / "batch.csv"
    pd.DataFrame([
        {"input": str(acquisition), "microscope": "luxendo", "output": str(tmp_path / "a.ome.zarr"),
         "downsample": "1 2 2", "projection": "none", "blend": "midline", "num_levels": 2,
         "shards": "auto", "channels": "RFP", "overwrite": "true"},
        {"input": str(acquisition), "microscope": "", "output": str(tmp_path / "b.ome.zarr"),
         "downsample": "", "projection": "max", "blend": "", "num_levels": "",
         "shards": "none", "channels": "", "overwrite": ""},
    ]).to_csv(csv, index=False)
    _run(monkeypatch, "fuse", "--batch", str(csv))
    a = mm.ZarrManager(str(tmp_path / "a.ome.zarr"))
    assert a.metadata["axes"] == "zyx" and len(a.data) == 2
    b = mm.ZarrManager(str(tmp_path / "b.ome.zarr"))
    assert b.metadata["axes"] == "cyx" and b.metadata["channel_names"] == ["GFP", "RFP"]

    # a CSV written before the rename (read_downsample column) still runs (row a overwrites its output)
    only_a = tmp_path / "only_a.csv"
    pd.read_csv(csv).iloc[:1].rename(columns={"downsample": "read_downsample"}).to_csv(only_a, index=False)
    _run(monkeypatch, "fuse", "--batch", str(only_a))
    assert mm.ZarrManager(str(tmp_path / "a.ome.zarr")).metadata["axes"] == "zyx"


def test_old_command_names_still_work(acquisition, tmp_path, monkeypatch, capsys):
    out = tmp_path / "old.ome.zarr"
    _run(monkeypatch, "mosaic2zarr", "-i", str(acquisition), "-z", str(out), "--read_downsample",
         "2", "-p", "max", "-ow", "true")
    assert "is now `pymif fuse`" in capsys.readouterr().err
    csv = tmp_path / "old.csv"
    pd.DataFrame([{"input": str(acquisition), "microscope": "luxendo", "output": str(out),
                   "read_downsample": "2", "projection": "max", "overwrite": "true"}]).to_csv(csv, index=False)
    for old in ("batch-mosaic2zarr", "batchmosaic2zarr"):
        _run(monkeypatch, old, "-i", str(csv))
        assert "is now `pymif fuse --batch`" in capsys.readouterr().err

    src = tmp_path / "src.zarr"
    mm.ZarrManager(str(out)).to_zarr(str(src))
    _run(monkeypatch, "2zarr", "-i", str(src), "-z", str(tmp_path / "conv.zarr"), "-sh", "none")
    assert "is now `pymif convert`" in capsys.readouterr().err
    _run(monkeypatch, "convert", "-i", str(src), "-z", str(tmp_path / "conv2.zarr"), "-sh", "none")
    assert "is now" not in capsys.readouterr().err


def test_convert_and_fuse_need_input_or_batch(monkeypatch, tmp_path):
    for argv in (["convert"], ["fuse"], ["fuse", "-i", str(tmp_path)], ["fuse", "--batch", __file__, "-z", "x"]):
        with pytest.raises(SystemExit):
            _run(monkeypatch, *argv)


def test_subset_to_crop():
    crop, channels = cli._subset_to_crop("c=0,2;z=1:5;x=3", ["a", "b", "c"])
    assert channels == ["a", "c"]
    assert crop == {"z": slice(1, 5), "x": 3}
    with pytest.raises(ValueError):
        cli._subset_to_crop("t=0", ["a"])
    with pytest.raises(ValueError):
        cli._subset_to_crop("y=1,2,3", ["a"])


def test_fuse_only_some_tiles(acquisition, tmp_path, monkeypatch):
    out = tmp_path / "right.ome.zarr"
    _run(monkeypatch, "fuse", "-i", str(acquisition), "-z", str(out), "-p", "max", "-sh", "none",
         "-t", "2-x00-y00", "2-x01-y00")
    img = mm.ZarrManager(str(out)).data[0].compute()
    # only the R tiles (x from 35 to 95 um, 2 um voxels): the output fits them
    assert img.shape[-1] == (95 - 35) // 2 and img.min() > 0

    csv = tmp_path / "batch.csv"
    pd.DataFrame([{"input": str(acquisition), "output": str(tmp_path / "b.ome.zarr"), "projection": "max",
                   "shards": "none", "channels": "GFP", "tiles": "1-x00-y00"}]).to_csv(csv, index=False)
    _run(monkeypatch, "fuse", "--batch", str(csv))
    assert mm.ZarrManager(str(tmp_path / "b.ome.zarr")).data[0].shape[-1] == 32 // 2
