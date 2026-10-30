from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import zarr

import zarrsmith.microscope_manager as mm
from zarrsmith.cli.__arguments import parse_bool
from zarrsmith.cli.zarrsmith import convert_batch, zarr_convert


def _manager(shape=(1, 2, 1, 64, 64), axes="tczyx", chunks=None):
    data = np.arange(np.prod(shape), dtype="uint16").reshape(shape)
    metadata = {
        "axes": axes,
        "scales": [(1.0, 1.0, 1.0)],
        "units": ("micrometer",) * 3,
        "time_increment": 1.0,
        "time_increment_unit": "second",
        "channel_names": ["A", "B"],
        "channel_colors": ["FF0000", "00FF00"],
    }
    return mm.ArrayManager(data, metadata, chunks=chunks or shape)


def _axes_of(path):
    group = zarr.open_group(str(path), mode="r")
    attrs = group.attrs.asdict()
    ms = attrs.get("ome", attrs)["multiscales"][0]
    return "".join(a["name"] for a in ms["axes"]), group["0"]


def test_drop_singleton_is_default(tmp_path):
    out = tmp_path / "a.zarr"
    _manager().to_zarr(str(out), ngff_version="0.5", zarr_format=3)
    axes, arr = _axes_of(out)
    assert axes == "cyx"
    assert arr.shape == (2, 64, 64)


def test_drop_singleton_false_keeps_axes(tmp_path):
    out = tmp_path / "b.zarr"
    _manager().to_zarr(str(out), ngff_version="0.5", zarr_format=3, drop_singleton=False)
    axes, arr = _axes_of(out)
    assert axes == "tczyx"
    assert arr.shape == (1, 2, 1, 64, 64)


def test_never_drops_spatial_axes(tmp_path):
    out = tmp_path / "c.zarr"
    _manager(shape=(1, 1, 1, 1, 64), chunks=(1, 1, 1, 1, 64)).to_zarr(
        str(out), ngff_version="0.5", zarr_format=3
    )
    axes, arr = _axes_of(out)
    assert axes == "yx"
    assert arr.shape == (1, 64)


def test_explicit_chunks_and_shards_are_trimmed(tmp_path):
    out = tmp_path / "d.zarr"
    _manager(shape=(1, 2, 1, 64, 64)).to_zarr(
        str(out),
        ngff_version="0.5",
        zarr_format=3,
        chunks=(1, 1, 1, 32, 32),
        shards=(1, 1, 1, 64, 64),
    )
    _, arr = _axes_of(out)
    assert arr.chunks == (1, 32, 32)
    assert arr.shards == (1, 64, 64)


def test_zarr_v2_drops_singleton(tmp_path):
    out = tmp_path / "e.zarr"
    _manager().to_zarr(str(out), ngff_version="0.4", zarr_format=2)
    axes, arr = _axes_of(out)
    assert axes == "cyx"
    assert arr.shape == (2, 64, 64)


def test_parse_bool():
    assert parse_bool("true") is True
    assert parse_bool("False") is False
    assert parse_bool("0") is False
    with pytest.raises(Exception):
        parse_bool("maybe")


def _write_source(path):
    _manager().to_zarr(str(path), ngff_version="0.5", zarr_format=3, drop_singleton=False)


def test_zarr_convert_drop_singleton(tmp_path):
    src, out = tmp_path / "src.zarr", tmp_path / "out.zarr"
    _write_source(src)
    zarr_convert(
        input_path=str(src), zarr_path=str(out), microscope="zarr",
        chunk_size=[1, 1, 1, 32, 32], zarr_format=3, num_levels=1,
    )
    axes, _ = _axes_of(out)
    assert axes == "cyx"


def test_zarr_convert_keep_singleton(tmp_path):
    src, out = tmp_path / "src.zarr", tmp_path / "out.zarr"
    _write_source(src)
    zarr_convert(
        input_path=str(src), zarr_path=str(out), microscope="zarr",
        chunk_size=[1, 1, 1, 32, 32], zarr_format=3, num_levels=1,
        drop_singleton=False,
    )
    axes, _ = _axes_of(out)
    assert axes == "tczyx"


def test_batch_csv_drop_singleton_column(tmp_path):
    src = tmp_path / "src.zarr"
    _write_source(src)
    csv = tmp_path / "batch.csv"
    pd.DataFrame(
        [
            {"input": str(src), "microscope": "zarr", "output": str(tmp_path / "o1.zarr"),
             "chunk_size": "1 1 1 32 32", "num_levels": 1, "drop_singleton": "false"},
            {"input": str(src), "microscope": "zarr", "output": str(tmp_path / "o2.zarr"),
             "chunk_size": "1 1 1 32 32", "num_levels": 1, "drop_singleton": ""},
        ]
    ).to_csv(csv, index=False)

    class Args:
        input_file = str(csv)

    convert_batch(Args())
    assert _axes_of(tmp_path / "o1.zarr")[0] == "tczyx"
    assert _axes_of(tmp_path / "o2.zarr")[0] == "cyx"
