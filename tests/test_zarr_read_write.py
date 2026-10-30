# tests/test_zarr_read_write.py
from __future__ import annotations

import zarr
import numpy as np
import pytest

import zarrsmith.microscope_manager as mm


def _assert_metadata_equal_basic(m1, m2):
    assert m2["axes"] == m1["axes"]
    assert m2["dtype"] == m1["dtype"]
    assert m2["channel_names"] == m1["channel_names"]
    assert m2["channel_colors"] == m1["channel_colors"]
    assert tuple(m2["units"]) == tuple(m1["units"])
    assert pytest.approx(m2["time_increment"]) == m1["time_increment"]
    assert m2["time_increment_unit"] in ("s", "second")
    assert len(m2["size"]) == len(m1["size"])
    assert len(m2["scales"]) == len(m1["scales"])


def test_write_v04_and_reopen(tmp_path, image_pyramid, metadata):
    out = tmp_path / "test_v04.zarr"

    writer = mm.ArrayManager(image_pyramid, metadata)

    writer.to_zarr(
        str(out),
        ngff_version="0.4",
        zarr_format=2,
        compressor="blosc",
        compressor_level=1,
        overwrite=True,
    )

    # low-level structural check
    root = zarr.open_group(str(out), mode="r")
    attrs = root.attrs.asdict()
    assert "multiscales" in attrs
    assert "omero" in attrs
    assert "ome" not in attrs

    # high-level reopen
    d = mm.ZarrManager(str(out), mode="r")
    assert len(d.data) == 3
    _assert_metadata_equal_basic(metadata, d.metadata)


def test_write_v05_and_reopen(tmp_path, image_pyramid, metadata):
    out = tmp_path / "test_v05.zarr"

    writer = mm.ArrayManager(image_pyramid, metadata)
    
    writer.to_zarr(
        str(out),
        ngff_version="0.5",
        zarr_format=3,
        compressor="blosc",
        compressor_level=1,
        overwrite=True,
    )

    root = zarr.open_group(str(out), mode="r")
    attrs = root.attrs.asdict()
    assert "ome" in attrs
    assert attrs["ome"]["version"] == "0.5"
    assert "multiscales" in attrs["ome"]
    assert "omero" in attrs["ome"]

    d = mm.ZarrManager(str(out), mode="r")
    assert len(d.data) == 3
    _assert_metadata_equal_basic(metadata, d.metadata)


def test_round_trip_metadata_v04(tmp_path, image_pyramid, metadata):
    out = tmp_path / "roundtrip_v04.zarr"

    writer = mm.ArrayManager(image_pyramid, metadata)

    writer.to_zarr(
        str(out),
        ngff_version="0.4",
        zarr_format=2,
        overwrite=True,
    )

    reread = mm.ZarrManager(str(out), mode="r")
    _assert_metadata_equal_basic(metadata, reread.metadata)


def test_round_trip_metadata_v05(tmp_path, image_pyramid, metadata):
    out = tmp_path / "roundtrip_v05.zarr"

    writer = mm.ArrayManager(image_pyramid, metadata)

    writer.to_zarr(
        str(out),
        ngff_version="0.5",
        zarr_format=3,
        overwrite=True,
    )

    reread = mm.ZarrManager(str(out), mode="r")
    _assert_metadata_equal_basic(metadata, reread.metadata)