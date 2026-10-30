"""Defaults and registries shared by the CLI and the napari widgets."""
from __future__ import annotations

import numpy as np
import pytest

import pymif.microscope_manager as mm
from pymif.utils.axes import metadata_axes, metadata_axis_size
from pymif.utils.chunks import default_chunk_size, default_num_levels


def _metadata(axes, size, dtype="uint16"):
    return {"axes": axes, "size": [size], "dtype": dtype}


def test_metadata_axes_and_sizes():
    md = _metadata("TZYX", (3, 10, 20, 30))
    assert metadata_axes(md) == "tzyx" and metadata_axes({}) == "tczyx"
    assert metadata_axis_size(md, "y") == 20 and metadata_axis_size(md, "c") == 1


def test_default_chunk_size_follows_the_dtype():
    chunks, size_mb, n_chunks = default_chunk_size(_metadata("tczyx", (1, 2, 100, 4096, 4096)), max_size_mb=100)
    assert chunks[:2] == (1, 1) and size_mb <= 100
    assert n_chunks["z"] == 2 * n_chunks["y"]  # z split twice as much as y and x
    wide, wide_mb, _ = default_chunk_size(_metadata("tczyx", (1, 2, 100, 4096, 4096), "float32"), max_size_mb=100)
    assert np.prod(wide) <= np.prod(chunks) and wide_mb <= 100  # 4 bytes a voxel: smaller chunks
    assert default_chunk_size(_metadata("yx", (50, 60)))[0] == (50, 60)  # small image: one chunk


def test_default_num_levels():
    assert default_num_levels(_metadata("zyx", (10, 1000, 1000))) == 1
    assert default_num_levels(_metadata("zyx", (10, 8192, 8192))) == 3  # 8192 -> 4096 -> 2048
    # 2D image with a Z, Y, X factor: the Y and X factors apply (not Z's)
    assert default_num_levels(_metadata("yx", (8192, 8192)), (1, 4, 4)) == 2
    # a factor of 1 never makes an axis smaller: no endless loop
    assert default_num_levels(_metadata("zyx", (4096, 100, 100)), (1, 2, 2)) == 1


def test_readers_shared_by_the_cli_and_napari(tmp_path):
    from pymif.cli.pymif import _resolve_zarr_manager
    from pymif.napari._convert_widget import dataset_reader

    assert dataset_reader("zeiss") is mm.READERS["zeiss"] is mm.ZeissManager
    with pytest.raises(ValueError, match="Unsupported"):
        dataset_reader("nope")
    store = tmp_path / "a.zarr"
    store.mkdir()
    (store / "zarr.json").write_text("{}")
    assert mm.detect_reader(store) == (mm.ZarrManager, "zarr")
    assert _resolve_zarr_manager(str(tmp_path / "x.czi")) == (mm.ZeissManager, "zeiss")
    assert mm.detect_reader(tmp_path / "unknown.xyz") is None
    with pytest.raises(TypeError, match="Could not auto-detect"):
        _resolve_zarr_manager(str(tmp_path / "unknown.xyz"))


def test_cli_color_errors_are_argparse_errors():
    import argparse

    from pymif.cli.__arguments import parse_color

    assert parse_color("#ff00ff") == "FF00FF" and parse_color("cyan") == "00FFFF"
    with pytest.raises(argparse.ArgumentTypeError, match="Invalid color"):
        parse_color("notacolor")


def test_add_layers_returns_the_layers():
    pytest.importorskip("napari")
    import dask.array as da
    from napari.components import ViewerModel

    from pymif.utils.visualize import add_layers

    md = {"axes": "czyx", "scales": [(2.0, 0.5, 0.5)], "units": ("micrometer",) * 3,
          "channel_names": ["A", "B"], "channel_colors": ["FF0000", "00FF00"]}
    viewer = ViewerModel()
    layers = add_layers(viewer, [da.zeros((2, 4, 8, 8), np.uint16)], md, name="img", blending="additive")
    assert [layer.name for layer in layers] == ["img: A", "img: B"]
    assert all(layer.blending == "additive" for layer in layers)
    assert tuple(layers[0].scale) == (2.0, 0.5, 0.5)
