# tests/test_channel_colors.py
from __future__ import annotations

import pytest

import zarrsmith.microscope_manager as mm
from zarrsmith.utils.ngff import ome_color_to_hex, parse_color


@pytest.mark.parametrize(
    "value, expected",
    [("magenta", "FF00FF"), ("Cyan", "00FFFF"), ("#ff0000", "FF0000"), (" 00ff00 ", "00FF00")],
)
def test_parse_color(value, expected):
    assert parse_color(value) == expected


def test_parse_color_invalid():
    with pytest.raises(TypeError):
        parse_color("notacolor")


@pytest.mark.parametrize(
    "value, expected",
    [(-1, "FFFFFF"), ("-16776961", "FF0000"), (16711935, "00FF00"), (65535, "0000FF")],
)
def test_ome_color_to_hex(value, expected):
    assert ome_color_to_hex(value) == expected


def test_named_channel_colors_written_to_omero(tmp_path, image_pyramid, metadata):
    out = tmp_path / "named_colors.zarr"
    md = dict(metadata, channel_colors=["magenta", "cyan"])

    mm.ArrayManager(image_pyramid, md).to_zarr(str(out), overwrite=True)

    reader = mm.ZarrManager(str(out))
    assert reader.metadata["channel_colors"] == ["FF00FF", "00FFFF"]
