# tests/test_evident_manager.py
"""EvidentManager tests with a fake slideio module (no .vsi file needed)."""
from __future__ import annotations

import json
import pickle
import sys
import types

import numpy as np
import pytest

import pymif.microscope_manager as mm
from pymif.cli.pymif import _resolve_zarr_manager


def _number(value, units="10^-6m^1"):
    return [
        {"name": "Units", "tag": 268435456, "value": units},
        {"name": "Value", "tag": 268435458, "value": f"{value:.6f}"},
    ]


def _raw_metadata(stack_name, channel_names, luts, z_positions=None):
    """Minimal VSI metadata tree, with the node names and tags slideio produces."""
    channels = [
        {"name": "Channel properties", "tag": 2008, "value": [
            {"name": "Channel name", "tag": 2419, "value": name},
            {"name": "Stack display LUT", "tag": 2004, "value": [
                {"name": "Value", "tag": 268435458, "value": f"red = {r}, green = {g}, blue = {b} [256 entries]"},
            ]},
        ]}
        for name, (r, g, b) in zip(channel_names, luts)
    ]
    frames = [
        {"name": "Frame properties", "tag": 2006, "value": [{"name": "Unknown tag", "tag": 2014, "value": _number(z)}]}
        for z in (z_positions or [])
    ]
    volume = {"name": "Multidimensional image volume", "tag": 2001, "value": [
        {"name": "Multidimensional stack properties", "tag": 2005, "value": [
            {"name": "Stack name", "tag": 2030, "value": stack_name},
        ]},
        {"name": "Volume for dimension 0 description", "tag": 2007, "value": channels},
        {"name": "Image frame volume", "tag": 2002, "value": frames},
    ]}
    return json.dumps({"name": "root", "tag": -2, "value": [
        {"name": "Collection volume", "tag": 2000, "value": [volume]},
    ]})


class FakeScene:
    """Mimics slideio.Scene: read_block returns (Z, Y, X, C) and drops singleton Z and C."""

    def __init__(self, name, data, channel_names, resolution=(2.6e-7, 2.6e-7), z_resolution=1e-6):
        self.name = name
        self.data = data  # TCZYX
        self.channel_names = channel_names
        self.resolution = resolution
        self.z_resolution = z_resolution
        self.t_resolution = 0.0

    @property
    def size(self):
        return self.data.shape[4], self.data.shape[3]

    num_t_frames = property(lambda self: self.data.shape[0])
    num_channels = property(lambda self: self.data.shape[1])
    num_z_slices = property(lambda self: self.data.shape[2])

    def get_channel_data_type(self, c):
        return self.data.dtype

    def get_channel_name(self, c):
        return self.channel_names[c]

    def read_block(self, rect=(0, 0, 0, 0), size=(0, 0), channel_indices=None, slices=(0, 1), frames=(0, 1)):
        x, y, w, h = rect
        channels = channel_indices or list(range(self.num_channels))
        block = self.data[frames[0], channels, slices[0]:slices[1], y:y + h, x:x + w]  # CZYX
        block = np.moveaxis(block, 0, -1)  # ZYXC
        if block.shape[-1] == 1:
            block = block[..., 0]
        if block.shape[0] == 1:
            block = block[0]
        return block


class FakeSlide:
    def __init__(self, scenes, raw_metadata):
        self.scenes = scenes
        self.raw_metadata = raw_metadata

    num_scenes = property(lambda self: len(self.scenes))

    def get_scene(self, i):
        return self.scenes[i]


@pytest.fixture
def fake_vsi(monkeypatch, tmp_path):
    """Install a fake slideio serving one 2-channel z-stack; returns (path, data, slide)."""
    rng = np.random.default_rng(0)
    data = rng.integers(0, 4096, size=(1, 2, 5, 37, 53), dtype=np.uint16)
    names = ["C640", "C405"]
    slide = FakeSlide(
        [FakeScene("C640, C405", data, names, z_resolution=0.0)],
        _raw_metadata("C640, C405", names, [(255, 0, 0), (0, 0, 255)],
                      z_positions=[9061.88 + 0.96 * z for z in range(5) for _ in names]),
    )
    module = types.ModuleType("slideio")
    module.open_slide = lambda path, driver: slide
    monkeypatch.setitem(sys.modules, "slideio", module)
    path = tmp_path / "sample.vsi"
    path.write_bytes(b"")
    return path, data, slide


def test_evident_metadata(fake_vsi):
    path, data, _ = fake_vsi
    vsi = mm.EvidentManager(path)

    md = vsi.metadata
    assert md["size"] == [data.shape]
    assert md["axes"] == "tczyx"
    assert md["dtype"] == "uint16"
    assert md["channel_names"] == ["C640", "C405"]
    assert md["channel_colors"] == ["FF0000", "0000FF"]
    # z step from the per-frame stage positions, since the scene has no z calibration
    assert md["scales"][0] == pytest.approx((0.96, 0.26, 0.26))
    assert md["units"] == ("micrometer",) * 3
    assert vsi.scenes == ["C640, C405"]


def test_evident_z_step_from_calibration(fake_vsi):
    path, _, slide = fake_vsi
    slide.scenes[0].z_resolution = 1e-6
    assert mm.EvidentManager(path).metadata["scales"][0][0] == pytest.approx(1.0)


def test_evident_missing_z_step_warns(fake_vsi):
    path, _, slide = fake_vsi
    slide.raw_metadata = _raw_metadata("C640, C405", ["C640", "C405"], [(255, 0, 0), (0, 0, 255)])
    with pytest.warns(UserWarning, match="no z step"):
        md = mm.EvidentManager(path).metadata
    assert md["scales"][0][0] == 1.0


def test_evident_lazy_reads(fake_vsi):
    path, data, _ = fake_vsi
    vsi = mm.EvidentManager(path, chunks=(1, 1, 2, 16, 16))
    arr = vsi.data[0]

    assert arr.chunksize == (1, 1, 2, 16, 16)
    np.testing.assert_array_equal(arr.compute(), data)
    np.testing.assert_array_equal(arr[0, 1, 3, 5:30:3, ::-2].compute(), data[0, 1, 3, 5:30:3, ::-2])
    np.testing.assert_array_equal(arr[:, :, 2].compute(), data[:, :, 2])


def test_evident_array_is_picklable(fake_vsi):
    path, data, _ = fake_vsi
    array = mm.EvidentManager(path).data[0]
    restored = pickle.loads(pickle.dumps(array))
    np.testing.assert_array_equal(restored.compute(), data)


def test_evident_scene_selection(fake_vsi):
    path, data, slide = fake_vsi
    overview = np.zeros((1, 3, 1, 8, 8), dtype=np.uint8)
    slide.scenes.insert(0, FakeScene("Overview", overview, ["R", "G", "B"]))

    vsi = mm.EvidentManager(path, scene_name="C640, C405")
    assert vsi.scene_index == 1
    assert vsi.metadata["size"] == [data.shape]

    vsi.read(scene_index=0)
    assert vsi.scene_name == "Overview"
    assert vsi.metadata["size"] == [overview.shape]
    assert vsi.metadata["dtype"] == "uint8"

    with pytest.raises(ValueError, match="Invalid scene"):
        mm.EvidentManager(path, scene_name="missing")
    with pytest.raises(ValueError, match="Invalid scene index"):
        vsi.read(scene_index=5)


def test_evident_zarr_roundtrip(fake_vsi, tmp_path):
    path, data, _ = fake_vsi
    vsi = mm.EvidentManager(path)
    vsi.build_pyramid(num_levels=2)
    out = tmp_path / "sample.zarr"
    vsi.to_zarr(str(out))

    back = mm.ZarrManager(str(out))
    np.testing.assert_array_equal(np.asarray(back.data[0]).reshape(data.shape), data)
    assert back.metadata["channel_names"] == ["C640", "C405"]
    assert back.metadata["scales"][0][-3:] == pytest.approx((0.96, 0.26, 0.26))


def test_cli_detects_vsi(tmp_path):
    assert _resolve_zarr_manager(str(tmp_path / "sample.vsi")) == (mm.EvidentManager, "evident")
    assert _resolve_zarr_manager("anything", "evident") == (mm.EvidentManager, "evident")
