from __future__ import annotations

import json

import h5py
import numpy as np

from pymif.mosaic_manager import LuxendoMosaicManager


def _write_tile(folder, name, desc, stack, tx, ty, tz, shape=(4, 8, 6), value=1):
    folder.mkdir(parents=True, exist_ok=True)
    with h5py.File(folder / f"{name}.lux.h5", "w") as f:
        f["Data"] = np.full(shape, value, np.uint16)
    info = {
        "channel": "1",
        "channel_description": desc,
        "stack": stack,
        "stack_description": "brain",
        "voxel_size_um": {"width": 2.0, "height": 2.0, "depth": 3.0},
        "image_size_vx": {"width": shape[2], "height": shape[1], "depth": shape[0]},
        "affine_to_sample": [
            {"matrix": np.eye(3).tolist(), "translation": [0, 0, 0]},
            {"matrix": [[2.0, 0, 0], [0, 2.0, 0], [0, 0, -3.0]], "translation": [tx, ty, tz]},
        ],
    }
    (folder / f"{name}.json").write_text(json.dumps({"processingInformation": info}))


def test_luxendo_scan(tmp_path):
    raw = tmp_path / "raw"
    _write_tile(raw / "stack_1-x00-y00_ch_GFP-L", "Cam_bottom_00000", "GFP-L", "1-x00-y00", 0.0, 0.0, 30.0)
    # a re-acquired stack: only the retake (_R_1) is used
    _write_tile(raw / "stack_2-x00-y00_ch_GFP-R", "Cam_bottom_00000", "GFP-R", "2-x00-y00", 100.0, 0.0, 30.0, value=5)
    _write_tile(raw / "stack_2-x00-y00_ch_GFP-R", "Cam_bottom_00000_R_1", "GFP-R", "2-x00-y00", 10.0, 0.0, 30.0, value=7)
    # files that are not tiles
    with h5py.File(raw / "main_raw.lux.h5", "w") as f:
        f["x"] = 1
    _write_tile(tmp_path / "ims" / "elsewhere", "Cam_bottom_00000", "GFP-L", "9-x00-y00", 0.0, 0.0, 0.0)

    m = LuxendoMosaicManager(tmp_path, progress=False)
    assert len(m.tiles) == 2  # ims/ is not scanned
    assert m.channels == ["GFP"] and m.groups == ["L", "R"]

    left, right = m.tiles
    assert right.attrs["path"].endswith("Cam_bottom_00000_R_1.lux.h5")
    assert right.data[0, 0, 0] == 7
    assert left.shape == (4, 8, 6) and left.dtype == np.uint16
    assert left.voxel_size == (3.0, 2.0, 2.0)
    # tile centred on the translation in x/y; z = translation / signed plane step * plane size
    assert left.origin == (-30.0, -7.0, -5.0)
    assert right.origin[2] == 5.0
    assert left.attrs["grid"] == (0, 0)

    zm = m.fuse(tmp_path / "f.zarr", blend="midline", chunks=(1, 4, 8, 8), progress=False)
    assert zm.metadata["channel_names"] == ["GFP"]
