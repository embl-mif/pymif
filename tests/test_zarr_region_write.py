# tests/test_zarr_region_write.py
from __future__ import annotations

import numpy as np
import zarr

import zarrsmith.microscope_manager as mm


def test_write_image_region_and_reopen(tmp_path, image_pyramid, metadata):
    path = tmp_path / "image_region.zarr"

    writer = mm.ArrayManager(image_pyramid, metadata)
    writer.to_zarr(
        str(path),
        ngff_version="0.5",
        zarr_format=3,
        overwrite=True,
    )

    d = mm.ZarrManager(str(path), mode="a")

    patch = np.full((1, 1, 2, 4, 4), 999, dtype=np.uint16)
    d.write_image_region(
        patch,
        t=slice(0, 1),
        c=slice(0, 1),
        z=slice(1, 3),
        y=slice(4, 8),
        x=slice(5, 9),
        level=0,
    )

    reread = mm.ZarrManager(str(path), mode="r")
    block = reread.data[0][0:1, 0:1, 1:3, 4:8, 5:9].compute()
    assert np.all(block == 999)


def test_write_label_region_and_reopen(tmp_path, image_pyramid, label_pyramid, metadata):
    path = tmp_path / "label_region.zarr"

    writer = mm.ArrayManager(image_pyramid, metadata)
    
    writer.to_zarr(
        str(path),
        ngff_version="0.5",
        zarr_format=3,
        overwrite=True,
    )

    d = mm.ZarrManager(str(path), mode="a")
    d.create_empty_group("nuclei", metadata, is_label=True)

    patch = np.full((1, 2, 4, 4), 5, dtype=np.uint16)
    d.write_label_region(
        patch,
        t=slice(0, 1),
        z=slice(1, 3),
        y=slice(4, 8),
        x=slice(4, 8),
        level=0,
        group="labels/nuclei",
    )

    root = zarr.open_group(str(path), mode="r")
    arr = root["labels"]["nuclei"]["0"][0:1, 1:3, 4:8, 4:8]
    assert np.all(arr == 5)