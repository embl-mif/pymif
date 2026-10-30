# tests/test_zarr_empty_creation.py
from __future__ import annotations

import zarr
import pytest

import zarrsmith.microscope_manager as mm


def test_create_empty_dataset_from_metadata_v04(tmp_path, metadata):
    path = tmp_path / "empty_v04.zarr"
    d = mm.ZarrManager(str(path), metadata={**metadata, "ngff_version": "0.4", "zarr_format": 2}, mode="a")

    root = zarr.open_group(str(path), mode="r")
    attrs = root.attrs.asdict()
    assert "multiscales" in attrs
    assert "omero" in attrs
    assert "ome" not in attrs
    assert "0" in root
    assert tuple(root["0"].shape) == metadata["size"][0]


def test_create_empty_dataset_from_metadata_v05(tmp_path, metadata):
    path = tmp_path / "empty_v05.zarr"
    d = mm.ZarrManager(str(path), metadata={**metadata, "ngff_version": "0.5", "zarr_format": 3}, mode="a")

    root = zarr.open_group(str(path), mode="r")
    attrs = root.attrs.asdict()
    assert "ome" in attrs
    assert attrs["ome"]["version"] == "0.5"
    assert "0" in root
    assert tuple(root["0"].shape) == metadata["size"][0]


def test_create_empty_group_inherits_root_format_v04(tmp_path, image_pyramid, metadata):
    root_path = tmp_path / "root_v04.zarr"

    writer = mm.ArrayManager(image_pyramid, metadata)
    writer.to_zarr(
        str(root_path),
        ngff_version="0.4",
        zarr_format=2,
        overwrite=True,
    )

    d = mm.ZarrManager(str(root_path), mode="a")
    d.create_empty_group("processed", metadata)

    root = zarr.open_group(str(root_path), mode="r")
    grp = root["processed"]
    attrs = grp.attrs.asdict()

    assert "multiscales" in attrs
    assert "omero" in attrs
    assert "ome" not in attrs


def test_create_empty_group_inherits_root_format_v05(tmp_path, image_pyramid, metadata):
    root_path = tmp_path / "root_v05.zarr"

    writer = mm.ArrayManager(image_pyramid, metadata)
    
    writer.to_zarr(
        str(root_path),
        ngff_version="0.5",
        zarr_format=3,
        overwrite=True,
    )

    d = mm.ZarrManager(str(root_path), mode="a")
    d.create_empty_group("processed", metadata)

    root = zarr.open_group(str(root_path), mode="r")
    grp = root["processed"]
    attrs = grp.attrs.asdict()

    assert "ome" in attrs
    assert attrs["ome"]["version"] == "0.5"


def test_create_empty_label_group(tmp_path, image_pyramid, metadata):
    root_path = tmp_path / "with_label.zarr"

    writer = mm.ArrayManager(image_pyramid, metadata)
    
    writer.to_zarr(
        str(root_path),
        ngff_version="0.5",
        zarr_format=3,
        overwrite=True,
    )

    d = mm.ZarrManager(str(root_path), mode="a")
    d.create_empty_group("nuclei", metadata, is_label=True)

    root = zarr.open_group(str(root_path), mode="r")
    assert "labels" in root
    assert "labels" not in root.attrs.asdict()["ome"]
    assert root["labels"].attrs.asdict()["ome"]["labels"] == ["nuclei"]
    assert "nuclei" in root["labels"]
    label_grp = root["labels"]["nuclei"]
    assert "0" in label_grp
    # Nested under <image>/labels/<name>: the source image is two levels up.
    assert label_grp.attrs.asdict()["ome"]["image-label"] == {"source": {"image": "../../"}}


def test_create_empty_label_group_v04_registers_labels_group(tmp_path, image_pyramid, metadata):
    root_path = tmp_path / "with_label_v04.zarr"

    writer = mm.ArrayManager(image_pyramid, metadata)

    writer.to_zarr(
        str(root_path),
        ngff_version="0.4",
        zarr_format=2,
        overwrite=True,
    )

    d = mm.ZarrManager(str(root_path), mode="a")
    d.create_empty_group("nuclei", metadata, is_label=True)

    root = zarr.open_group(str(root_path), mode="r")
    assert "labels" not in root.attrs.asdict()
    assert root["labels"].attrs.asdict()["labels"] == ["nuclei"]
