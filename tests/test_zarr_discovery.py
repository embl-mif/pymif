# tests/test_zarr_discovery.py
from __future__ import annotations

import zarr

import zarrsmith.microscope_manager as mm


def test_subgroup_and_label_discovery(tmp_path, image_pyramid, metadata):
    path = tmp_path / "discovery.zarr"

    writer = mm.ArrayManager(image_pyramid, metadata)

    writer.to_zarr(
        str(path),
        ngff_version="0.5",
        zarr_format=3,
        overwrite=True,
    )

    d = mm.ZarrManager(str(path), mode="a")
    d.create_empty_group("processed", metadata, is_label=False)
    d.create_empty_group("nuclei", metadata, is_label=True)

    reread = mm.ZarrManager(str(path), mode="r")

    assert "processed" in reread.groups
    assert "nuclei" in reread.labels
    assert len(reread.groups["processed"]) == len(metadata["size"])
    assert len(reread.labels["nuclei"]) == len(metadata["size"])


def test_rewrite_with_labels_uses_labels_group_registry(tmp_path, image_pyramid, metadata):
    source = tmp_path / "source.zarr"
    rewritten = tmp_path / "rewritten.zarr"

    writer = mm.ArrayManager(image_pyramid, metadata)
    writer.to_zarr(
        str(source),
        ngff_version="0.5",
        zarr_format=3,
        overwrite=True,
    )

    d = mm.ZarrManager(str(source), mode="a")
    d.create_empty_group("nuclei", metadata, is_label=True)
    d.to_zarr(
        str(rewritten),
        ngff_version="0.5",
        zarr_format=3,
        overwrite=True,
    )

    root = zarr.open_group(str(rewritten), mode="r")
    assert "labels" not in root.attrs.asdict()["ome"]
    assert root["labels"].attrs.asdict()["ome"]["labels"] == ["nuclei"]
