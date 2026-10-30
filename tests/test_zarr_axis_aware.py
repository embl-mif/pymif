from __future__ import annotations

import numpy as np
import dask.array as da
import pytest
import zarr

import zarrsmith.microscope_manager as mm


def _yx_metadata(dtype="uint16", data_type="intensity"):
    return {
        "size": [(8, 9), (4, 5)],
        "chunksize": [(4, 5), (2, 3)],
        "scales": [(0.25, 0.25), (0.5, 0.5)],
        "units": ("micrometer", "micrometer"),
        "time_increment": None,
        "time_increment_unit": None,
        "channel_names": [],
        "channel_colors": [],
        "dtype": dtype,
        "axes": "yx",
        "data_type": data_type,
        "name": f"{data_type}_yx",
    }


def test_yx_intensity_roundtrip_v05(tmp_path):
    path = tmp_path / "yx_intensity.zarr"
    base = np.arange(8 * 9, dtype=np.uint16).reshape(8, 9)
    levels = [
        da.from_array(base, chunks=(4, 5)),
        da.from_array(base[::2, ::2], chunks=(2, 3)),
    ]
    metadata = _yx_metadata(data_type="intensity")

    writer = mm.ArrayManager(levels, metadata)
    writer.to_zarr(path, ngff_version="0.5", zarr_format=3, overwrite=True)

    root = zarr.open_group(str(path), mode="r")
    ome = root.attrs.asdict()["ome"]
    assert ome["data_type"] == "intensity"
    assert [axis["name"] for axis in ome["multiscales"][0]["axes"]] == ["y", "x"]
    assert root["0"].attrs.asdict()["dimension_names"] == ["y", "x"]
    assert "omero" in ome

    reader = mm.ZarrManager(path, mode="r")
    assert reader.metadata["axes"] == "yx"
    assert reader.metadata["data_type"] == "intensity"
    assert reader.metadata["time_increment"] is None
    assert reader.metadata["channel_names"] == []
    np.testing.assert_array_equal(reader.data[0].compute(), base)


def test_yx_label_roundtrip_v05_has_label_metadata(tmp_path):
    path = tmp_path / "yx_label.zarr"
    labels = np.zeros((8, 9), dtype=np.uint16)
    labels[2:6, 3:7] = 4
    levels = [
        da.from_array(labels, chunks=(4, 5)),
        da.from_array(labels[::2, ::2], chunks=(2, 3)),
    ]
    metadata = _yx_metadata(data_type="label")

    writer = mm.ArrayManager(levels, metadata)
    writer.to_zarr(path, ngff_version="0.5", zarr_format=3, overwrite=True)

    root = zarr.open_group(str(path), mode="r")
    ome = root.attrs.asdict()["ome"]
    assert ome["data_type"] == "label"
    assert ome["multiscales"][0]["type"] == "label"
    # Standalone label store: no source image to point at.
    assert ome["image-label"] == {}
    assert root["0"].attrs.asdict()["dimension_names"] == ["y", "x"]
    assert "omero" not in ome

    reader = mm.ZarrManager(path, mode="r")
    assert reader.metadata["axes"] == "yx"
    assert reader.metadata["data_type"] == "label"
    assert reader.metadata["is_label"] is True
    np.testing.assert_array_equal(reader.data[0].compute(), labels)


def test_invalid_axis_labels_are_rejected():
    arr = da.from_array(np.zeros((2, 3, 4), dtype=np.uint16), chunks=(1, 3, 4))
    metadata = {
        "size": [(2, 3, 4)],
        "chunksize": [(1, 3, 4)],
        "scales": [(1.0, 1.0)],
        "units": ("micrometer", "micrometer"),
        "dtype": "uint16",
        "axes": "tyq",
    }
    with pytest.raises(ValueError, match="Invalid axis"):
        mm.ArrayManager([arr], metadata)


def test_legacy_label_group_creation_drops_c_axis(tmp_path, image_pyramid, metadata):
    path = tmp_path / "legacy_label.zarr"
    mm.ArrayManager(image_pyramid, metadata).to_zarr(path, ngff_version="0.5", zarr_format=3, overwrite=True)

    reader = mm.ZarrManager(path, mode="a")
    reader.create_empty_group("nuclei", metadata, is_label=True)

    root = zarr.open_group(str(path), mode="r")
    label_group = root["labels"]["nuclei"]
    assert tuple(label_group["0"].shape) == (2, 4, 16, 16)
    root_ome = root.attrs.asdict()["ome"]
    assert "labels" not in root_ome
    labels_ome = root["labels"].attrs.asdict()["ome"]
    assert labels_ome["labels"] == ["nuclei"]
    axes = label_group.attrs.asdict()["ome"]["multiscales"][0]["axes"]
    assert [axis["name"] for axis in axes] == ["t", "z", "y", "x"]


def test_explicit_channelled_label_group_keeps_c_axis(tmp_path, image_pyramid, metadata):
    path = tmp_path / "channelled_label.zarr"
    mm.ArrayManager(image_pyramid, metadata).to_zarr(path, ngff_version="0.5", zarr_format=3, overwrite=True)

    label_metadata = {**metadata, "data_type": "label"}
    reader = mm.ZarrManager(path, mode="a")
    reader.create_empty_group("classes", label_metadata, data_type="label")

    root = zarr.open_group(str(path), mode="r")
    label_group = root["labels"]["classes"]
    root_ome = root.attrs.asdict()["ome"]
    assert "labels" not in root_ome
    labels_ome = root["labels"].attrs.asdict()["ome"]
    assert labels_ome["labels"] == ["classes"]
    assert tuple(label_group["0"].shape) == metadata["size"][0]
    ome = label_group.attrs.asdict()["ome"]
    assert ome["data_type"] == "label"
    assert [axis["name"] for axis in ome["multiscales"][0]["axes"]] == list("tczyx")


def test_axis_aware_subset_without_time_or_channel(tmp_path):
    path = tmp_path / "subset_yx.zarr"
    base = np.arange(8 * 9, dtype=np.uint16).reshape(8, 9)
    levels = [
        da.from_array(base, chunks=(4, 5)),
        da.from_array(base[::2, ::2], chunks=(2, 3)),
    ]
    metadata = _yx_metadata(data_type="intensity")
    mm.ArrayManager(levels, metadata).to_zarr(path, ngff_version="0.5", zarr_format=3, overwrite=True)

    reader = mm.ZarrManager(path, mode="r")
    reader.subset_dataset(Y=[1, 3, 5], X=slice(2, 7))

    assert reader.metadata["axes"] == "yx"
    assert reader.metadata["size"][0] == (3, 5)
    np.testing.assert_array_equal(reader.data[0].compute(), base[[1, 3, 5], 2:7])


def test_region_write_pyramid_odd_spatial_size_uses_ceil_downsampling():
    from zarrsmith.utils.write_image_region import _generate_pyramid

    labels = np.zeros((2, 13, 128, 128), dtype=np.uint16)
    pyramid = _generate_pyramid(
        labels,
        total_levels=3,
        axes="tzyx",
        level_scale_ratios=[(1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (4.0, 4.0, 4.0)],
    )

    assert pyramid[0].shape == (2, 13, 128, 128)
    assert pyramid[1].shape == (2, 7, 64, 64)
    assert pyramid[2].shape == (2, 4, 32, 32)
