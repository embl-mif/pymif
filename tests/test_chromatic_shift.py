# tests/test_chromatic_shift.py
from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
import zarr
from scipy import ndimage as ndi

import pymif.microscope_manager as mm
from pymif.utils.chromatic_shift import (
    PROVENANCE_KEY,
    plan_z_extent,
    shift_along_axis,
)

SPACING = (2.0, 0.5, 0.5)
# Content displacement of each channel in Z planes: I_c(z) = I_ref(z - d).
DISPLACEMENTS = (0.0, 2.6, -1.3)


def _synthetic_sample(shape=(48, 96, 96), displacements=DISPLACEMENTS, seed=0):
    """Ellipsoid sample with a soft edge and different internal structure per channel."""
    rng = np.random.default_rng(seed)
    nz, ny, nx = shape
    z, y, x = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    center = ((nz - 1) / 2, (ny - 1) / 2, (nx - 1) / 2)
    radii = (nz * 0.3, ny * 0.35, nx * 0.35)

    out = np.empty((len(displacements), *shape), dtype=np.uint16)
    for c, d in enumerate(displacements):
        r = np.sqrt(
            ((z - d - center[0]) / radii[0]) ** 2
            + ((y - center[1]) / radii[1]) ** 2
            + ((x - center[2]) / radii[2]) ** 2
        )
        envelope = 1 / (1 + np.exp((r - 1) / 0.08))
        blobs = ndi.gaussian_filter(rng.random(shape), 2.0)
        blobs = (blobs - blobs.min()) / (blobs.max() - blobs.min())
        blobs = ndi.shift(blobs, (d, 0, 0), order=1, mode="nearest")
        signal = 100 + envelope * (200 * (c + 1) + 2000 * blobs**4)
        out[c] = np.clip(rng.poisson(signal), 0, 65535)
    return out


def _write_store(path, arr, *, downscale=(1, 2, 2), ngff_version="0.5", labels=False):
    metadata = {
        "axes": "czyx",
        "scales": [SPACING],
        "units": ("micrometer",) * 3,
        "channel_names": [f"ch{c}" for c in range(arr.shape[0])],
        "channel_colors": ["FF0000", "00FF00", "0000FF"][: arr.shape[0]],
    }
    writer = mm.ArrayManager(da.from_array(arr, chunks=(1, 8, 32, 32)), metadata)
    writer.build_pyramid(num_levels=3, downscale_factor=downscale)
    kwargs = dict(ngff_version=ngff_version, compressor="blosc", progress=False, overwrite=True)
    kwargs["shards"] = (1, 16, 64, 64) if ngff_version == "0.5" else None
    writer.to_zarr(str(path), **kwargs)

    z = mm.ZarrManager(str(path), mode="r+" if labels else "r")
    if labels:
        mask = (arr[0] > 400).astype(np.uint16)
        meta = dict(z.metadata, axes="zyx", dtype="uint16", data_type="label")
        meta["size"] = [s[1:] for s in meta["size"]]
        meta["chunksize"] = [c[1:] for c in meta["chunksize"]]
        meta.pop("shards", None)
        z.create_empty_group("mask", meta, is_label=True)
        z.write_label_region(mask, level=0, group="labels/mask", downscale_factor=downscale)
        z = mm.ZarrManager(str(path), mode="r")
    return z


@pytest.fixture(scope="module")
def sample():
    return _synthetic_sample()


@pytest.fixture
def store(tmp_path, sample):
    return _write_store(tmp_path / "in.zarr", sample)


def _expected_shifts(displacements=DISPLACEMENTS):
    return [-d * SPACING[0] for d in displacements]


# ----------------------------------------------------------------------
# Estimation
# ----------------------------------------------------------------------

def test_estimate_recovers_subplane_z_shift(store):
    report = store.estimate_chromatic_shift(max_shift=10.0)

    assert report["shifts"][0] == 0.0
    np.testing.assert_allclose(report["shifts"], _expected_shifts(), atol=0.25 * SPACING[0])
    assert report["unit"] == "micrometer"
    assert all(n > 0 for n in report["n_regions_used"][1:])
    # Refinement crops come from both the lower and the upper surface of the sample.
    assert {r["surface"] for r in report["regions"]} == {"lower", "upper"}


def test_estimate_with_z_downsampled_pyramid(tmp_path):
    arr = _synthetic_sample(shape=(64, 96, 96))
    z = _write_store(tmp_path / "zdown.zarr", arr, downscale=2)

    report = z.estimate_chromatic_shift(max_shift=10.0, coarse_level=1)

    np.testing.assert_allclose(report["shifts"], _expected_shifts(), atol=0.25 * SPACING[0])


def test_estimate_other_reference_channel(store):
    report = store.estimate_chromatic_shift(reference_channel=1, max_shift=10.0)

    expected = [-(d - DISPLACEMENTS[1]) * SPACING[0] for d in DISPLACEMENTS]
    assert report["shifts"][1] == 0.0
    np.testing.assert_allclose(report["shifts"], expected, atol=0.25 * SPACING[0])


def test_coarse_only_estimate(store):
    report = store.estimate_chromatic_shift(max_shift=10.0, refine_level=None)

    assert report["regions"] == []
    assert report["shifts"] == report["coarse_shifts"]
    np.testing.assert_allclose(report["shifts"], _expected_shifts(), atol=0.5 * SPACING[0])


def test_default_search_range_is_relative_to_stack_depth(store):
    report = store.estimate_chromatic_shift()

    assert report["max_shift"] == pytest.approx(0.1 * 48 * SPACING[0])
    assert report["warnings"] == []
    np.testing.assert_allclose(report["shifts"], _expected_shifts(), atol=0.25 * SPACING[0])


def test_search_limit_is_reported(store):
    # ±2 coarse planes (4 um) cannot reach channel 1's 5.2 um.
    with pytest.warns(UserWarning, match="search limit"):
        report = store.estimate_chromatic_shift(max_shift=1.0, refine_level=None)

    assert any("Channel 1" in w and "search limit" in w for w in report["warnings"])


def test_estimate_fails_without_z_surfaces(tmp_path):
    # Sample filling the whole stack in Z: only lateral edges exist.
    arr = np.full((2, 24, 64, 64), 100, dtype=np.uint16)
    arr[:, :, 16:48, 16:48] = 1000
    z = _write_store(tmp_path / "noz.zarr", arr)

    with pytest.raises(ValueError, match="No Z-facing sample surface"):
        z.estimate_chromatic_shift(max_shift=6.0)


# ----------------------------------------------------------------------
# Resampling helpers
# ----------------------------------------------------------------------

def test_shift_along_axis_integer_and_fractional():
    arr = da.from_array(np.arange(1, 11, dtype=np.uint16).reshape(10, 1), chunks=(4, 1))

    out = shift_along_axis(arr, 0, 12, 2, order=1).compute()[:, 0]
    np.testing.assert_array_equal(out, [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    out = shift_along_axis(arr, 0, 10, -3, order=1).compute()[:, 0]
    np.testing.assert_array_equal(out, [4, 5, 6, 7, 8, 9, 10, 0, 0, 0])

    out = shift_along_axis(arr.astype(np.float32), 0, 10, 0.5, order=1).compute()[:, 0]
    np.testing.assert_allclose(out[1:], np.arange(1, 10) + 0.5)
    assert out[0] == 0.5  # half of the first plane blended with the zero padding

    out = shift_along_axis(arr, 0, 10, 1.6, order=0).compute()[:, 0]
    np.testing.assert_array_equal(out, [0, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    assert out.dtype == np.uint16


def test_plan_z_extent_keeps_levels_aligned():
    z_spacings = [2.0, 4.0, 8.0]  # Z downsampled by 2 per level
    shifts = np.array([0.0, -5.2, 2.6])

    below, total = plan_z_extent(64, z_spacings, shifts, "union")
    # 5.2 / 2 = 2.6 planes below -> 4 (multiple of 4); 1.3 above -> 4.
    assert (below, total) == (8.0, 16.0)

    below, total = plan_z_extent(64, z_spacings, shifts, "intersection")
    assert (below, total) == (-8.0, -16.0)

    assert plan_z_extent(64, z_spacings, shifts, "same") == (0.0, 0.0)

    with pytest.raises(ValueError, match="no Z planes"):
        plan_z_extent(4, z_spacings, np.array([0.0, 40.0]), "intersection")


# ----------------------------------------------------------------------
# Writing the corrected store
# ----------------------------------------------------------------------

def _array_layout(arr: zarr.Array):
    return {
        "chunks": arr.chunks,
        "shards": arr.shards,
        "compressors": [c.to_dict() if hasattr(c, "to_dict") else c.get_config() for c in arr.compressors],
        "dtype": arr.dtype,
        "fill_value": arr.fill_value,
    }


def test_apply_preserves_structure_and_metadata(tmp_path, store):
    shifts = _expected_shifts()
    out = store.apply_chromatic_shift(tmp_path / "out.zarr", shifts, progress=False)

    src_root, dst_root = store.root, out.root
    for level in range(3):
        assert _array_layout(dst_root[str(level)]) == _array_layout(src_root[str(level)])

    # Z grows by ceil(2.6) planes below and ceil(1.3) above; Z is not downsampled.
    assert out.metadata["size"][0] == (3, 48 + 3 + 2, 96, 96)
    for level in range(3):
        src_shape = store.metadata["size"][level]
        assert out.metadata["size"][level] == (3, src_shape[1] + 5, *src_shape[2:])

    assert out.metadata["scales"] == store.metadata["scales"]
    assert out.metadata["units"] == store.metadata["units"]
    assert out.metadata["channel_names"] == store.metadata["channel_names"]
    assert out.metadata["channel_colors"] == store.metadata["channel_colors"]
    assert out.ngff_version == store.ngff_version

    ome_in = src_root.attrs["ome"]
    ome_out = dst_root.attrs["ome"]
    assert ome_out == ome_in
    provenance = dst_root.attrs[PROVENANCE_KEY]
    assert provenance["shifts"] == shifts
    assert provenance["z_offset"] == 3 * SPACING[0]
    assert provenance["extent"] == "union"


def test_apply_aligns_channels(tmp_path, store):
    report = store.estimate_chromatic_shift(max_shift=10.0)
    out = store.apply_chromatic_shift(tmp_path / "out.zarr", report, progress=False)

    residual = out.estimate_chromatic_shift(max_shift=10.0)
    np.testing.assert_allclose(residual["shifts"], 0.0, atol=0.25 * SPACING[0])
    assert out.root.attrs[PROVENANCE_KEY]["estimate"]["n_regions_used"] == report["n_regions_used"]


def test_apply_integer_shift_is_exact(tmp_path, store, sample):
    out = store.apply_chromatic_shift(
        tmp_path / "out.zarr", [0.0, 4.0, -2.0], order=0, extent="same", progress=False
    )
    data = np.asarray(out.data[0])

    assert data.shape == sample.shape
    np.testing.assert_array_equal(data[0], sample[0])
    np.testing.assert_array_equal(data[1, 2:], sample[1, :-2])   # +4 um = +2 planes
    np.testing.assert_array_equal(data[1, :2], 0)
    np.testing.assert_array_equal(data[2, :-1], sample[2, 1:])   # -2 um = -1 plane
    np.testing.assert_array_equal(data[2, -1], 0)


def test_apply_intersection_crops(tmp_path, store, sample):
    out = store.apply_chromatic_shift(
        tmp_path / "out.zarr", [0.0, 4.0, -2.0], order=0, extent="intersection", progress=False
    )
    data = np.asarray(out.data[0])

    # Channel 1 is empty in its first 2 planes, channel 2 in its last one.
    assert data.shape[1] == sample.shape[1] - 3
    np.testing.assert_array_equal(data[0], sample[0, 2:-1])


def test_apply_pads_labels_with_reference(tmp_path, sample):
    store = _write_store(tmp_path / "in.zarr", sample, labels=True)
    out = store.apply_chromatic_shift(tmp_path / "out.zarr", [0.0, -5.2, 2.6], order=0, progress=False)

    label_in = np.asarray(store.labels["mask"].data[0])
    assert label_in.any()
    label_out = np.asarray(out.labels["mask"].data[0])
    raw_out = np.asarray(out.data[0])

    # 3 planes added below (channel 1 moves down 2.6 planes), 2 above.
    assert label_out.shape == (label_in.shape[0] + 5, *label_in.shape[1:])
    assert raw_out.shape[1] == label_out.shape[0]
    np.testing.assert_array_equal(label_out[3:-2], label_in)
    np.testing.assert_array_equal(raw_out[0, 3:-2], sample[0])
    assert out.root["labels"].attrs.asdict() == store.root["labels"].attrs.asdict()
    for level in range(3):
        src = store.root[f"labels/mask/{level}"]
        dst = out.root[f"labels/mask/{level}"]
        assert _array_layout(dst) == _array_layout(src)


def test_apply_zarr_v2(tmp_path, sample):
    store = _write_store(tmp_path / "in_v2.zarr", sample, ngff_version="0.4")
    out = store.apply_chromatic_shift(tmp_path / "out_v2.zarr", [0.0, 4.0, -2.0], order=0, progress=False)

    for level in range(3):
        src, dst = store.root[str(level)], out.root[str(level)]
        assert dst.metadata.zarr_format == 2
        assert dst.chunks == src.chunks
        assert dst.compressors == src.compressors
        assert dst.metadata.dimension_separator == src.metadata.dimension_separator
    assert out.ngff_version == "0.4"
    assert out.metadata["size"][0] == (3, 48 + 1 + 2, 96, 96)


def test_apply_rejects_bad_input(tmp_path, store):
    with pytest.raises(ValueError, match="one shift per channel"):
        store.apply_chromatic_shift(tmp_path / "out.zarr", [0.0, 1.0], progress=False)
    with pytest.raises(ValueError, match="different from the input"):
        store.apply_chromatic_shift(store.path, [0.0, 0.0, 0.0], progress=False)

    store.subset_dataset(Y=slice(0, 48))
    with pytest.raises(ValueError, match="in-memory changes"):
        store.apply_chromatic_shift(tmp_path / "out2.zarr", [0.0, 0.0, 0.0], progress=False)


def test_correct_chromatic_shift_one_call(tmp_path, store):
    out = store.correct_chromatic_shift(tmp_path / "out.zarr", max_shift=10.0, progress=False)

    shifts = out.root.attrs[PROVENANCE_KEY]["shifts"]
    np.testing.assert_allclose(shifts, _expected_shifts(), atol=0.25 * SPACING[0])
