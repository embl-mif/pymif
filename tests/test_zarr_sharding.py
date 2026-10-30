# tests/test_zarr_sharding.py
from __future__ import annotations

import zarr
import numpy as np
import pytest

import zarrsmith.microscope_manager as mm
from zarrsmith.utils.ngff import _resolve_shards_for_levels


def test_auto_shards_never_merge_t_or_c_by_default(tmp_path, image_pyramid, metadata):
    """`shards="auto"` must not group chunks along t or c unless asked to."""
    out = tmp_path / "auto_no_tc.zarr"

    writer = mm.ArrayManager(image_pyramid, metadata)
    writer.to_zarr(
        str(out),
        ngff_version="0.5",
        zarr_format=3,
        shards="auto",
        shard_target_mb=1000,  # generous enough to merge everything if allowed
        overwrite=True,
    )

    root = zarr.open_group(str(out), mode="r")
    axes = "tczyx"
    t_idx, c_idx = axes.index("t"), axes.index("c")
    for i in range(3):
        z = root[str(i)]
        assert z.shards is not None
        # t and c must stay at exactly one chunk each (no merging).
        assert z.shards[t_idx] == z.chunks[t_idx]
        assert z.shards[c_idx] == z.chunks[c_idx]
        # z, y or x should still have grown, otherwise nothing was sharded.
        assert any(
            z.shards[i] != z.chunks[i] for i in range(len(z.chunks)) if i not in (t_idx, c_idx)
        )


def test_shard_exclude_axes_override_allows_t_and_c(tmp_path, image_pyramid, metadata):
    """Passing shard_exclude_axes=() restores the old grow-every-axis behavior."""
    out = tmp_path / "auto_allow_tc.zarr"

    writer = mm.ArrayManager(image_pyramid, metadata)
    writer.to_zarr(
        str(out),
        ngff_version="0.5",
        zarr_format=3,
        shards="auto",
        shard_target_mb=1000,
        shard_exclude_axes=(),
        overwrite=True,
    )

    root = zarr.open_group(str(out), mode="r")
    axes = "tczyx"
    t_idx, c_idx = axes.index("t"), axes.index("c")
    z0 = root["0"]
    assert z0.shards[t_idx] > z0.chunks[t_idx]
    assert z0.shards[c_idx] > z0.chunks[c_idx]


def test_chunks_override_rechunks_before_writing(tmp_path, image_pyramid, metadata):
    """`chunks=` controls the on-disk chunk shape, independent of the input dask chunking."""
    out = tmp_path / "custom_chunks.zarr"

    # image_pyramid level 0 is chunked (1,1,2,8,8); ask for a coarser chunk.
    writer = mm.ArrayManager(image_pyramid, metadata)
    writer.to_zarr(
        str(out),
        ngff_version="0.5",
        zarr_format=3,
        chunks=(2, 2, 4, 16, 16),
        overwrite=True,
    )

    root = zarr.open_group(str(out), mode="r")
    assert root["0"].chunks == (2, 2, 4, 16, 16)

    # Data must still round-trip correctly after rechunking.
    d = mm.ZarrManager(str(out), mode="r")
    np.testing.assert_array_equal(np.asarray(d.data[0]), image_pyramid[0].compute())


def test_chunks_per_level_list(tmp_path, image_pyramid, metadata):
    out = tmp_path / "custom_chunks_per_level.zarr"

    per_level = [
        (1, 1, 4, 16, 16),
        (1, 1, 2, 8, 8),
        (1, 1, 1, 4, 4),
    ]
    writer = mm.ArrayManager(image_pyramid, metadata)
    writer.to_zarr(
        str(out),
        ngff_version="0.5",
        zarr_format=3,
        chunks=per_level,
        overwrite=True,
    )

    root = zarr.open_group(str(out), mode="r")
    for i, expected in enumerate(per_level):
        assert root[str(i)].chunks == tuple(expected)


def test_chunks_clip_to_level_extent(tmp_path, image_pyramid, metadata):
    """An oversized chunk request is clipped to each level's own shape."""
    out = tmp_path / "custom_chunks_clip.zarr"

    writer = mm.ArrayManager(image_pyramid, metadata)
    writer.to_zarr(
        str(out),
        ngff_version="0.5",
        zarr_format=3,
        chunks=(1000, 1000, 1000, 1000, 1000),
        overwrite=True,
    )

    root = zarr.open_group(str(out), mode="r")
    for i, shape in enumerate([(2, 2, 4, 16, 16), (2, 2, 2, 8, 8), (2, 2, 1, 4, 4)]):
        assert root[str(i)].chunks == tuple(shape)


def test_no_chunks_override_keeps_existing_chunking(tmp_path, image_pyramid, metadata):
    out = tmp_path / "no_chunks_override.zarr"

    writer = mm.ArrayManager(image_pyramid, metadata)
    writer.to_zarr(str(out), ngff_version="0.5", zarr_format=3, overwrite=True)

    root = zarr.open_group(str(out), mode="r")
    for i, expected in enumerate(metadata["chunksize"]):
        assert root[str(i)].chunks == tuple(expected)


def test_shards_auto_by_default(tmp_path, image_pyramid, metadata):
    """Omitting `shards` is the same as shards="auto"."""
    out = tmp_path / "default_shards.zarr"

    writer = mm.ArrayManager(image_pyramid, metadata)
    writer.to_zarr(str(out), ngff_version="0.5", zarr_format=3, overwrite=True)

    root = zarr.open_group(str(out), mode="r")
    for i in range(3):
        assert root[str(i)].shards is not None


def test_shards_none_disables_sharding(tmp_path, image_pyramid, metadata):
    out = tmp_path / "no_shards.zarr"

    writer = mm.ArrayManager(image_pyramid, metadata)
    writer.to_zarr(str(out), ngff_version="0.5", zarr_format=3, overwrite=True, shards=None)

    root = zarr.open_group(str(out), mode="r")
    for i in range(3):
        assert root[str(i)].shards is None


def test_auto_shards_reduce_file_count(tmp_path, image_pyramid, metadata):
    out = tmp_path / "auto_shards.zarr"

    writer = mm.ArrayManager(image_pyramid, metadata)
    writer.to_zarr(
        str(out),
        ngff_version="0.5",
        zarr_format=3,
        shards="auto",
        shard_target_mb=1,
        overwrite=True,
    )

    root = zarr.open_group(str(out), mode="r")
    for i in range(3):
        z = root[str(i)]
        # Every axis of a resolved shard must be a whole multiple of the chunk.
        assert z.shards is not None
        for shard_dim, chunk_dim in zip(z.shards, z.chunks):
            assert shard_dim % chunk_dim == 0

    # Data still round-trips correctly through the high-level reader.
    d = mm.ZarrManager(str(out), mode="r")
    np.testing.assert_array_equal(np.asarray(d.data[0]), image_pyramid[0].compute())


def test_auto_shards_reported_in_read_metadata(tmp_path, image_pyramid, metadata):
    """metadata['shards'] should reflect the actual on-disk shard shapes."""
    out = tmp_path / "auto_shards_metadata.zarr"

    writer = mm.ArrayManager(image_pyramid, metadata)
    writer.to_zarr(
        str(out),
        ngff_version="0.5",
        zarr_format=3,
        shards="auto",
        shard_target_mb=1,
        overwrite=True,
    )

    d = mm.ZarrManager(str(out), mode="r")
    root = zarr.open_group(str(out), mode="r")

    reported = d.metadata["shards"]
    assert reported is not None
    assert len(reported) == 3
    for i, shard_shape in enumerate(reported):
        assert shard_shape is not None
        assert tuple(shard_shape) == tuple(root[str(i)].shards)


def test_unsharded_read_metadata_reports_none(tmp_path, image_pyramid, metadata):
    out = tmp_path / "no_shards_metadata.zarr"

    writer = mm.ArrayManager(image_pyramid, metadata)
    writer.to_zarr(str(out), ngff_version="0.5", zarr_format=3, overwrite=True, shards=None)

    d = mm.ZarrManager(str(out), mode="r")
    assert d.metadata["shards"] is None


def test_explicit_shard_shape_snaps_to_chunk_multiple(tmp_path, image_pyramid, metadata):
    out = tmp_path / "explicit_shards.zarr"

    writer = mm.ArrayManager(image_pyramid, metadata)
    # Level 0 chunks are (1,1,2,8,8); ask for a shard 3x too small in y/x and
    # oversized in z, it should still snap to a valid multiple of the chunk.
    writer.to_zarr(
        str(out),
        ngff_version="0.5",
        zarr_format=3,
        shards=(2, 2, 100, 10, 10),
        overwrite=True,
    )

    root = zarr.open_group(str(out), mode="r")
    z0 = root["0"]
    assert z0.shards is not None
    for shard_dim, chunk_dim in zip(z0.shards, z0.chunks):
        assert shard_dim % chunk_dim == 0
    # z0 shape is (2,2,4,16,16); the oversized z request must be clipped to it.
    assert z0.shards[2] <= z0.shape[2]


def test_per_level_shard_shapes(tmp_path, image_pyramid, metadata):
    out = tmp_path / "per_level_shards.zarr"

    per_level = [
        (2, 2, 4, 16, 16),
        (2, 2, 2, 8, 8),
        (2, 2, 1, 4, 4),
    ]
    writer = mm.ArrayManager(image_pyramid, metadata)
    writer.to_zarr(
        str(out),
        ngff_version="0.5",
        zarr_format=3,
        shards=per_level,
        overwrite=True,
    )

    root = zarr.open_group(str(out), mode="r")
    for i, expected_shape in enumerate(per_level):
        z = root[str(i)]
        assert z.shards is not None
        assert tuple(z.shards) == tuple(expected_shape)


@pytest.mark.parametrize("kwargs", [{}, {"shards": "auto"}, {"shards": (1, 1, 2, 8, 8)}])
def test_shards_with_zarr_v2_warn_and_are_disabled(tmp_path, image_pyramid, metadata, kwargs):
    out = tmp_path / "v2_shards.zarr"

    writer = mm.ArrayManager(image_pyramid, metadata)
    with pytest.warns(UserWarning, match="set to None"):
        writer.to_zarr(str(out), ngff_version="0.4", zarr_format=2, overwrite=True, **kwargs)

    root = zarr.open_group(str(out), mode="r")
    assert root["0"].metadata.zarr_format == 2


def test_no_warning_for_zarr_v2_when_sharding_disabled(tmp_path, image_pyramid, metadata):
    import warnings

    out = tmp_path / "v2_no_shards.zarr"
    writer = mm.ArrayManager(image_pyramid, metadata)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        writer.to_zarr(str(out), ngff_version="0.4", zarr_format=2, overwrite=True, shards=None)


def test_invalid_shards_string_rejected(tmp_path, image_pyramid, metadata):
    out = tmp_path / "bad_shards_value.zarr"

    writer = mm.ArrayManager(image_pyramid, metadata)
    with pytest.raises(ValueError, match="Unsupported shards value"):
        writer.to_zarr(
            str(out),
            ngff_version="0.5",
            zarr_format=3,
            shards="yes-please",
            overwrite=True,
        )


def test_create_empty_dataset_with_auto_shards(tmp_path):
    """metadata['shards'] drives sharding when creating an empty pyramid."""
    out = tmp_path / "empty_with_shards.zarr"

    metadata = {
        "axes": "cyx",
        "size": [(3, 1024, 1024), (3, 512, 512), (3, 256, 256)],
        "chunksize": [(1, 256, 256), (1, 256, 256), (1, 256, 256)],
        "dtype": "uint16",
        "scales": [(1.0, 1.0), (2.0, 2.0), (4.0, 4.0)],
        "shards": "auto",
        "shard_target_mb": 1,
    }

    manager = mm.ZarrManager(str(out), mode="w", metadata=metadata)
    assert manager.raw.metadata["zarr_format"] == 3

    root = zarr.open_group(str(out), mode="r")
    z0 = root["0"]
    assert z0.shards is not None
    for shard_dim, chunk_dim in zip(z0.shards, z0.chunks):
        assert shard_dim % chunk_dim == 0


def test_auto_shard_skips_when_chunk_already_large(tmp_path, image_pyramid, metadata):
    """A tiny shard_target_mb (smaller than one chunk) must not force sharding."""
    out = tmp_path / "auto_shards_skip.zarr"

    writer = mm.ArrayManager(image_pyramid, metadata)
    writer.to_zarr(
        str(out),
        ngff_version="0.5",
        zarr_format=3,
        shards="auto",
        shard_target_mb=1e-6,
        overwrite=True,
    )

    root = zarr.open_group(str(out), mode="r")
    for i in range(3):
        assert root[str(i)].shards is None


@pytest.mark.parametrize(
    "shape,chunk,target_mb,expect_none",
    [
        # Single chunk already covers the whole level -> nothing to gain.
        ((1, 8, 8), (1, 8, 8), 64, True),
        # Many small chunks and a generous budget -> shard shape should grow.
        ((1, 512, 512), (1, 16, 16), 4, False),
    ],
)
def test_resolve_shards_for_levels_auto(shape, chunk, target_mb, expect_none):
    result = _resolve_shards_for_levels(
        [shape],
        [chunk],
        np.dtype("uint16"),
        "auto",
        zarr_format=3,
        target_bytes=int(target_mb * 1024 * 1024),
    )
    assert len(result) == 1
    if expect_none:
        assert result[0] is None
    else:
        assert result[0] is not None
        for shard_dim, chunk_dim, shape_dim in zip(result[0], chunk, shape):
            assert shard_dim % chunk_dim == 0
            assert shard_dim <= shape_dim
