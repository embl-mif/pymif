from __future__ import annotations

from typing import Any

import dask.array as da
import zarr

from .axes import normalize_axes, normalize_data_type
from .ngff import (
    ZarrWriteConfig,
    _build_multiscales,
    _build_omero_metadata,
    _create_level_arrays,
    _resolve_creation_shards,
    _infer_ngff_version,
    _register_label_on_labels_group,
    _resolve_format,
    _set_group_ngff_metadata,
    _set_dimension_names,
    _validate_metadata,
)

def _without_axis(values, axis_index: int):
    return tuple(v for i, v in enumerate(values) if i != axis_index)

def _metadata_for_legacy_label(metadata: dict[str, Any], requested_data_type: str, is_label: bool, explicit_data_type: str | None) -> dict[str, Any]:
    """Keep old ``is_label=True`` calls working by dropping an image C axis.

    New callers can request channelled labels explicitly with
    ``data_type='label'`` and metadata whose axes include ``c``.  Old PyMIF
    examples passed the intensity image metadata plus ``is_label=True``; those
    should still create ``tzyx`` labels.
    """
    out = dict(metadata)
    axes = normalize_axes(out.get("axes"), ndim=len(out["size"][0]))
    metadata_type = normalize_data_type(out.get("data_type"))
    legacy_label_request = (
        is_label
        and requested_data_type == "label"
        and explicit_data_type is None
        and metadata_type == "intensity"
        and "c" in axes
    )
    if not legacy_label_request:
        return out

    c_axis = axes.index("c")
    out["axes"] = "".join(ax for ax in axes if ax != "c")
    out["size"] = [_without_axis(tuple(size), c_axis) for size in out["size"]]
    out["chunksize"] = [_without_axis(tuple(chunks), c_axis) for chunks in out["chunksize"]]
    if out.get("shards") is not None and not isinstance(out["shards"], str):
        shards = out["shards"]
        out["shards"] = (
            [None if s is None else _without_axis(tuple(s), c_axis) for s in shards]
            if isinstance(shards[0], (list, tuple))
            else _without_axis(tuple(shards), c_axis)
        )
    out["channel_names"] = []
    out["channel_colors"] = []
    return out

def create_empty_group(
    root: zarr.Group,
    group_name: str,
    metadata: dict[str, Any],
    is_label: bool = False,
    ngff_version: str | None = None,
    zarr_format: int | None = None,
    compressor: str | None = None,
    compressor_level: int = 3,
    data_type: str | None = None,
    shards: Any = None,
    shard_target_mb: float = 1024.0,
    shard_exclude_axes: Any = None,
):
    """Create an empty image subgroup or label subgroup inside an existing root.

    The subgroup inherits the root NGFF/zarr version so the hierarchy stays
    internally consistent. When ``is_label`` is ``True`` the group is created
    below ``labels/`` and the root label registry is updated.
    """
    if not metadata:
        raise ValueError("Metadata is required to create an empty group.")

    root_ngff = _infer_ngff_version(root)
    root_zarr = 3 if root_ngff == "0.5" else 2

    if ngff_version is not None and ngff_version != root_ngff:
        raise ValueError(
            f"Cannot create a group with ngff_version={ngff_version} inside a root "
            f"dataset with ngff_version={root_ngff}."
        )

    if zarr_format is not None and zarr_format != root_zarr:
        raise ValueError(
            f"Cannot create a group with zarr_format={zarr_format} inside a root "
            f"dataset with zarr_format={root_zarr}."
        )
    
    requested_data_type = normalize_data_type(data_type or metadata.get("data_type"), is_label=True if is_label else None)
    is_label = is_label or requested_data_type == "label"
    metadata = _metadata_for_legacy_label(metadata, requested_data_type, is_label, data_type)

    cfg = ZarrWriteConfig(
        ngff_version=root_ngff,
        zarr_format=root_zarr,
        compressor=compressor,
        compressor_level=compressor_level,
    )
    ngff_version, zarr_format = _resolve_format(cfg)

    parent = root.require_group("labels") if is_label else root
    if group_name in parent:
        del parent[group_name]
    grp = parent.create_group(group_name)

    sizes = [tuple(s) for s in metadata["size"]]
    chunks = [tuple(c) for c in metadata["chunksize"]]
    dtype = metadata.get("dtype", "uint16")
    axes = normalize_axes(metadata.get("axes"), ndim=len(sizes[0]))

    effective_metadata = dict(metadata)
    effective_metadata["axes"] = "".join(axes)
    effective_metadata["data_type"] = requested_data_type

    dummy_levels = [da.empty(shape=s, dtype=dtype, chunks=c) for s, c in zip(sizes, chunks)]
    _validate_metadata(dummy_levels, effective_metadata, axes)

    shard_shapes = _resolve_creation_shards(
        effective_metadata, sizes, chunks, axes,
        zarr_format=zarr_format,
        shards=shards,
        shard_target_mb=shard_target_mb,
        shard_exclude_axes=shard_exclude_axes,
    )
    _create_level_arrays(
        grp, sizes, chunks, dtype, shard_shapes,
        zarr_format=zarr_format, compressor=compressor, compressor_level=compressor_level,
    )

    data_type = normalize_data_type(effective_metadata.get("data_type"))
    multiscales = _build_multiscales(effective_metadata, axes, group_name)

    _set_dimension_names(grp, multiscales["datasets"], axes, zarr_format=zarr_format)

    if data_type == "label":
        _set_group_ngff_metadata(
            grp,
            ngff_version=ngff_version,
            multiscales=multiscales,
            omero=None,
            data_type=data_type,
            extra={"image-label": {"source": {"image": "../../"}}},
        )
        _register_label_on_labels_group(root, group_name, ngff_version)
    else:
        omero = _build_omero_metadata(dummy_levels[0], axes, effective_metadata)
        _set_group_ngff_metadata(
            grp,
            ngff_version=ngff_version,
            multiscales=multiscales,
            omero=omero,
            data_type=data_type,
            extra=None,
        )

    return grp
