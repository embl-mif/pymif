from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence

import dask.array as da
import numpy as np
import zarr

from .axes import normalize_axes, normalize_data_type, spatial_axes_in_order
from .ngff import (
    ZarrWriteConfig,
    _build_multiscales,
    _build_omero_metadata,
    _resolve_format,
    _set_group_ngff_metadata,
    _set_dimension_names,
    _validate_metadata,
    _write_pyramid_v2,
    _write_pyramid_v3,
)

_DROPPABLE_AXES = ("t", "c", "z")


def _drop_axes_from_config_shapes(value, keep: list[int], ndim: int):
    """Drop entries of a chunks/shards spec that refer to removed axes.

    Handles a single shape or one shape per level; anything that does not look
    like a full-``ndim`` shape (``None``, ``"auto"``) is returned unchanged.
    """
    if value is None or isinstance(value, str):
        return value
    seq = list(value)
    if seq and all(isinstance(v, (int, np.integer)) for v in seq):
        return tuple(seq[i] for i in keep) if len(seq) == ndim else value
    return [
        tuple(s[i] for i in keep) if len(s) == ndim else s
        for s in seq
    ]


def _drop_singleton_axes(
    data_levels: Sequence[da.Array],
    metadata: dict,
    config: ZarrWriteConfig,
) -> tuple[list[da.Array], dict, ZarrWriteConfig]:
    """Remove singleton ``t``/``c``/``z`` axes from the data and metadata.

    ``y`` and ``x`` are never dropped. Returns the inputs unchanged when
    ``config.drop_singleton`` is false or no axis qualifies.
    """
    data_levels = list(data_levels)
    if not config.drop_singleton or not data_levels:
        return data_levels, metadata, config

    ndim = data_levels[0].ndim
    axes = normalize_axes(metadata.get("axes"), ndim=ndim)
    drop = [
        i for i, ax in enumerate(axes)
        if ax in _DROPPABLE_AXES and all(arr.shape[i] == 1 for arr in data_levels)
    ]
    if not drop:
        return data_levels, metadata, config

    keep = [i for i in range(ndim) if i not in drop]
    dropped_axes = {axes[i] for i in drop}
    new_axes = "".join(axes[i] for i in keep)

    new_levels = [arr.reshape(tuple(arr.shape[i] for i in keep)) for arr in data_levels]

    out = dict(metadata)
    out["axes"] = new_axes
    for key in ("size", "chunksize"):
        if out.get(key) is not None:
            out[key] = [
                tuple(v[i] for i in keep) if len(v) == ndim else tuple(v)
                for v in out[key]
            ]
    # scales/units only cover spatial axes, so drop the matching positions.
    spatial = spatial_axes_in_order(axes)
    keep_spatial = [i for i, ax in enumerate(spatial) if ax not in dropped_axes]
    if len(keep_spatial) != len(spatial):
        if out.get("scales") is not None:
            out["scales"] = [tuple(s[i] for i in keep_spatial) for s in out["scales"]]
        if out.get("units"):
            out["units"] = tuple(out["units"][i] for i in keep_spatial)
    if "t" in dropped_axes:
        out.pop("time_increment", None)
        out.pop("time_increment_unit", None)

    new_config = replace(
        config,
        chunks=_drop_axes_from_config_shapes(config.chunks, keep, ndim),
        shards=_drop_axes_from_config_shapes(config.shards, keep, ndim),
        shard_exclude_axes=tuple(
            ax for ax in (config.shard_exclude_axes or ()) if ax not in dropped_axes
        ),
    )
    return new_levels, out, new_config


def _metadata_for_write(
    metadata: dict,
    axes: tuple[str, ...],
    *,
    config: ZarrWriteConfig,
    is_label: bool | None = None,
) -> dict:
    out = dict(metadata)
    data_type = normalize_data_type(config.data_type or metadata.get("data_type"), is_label=is_label)
    out["axes"] = "".join(axes)
    out["data_type"] = data_type
    return out

def to_zarr(
    path: str | Path,
    data_levels: Sequence[da.Array],
    metadata: dict,
    *,
    config: ZarrWriteConfig | None = None,
):
    """Write a pyramid of dask arrays to an OME-Zarr root group.

    Parameters
    ----------
    path : str | Path
        Destination zarr store.
    data_levels : sequence of dask.array.Array
        Pyramid levels ordered from finest to coarsest resolution.
    metadata : dict
        Normalized PyMIF metadata dictionary describing axes, scales, channel
        metadata and units.
    config : ZarrWriteConfig | None
        Output configuration controlling NGFF version, zarr format, overwrite
        behaviour and compression.
    """
    cfg = config or ZarrWriteConfig()
    if not data_levels:
        raise ValueError("data_levels cannot be empty.")

    data_levels, metadata, cfg = _drop_singleton_axes(data_levels, metadata, cfg)
    axes = normalize_axes(metadata.get("axes"), ndim=data_levels[0].ndim)
    effective_metadata = _metadata_for_write(metadata, axes, config=cfg)
    _validate_metadata(data_levels, effective_metadata, axes)

    ngff_version, zarr_format = _resolve_format(cfg)
    root = zarr.open_group(
        str(Path(path)),
        mode="w" if cfg.overwrite else "w-",
        zarr_format=zarr_format,
    )

    if zarr_format == 3:
        delayed = _write_pyramid_v3(root=root, data_levels=data_levels, cfg=cfg, axes=axes, desc=Path(path).name)
    else:
        delayed = _write_pyramid_v2(root=root, data_levels=data_levels, cfg=cfg, desc=Path(path).name)

    multiscales = _build_multiscales(
        effective_metadata, axes, effective_metadata.get("name") or "dataset"
    )

    _set_dimension_names(root, multiscales["datasets"], axes, zarr_format=zarr_format)

    data_type = normalize_data_type(effective_metadata.get("data_type"))
    omero = None
    extra = None
    if data_type == "intensity":
        omero = _build_omero_metadata(data_levels[0], axes, effective_metadata)
    else:
        extra = {"image-label": {}}  # standalone label store: no source image

    _set_group_ngff_metadata(
        root,
        ngff_version=ngff_version,
        multiscales=multiscales,
        omero=omero,
        data_type=data_type,
        extra=extra,
    )

    return root if cfg.compute else delayed


def write_multiscale_to_group(
    group: zarr.Group,
    data_levels: Sequence[da.Array],
    metadata: dict,
    *,
    config: ZarrWriteConfig | None = None,
    name: str | None = None,
    is_label: bool = False,
):
    """Write a pyramid of Dask arrays into an existing zarr group.

    Used by :class:`pymif.microscope_manager.ZarrManager` for raw data, image
    subgroups and label groups.  The axes may be any subset of ``tczyx``.
    """
    cfg = config or ZarrWriteConfig()
    if not data_levels:
        raise ValueError("data_levels cannot be empty.")

    data_levels, metadata, cfg = _drop_singleton_axes(data_levels, metadata, cfg)
    axes = normalize_axes(metadata.get("axes"), ndim=data_levels[0].ndim)
    effective_metadata = _metadata_for_write(
        metadata,
        axes,
        config=cfg,
        is_label=True if is_label else None,
    )
    _validate_metadata(data_levels, effective_metadata, axes)

    ngff_version, zarr_format = _resolve_format(cfg)

    if cfg.overwrite:
        for key in list(group.array_keys()):
            if str(key).isdigit():
                del group[key]

    if zarr_format == 3:
        label = name or group.name.strip('/') or 'raw'
        delayed = _write_pyramid_v3(root=group, data_levels=data_levels, cfg=cfg, axes=axes, desc=label)
    else:
        label = name or group.name.strip('/') or 'raw'
        delayed = _write_pyramid_v2(root=group, data_levels=data_levels, cfg=cfg, desc=label)

    multiscales = _build_multiscales(
        effective_metadata, axes, name or effective_metadata.get("name") or "dataset"
    )

    _set_dimension_names(group, multiscales["datasets"], axes, zarr_format=zarr_format)

    data_type = normalize_data_type(effective_metadata.get("data_type"))
    omero = None
    extra = None
    if data_type == "intensity":
        omero = _build_omero_metadata(data_levels[0], axes, effective_metadata)
    else:
        # Only labels nested in ``<image>/labels/<name>`` have a source image.
        image_label = {"source": {"image": "../../"}} if is_label else {}
        extra = {"image-label": image_label}

    _set_group_ngff_metadata(
        group,
        ngff_version=ngff_version,
        multiscales=multiscales,
        omero=omero,
        data_type=data_type,
        extra=extra,
    )

    return group if cfg.compute else delayed
