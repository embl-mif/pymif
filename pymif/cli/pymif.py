from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import pymif.microscope_manager as mm
from pymif.cli.__arguments import (
    _parse_arguments,
    parse_bool,
    parse_color,
    parse_downscale_factor,
    parse_shard_exclude_axes_spec,
    parse_shards_spec,
    parse_subset_spec,
)


def _axes(metadata):
    return str(metadata.get("axes", "tczyx")).lower()


def _axis_size(metadata, axis, default=1):
    axes = _axes(metadata)
    return int(metadata["size"][0][axes.index(axis)]) if axis in axes else default


def _dataset_size_mb(metadata):
    dtype = np.dtype(metadata.get("dtype", "uint16"))
    nbytes = dtype.itemsize
    for size in metadata["size"][0]:
        nbytes *= int(size)
    return nbytes / 1024 / 1024


def _select_chunk_size(metadata, max_size):
    axes = _axes(metadata)
    size_map = dict(zip(axes, metadata["size"][0]))
    n_chunks = {"z": 2, "y": 1, "x": 1}
    chunk_map = {"t": 1, "c": 1}
    while True:
        chunk_map.update({
            "z": int(size_map.get("z", 1) / n_chunks["z"] + 1),
            "y": int(size_map.get("y", 1) / n_chunks["y"] + 1),
            "x": int(size_map.get("x", 1) / n_chunks["x"] + 1),
        })
        dtype_size = np.dtype(metadata.get("dtype", "uint16")).itemsize
        size_mb = dtype_size * chunk_map["z"] * chunk_map["y"] * chunk_map["x"] / 1024 / 1024
        if size_mb <= max_size:
            break
        n_chunks = {key: value * 2 for key, value in n_chunks.items()}
    chunks = tuple(int(max(1, min(size_map.get(ax, 1), chunk_map[ax]))) for ax in axes)
    return chunks, size_mb, n_chunks


def _check_chunk_size(chunk_size, metadata):
    """Return ``chunk_size`` as a tuple, or raise if it does not have one value per dataset axis."""
    axes = _axes(metadata)
    chunk_size = tuple(int(c) for c in chunk_size)
    if len(chunk_size) != len(axes):
        raise ValueError(
            f"chunk_size={chunk_size} has {len(chunk_size)} values, but the dataset has "
            f"{len(axes)} axes ({axes}, shape {tuple(metadata['size'][0])}). "
            f"Provide one chunk size per axis, in {axes.upper()} order."
        )
    return chunk_size


def _estimate_levels(metadata, downscale_factor):
    axes = _axes(metadata)
    size_map = dict(zip(axes, metadata["size"][0]))
    shape = [int(size_map[ax]) for ax in axes if ax in "zyx"]
    if not shape:
        return 1
    n = 1
    print(f"Layer {n}, shape {shape}")
    while any(s > 2048 for s in shape):
        n += 1
        shape = [max(1, s // downscale_factor[i]) for i, s in enumerate(shape)]
        print(f"Layer {n}, shape {shape}")
    return n


def _normalize_downscale_factor(value):
    if value is None:
        return tuple([2] * 3)
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return tuple([int(value[0])] * 3)
        return tuple(int(v) for v in value)
    return tuple([int(value)] * 3)


def _present(value):
    return not (
        pd.isna(value)
        or (isinstance(value, str) and value.strip() == "")
        or value == "-1"
    )


def _normalize_shards(value):
    """Normalize a `shards` CLI/CSV/kwarg value into ``None``, ``"auto"`` or a tuple."""
    if value is None:
        return None
    if isinstance(value, str):
        return parse_shards_spec(value)
    if isinstance(value, (list, tuple)):
        return parse_shards_spec(value)
    raise TypeError(
        f"shards must be 'auto', a shard shape, or None; got {value!r}."
    )


def _normalize_shard_exclude_axes(value):
    """Normalize a `shard_exclude_axes` CLI/CSV/kwarg value into ``None`` or a tuple."""
    if value is None:
        return None
    if isinstance(value, (str, list, tuple)):
        return parse_shard_exclude_axes_spec(value)
    raise TypeError(
        f"shard_exclude_axes must be a sequence of axis names, 'none', or None; got {value!r}."
    )


def _normalize_subset(value):
    """Normalize a subset specification from the CLI or a batch CSV."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k).upper(): v for k, v in value.items()}
    if isinstance(value, str):
        parsed = parse_subset_spec(value)
        return None if parsed is None else parsed
    raise TypeError(
        "subset must be either a dict of axis selections or a compact subset string"
    )

def _manager_name_map() -> Dict[str, Any]:
    return {
        'luxendo': mm.LuxendoManager,
        'opera': mm.OperaManager,
        'viventis': mm.ViventisManager,
        'zeiss': mm.ZeissManager,
        'evident': mm.EvidentManager,
        'zarrv04': mm.ZarrV04Manager,
        'zarr': mm.ZarrManager,
        'scape': mm.ScapeManager,
    }

def _resolve_zarr_manager(input_path: str, microscope: Optional[str] = None):
    """Resolve a manager class from an explicit microscope name or by inspecting the path."""

    if _present(microscope):
        key = str(microscope).strip().lower()
        manager = _manager_name_map().get(key)
        if manager is None:
            raise TypeError(
                f'Microscope {microscope} not recognized. Should be one of '
                f'"luxendo", "opera", "viventis", "zeiss", "evident", "zarrv04", "zarr", "scape".'
            )
        return manager, key

    path = Path(input_path)

    # Zarr stores: prefer explicit v3/v0.5 when zarr.json exists, otherwise legacy v0.4.
    if path.is_dir():
        if (path / 'zarr.json').exists():
            return mm.ZarrManager, 'zarr'
        if (path / '.zgroup').exists() or (path / '.zattrs').exists():
            return mm.ZarrManager, 'zarrv04'

        if any(path.glob('*.lux.h5')) and any(path.glob('*.xml')):
            return mm.LuxendoManager, 'luxendo'

        if any(path.glob('*.ome')) and any(path.glob('*.tif')):
            return mm.ViventisManager, 'viventis'

    # File-based formats.
    suffixes = ''.join(path.suffixes).lower()
    if suffixes.endswith('.czi'):
        return mm.ZeissManager, 'zeiss'

    if suffixes.endswith('.vsi'):
        return mm.EvidentManager, 'evident'

    if suffixes.endswith('.ome.tif') or suffixes.endswith('.ome.tiff') or path.suffix.lower() in {'.tif', '.tiff'}:
        metadata_dir = path.parent / 'Metadata'
        if metadata_dir.exists() and any(metadata_dir.glob('*.xlif')):
            return mm.ScapeManager, 'scape'
        return mm.OperaManager, 'opera'

    raise TypeError(
        'Could not auto-detect the microscope from the input path. '
        'Pass --microscope explicitly with one of "luxendo", "opera", "viventis", "zeiss", "evident", "zarrv04", "zarr", "scape".'
    )

def zarr_convert(
    input_path, 
    zarr_path, 
    microscope: Optional[str] = None, 
    max_size : Optional[int] = 100, 
    chunk_size : Optional[List[int]] = None,
    scene_index : Optional[int] = None,
    channel_names : Optional[List[str]] = None,
    channel_colors : Optional[List[str]] = None,
    zarr_format : Optional[int] = 3,
    downscale_factor: Optional[int] = 2,
    num_levels: Optional[int] = None,
    subset: Optional[dict] = None,
    shards: Optional[Any] = "auto",
    shard_target_mb: Optional[float] = None,
    shard_exclude_axes: Optional[Any] = None,
    drop_singleton: Optional[bool] = True,
    num_workers: Optional[int] = None,
):
    """Helper function for CLI to convert a dataset to zarr given some parameters.

    Parameters
    ----------
        input_path : str
            Input path for the data to be converted.
        zarr_path : str
            Output .zarr path.
        microscope : str
            Microscope used to acquire input data.
            One of \"luxendo\", \"opera\", \"viventis\", \"zeiss\", \"evident\", \"zarrv04\", \"zarr\".
        max_size : Optional[int]
            Max chunk size in MB. \n
            Default: 100
        chunk_size : Optional[List[int]]
            Chunk size in TCZYX format, or whatever axes are present in the dataset.
            Default: None
        scene_index : Optional[int]
            Scene index for .czi and .vsi files. \n
            Default: None (scene 1 for .czi, scene 0 for .vsi)
        channel_names : Optional[List[str]]
            Name of channels.\n
            Example: \"-cn bf gfp rfp\"\n
            Default: None
        channel_colors : Optional[List[str]]
            Colors of channels (hex or matplotlib color name)\n
            Example: \"-cc 0000FF cyan 00ff00\")\n
            Default: None
        zarr_format : Optional[int]
            Output zarr format. Zarr v2 maps to NGFF 0.4 and Zarr v3 maps to NGFF 0.5.\n
            Default: 3
        downscale_factor : Optional[int]
            Pyramid downsampling factor.\n
            Example: \"-df 1 2 2\")\n
            Default: 2
        num_levels : Optional[int]
            Number of pyramid levels.\n
            Example: \"-nl 3\")\n
            Default: None
        subset : Optional[dict]
            Axis subset to apply before chunk selection and pyramid generation.\n
            Example: "--subset y=10:100:2;x=20:80"\n
            Default: None
        shards : Optional[str | List[int]]
            Zarr v3 sharding for the output.

            "auto" sizes shards automatically; a shape (matching the dataset's
            axes) requests that shard shape explicitly; "none" disables sharding.
            With zarr_format=2 sharding is set to none with a warning.

            Example: \"-sh auto\", \"-sh 1 1 8 2160 4096\" or \"-sh none\"

            Default: "auto"
        shard_target_mb : Optional[float]
            Target uncompressed shard size in MB, used when shards="auto".\n
            Default: 1024 (pymif's own default; unset here means "let pymif choose")
        shard_exclude_axes : Optional[List[str]]
            Axes that shards="auto" never merges chunks along.\n
            Example: \"-sea t c\" (default) or \"-sea none\" to allow every axis to merge.\n
            Default: None (falls back to pymif's own default of t, c)
        drop_singleton : Optional[bool]
            Remove singleton t/c/z axes from the output dataset.\n
            Example: \"-ds false\" to keep them.\n
            Default: True
        num_workers : Optional[int]
            Number of threads used to write the zarr in parallel.

            Example: \"-nw 4\"

            Default: None (half of the available cores, capped at 8)
    """

    manager, resolved_microscope = _resolve_zarr_manager(input_path, microscope)
    print(f'\n--->Using manager: {manager.__name__} ({resolved_microscope})')

    downscale_factor = _normalize_downscale_factor(downscale_factor)
    
    # --- Figure out chunks dimensions ---
    if resolved_microscope.lower() == "zeiss":
        dataset = manager(path=input_path, scene_index=1 if scene_index is None else scene_index)
    elif resolved_microscope.lower() == "evident":
        dataset = manager(path=input_path, scene_index=0 if scene_index is None else scene_index)
    else:
        dataset = manager(path=input_path)
        
    # --- Show metadata summary ---
    print("\n--->Input dataset")
    for i in dataset.metadata:
        print(f"{i.upper()}: {dataset.metadata[i]}")
    print("CHUNK SIZE:", dataset.chunks)
    print("DATASET SIZE (MB):", _dataset_size_mb(dataset.metadata))

    subset_kwargs = _normalize_subset(subset)
    if subset_kwargs:
        print(f"\n--->Applying subset: {subset_kwargs}")
        dataset.subset_dataset(**subset_kwargs, rebuild_pyramid=False)
        print("\n--->Dataset after subsetting")
        for i in dataset.metadata:
            print(f"{i.upper()}: {dataset.metadata[i]}")
        print("DATASET SIZE (MB):", _dataset_size_mb(dataset.metadata))
        
    # --- Select chunk size ---
    print(f"\n--->Select chunks.")
    if chunk_size is not None:
        chunk_size = _check_chunk_size(chunk_size, dataset.metadata)
        print(f"\tUsing user-provided chunk size: {chunk_size} ({_axes(dataset.metadata)})")
        size_mb = _dataset_size_mb(dataset.metadata) * np.prod(chunk_size) / np.prod(dataset.metadata["size"][0])
        n_chunks = {ax: int(np.ceil(size / chunk)) for ax, size, chunk in zip(_axes(dataset.metadata), dataset.metadata["size"][0], chunk_size)}
    else:
        print(f"\tUsing max_size={max_size} MB to select chunk size.")
        chunk_size, size_mb, n_chunks = _select_chunk_size(dataset.metadata, max_size)

    print(f"Chunk size: {chunk_size}, {size_mb} MB.")
    print(f"N chunks: {n_chunks}.")

    dataset.data = [
        arr.rechunk(chunk_size) if arr.ndim == len(chunk_size) else arr
        for arr in dataset.data
    ]
    dataset.metadata["chunksize"] = [tuple(arr.chunksize) for arr in dataset.data]
    if hasattr(dataset, "chunks"):
        dataset.chunks = chunk_size

    # --- Build pyramid ---
    print(f"\n--->Selected pyramidal layers, lower layer should have dims<2048")
    if num_levels is None:
        num_levels = _estimate_levels(dataset.metadata, downscale_factor)

    dataset.build_pyramid(
        num_levels=num_levels, 
        downscale_factor=downscale_factor
    )

    # --- Modify metadata according to optional parameters ---
    
    # Metadata format:
    # metadata = {
    #         # "size": [(size_t, size_c, size_z, size_y, size_x)], # can't change
    #         # "scales": scales, # can't change
    #         # "units": units, # can't change
    #         # "time_increment": time_increment, # can't change
    #         # "time_increment_unit": time_unit, # can't change
    #         "channel_names": channel_names,
    #         "channel_colors": channel_colors,
    #         # "dtype": pixels.attrib.get("Type", "uint16"), # can't change
    #         # "axes": "tczyx" # can't change
    #     }

    print("\n--->Updating metadata to selected channel_names and channel_colors")
    metadata = {}
    if channel_names:
        if "c" not in _axes(dataset.metadata):
            raise TypeError("channel_names were provided, but the dataset has no channel axis.")
        n_ch = _axis_size(dataset.metadata, "c")
        if len(channel_names)!=n_ch:
            raise TypeError(f"Length of channel_names={channel_names} does not match dataset channels of length={n_ch}.")
        metadata["channel_names"] = channel_names
        if channel_colors:
            if len(channel_colors)!=n_ch:
                raise TypeError(f"Length of channel_colors={channel_colors} does not match dataset channels of length={n_ch}.")
            metadata["channel_colors"] = channel_colors
    dataset.update_metadata(metadata)

    print("\n--->Updating metadata to selected zarr_format and downscale_factor")
    ngff_version = '0.4' if int(zarr_format) == 2 else '0.5'

    # --- Resolve sharding options ---
    shards = _normalize_shards(shards)
    shard_exclude_axes = _normalize_shard_exclude_axes(shard_exclude_axes)
    drop_singleton = True if drop_singleton is None else parse_bool(drop_singleton)
    to_zarr_kwargs = {
        "zarr_format": int(zarr_format),
        "ngff_version": ngff_version,
        "drop_singleton": drop_singleton,
        "shards": shards,
    }
    if num_workers is not None:
        to_zarr_kwargs["num_workers"] = int(num_workers)
    if shards is not None:
        if shard_target_mb is not None:
            to_zarr_kwargs["shard_target_mb"] = float(shard_target_mb)
        if shard_exclude_axes is not None:
            to_zarr_kwargs["shard_exclude_axes"] = shard_exclude_axes

    # --- Show metadata summary ---
    print("\n--->Input dataset after adjustments")
    for i in dataset.metadata:
        print(f"{i.upper()}: {dataset.metadata[i]}")
    print(f"CHUNK SIZE: {dataset.chunks} , {size_mb} MB.")
    print(f"N CHUNKS: {n_chunks}.")
    print(f"PYRAMID LEVELS: {num_levels}.")
    print(f"ZARR FORMAT: {zarr_format}, NGFF VERSION: {ngff_version}.")
    print(f"DROP_SINGLETON: {drop_singleton}.")
    print(f"NUM_WORKERS: {num_workers if num_workers is not None else '(pymif default)'}.")
    print(f"SHARDS: {shards}.")
    if shards is not None:
        print(f"SHARD_TARGET_MB: {to_zarr_kwargs.get('shard_target_mb', '(pymif default: 1024)')}.")
        print(f"SHARD_EXCLUDE_AXES: {shard_exclude_axes if shard_exclude_axes is not None else '(pymif default: t, c)'}.")

    # --- Write to OME-Zarr format ---
    print("\n--->Writing to zarr")
    dataset.to_zarr(zarr_path, **to_zarr_kwargs)

    # --- Show metadata summary for updated dataset ---
    dataset = mm.ZarrManager(path=zarr_path)

def _batch_row_kwargs(v, columns) -> Dict[str, Any]:
    """``zarr_convert`` options of one batch CSV row (shared by ``convert --batch`` and ``fuse --batch``)."""
    kwargs: Dict[str, Any] = {}

    # Prioritize chunk_size over max_size if both are present
    if "chunk_size" in columns and _present(v.get("chunk_size")):
        kwargs["chunk_size"] = parse_downscale_factor(v["chunk_size"])
    elif "max_size(MB)" in columns and _present(v.get("max_size(MB)")):
        kwargs["max_size"] = float(v["max_size(MB)"])

    if "scene_index" in columns and _present(v.get("scene_index")):
        kwargs["scene_index"] = int(v["scene_index"])

    if "channel_names" in columns and _present(v.get("channel_names")):
        kwargs["channel_names"] = [c.strip() for c in str(v["channel_names"]).split(" ")]

    if "channel_colors" in columns and _present(v.get("channel_colors")):
        kwargs["channel_colors"] = [parse_color(c.strip()) for c in str(v["channel_colors"]).split(" ")]

    if "zarr_format" in columns and _present(v.get("zarr_format")):
        kwargs["zarr_format"] = int(v["zarr_format"])

    if "downscale_factor" in columns and _present(v.get("downscale_factor")):
        kwargs["downscale_factor"] = parse_downscale_factor(v["downscale_factor"])

    if "num_levels" in columns and _present(v.get("num_levels")):
        kwargs["num_levels"] = int(v["num_levels"])

    for key in ("subset", "shards", "shard_exclude_axes"):
        if key in columns and _present(v.get(key)):
            kwargs[key] = v[key]

    if "shard_target_mb" in columns and _present(v.get("shard_target_mb")):
        kwargs["shard_target_mb"] = float(v["shard_target_mb"])

    if "drop_singleton" in columns and _present(v.get("drop_singleton")):
        kwargs["drop_singleton"] = parse_bool(v["drop_singleton"])

    if "num_workers" in columns and _present(v.get("num_workers")):
        kwargs["num_workers"] = int(v["num_workers"])

    return kwargs


#: Parsed arguments that are not options of the conversion functions.
_NOT_CONVERSION_OPTIONS = {"runmode", "mode", "command", "renamed_to", "batch", "input_file"}


def convert_batch(args):
    """Runmode to convert batch of imaged to zarr

    Args:
        args (args): parsed arguments
    """
    csv_path = getattr(args, "input_file", args)  # parsed arguments or a path
    print(f"Converting batch.\nRunning through: pymif convert --batch {csv_path}")

    database = pd.read_csv(csv_path)
    print(database)

    for i, v in database.iterrows():
        print("-" * 20)

        input_f = v["input"]
        output = v["output"]
        microscope = v["microscope"]

        conv_kwargs = {
            "input_path": input_f,
            "zarr_path": output,
            "microscope": microscope,
            **_batch_row_kwargs(v, database.columns),
        }

        zarr_convert(**conv_kwargs)

def convert_single(args):
    """Runmode to convert a single image to zarr

    Args:
        args (args): parsed arguments
    """
    cli = (
        f'pymif convert --input_path {args.input_path} --zarr_path {args.zarr_path} '
        f'--microscope {args.microscope} --max_size {args.max_size} '
        f'--scene_index {args.scene_index} --channel_names {args.channel_names} '
        f'--channel_colors {args.channel_colors} --zarr_format {args.zarr_format} '
        f'--num_levels {args.num_levels} --downscale_factor {args.downscale_factor} '
        f'--chunk_size {args.chunk_size} --subset {args.subset} '
        f'--shards {args.shards} --shard_target_mb {args.shard_target_mb} '
        f'--shard_exclude_axes {args.shard_exclude_axes} '
        f'--drop_singleton {args.drop_singleton} --num_workers {args.num_workers}'
    )
    print(f'Converting single file.\nRunning through: {cli}')
    exclude = _NOT_CONVERSION_OPTIONS
    conv_kwargs = {
        k: v
        for k, v in vars(args).items()
        if k not in exclude and v is not None and not (isinstance(v, str) and v.strip() == "")
    }
    if "chunk_size" in conv_kwargs:
        conv_kwargs.pop("max_size", None)

    # Convert 2 zarr
    zarr_convert(**conv_kwargs)


# ---------------------------------------------------------------------------
# Mosaics: fuse / fuse --batch
# ---------------------------------------------------------------------------

def _mosaic_manager_name_map() -> Dict[str, Any]:
    from pymif.mosaic_manager import MOSAIC_READERS

    return dict(MOSAIC_READERS)


def _resolve_mosaic_manager(input_path: str, microscope: Optional[str] = None):
    """Mosaic reader class from an explicit microscope name or by inspecting the folder.

    An explicit microscope that does not match a recognised folder is refused when
    the reader is created (:class:`~pymif.mosaic_manager.WrongReaderError`)."""
    from pymif.mosaic_manager import detect_microscope

    readers = _mosaic_manager_name_map()
    if _present(microscope):
        key = str(microscope).strip().lower()
        if key not in readers:
            raise TypeError(f'Mosaic microscope {microscope} not recognized. Should be one of {sorted(readers)}.')
        return readers[key], key
    key = detect_microscope(input_path)
    if key is not None:
        return readers[key], key
    raise TypeError(
        'Could not auto-detect the mosaic type from the input path. '
        f'Pass --microscope explicitly with one of {sorted(readers)}.'
    )


def _int_triplet(value, default=(1, 1, 1)) -> Tuple[int, int, int]:
    """``None`` -> default, ``n`` -> ``(n, n, n)``, ``(y, x)`` -> ``(1, y, x)``, ``(z, y, x)`` as is."""
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        value = parse_downscale_factor(value)
    if isinstance(value, (int, np.integer)):
        return (int(value),) * 3
    value = tuple(int(v) for v in value)
    if len(value) == 1:
        return value * 3
    if len(value) == 2:
        return (1,) + value
    if len(value) == 3:
        return value
    raise ValueError(f"Expected 1, 2 or 3 integers, got {value}.")


def _subset_to_crop(subset, channel_names: List[str]):
    """Split a convert-style subset into a fusion crop (z/y/x) and a channel selection (c)."""
    subset = _normalize_subset(subset) or {}
    if "T" in subset:
        raise ValueError("A mosaic has no time axis: remove t from --subset.")
    channels = list(channel_names)
    if "C" in subset:
        sel = subset["C"]
        if isinstance(sel, slice):
            channels = channels[sel]
        elif isinstance(sel, list):
            channels = [channels[i] for i in sel]
        else:
            channels = [channels[int(sel)]]
    crop = {}
    for axis in "ZYX":
        if axis in subset:
            sel = subset[axis]
            if isinstance(sel, list):
                raise ValueError(f"--subset {axis.lower()}: use a range such as 10:200 (index lists are not supported for mosaics).")
            crop[axis.lower()] = sel
    return crop or None, channels


def mosaic_convert(
    input_path,
    zarr_path,
    microscope: Optional[str] = None,
    max_size: Optional[float] = 100,
    chunk_size: Optional[List[int]] = None,
    scene_index: Optional[int] = None,
    channel_names: Optional[List[str]] = None,
    channel_colors: Optional[List[str]] = None,
    zarr_format: Optional[int] = 3,
    downscale_factor: Optional[Any] = 2,
    num_levels: Optional[int] = None,
    subset: Optional[Any] = None,
    shards: Optional[Any] = "auto",
    shard_target_mb: Optional[float] = None,
    shard_exclude_axes: Optional[Any] = None,
    drop_singleton: Optional[bool] = True,
    num_workers: Optional[int] = None,
    read_downsample: Optional[Any] = None,
    projection: Optional[str] = "none",
    blend: str = "sigmoid",
    sigmoid_steepness: Optional[float] = None,
    channels: Optional[List[str]] = None,
    groups: Optional[List[str]] = None,
    fields: Optional[List[str]] = None,
    tiles: Optional[List[str]] = None,
    skip_tiles: Optional[List[str]] = None,
    midline_axis: str = "x",
    out_dtype: Optional[str] = None,
    read_workers: int = 4,
    temp_dir: Optional[str] = None,
    overwrite: bool = False,
    flatfield: bool = True,
):
    """Fuse a mosaic into an OME-Zarr image (CLI helper for ``pymif fuse``).

    The ``convert`` options (``max_size`` ... ``num_workers``) have the same meaning as in
    :func:`zarr_convert`, applied to the fused output; ``subset`` crops it (z/y/x in
    output voxels) and selects channels (c). ``tiles`` lists the only tiles to fuse
    (ids as for ``skip_tiles``); the output then fits them. ``fields`` keeps only the
    tiles of these fields (connected groups of touching tiles, ``field0``, ``field1``...);
    without ``tiles`` and ``fields`` the first field is fused. Tiles from several fields
    are not fused: an error is printed and ``False`` returned (``True`` once the output
    is saved). The other options are those of
    :func:`pymif.mosaic_manager.fuse_tiles`: ``read_downsample`` is its ``ds`` and
    ``read_workers`` its ``workers``. ``flatfield`` is passed to readers that correct
    the tiles (Opera); the others ignore it.
    """
    from pymif.mosaic_manager import fuse_tiles, plan_fusion, reader_options
    from pymif.mosaic_manager.fusion import SIGMOID_STEEPNESS

    reader, resolved = _resolve_mosaic_manager(input_path, microscope)
    print(f"\n--->Using mosaic manager: {reader.__name__} ({resolved})")
    if scene_index is not None and resolved != "zeiss":
        print(f"\tscene_index={scene_index} is ignored for {resolved} mosaics.")
    options = reader_options(reader, flatfield=flatfield)
    if not flatfield and "flatfield" not in options:
        print(f"\t--no_flatfield is ignored for {resolved} mosaics (no flat-field correction).")
    mosaic = reader(input_path, **options)
    print(f"{len(mosaic.tiles)} tiles, channels {mosaic.channels}, groups {mosaic.groups}, "
          f"fields {mosaic.fields}")
    if len(mosaic.fields) > 1:
        for f in mosaic.fields:
            stacks = sorted({mosaic.tile_id(t).split("/", 1)[1] for t in mosaic.field_tiles(f)})
            print(f"\t{f}: {len(stacks)} stack position(s): {' '.join(stacks)}")

    use = mosaic.fusion_tiles(list(tiles) if tiles else None, list(fields) if fields else None)
    if tiles:
        print(f"Fusing only {len(use)} tile(s) of {list(tiles)}")
    if fields:
        print(f"Fusing only fields {list(fields)}: {len(use)} tile(s)")
    elif not tiles and len(mosaic.fields) > 1:
        print(f"Fusing the first field, {mosaic.fields[0]} ({len(use)} tiles); choose another with --fields.")
    if groups:
        wanted = {str(g) for g in groups}
        use = [t for t in use if str(t.group) in wanted]
        if not use:
            raise ValueError(f"No tile in groups {sorted(wanted)}; available: {mosaic.groups}.")

    skip = mosaic.tiles_by_ids(list(skip_tiles)) if skip_tiles else None
    if skip:
        print(f"Leaving {len(skip)} tile(s) blank: {[mosaic.tile_id(t) for t in skip]}")

    ds = _int_triplet(read_downsample)
    projection = None if projection in (None, "none", "None", "") else projection
    crop, selected = _subset_to_crop(subset, channels or list(dict.fromkeys(t.channel for t in use)))
    plan = plan_fusion(use, ds=ds, projection=projection, blend=blend, channels=selected,
                       crop=crop, skip=skip, out_dtype=out_dtype, midline_axis=midline_axis)
    print(f"\n--->Output: {plan.summary()}")
    if plan.error:
        print(f"ERROR: {plan.error}")
        return False

    plan_md = {"axes": plan.axes, "size": [plan.shape], "dtype": str(plan.dtype)}
    if chunk_size is not None:
        chunks = tuple(int(c) for c in chunk_size)
        print(f"\tUsing user-provided chunk size: {chunks}")
    else:
        chunks, size_mb, _ = _select_chunk_size(plan_md, max_size if max_size is not None else 100)
        print(f"\tUsing max_size={max_size} MB: chunk size {chunks}, {size_mb:.1f} MB.")

    factor = _normalize_downscale_factor(downscale_factor)
    if num_levels is None:
        num_levels = _estimate_levels(plan_md, factor[1:] if projection else factor)

    shards = _normalize_shards(shards)
    shard_exclude_axes = _normalize_shard_exclude_axes(shard_exclude_axes)
    write_kwargs = {
        "zarr_format": int(zarr_format),
        "ngff_version": "0.4" if int(zarr_format) == 2 else "0.5",
        "drop_singleton": True if drop_singleton is None else parse_bool(drop_singleton),
        "shards": shards,
        "chunks": chunks,
    }
    if num_workers is not None:
        write_kwargs["num_workers"] = int(num_workers)
    if shards is not None:
        if shard_target_mb is not None:
            write_kwargs["shard_target_mb"] = float(shard_target_mb)
        if shard_exclude_axes is not None:
            write_kwargs["shard_exclude_axes"] = shard_exclude_axes

    print(f"PYRAMID LEVELS: {num_levels}, DOWNSCALE FACTOR: {factor}.")
    print(f"WRITE OPTIONS: {write_kwargs}")
    print("\n--->Fusing to zarr")
    fuse_tiles(
        use,
        zarr_path,
        ds=ds,
        projection=projection,
        blend=blend,
        channels=plan.channels,
        channel_names=channel_names,
        channel_colors=channel_colors,
        crop=crop,
        skip=skip,
        levels=int(num_levels),
        downscale_factor=factor,
        workers=int(read_workers),
        out_dtype=out_dtype,
        midline_axis=midline_axis,
        sigmoid_steepness=SIGMOID_STEEPNESS if sigmoid_steepness is None else float(sigmoid_steepness),
        overwrite=bool(overwrite),
        temp_dir=temp_dir,
        **write_kwargs,
    )
    print(f"Saved {zarr_path}")
    return True


def convert_mosaic_single(args):
    """Runmode to fuse one mosaic to zarr."""
    print(f"Fusing mosaic {args.input_path} -> {args.zarr_path}")
    kwargs = {
        k: v
        for k, v in vars(args).items()
        if k not in _NOT_CONVERSION_OPTIONS and v is not None and not (isinstance(v, str) and v.strip() == "")
    }
    if "chunk_size" in kwargs:
        kwargs.pop("max_size", None)
    if not mosaic_convert(**kwargs):
        raise SystemExit(1)


def convert_mosaic_batch(args):
    """Runmode to fuse a batch of mosaics listed in a CSV file."""
    csv_path = getattr(args, "input_file", args)  # parsed arguments or a path
    print(f"Fusing a batch of mosaics from {csv_path}")
    database = pd.read_csv(csv_path)
    print(database)
    columns = database.columns
    not_fused = []
    for _, v in database.iterrows():
        print("-" * 20)
        kwargs = {
            "input_path": v["input"],
            "zarr_path": v["output"],
            "microscope": v["microscope"] if "microscope" in columns and _present(v.get("microscope")) else None,
            **_batch_row_kwargs(v, columns),
        }
        for key in ("downsample", "read_downsample"):  # read_downsample: the earlier column name
            if key in columns and _present(v.get(key)):
                kwargs["read_downsample"] = _int_triplet(str(v[key]))
        for key in ("projection", "blend", "midline_axis", "out_dtype", "temp_dir"):
            if key in columns and _present(v.get(key)):
                kwargs[key] = str(v[key]).strip()
        for key in ("channels", "groups", "fields", "tiles", "skip_tiles"):
            if key in columns and _present(v.get(key)):
                kwargs[key] = str(v[key]).split()
        if "read_workers" in columns and _present(v.get("read_workers")):
            kwargs["read_workers"] = int(v["read_workers"])
        if "sigmoid_steepness" in columns and _present(v.get("sigmoid_steepness")):
            kwargs["sigmoid_steepness"] = float(v["sigmoid_steepness"])
        if "overwrite" in columns and _present(v.get("overwrite")):
            kwargs["overwrite"] = parse_bool(v["overwrite"])
        if "flatfield" in columns and _present(v.get("flatfield")):
            kwargs["flatfield"] = parse_bool(v["flatfield"])
        if not mosaic_convert(**kwargs):
            not_fused.append(str(v["output"]))
    if not_fused:
        print(f"ERROR: {len(not_fused)} row(s) not fused (tiles from several fields): {', '.join(not_fused)}")


# ---------------------------------------------------------------------------
# Chromatic shift: chromatic-shift / chromatic-shift --batch
# ---------------------------------------------------------------------------

#: Options of ``estimate_chromatic_shift`` exposed on the command line.
_ESTIMATE_OPTIONS = ("timepoint", "max_shift", "coarse_level", "refine_level", "n_regions", "crop_yx",
                     "min_z_normal", "min_correlation", "threshold")


def _channel_index(channel, channel_names: List[str]) -> int:
    """Channel given as an index (``1``) or a name (``"cFos-647"``)."""
    text = str(channel).strip()
    if text.lstrip("-").isdigit():
        return int(text)
    if text in channel_names:
        return channel_names.index(text)
    raise ValueError(f"Reference channel {channel!r} not found; channels are {channel_names}.")


def _parse_shifts(value) -> Any:
    """``--shifts`` values or batch cell: one number per channel, or the path to a JSON report."""
    tokens = [str(v) for v in value] if isinstance(value, (list, tuple)) else str(value).split()
    if len(tokens) == 1 and Path(tokens[0]).is_file():
        import json

        return json.loads(Path(tokens[0]).read_text())
    try:
        return [float(t) for t in tokens]
    except ValueError as exc:
        raise ValueError(
            f"Invalid shifts {value!r}: give one number per channel or the path to a JSON report."
        ) from exc


def _print_shift_report(report: Dict[str, Any]) -> None:
    names = report.get("channel_names") or []
    unit = report.get("unit") or ""
    ref = report.get("reference_channel", 0)
    for c, shift in enumerate(report["shifts"]):
        name = names[c] if c < len(names) else f"channel {c}"
        if c == ref:
            print(f"  {name}: reference")
            continue
        surfaces = (report.get("surface_shifts") or [None] * (c + 1))[c] or {}
        print(
            f"  {name}: {shift:+.2f} {unit} (coarse {report['coarse_shifts'][c]:+.2f}, "
            f"{report['n_regions_used'][c]} regions, spread {report['spread'][c]}, "
            f"lower {surfaces.get('lower')}, upper {surfaces.get('upper')})"
        )
    for warning in report.get("warnings") or []:
        print(f"  WARNING: {warning}")


def chromatic_shift_correct(
    input_path: str,
    zarr_path: Optional[str] = None,
    *,
    report_path: Optional[str] = None,
    shifts: Any = None,
    reference_channel: Any = 0,
    extent: str = "union",
    order: int = 1,
    include_groups: bool = True,
    include_labels: bool = True,
    overwrite: bool = False,
    num_workers: Optional[int] = None,
    fail_on_warning: bool = False,
    **estimate_kwargs,
) -> Dict[str, Any]:
    """Measure the Z chromatic shift of an OME-Zarr image and write a corrected copy.

    Without ``zarr_path`` only the estimate runs (and is saved to
    ``report_path`` when given). ``shifts`` (one value per channel, a report
    dict or the path to a JSON report) skips the estimate. With
    ``fail_on_warning`` a report with warnings stops before anything is
    written except the report. Returns the report, or the given shifts.
    """
    if zarr_path is not None and Path(zarr_path).exists() and not overwrite:
        raise FileExistsError(f"{zarr_path} already exists; pass --overwrite to replace it.")

    z = mm.ZarrManager(path=str(input_path))
    names = list(z.metadata.get("channel_names") or [])

    if shifts is not None:
        report = shifts if isinstance(shifts, dict) else _parse_shifts(shifts)
        print(f"Using the given shifts: {report['shifts'] if isinstance(report, dict) else report}")
    else:
        ref = _channel_index(reference_channel, names)
        options = {k: v for k, v in estimate_kwargs.items() if k in _ESTIMATE_OPTIONS and v is not None}
        if options.get("refine_level") == "none":
            options["refine_level"] = None
        print(f"\n--->Estimating the chromatic shift of {input_path} (reference: {names[ref] if ref < len(names) else ref})")
        report = z.estimate_chromatic_shift(reference_channel=ref, **options)
        _print_shift_report(report)

    if report_path is not None:
        import json

        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(json.dumps(report, indent=2))
        print(f"Report saved to {report_path}")

    if fail_on_warning and isinstance(report, dict) and report.get("warnings"):
        raise SystemExit(f"Chromatic shift estimate of {input_path} has warnings; nothing written (--fail_on_warning).")

    if zarr_path is None:
        return report

    print(f"\n--->Writing the corrected dataset to {zarr_path}")
    z.apply_chromatic_shift(
        zarr_path,
        report,
        extent=extent,
        order=int(order),
        include_groups=bool(include_groups),
        include_labels=bool(include_labels),
        overwrite=True,  # an existing output was refused above unless overwrite was asked for
        num_workers=num_workers,
    )
    print(f"Saved {zarr_path}")
    return report


def _chromatic_shift_kwargs(args) -> Dict[str, Any]:
    kwargs = {
        k: v
        for k, v in vars(args).items()
        if k not in _NOT_CONVERSION_OPTIONS and v is not None and not (isinstance(v, str) and v.strip() == "")
    }
    if "report" in kwargs:
        kwargs["report_path"] = kwargs.pop("report")
    return kwargs


def chromatic_shift_single(args):
    """Runmode to correct the chromatic shift of one OME-Zarr image."""
    chromatic_shift_correct(**_chromatic_shift_kwargs(args))


def chromatic_shift_batch(args):
    """Runmode to correct the chromatic shift of every OME-Zarr image listed in a CSV file."""
    csv_path = getattr(args, "input_file", args)  # parsed arguments or a path
    print(f"Correcting the chromatic shift of a batch from {csv_path}")
    database = pd.read_csv(csv_path)
    print(database)
    columns = database.columns

    def cell(v, key):
        return v.get(key) if key in columns and _present(v.get(key)) else None

    for _, v in database.iterrows():
        print("-" * 20)
        kwargs: Dict[str, Any] = {"input_path": v["input"], "zarr_path": cell(v, "output")}
        for key, cast in (("report", str), ("shifts", str), ("reference_channel", str), ("extent", str),
                          ("refine_level", str), ("timepoint", int), ("coarse_level", int), ("n_regions", int),
                          ("crop_yx", int), ("order", int), ("num_workers", int), ("max_shift", float),
                          ("min_z_normal", float), ("min_correlation", float), ("threshold", float)):
            value = cell(v, key)
            if value is not None:
                kwargs["report_path" if key == "report" else key] = cast(value).strip() if cast is str else cast(value)
        if kwargs.get("refine_level") not in (None, "none"):
            kwargs["refine_level"] = int(kwargs["refine_level"])
        for key in ("include_groups", "include_labels", "overwrite", "fail_on_warning"):
            value = cell(v, key)
            if value is not None:
                kwargs[key] = parse_bool(value)
        chromatic_shift_correct(**kwargs)


def main(argv=None):
    """Entry point of the ``pymif`` command: ``convert``, ``fuse`` or ``chromatic-shift``."""
    args = _parse_arguments(argv)
    if args.renamed_to:
        print(f"Note: `pymif {args.command}` is now `{args.renamed_to}`; the old name still works for now.",
              file=sys.stderr)
    runners = {
        "convert": convert_single,
        "convert_batch": convert_batch,
        "fuse": convert_mosaic_single,
        "fuse_batch": convert_mosaic_batch,
        "chromatic_shift": chromatic_shift_single,
        "chromatic_shift_batch": chromatic_shift_batch,
    }
    runners[args.mode](args)

if __name__ == "__main__":
    main()