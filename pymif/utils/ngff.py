from __future__ import annotations

import contextlib
import math
import os
import warnings
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import dask
import dask.array as da
import numpy as np
import zarr
from numcodecs import Blosc, GZip

from .chunks import get_chunks as _get_chunks
from .chunks import rechunk_to_shape as _rechunk_to_shape
from .chunks import shape_tuple as _shape_tuple
from .colors import (  # noqa: F401  (re-exported: historical import location)
    default_channel_color,
    ome_color_to_hex,
    parse_channel_color,
    parse_color,
)
from .units import normalize_unit
from .axes import (
    DATA_TYPES,
    SPATIAL_AXIS_SET,
    normalize_axes,
    normalize_data_type,
    spatial_axes_in_order,
)

SPATIAL_AXES = SPATIAL_AXIS_SET

# Axes that `shards="auto"` never merges chunks along by default. Timepoints
# and channels are typically read/processed independently, so keeping each
# one in its own shard (or unsharded chunk) preserves that access pattern;
# the spatial axes (z/y/x) get consolidated automatically.
DEFAULT_SHARD_EXCLUDE_AXES: tuple[str, ...] = ("t", "c")

# Default write parallelism: this fraction of the available cores, capped.
# Beyond ~8 threads writes stop getting faster (compression and disk I/O
# saturate) while memory use keeps growing.
DEFAULT_WORKER_FRACTION = 0.5
DEFAULT_MAX_WORKERS = 8
# Share of free RAM the in-flight shard-sized blocks may use, and the fallback
# budget (MB) when psutil is unavailable.
DEFAULT_MEMORY_FRACTION = 0.5
FALLBACK_MEMORY_BUDGET_MB = 4096.0


@dataclass(slots=True)
class ZarrWriteConfig:
    """Configuration container for NGFF/OME-Zarr writing operations.

    Parameters
    ----------
    ngff_version, zarr_format
        ``0.4``/Zarr v2 and ``0.5``/Zarr v3 are the supported pairs.
    data_type
        Optional dataset semantic type.  Use ``"intensity"`` for regular image
        intensities or ``"label"`` for integer segmentation/annotation data.
    chunks
        Requested on-disk chunk shape, overriding whatever chunking the input
        dask arrays already have. One of:

        - ``None`` (default): write each level with its existing dask
          chunking, unchanged.
        - A single chunk shape matching the array's ndim: used for every
          level, clipped so no chunk axis exceeds that level's extent.
        - A sequence of chunk shapes, one per pyramid level.

        Applies to both zarr v2 and v3. This only changes the write-time
        chunk shape; it does not touch ``metadata['chunksize']``, which is
        only used when a pyramid is built via
        :func:`~pymif.utils.pyramid.build_pyramid`.
    shards
        Zarr v3 sharding configuration, applied per pyramid level. One of:

        - ``"auto"`` (default): pick a shard shape per level automatically so each
          shard is close to ``shard_target_mb`` in (uncompressed) size,
          grouping whole chunks along the axes with the most chunks first.
          Levels that are already small (few chunks, or a chunk already at
          or above the target size) are left unsharded.
        - A single shape tuple matching the array's ndim: used as the
          requested shard shape for every level, snapped to the nearest
          multiple of that level's chunk shape and clipped so it never
          exceeds the level's extent.
        - A sequence of shape tuples, one per pyramid level, each resolved
          the same way.
        - ``None``: no sharding, one file per chunk.

        Only valid when writing zarr v3 (``ngff_version="0.5"``); sharding
        has no zarr v2 equivalent, so for zarr v2 a warning is issued and
        sharding is set to ``None``. ``shard_exclude_axes`` only constrains
        ``"auto"``; an explicit shard shape always applies exactly as given.
    shard_target_mb
        Target *uncompressed* shard size in megabytes used by
        ``shards="auto"``.
    shard_exclude_axes
        Axis names that ``shards="auto"`` never merges chunks along, even if
        doing so would help reach ``shard_target_mb``. Defaults to
        ``("t", "c")`` so each timepoint and channel stays independently
        addressable; the spatial axes (``z``/``y``/``x``) are consolidated
        automatically. Pass ``()`` to allow every axis to grow.
    drop_singleton
        If ``True`` (default), ``t``, ``c`` and ``z`` axes of size 1 are
        removed from the written arrays and metadata (``y`` and ``x`` are
        always kept). ``chunks``/``shards`` given for the full axis set are
        trimmed accordingly. Pass ``False`` to keep every axis.
    num_workers
        Number of threads used to compress and write blocks in parallel.
        ``None`` (default) uses half of the available cores, capped at 8.
        For sharded levels the count is further reduced so the shard-sized
        blocks held in memory at once fit in about half of the free RAM.
        Only applies when ``compute=True``; with ``compute=False`` the caller
        controls the scheduler when computing the returned tasks.
    progress
        If ``True`` (default), show a ``tqdm`` progress bar (terminal or
        notebook) while the arrays are written. Only applies when
        ``compute=True`` and a local (threaded) dask scheduler is used.
        Pass ``False`` for silent writes, e.g. in batch jobs or tests.
    """

    ngff_version: Literal["0.4", "0.5"] | None = None
    zarr_format: Literal[2, 3] | None = None
    overwrite: bool = True
    compute: bool = True
    storage_options: dict[str, Any] | None = None
    compressor: Literal["blosc", "gzip"] | None = None
    compressor_level: int = 3
    data_type: Literal["intensity", "label"] | None = None
    chunks: Sequence[int] | Sequence[Sequence[int]] | None = None
    shards: Literal["auto"] | Sequence[int] | Sequence[Sequence[int]] | None = "auto"
    shard_target_mb: float = 1024.0 # in MB (1 GB)
    shard_exclude_axes: Sequence[str] = DEFAULT_SHARD_EXCLUDE_AXES
    drop_singleton: bool = True
    num_workers: int | None = None
    progress: bool = True

def _infer_ngff_version(group: zarr.Group) -> str:
    """Infer the NGFF metadata layout used by an existing group."""
    attrs = group.attrs.asdict()
    if "ome" in attrs:
        return attrs["ome"].get("version", "0.5")
    return "0.4"

def _label_entry(label_name: str) -> str:
    return label_name


def _labels_contains(labels: Sequence[Any], label_name: str) -> bool:
    label_path = f"labels/{label_name}"
    for item in labels:
        if item == label_name or item == label_path:
            return True
        if isinstance(item, dict) and item.get("name") == label_name:
            return True
        if isinstance(item, dict) and item.get("path") == label_path:
            return True
    return False

def _register_label_on_labels_group(root: zarr.Group, label_name: str, ngff_version: str) -> None:
    """Register a label image on the ``labels`` container group."""
    labels_group = root.require_group("labels")
    attrs = labels_group.attrs.asdict()
    entry = _label_entry(label_name)

    if ngff_version == "0.5":
        ome = dict(attrs.get("ome", {}))
        ome.setdefault("version", "0.5")
        labels = list(ome.get("labels", []))
        if not _labels_contains(labels, label_name):
            labels.append(entry)
        ome["labels"] = labels
        labels_group.attrs["ome"] = ome

        root_attrs = root.attrs.asdict()
        root_ome = root_attrs.get("ome")
        if isinstance(root_ome, dict) and "labels" in root_ome:
            root_ome = dict(root_ome)
            root_ome.pop("labels", None)
            root.attrs["ome"] = root_ome
    else:
        labels = list(attrs.get("labels", []))
        if not _labels_contains(labels, label_name):
            labels.append(entry)
        labels_group.attrs["labels"] = labels

        if "labels" in root.attrs.asdict():
            del root.attrs["labels"]


def _get_group_ome_attrs(group: zarr.Group) -> dict[str, Any]:
    """Return the effective OME-NGFF metadata mapping for a group."""
    attrs = group.attrs.asdict()
    ome = attrs.get("ome")
    return ome if isinstance(ome, dict) else attrs

def _get_multiscales(group: zarr.Group) -> list[dict[str, Any]]:
    """Return the raw ``multiscales`` list from a group across NGFF versions."""
    return _get_group_ome_attrs(group).get("multiscales", [])

def _get_group_multiscales(group: zarr.Group):
    """Compatibility helper returning the stored multiscales block for a group."""
    return _get_group_ome_attrs(group).get("multiscales")

def _infer_data_type_from_group(group: zarr.Group) -> str:
    """Infer ``intensity`` or ``label`` from explicit and legacy metadata."""
    attrs = group.attrs.asdict()
    image_meta = _get_group_ome_attrs(group)

    explicit = image_meta.get("data_type") or attrs.get("data_type")
    if explicit is not None:
        return normalize_data_type(explicit)

    if "image-label" in image_meta or "image-label" in attrs:
        return "label"

    multiscales = image_meta.get("multiscales", [])
    if multiscales:
        ms_type = multiscales[0].get("type")
        if ms_type in DATA_TYPES or ms_type in {"image", "labels"}:
            return normalize_data_type(ms_type)

    return "intensity"

def _set_group_ngff_metadata(
    group: zarr.Group,
    *,
    ngff_version: str,
    multiscales: dict[str, Any],
    omero: dict[str, Any] | None = None,
    data_type: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write NGFF metadata to ``group`` using either the v0.4 or v0.5 layout."""
    extra = dict(extra or {})
    normalized_data_type = normalize_data_type(data_type)
    multiscales = dict(multiscales)
    multiscales.setdefault("type", "label" if normalized_data_type == "label" else "image")

    if normalized_data_type == "label":
        # No ``source`` unless the caller knows where the source image lives
        # (labels nested under ``<image>/labels/<name>`` pass ``../../``).
        extra.setdefault("image-label", {})

    if ngff_version == "0.5":
        payload = {
            "version": "0.5",
            "data_type": normalized_data_type,
            "multiscales": [multiscales],
        }
        if omero is not None and normalized_data_type == "intensity":
            payload["omero"] = omero
        payload.update(extra)
        group.attrs["ome"] = payload
    else:
        group.attrs["data_type"] = normalized_data_type
        group.attrs["multiscales"] = [multiscales]
        if omero is not None and normalized_data_type == "intensity":
            group.attrs["omero"] = omero
        for key, value in extra.items():
            group.attrs[key] = value

def _set_dimension_names(
    group: zarr.Group,
    datasets: Sequence[dict[str, Any]],
    axes: Sequence[str],
    *,
    zarr_format: int,
) -> None:
    """Write Zarr v3 array-level dimension names required by NGFF v0.5."""
    if zarr_format != 3:
        return
    names = [str(axis) for axis in axes]
    for dataset in datasets:
        path = dataset.get("path")
        if path in group:
            group[path].attrs["dimension_names"] = names


def _resolve_format(cfg: ZarrWriteConfig) -> tuple[str, int]:
    """Resolve and validate the NGFF version / zarr format pair to use."""
    ngff_version = cfg.ngff_version or ("0.5" if cfg.zarr_format in (None, 3) else "0.4")
    zarr_format = cfg.zarr_format or (3 if ngff_version == "0.5" else 2)

    if (ngff_version, zarr_format) not in {("0.4", 2), ("0.5", 3)}:
        raise ValueError(
            f"Incompatible ngff_version/zarr_format pair: {ngff_version}/{zarr_format}. "
            "Use 0.4 with zarr v2 or 0.5 with zarr v3."
        )

    return ngff_version, zarr_format


def _resolve_write_chunks(
    data_levels: Sequence[da.Array],
    chunks: Sequence[int] | Sequence[Sequence[int]] | None,
) -> list[da.Array]:
    """Rechunk each pyramid level to the requested write-time ``chunks``.

    ``chunks`` is either ``None`` (levels are written with whatever chunking
    they already have), a single chunk shape applied to every level, or one
    chunk shape per level. Mirrors how ``shards`` accepts a single shape or a
    per-level list.
    """
    if chunks is None:
        return list(data_levels)

    n_levels = len(data_levels)
    ndim = data_levels[0].ndim

    flat = _shape_tuple(chunks, ndim)
    if flat is not None:
        return [_rechunk_to_shape(arr, flat) for arr in data_levels]

    if len(chunks) != n_levels:
        raise ValueError(
            "chunks must be a single chunk shape or contain one entry per "
            f"pyramid level ({n_levels}), got {len(chunks)}."
        )
    return [_rechunk_to_shape(arr, c) for arr, c in zip(data_levels, chunks)]


def _available_memory_mb() -> float:
    """Free RAM in MB, or a fixed fallback budget when psutil is missing."""
    try:
        import psutil

        return psutil.virtual_memory().available / (1024 * 1024)
    except Exception:
        return FALLBACK_MEMORY_BUDGET_MB / DEFAULT_MEMORY_FRACTION


def _resolve_num_workers(cfg: ZarrWriteConfig, block_bytes: int = 0) -> int:
    """Number of write threads for ``cfg``, limited by the in-flight block size.

    ``block_bytes`` is the uncompressed size of the largest block a worker
    holds at once (a whole shard for sharded levels, ``0`` otherwise). Each
    worker may also hold a compressed copy, hence the factor of 2.
    """
    if cfg.num_workers is not None:
        if int(cfg.num_workers) < 1:
            raise ValueError(f"num_workers must be >= 1, got {cfg.num_workers}.")
        workers = int(cfg.num_workers)
    else:
        cores = os.cpu_count() or 1
        workers = min(max(1, int(cores * DEFAULT_WORKER_FRACTION)), DEFAULT_MAX_WORKERS)

    if block_bytes > 0:
        budget = _available_memory_mb() * DEFAULT_MEMORY_FRACTION * 1024 * 1024
        fit = max(1, int(budget // (2 * block_bytes)))
        if fit < workers:
            if cfg.num_workers is not None or fit == 1:
                warnings.warn(
                    f"Shard-sized blocks of {block_bytes / 1024**2:.0f} MB limit write "
                    f"parallelism to {fit} worker(s) (requested {workers}). Use smaller "
                    "shards (shard_target_mb / shards) to write in parallel.",
                    stacklevel=3,
                )
            workers = fit
    return workers


@contextlib.contextmanager
def _progress_bar(enabled: bool, desc: str | None):
    """Show a tqdm progress bar for the dask computation run inside the block."""
    if not enabled:
        yield
        return
    try:
        from tqdm.dask import TqdmCallback
    except ImportError:  # tqdm is optional at runtime: just skip the bar
        yield
        return
    with TqdmCallback(desc=desc or "Writing zarr", unit="task"):
        yield


def _run_stores(tasks, cfg: ZarrWriteConfig, block_bytes: int = 0, desc: str | None = None):
    """Run (or return) the per-level store tasks of one multiscale dataset.

    All levels are computed in a single dask pass so shared upstream work (for
    example reading the source file) happens once instead of once per level.
    With ``cfg.compute`` false the unevaluated tasks are returned unchanged.
    ``desc`` labels the progress bar (see ``ZarrWriteConfig.progress``).
    """
    if not cfg.compute:
        return list(tasks)
    with dask.config.set(num_workers=_resolve_num_workers(cfg, block_bytes)), _progress_bar(cfg.progress, desc):
        dask.compute(*tasks)
    return []


def _write_pyramid_v2(
    *,
    root: zarr.Group,
    data_levels: Sequence[da.Array],
    cfg: ZarrWriteConfig,
    desc: str | None = None,
):
    """Create and populate zarr v2 arrays for each pyramid level."""
    if cfg.shards is not None:
        warnings.warn(
            f"Sharding (shards={cfg.shards!r}) is only supported for zarr_format=3 "
            "(NGFF v0.5) datasets; it will be set to None (no sharding) for "
            "zarr_format=2.",
            UserWarning,
            stacklevel=4,
        )

    data_levels = _resolve_write_chunks(data_levels, cfg.chunks)
    tasks = []

    for i, arr in enumerate(data_levels):
        create_kwargs = _array_create_kwargs(
            str(i), arr.shape, arr.dtype, _get_chunks(arr),
            zarr_format=2,
            compressor=cfg.compressor,
            compressor_level=cfg.compressor_level,
            storage_options=cfg.storage_options,
        )

        z = root.create_array(**create_kwargs)

        tasks.append(da.store(arr, z, lock=False, compute=False))

    return _run_stores(tasks, cfg, desc=desc)


def _write_pyramid_v3(
    *,
    root: zarr.Group,
    data_levels: Sequence[da.Array],
    cfg: ZarrWriteConfig,
    axes: Sequence[str] | None = None,
    desc: str | None = None,
):
    """Create and populate zarr v3 arrays for each pyramid level."""
    data_levels = _resolve_write_chunks(data_levels, cfg.chunks)
    tasks = []
    block_bytes = 0

    chunks_per_level = [_get_chunks(arr) for arr in data_levels]
    shard_shapes = _resolve_shards_for_levels(
        [tuple(arr.shape) for arr in data_levels],
        chunks_per_level,
        data_levels[0].dtype,
        cfg.shards,
        zarr_format=3,
        target_bytes=int(cfg.shard_target_mb * 1024 * 1024),
        axes=axes,
        exclude_axes=cfg.shard_exclude_axes,
    )

    for i, arr in enumerate(data_levels):
        create_kwargs = _array_create_kwargs(
            str(i), arr.shape, arr.dtype, chunks_per_level[i],
            zarr_format=3,
            compressor=cfg.compressor,
            compressor_level=cfg.compressor_level,
            shards=shard_shapes[i],
            storage_options=cfg.storage_options,
            always_pass_compressors=True,
        )

        z = root.create_array(**create_kwargs)

        # When a level is sharded, several chunks land in the same shard file,
        # so threads writing different chunks of one shard would race (zarr
        # read-modify-writes the shard on every partial write). Instead make
        # each dask block exactly one shard: a block then encodes and writes a
        # whole shard file, and no two blocks ever touch the same file, so no
        # lock is needed.
        to_store = arr
        if shard_shapes[i] is not None:
            to_store = arr.rechunk(tuple(shard_shapes[i]))
            block_bytes = max(
                block_bytes,
                math.prod(int(c) for c in shard_shapes[i]) * np.dtype(arr.dtype).itemsize,
            )
        tasks.append(da.store(to_store, z, lock=False, compute=False))

    return _run_stores(tasks, cfg, block_bytes=block_bytes, desc=desc)


def _auto_shard_for_level(
    shape: Sequence[int],
    chunk: Sequence[int],
    itemsize: int,
    target_bytes: int,
    *,
    axes: Sequence[str] | None = None,
    exclude_axes: Sequence[str] = (),
) -> tuple[int, ...] | None:
    """Pick a shard shape for one pyramid level targeting ``target_bytes`` per shard.

    Whole chunks are grouped into a shard, growing the axes with the most
    available chunks first (typically the spatial ``z``/``y``/``x`` axes).
    Axes whose name is in ``exclude_axes`` (requires ``axes`` to be given)
    are never grown, so e.g. timepoints or channels can be kept one-per-shard
    even while z/y/x is merged. Levels with no growable axis holding more
    than one chunk, or whose chunk is already at or above the target size,
    are left unsharded (``None``).
    """
    ndim = len(chunk)
    chunks_per_axis = [max(1, -(-int(s) // int(c))) for s, c in zip(shape, chunk)]

    if axes is not None and exclude_axes:
        growable = [i for i in range(ndim) if axes[i] not in exclude_axes]
    else:
        growable = list(range(ndim))

    if not growable or all(chunks_per_axis[i] <= 1 for i in growable):
        return None

    chunk_bytes = itemsize
    for c in chunk:
        chunk_bytes *= int(c)
    if chunk_bytes <= 0 or chunk_bytes >= target_bytes:
        return None

    want_chunks = max(2, round(target_bytes / chunk_bytes))

    multiplier = [1] * ndim
    order = sorted(growable, key=lambda a: chunks_per_axis[a], reverse=True)

    def _total() -> int:
        total = 1
        for m in multiplier:
            total *= m
        return total

    progressed = True
    while _total() < want_chunks and progressed:
        progressed = False
        for axis in order:
            if multiplier[axis] < chunks_per_axis[axis]:
                multiplier[axis] += 1
                progressed = True
                if _total() >= want_chunks:
                    break

    if all(m == 1 for m in multiplier):
        return None

    return tuple(int(c) * int(m) for c, m in zip(chunk, multiplier))


def _snap_shard_shape(
    spec: Any,
    shape: Sequence[int],
    chunk: Sequence[int],
) -> tuple[int, ...] | None:
    """Round a requested shard shape to a valid multiple of ``chunk``.

    The request is rounded to the nearest whole number of chunks per axis
    (minimum one) and clipped so the shard never spans more chunks than the
    level actually has. Returns ``None`` when the result collapses back to
    the plain chunk shape (i.e. sharding would add no value at this level).
    """
    ndim = len(chunk)
    parsed = _shape_tuple(spec, ndim)
    if parsed is None:
        raise ValueError(
            f"Invalid shard shape {spec!r}; expected a length-{ndim} sequence "
            "of positive ints."
        )

    out = []
    for value, c, s in zip(parsed, chunk, shape):
        c = int(c)
        s = int(s)
        chunks_per_axis = max(1, -(-s // c))
        mult = min(max(1, round(value / c)), chunks_per_axis)
        out.append(c * mult)

    result = tuple(out)
    if result == tuple(int(c) for c in chunk):
        return None
    return result


def _resolve_shards_for_levels(
    shapes: Sequence[Sequence[int]],
    chunks: Sequence[Sequence[int]],
    dtype: Any,
    shards: Literal["auto"] | Sequence[int] | Sequence[Sequence[int]] | None,
    *,
    zarr_format: int,
    target_bytes: int,
    axes: Sequence[str] | None = None,
    exclude_axes: Sequence[str] = (),
) -> list[tuple[int, ...] | None]:
    """Resolve a per-pyramid-level shard shape (or ``None``) for zarr v3 writes.

    ``axes``/``exclude_axes`` only affect ``shards="auto"``; an explicit
    shard shape (single tuple or per-level list) always applies exactly as
    requested.
    """
    n_levels = len(shapes)
    if shards is None:
        return [None] * n_levels

    if zarr_format != 3:
        raise ValueError(
            "Sharding is only supported for zarr_format=3 (NGFF v0.5) datasets; "
            f"got zarr_format={zarr_format}."
        )

    ndim = len(chunks[0])
    itemsize = np.dtype(dtype).itemsize

    if isinstance(shards, str):
        if shards != "auto":
            raise ValueError(
                f"Unsupported shards value {shards!r}; use 'auto', an explicit "
                "shard shape, or one shard shape per pyramid level."
            )
        return [
            _auto_shard_for_level(
                shape, chunk, itemsize, target_bytes,
                axes=axes, exclude_axes=exclude_axes,
            )
            for shape, chunk in zip(shapes, chunks)
        ]

    flat = _shape_tuple(shards, ndim)
    if flat is not None:
        return [_snap_shard_shape(flat, shape, chunk) for shape, chunk in zip(shapes, chunks)]

    if len(shards) != n_levels:
        raise ValueError(
            "shards must be 'auto', a single shard shape, or contain one "
            f"entry per pyramid level ({n_levels}), got {len(shards)}."
        )
    return [
        _snap_shard_shape(spec, shape, chunk)
        for spec, shape, chunk in zip(shards, shapes, chunks)
    ]


def _build_v2_compressor(compressor: str | None, level: int):
    """Construct a zarr v2-compatible compressor configuration."""
    if compressor is None:
        return None
    if compressor == "blosc":
        return Blosc(cname="zstd", clevel=level, shuffle=Blosc.BITSHUFFLE)
    if compressor == "gzip":
        return GZip(level=level)
    raise ValueError(f"Unsupported compressor for zarr v2: {compressor}")


def _build_v3_compressors(compressor: str | None, level: int):
    """Construct a zarr v3-compatible compressor chain."""
    if compressor is None:
        return None
    if compressor == "blosc":
        return [
            zarr.codecs.BloscCodec(
                cname="zstd",
                clevel=level,
                shuffle=zarr.codecs.BloscShuffle.bitshuffle,
            )
        ]
    raise ValueError(f"Unsupported compressor for zarr v3: {compressor}")


def _array_create_kwargs(
    name: str,
    shape: Sequence[int],
    dtype: Any,
    chunks: Sequence[int],
    *,
    zarr_format: int,
    compressor: str | None,
    compressor_level: int,
    shards: Sequence[int] | None = None,
    storage_options: dict[str, Any] | None = None,
    always_pass_compressors: bool = False,
) -> dict[str, Any]:
    """Keyword arguments for ``Group.create_array`` of one pyramid level.

    Single place that knows the differences between zarr v2 (compressor +
    ``/``-separated v2 chunk keys) and v3 (codec chain + optional sharding).
    For v3, ``compressors`` is only passed when a compressor is requested,
    unless ``always_pass_compressors`` is set (the writer passes ``None``
    explicitly, which stores uncompressed chunks).
    """
    kwargs: dict[str, Any] = {
        "name": name,
        "shape": tuple(shape),
        "dtype": dtype,
        "chunks": tuple(chunks),
    }
    if zarr_format == 2:
        kwargs["compressor"] = _build_v2_compressor(compressor, compressor_level)
        kwargs["chunk_key_encoding"] = {"name": "v2", "separator": "/"}
    else:
        compressors = _build_v3_compressors(compressor, compressor_level)
        if compressors is not None or always_pass_compressors:
            kwargs["compressors"] = compressors
        if shards is not None:
            kwargs["shards"] = tuple(shards)
    if storage_options is not None:
        kwargs.update(storage_options)
    return kwargs


def _resolve_creation_shards(
    metadata: dict[str, Any],
    sizes: Sequence[Sequence[int]],
    chunks: Sequence[Sequence[int]],
    axes: Sequence[str],
    *,
    zarr_format: int,
    shards: Any = None,
    shard_target_mb: float = 1024.0,
    shard_exclude_axes: Sequence[str] | None = None,
) -> list[tuple[int, ...] | None]:
    """Per-level shard shapes for creating empty arrays from metadata.

    Falls back to ``metadata['shards']`` / ``metadata['shard_exclude_axes']``
    when not given explicitly. Shards inherited from metadata whose rank does
    not match the new arrays (for example image metadata reused for a label
    group without channel axis) are ignored with a warning instead of failing.
    """
    inherited = shards is None and metadata.get("shards") is not None
    if shards is None:
        shards = metadata.get("shards")
    if inherited and not isinstance(shards, str):
        ndim = len(chunks[0])
        per_level = shards if isinstance(shards[0], (list, tuple)) else [shards]
        if any(s is not None and len(s) != ndim for s in per_level):
            warnings.warn(
                f"Ignoring metadata['shards'] {shards!r}: it does not match the "
                f"{ndim} axes of the new arrays.",
                UserWarning,
                stacklevel=3,
            )
            shards = None
    if shard_exclude_axes is None:
        shard_exclude_axes = metadata.get("shard_exclude_axes", DEFAULT_SHARD_EXCLUDE_AXES)
    return _resolve_shards_for_levels(
        sizes,
        chunks,
        metadata.get("dtype", "uint16"),
        shards,
        zarr_format=zarr_format,
        target_bytes=int(shard_target_mb * 1024 * 1024),
        axes=axes,
        exclude_axes=shard_exclude_axes,
    )


def _create_level_arrays(
    parent: zarr.Group,
    sizes: Sequence[Sequence[int]],
    chunks: Sequence[Sequence[int]],
    dtype: Any,
    shard_shapes: Sequence[Sequence[int] | None],
    *,
    zarr_format: int,
    compressor: str | None,
    compressor_level: int,
) -> None:
    """Create the empty array of every pyramid level (``0``, ``1``, ...) in ``parent``."""
    for i, (shape, chunk) in enumerate(zip(sizes, chunks)):
        parent.create_array(
            **_array_create_kwargs(
                str(i), shape, dtype, chunk,
                zarr_format=zarr_format,
                compressor=compressor,
                compressor_level=compressor_level,
                shards=shard_shapes[i],
            )
        )


def _build_multiscales(
    metadata: dict[str, Any],
    axes: Sequence[str],
    name: str,
) -> dict[str, Any]:
    """NGFF ``multiscales`` entry (axes + per-level transforms) for ``metadata``."""
    data_type = normalize_data_type(metadata.get("data_type"))
    return {
        "name": name,
        "axes": _build_axes(axes, metadata),
        "datasets": [
            {"path": str(i), "coordinateTransformations": ct}
            for i, ct in enumerate(
                _build_coordinate_transformations(
                    axes=axes,
                    scales=metadata["scales"],
                    time_increment=metadata.get("time_increment"),
                )
            )
        ],
        "type": "label" if data_type == "label" else "image",
    }


def _validate_metadata(
    data_levels: Sequence[da.Array],
    metadata: dict[str, Any],
    axes: tuple[str, ...],
) -> None:
    """Validate the minimal metadata contract required for NGFF writing."""
    if not data_levels:
        raise ValueError("data_levels cannot be empty.")

    ndim = data_levels[0].ndim
    axes = normalize_axes(axes, ndim=ndim)

    for arr in data_levels[1:]:
        if arr.ndim != ndim:
            raise ValueError("All pyramid levels must have the same ndim.")

    data_type = normalize_data_type(metadata.get("data_type"))
    if data_type == "label" and not np.issubdtype(np.dtype(data_levels[0].dtype), np.integer):
        raise ValueError("Label datasets must use an integer dtype.")

    sizes = metadata.get("size")
    if sizes is not None and len(sizes) != len(data_levels):
        raise ValueError("metadata['size'] must contain one entry per pyramid level.")

    chunks = metadata.get("chunksize")
    if chunks is not None and len(chunks) != len(data_levels):
        raise ValueError("metadata['chunksize'] must contain one entry per pyramid level.")

    scales = metadata.get("scales")
    if scales is None or len(scales) != len(data_levels):
        raise ValueError("metadata['scales'] must contain one entry per pyramid level.")

    spatial_axes = spatial_axes_in_order(axes)
    for scale in scales:
        if len(scale) != len(spatial_axes):
            raise ValueError(
                "Each scale entry must match the number of spatial axes "
                f"({len(spatial_axes)}), got {len(scale)}."
            )

    spatial_units = list(metadata.get("units", ()))
    if spatial_units and len(spatial_units) != len(spatial_axes):
        raise ValueError(
            "metadata['units'] must match the number of spatial axes "
            f"({len(spatial_axes)})."
        )

    if "t" in axes and metadata.get("time_increment") is None:
        raise ValueError("A 't' axis requires metadata['time_increment'].")


def _build_axes(axes: tuple[str, ...], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the NGFF ``axes`` description from the normalized PyMIF metadata."""
    axes = normalize_axes(axes)
    axis_types = {"t": "time", "c": "channel", "z": "space", "y": "space", "x": "space"}
    spatial_labels = spatial_axes_in_order(axes)
    spatial_units = [
        normalize_unit(u) for u in metadata.get("units", [None] * len(spatial_labels))
    ]
    spatial_unit_map = dict(zip(spatial_labels, spatial_units))
    time_unit = normalize_unit(metadata.get("time_increment_unit"))

    out = []
    for ax in axes:
        entry = {"name": ax, "type": axis_types[ax]}
        if ax == "t" and time_unit:
            entry["unit"] = time_unit
        elif ax in spatial_unit_map and spatial_unit_map[ax]:
            entry["unit"] = spatial_unit_map[ax]
        out.append(entry)
    return out


def _build_coordinate_transformations(
    *,
    axes: tuple[str, ...] | str,
    scales: Sequence[Sequence[float]],
    time_increment: float | None,
) -> list[list[dict[str, Any]]]:
    """Generate one NGFF scale transformation entry per pyramid level."""
    axes = normalize_axes(axes)
    out = []
    for spatial_scale in scales:
        spatial_iter = iter(spatial_scale)
        full_scale = []
        for ax in axes:
            if ax == "t":
                full_scale.append(float(time_increment if time_increment is not None else 1.0))
            elif ax == "c":
                full_scale.append(1.0)
            elif ax in SPATIAL_AXIS_SET:
                full_scale.append(float(next(spatial_iter)))
            else:
                full_scale.append(1.0)
        out.append([{"type": "scale", "scale": full_scale}])
    return out


def _build_omero_metadata(
    arr: da.Array,
    axes: tuple[str, ...] | str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Create OMERO channel display metadata for an intensity array."""
    axes = normalize_axes(axes, ndim=arr.ndim)
    c_size = arr.shape[axes.index("c")] if "c" in axes else 1
    ch_names = list(metadata.get("channel_names") or [])
    ch_colors = list(metadata.get("channel_colors") or [])

    lo, hi = _default_window(arr.dtype)
    channels = []
    for i in range(c_size):
        label = ch_names[i] if i < len(ch_names) else f"channel_{i}"
        color = ch_colors[i] if i < len(ch_colors) else default_channel_color(i)
        channels.append(
            {
                "label": label,
                "color": parse_color(color),
                "window": {"start": lo, "end": hi, "min": lo, "max": hi},
                "active": True,
                "inverted": False,
                "coefficient": 1.0,
                "family": "linear",
            }
        )

    return {"channels": channels, "rdefs": {"model": "color"}}


def _default_window(dtype: np.dtype | str) -> tuple[float, float]:
    """Return a default display range for the provided dtype."""
    dt = np.dtype(dtype)
    if np.issubdtype(dt, np.bool_):
        return 0.0, 1.0
    if np.issubdtype(dt, np.integer):
        info = np.iinfo(dt)
        return float(info.min), float(info.max)
    return 0.0, 1.0
