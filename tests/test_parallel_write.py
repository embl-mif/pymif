from __future__ import annotations

import warnings

import dask
import dask.array as da
import numpy as np
import pandas as pd
import pytest
import zarr

import zarrsmith.microscope_manager as mm
from zarrsmith.cli.zarrsmith import convert_batch
from zarrsmith.utils import ngff
from zarrsmith.utils.ngff import ZarrWriteConfig, _resolve_num_workers


def _manager(shape=(1, 2, 8, 128, 128), chunks=(1, 1, 2, 32, 32), source=None):
    data = np.arange(np.prod(shape), dtype="uint16").reshape(shape)
    arr = da.from_array(data, chunks=chunks) if source is None else source(data, chunks)
    metadata = {
        "axes": "tczyx",
        "scales": [(1.0, 1.0, 1.0)],
        "units": ("micrometer",) * 3,
        "time_increment": 1.0,
        "time_increment_unit": "second",
    }
    return mm.ArrayManager(arr, metadata, chunks=chunks), data


# --- worker resolution ---

def test_default_workers_is_fraction_of_cores_capped(monkeypatch):
    monkeypatch.setattr(ngff.os, "cpu_count", lambda: 24)
    assert _resolve_num_workers(ZarrWriteConfig()) == 8
    monkeypatch.setattr(ngff.os, "cpu_count", lambda: 4)
    assert _resolve_num_workers(ZarrWriteConfig()) == 2
    monkeypatch.setattr(ngff.os, "cpu_count", lambda: 1)
    assert _resolve_num_workers(ZarrWriteConfig()) == 1


def test_explicit_workers_and_validation():
    assert _resolve_num_workers(ZarrWriteConfig(num_workers=3)) == 3
    with pytest.raises(ValueError):
        _resolve_num_workers(ZarrWriteConfig(num_workers=0))


def test_workers_limited_by_shard_memory(monkeypatch):
    monkeypatch.setattr(ngff, "_available_memory_mb", lambda: 1000.0)
    # budget = 500 MB; each worker needs 2 * 100 MB -> 2 workers fit.
    with pytest.warns(UserWarning, match="limit write parallelism"):
        assert _resolve_num_workers(ZarrWriteConfig(num_workers=8), 100 * 1024**2) == 2


# --- correctness ---

@pytest.mark.parametrize("zarr_format,ngff_version", [(3, "0.5"), (2, "0.4")])
def test_unsharded_parallel_write_roundtrip(tmp_path, zarr_format, ngff_version):
    mgr, data = _manager()
    out = tmp_path / "o.zarr"
    mgr.build_pyramid(num_levels=3, downscale_factor=(1, 2, 2))
    mgr.to_zarr(str(out), zarr_format=zarr_format, ngff_version=ngff_version,
                num_workers=4, drop_singleton=False)
    root = zarr.open_group(str(out), mode="r")
    assert np.array_equal(root["0"][:], data)
    assert root["1"].shape == (1, 2, 8, 64, 64)
    assert root["2"].shape == (1, 2, 8, 32, 32)


def test_sharded_parallel_write_is_correct_with_many_chunks_per_shard(tmp_path):
    mgr, data = _manager()
    out = tmp_path / "s.zarr"
    mgr.build_pyramid(num_levels=3, downscale_factor=(1, 2, 2))
    # 4x4 chunks per shard in y/x, so dask chunks != shards before alignment.
    mgr.to_zarr(str(out), zarr_format=3, ngff_version="0.5", drop_singleton=False,
                shards=(1, 1, 2, 64, 64), num_workers=8)
    root = zarr.open_group(str(out), mode="r")
    assert root["0"].shards == (1, 1, 2, 64, 64)
    assert root["0"].chunks == (1, 1, 2, 32, 32)
    assert np.array_equal(root["0"][:], data)


def test_sharded_write_does_not_use_a_lock(tmp_path, monkeypatch):
    mgr, _ = _manager()
    locks = []
    real_store = da.store

    def spy(*a, **k):
        locks.append(k.get("lock"))
        return real_store(*a, **k)

    monkeypatch.setattr(ngff.da, "store", spy)
    mgr.to_zarr(str(tmp_path / "n.zarr"), zarr_format=3, ngff_version="0.5",
                shards=(1, 1, 2, 64, 64), drop_singleton=False)
    assert locks and all(lock is False for lock in locks)


def test_all_levels_computed_in_one_pass_reads_source_once(tmp_path):
    reads = []

    def counting_source(data, chunks):
        def load(block_info=None):
            loc = block_info[None]["array-location"]
            reads.append(tuple(map(tuple, loc)))
            return data[tuple(slice(a, b) for a, b in loc)]

        return da.map_blocks(load, chunks=da.from_array(data, chunks=chunks).chunks,
                             dtype=data.dtype, meta=np.empty((0,) * data.ndim, data.dtype))

    mgr, _ = _manager(source=counting_source)
    total_blocks = int(np.prod(mgr.data[0].numblocks))
    mgr.build_pyramid(num_levels=3, downscale_factor=(1, 2, 2))
    reads.clear()
    mgr.to_zarr(str(tmp_path / "r.zarr"), zarr_format=3, ngff_version="0.5",
                drop_singleton=False)
    # Each source block is read once, not once per pyramid level.
    assert len(reads) == total_blocks


def test_compute_false_returns_tasks_without_writing(tmp_path):
    mgr, data = _manager()
    out = tmp_path / "d.zarr"
    from zarrsmith.utils.to_zarr import to_zarr

    tasks = to_zarr(str(out), mgr.data, mgr.metadata,
                    config=ZarrWriteConfig(compute=False, drop_singleton=False))
    assert len(tasks) == len(mgr.data)
    assert not np.any(zarr.open_group(str(out), mode="r")["0"][:])
    dask.compute(*tasks)
    assert np.array_equal(zarr.open_group(str(out), mode="r")["0"][:], data)


# --- CLI batch ---

def test_batch_csv_num_workers_column(tmp_path):
    mgr, _ = _manager(shape=(1, 2, 4, 64, 64), chunks=(1, 1, 2, 32, 32))
    src = tmp_path / "src.zarr"
    mgr.to_zarr(str(src), zarr_format=3, ngff_version="0.5", drop_singleton=False)
    csv = tmp_path / "b.csv"
    pd.DataFrame([{"input": str(src), "microscope": "zarr", "output": str(tmp_path / "o.zarr"),
                   "chunk_size": "1 1 2 32 32", "num_levels": 1, "num_workers": 2}]).to_csv(csv, index=False)

    class Args:
        input_file = str(csv)

    convert_batch(Args())
    assert zarr.open_group(str(tmp_path / "o.zarr"), mode="r")["0"].shape[-2:] == (64, 64)
