from __future__ import annotations

from pathlib import Path
from typing import Tuple

import h5py
import numpy as np


class H5Array:
    """Lazy, picklable view of one HDF5 dataset.

    The file is opened for every read, so the object holds no handle: it can be
    sent to worker processes (which is what lets :func:`~zarrsmith.mosaic_manager.fuse_tiles`
    read many tiles in parallel despite h5py's per-process lock) and needs no
    ``close()``. It behaves like a read-only array: ``shape``, ``dtype``, ``ndim``
    and numpy-style slicing, so ``dask.array.from_array`` accepts it directly.
    """

    #: Tells ``fuse_tiles(executor="auto")`` that this data can be used in worker processes.
    picklable = True

    def __init__(self, path: str | Path, dataset: str = "Data", shape=None, dtype=None):
        """Pass ``shape`` and ``dtype`` when they are known (e.g. from metadata) to
        avoid opening the file, which is slow on network drives."""
        self.path = str(path)
        self.dataset = dataset
        if shape is None or dtype is None:
            with h5py.File(self.path, "r") as f:
                d = f[dataset]
                shape = d.shape if shape is None else shape
                dtype = d.dtype if dtype is None else dtype
        self.shape: Tuple[int, ...] = tuple(int(s) for s in shape)
        self.dtype = np.dtype(dtype)
        self.ndim = len(self.shape)

    def __getitem__(self, key):
        with h5py.File(self.path, "r") as f:
            return f[self.dataset][key]

    def __len__(self) -> int:
        return self.shape[0]

    def __repr__(self) -> str:
        return f"H5Array({self.path!r}, {self.dataset!r}, shape={self.shape}, dtype={self.dtype})"
