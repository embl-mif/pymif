from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import numpy as np


class TiffStackArray:
    """Lazy, picklable ``(z, y, x)`` stack of single-plane TIFF files (one file per plane).

    Like :class:`~pymif.mosaic_manager.H5Array`, it holds no open file, so it can be
    sent to worker processes and needs no ``close()``. Slicing reads only the planes it
    needs; every plane is decoded in full (``chunks`` is ``(1, y, x)``, which the fusion
    uses to size its regions).

    Parameters
    ----------
    paths
        One file per plane, in z order. ``None`` marks a missing plane, read as zeros.
    shape, dtype
        ``(z, y, x)`` shape and data type, from the metadata (no file is opened).
    folder
        Folder of relative ``paths`` (keeps memory low for plates with many planes).
    transform
        Optional picklable callable applied to every whole plane after reading (e.g. a
        flat-field correction). It returns floats, which are rounded and clipped to ``dtype``.
    """

    #: Tells ``fuse_tiles(executor="auto")`` that this data can be used in worker processes.
    picklable = True

    def __init__(self, paths: Sequence[Optional[str | Path]], shape, dtype, folder: Optional[str | Path] = None,
                 transform: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        self.paths = [None if p is None else str(p) for p in paths]
        self.folder = None if folder is None else str(folder)
        self.shape: Tuple[int, ...] = tuple(int(s) for s in shape)
        if len(self.shape) != 3 or self.shape[0] != len(self.paths):
            raise ValueError(f"shape {self.shape} does not match {len(self.paths)} plane file(s).")
        self.dtype = np.dtype(dtype)
        self.ndim = 3
        self.chunks = (1,) + self.shape[1:]
        self.transform = transform
        self._ready: Optional[Tuple[int, np.ndarray]] = None  # plane read ahead by prefetch()

    def path(self, z: int) -> Optional[str]:
        """Full path of plane ``z`` (``None`` for a missing plane)."""
        p = self.paths[z]
        return p if p is None or self.folder is None else os.path.join(self.folder, p)

    def prefetch(self, z: int) -> None:
        """Read plane ``z`` ahead (e.g. in a thread, for many tiles at once). The next
        read of this stack uses it, or drops it if it asks for another plane."""
        self._ready = (z, self._read(z))

    def _plane(self, z: int) -> np.ndarray:
        ready, self._ready = self._ready, None
        if ready is not None and ready[0] == z:
            return ready[1]
        return self._read(z)

    def _read(self, z: int) -> np.ndarray:
        path = self.path(z)
        if path is None:
            return np.zeros(self.shape[1:], self.dtype)
        import tifffile

        plane = tifffile.imread(path)
        if plane.shape != self.shape[1:]:
            raise ValueError(f"{path}: plane shape {plane.shape} does not match the metadata {self.shape[1:]}.")
        if self.transform is None:
            return plane.astype(self.dtype, copy=False)
        out = self.transform(plane)
        if np.issubdtype(self.dtype, np.integer):
            info = np.iinfo(self.dtype)
            out = np.clip(np.rint(out), info.min, info.max)
        return out.astype(self.dtype)

    def __getitem__(self, key) -> np.ndarray:
        key = key if isinstance(key, tuple) else (key,)
        if any(k is Ellipsis for k in key):
            i = key.index(Ellipsis)
            key = key[:i] + (slice(None),) * (3 - len(key) + 1) + key[i + 1:]
        key = key + (slice(None),) * (3 - len(key))
        zkey, yx = key[0], key[1:]
        if isinstance(zkey, (int, np.integer)):
            return self._plane(range(self.shape[0])[zkey])[yx]
        planes = range(self.shape[0])[zkey] if isinstance(zkey, slice) else np.arange(self.shape[0])[zkey]
        yx_shape = np.broadcast_to(np.zeros((), bool), self.shape[1:])[yx].shape  # no allocation
        out = np.empty((len(planes),) + yx_shape, self.dtype)
        for i, z in enumerate(planes):
            out[i] = self._plane(int(z))[yx]
        return out

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_ready"] = None
        return state

    def __len__(self) -> int:
        return self.shape[0]

    def __repr__(self) -> str:
        first = next((self.path(z) for z in range(len(self.paths)) if self.paths[z] is not None), None)
        return f"TiffStackArray({len(self.paths)} planes, first={first!r}, shape={self.shape}, dtype={self.dtype})"
