from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(eq=False)
class Tile:
    """One acquired tile of a mosaic: a lazy ``(z, y, x)`` array and where it sits.

    Placement is translation-only: ``origin`` is the position (in ``units``) of
    voxel ``[0, 0, 0]`` in the common sample frame and ``voxel_size`` the voxel
    spacing. All tiles of a mosaic must use the same units and voxel size.

    Parameters
    ----------
    data
        Anything with ``shape``, ``dtype`` and numpy-style slicing that returns
        array data: an ``h5py.Dataset``, a dask or a numpy array. It is only
        read when a fusion needs it.
    origin, voxel_size
        ``(z, y, x)`` position of the first voxel and voxel spacing.
    channel
        Channel (dye) name. Tiles with the same name are fused into one output
        channel.
    group
        Optional label of tiles that should be merged among themselves first,
        e.g. ``"L"``/``"R"`` illumination side. Used by ``blend="midline"``.
    attrs
        Free-form, reader specific metadata (stack, grid position, ...).
    """

    data: Any
    origin: Tuple[float, float, float]
    voxel_size: Tuple[float, float, float]
    channel: str
    group: Optional[str] = None
    attrs: Dict[str, Any] = field(default_factory=dict)
    units: str = "micrometer"

    def __post_init__(self) -> None:
        if len(self.data.shape) != 3:
            raise ValueError(f"Tile data must be 3D (z, y, x), got shape {tuple(self.data.shape)}.")
        self.origin = tuple(float(v) for v in self.origin)
        self.voxel_size = tuple(float(v) for v in self.voxel_size)
        if len(self.origin) != 3 or len(self.voxel_size) != 3:
            raise ValueError("origin and voxel_size must have three (z, y, x) values.")
        if any(v <= 0 for v in self.voxel_size):
            raise ValueError("voxel_size must be positive.")

    @property
    def shape(self) -> Tuple[int, int, int]:
        return tuple(int(s) for s in self.data.shape)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.data.dtype)

    @property
    def extent(self) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        """``(min, max)`` corners of the tile in sample coordinates."""
        end = tuple(o + n * v for o, n, v in zip(self.origin, self.shape, self.voxel_size))
        return self.origin, end
