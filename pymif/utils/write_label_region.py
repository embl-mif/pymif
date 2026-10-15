from __future__ import annotations

from typing import Union

import dask.array as da
import numpy as np
import zarr

from .downsampling import SpatialFactor
from .write_image_region import _scale_index as _scale_index_general, _write_region


def write_label_region(
    root: zarr.Group,
    mode: str,
    data: Union[np.ndarray, da.Array, list[Union[np.ndarray, da.Array]]],
    t: int | slice = slice(None),
    z: int | slice = slice(None),
    y: int | slice = slice(None),
    x: int | slice = slice(None),
    level: int = 0,
    group_name: str = "labels/nuclei",
    downscale_factor: SpatialFactor | None = None,
):
    """
    Internal function that writes label data (pyramid or single array) to a zarr group.

    If a single array is provided, the function automatically generates
    the complete pyramid by upsampling and downsampling from the given level.

    Parameters
    ----------
    root : zarr.Group
        The root Zarr group.
    mode : str
        Zarr open mode ("r+", "a", or "w"). Must allow writing.
    data : np.ndarray, da.Array, or list thereof
        Array(s) to write. If a list, each entry corresponds to one resolution level.
        If a single array, pyramid levels will be generated automatically.
    t, z, y, x : int or slice
        Indices/slices for each dimension.
    level : int
        Pyramid level of the provided `data` if `data` is a single array.
    group_name : str, optional
        Name of the label group inside the root (e.g. "labels/nuclei").
    """
    return _write_region(
        root=root,
        mode=mode,
        data=data,
        selectors={"t": t, "c": slice(None), "z": z, "y": y, "x": x},
        level=level,
        group_name=group_name,
        downscale_factor=downscale_factor,
        expected_data_type="label",
    )

def _scale_index(index_tuple, shape, scale_factor: SpatialFactor, axes="tzyx"):
    """Compatibility wrapper for tests and legacy callers."""
    return _scale_index_general(index_tuple, shape, scale_factor, axes=axes)
