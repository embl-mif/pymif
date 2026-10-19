"""Fields of a mosaic: groups of tiles that touch or overlap (connected components).

A mosaic can hold several separate fields, e.g. two samples or two regions of one
sample imaged in the same acquisition. Fusing tiles of different fields into one
image puts them on a single canvas that spans all of them, with blank space in
between. Every tile therefore gets a ``field`` label when a
:class:`~pymif.mosaic_manager.MosaicManager` is loaded, fusion uses the first
field by default, and fusing tiles of more than one field is refused
(:class:`MultipleFieldsError`).
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Sequence

import numpy as np

from .tile import Tile

logger = logging.getLogger("pymif")

#: Prefix of the field names: ``field0``, ``field1``, ...
FIELD_PREFIX = "field"


class MultipleFieldsError(ValueError):
    """Tiles of more than one field were given to one fusion, which is not allowed."""


def natural_key(name: str):
    """Sort key putting ``field2`` before ``field10``."""
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", str(name))]


def connected_components(tiles: Sequence[Tile], tolerance: Optional[Sequence[float]] = None) -> List[int]:
    """Component index of every tile: tiles whose boxes overlap or touch are connected.

    Two tiles touch when their ``(z, y, x)`` boxes (:attr:`Tile.extent`) overlap or
    are less than ``tolerance`` apart on every axis (default: one voxel). Components
    are numbered by their position: by the ``y`` then ``x`` then ``z`` start of their
    bounding box, so the numbering does not depend on the tile order.
    """
    n = len(tiles)
    if n == 0:
        return []
    lo = np.array([t.extent[0] for t in tiles], float)
    hi = np.array([t.extent[1] for t in tiles], float)
    tol = np.array(tolerance if tolerance is not None else np.max([t.voxel_size for t in tiles], axis=0), float)

    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components as _components

    # boxes i and j touch when they overlap, or are closer than tol, on every axis
    touch = np.all((lo[:, None, :] <= hi[None, :, :] + tol) & (lo[None, :, :] <= hi[:, None, :] + tol), axis=2)
    _, roots = _components(csr_matrix(touch), directed=False)

    starts: Dict[int, np.ndarray] = {}
    for i, r in enumerate(roots):
        starts[r] = lo[i] if r not in starts else np.minimum(starts[r], lo[i])
    order = sorted(starts, key=lambda r: (starts[r][1], starts[r][2], starts[r][0]))
    index = {r: k for k, r in enumerate(order)}
    return [index[r] for r in roots]


def label_fields(tiles: Sequence[Tile], tolerance: Optional[Sequence[float]] = None) -> List[str]:
    """Set ``tile.field`` on every tile to the name of its connected component.

    Tiles that already have a field (given by the reader) keep it. Returns the
    field names in order. See :func:`connected_components`.
    """
    tiles = list(tiles)
    if tiles and all(t.field is not None for t in tiles):
        return fields_of(tiles)
    todo = [t for t in tiles if t.field is None]
    for t, k in zip(todo, connected_components(todo, tolerance)):
        t.field = f"{FIELD_PREFIX}{k}"
    names = fields_of(tiles)
    if len(names) > 1:
        counts = {f: sum(t.field == f for t in tiles) for f in names}
        logger.info("Mosaic has %d separate fields: %s", len(names),
                    ", ".join(f"{f} ({c} tiles)" for f, c in counts.items()))
    return names


def fields_of(tiles: Sequence[Tile]) -> List[str]:
    """Names of the fields of ``tiles``, in natural order (tiles without one ignored)."""
    return sorted({t.field for t in tiles if t.field is not None}, key=natural_key)


def multiple_fields_message(fields: Sequence[str]) -> str:
    return (f"Tiles from {len(fields)} separate fields ({', '.join(fields)}) cannot be fused into one image: "
            "the fusion will not be done. Fuse one field at a time with field=... (CLI: --fields).")


def check_single_field(tiles: Sequence[Tile]) -> List[str]:
    """Raise :class:`MultipleFieldsError` (and log the error) if ``tiles`` come from
    more than one field. Returns the fields of ``tiles``."""
    fields = fields_of(tiles)
    if len(fields) > 1:
        message = multiple_fields_message(fields)
        logger.error(message)
        raise MultipleFieldsError(message)
    return fields
