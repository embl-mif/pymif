from __future__ import annotations

from typing import Any, Tuple

from .zarr_manager import ZarrManager


class ZarrV04Manager(ZarrManager):
    """Backward-compatible reader for legacy NGFF/OME-Zarr v0.4 stores.

    The original implementation assumed five-dimensional ``TCZYX`` data.  This
    compatibility wrapper now delegates to :class:`ZarrManager`, which supports
    both v0.4 and v0.5 metadata layouts and any unique subset of the image axes
    ``t``, ``c``, ``z``, ``y`` and ``x``.
    """

    def __init__(
        self,
        path,
        chunks: Tuple[int, ...] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        super().__init__(
            path=path,
            chunks=chunks,
            mode="r",
            metadata=metadata,
            ngff_version="0.4",
            zarr_format=2,
        )
