from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import dask.array as da
import numpy as np

from .microscope_manager import MicroscopeManager
from ..utils.ngff import parse_channel_color
from ..utils.axes import normalize_axes, infer_axes_from_ndim, spatial_axes_in_order, normalize_data_type


class ArrayManager(MicroscopeManager):
    """
    Create a MicroscopeManager instance from in-memory NumPy or Dask array(s)
    with user-defined metadata. Supports single resolution or multiscale pyramid.

    The default remains legacy TCZYX for 5D arrays, but ``metadata['axes']`` may
    be any unique combination of ``t``, ``c``, ``z``, ``y`` and ``x`` whose length
    matches the input arrays.
    """

    #: ``chunks=None`` means "let dask decide" for in-memory arrays.
    DEFAULT_CHUNKS = None

    def __init__(
        self,
        array: Union[np.ndarray, da.Array, List[Union[np.ndarray, da.Array]]],
        metadata: Dict[str, Any],
        chunks: Optional[Tuple[int, ...]] = (1, 1, 8, 4096, 4096),
    ):
        """Initialize ArrayManager with a single array or pyramid.

        Parameters
        ----------
        array
            A NumPy/Dask array or list of arrays.  All levels must have the same
            dimensionality.
        metadata
            Metadata dictionary.  ``axes`` may be any subset of ``tczyx`` and
            ``data_type`` may be ``"intensity"`` or ``"label"``.
        chunks
            Dask chunk shape for NumPy inputs.  When ``None`` or incompatible with
            the dimensionality, automatic chunking is used.
        """
        super().__init__(chunks=chunks)
        # The raw user input is kept apart from ``self.data`` (always a list of
        # dask arrays once :meth:`read` has run).
        self._source = array
        self._user_metadata = dict(metadata)
        self.read()

    def read(self) -> None:
        """Convert the inputs given to the constructor into dask levels and normalized metadata.

        Unlike the file-based managers there is nothing to load from disk; the
        arrays and metadata come from the constructor. The data is built first
        because the metadata defaults (size, dtype, channel count) depend on it.
        """
        self.data = self._build_dask_array()
        self.metadata = self._parse_metadata()

    def _build_dask_array(self) -> List[da.Array]:
        """Validate the input levels and return them as dask arrays."""
        levels = self._source if isinstance(self._source, list) else [self._source]
        if not levels:
            raise ValueError("array cannot be empty.")

        axes = normalize_axes(
            self._user_metadata.get("axes") or infer_axes_from_ndim(levels[0].ndim),
            ndim=levels[0].ndim,
        )
        out = []
        for level in levels:
            if getattr(level, "ndim", None) != len(axes):
                raise ValueError(
                    f"Each level ndim must match metadata['axes']={''.join(axes)!r}."
                )
            if isinstance(level, np.ndarray):
                level = da.from_array(level, chunks=self._normalize_chunks(self.chunks, level.shape))
            elif not isinstance(level, da.Array):
                raise TypeError("array levels must be NumPy arrays or Dask arrays.")
            out.append(level)
        return out

    def _parse_metadata(self) -> Dict[str, Any]:
        """Fill in defaults of the user metadata from the already built ``self.data``."""
        metadata = dict(self._user_metadata)
        axes = normalize_axes(metadata.get("axes") or infer_axes_from_ndim(self.data[0].ndim), ndim=self.data[0].ndim)
        metadata["axes"] = "".join(axes)
        metadata.setdefault("dtype", str(self.data[0].dtype))
        metadata["data_type"] = normalize_data_type(metadata.get("data_type"))
        metadata.setdefault("size", [tuple(level.shape) for level in self.data])
        metadata.setdefault("chunksize", [tuple(level.chunksize) for level in self.data])

        spatial_axes = spatial_axes_in_order(axes)
        user_scales = metadata.get("scales", [])
        if isinstance(user_scales, list) and len(user_scales) == len(self.data):
            scales = [tuple(scale) for scale in user_scales]
        else:
            if user_scales:
                warnings.warn(
                    "Metadata 'scales' length does not match pyramid levels. "
                    "Falling back to automatic scale generation.",
                    stacklevel=3,
                )
            base_scale = tuple(user_scales[0]) if user_scales else tuple(1.0 for _ in spatial_axes)
            if len(base_scale) != len(spatial_axes):
                base_scale = tuple(1.0 for _ in spatial_axes)
            scales = [tuple(float(s) * (2**i) for s in base_scale) for i in range(len(self.data))]
        metadata["scales"] = scales

        metadata.setdefault("units", tuple("micrometer" for _ in spatial_axes))
        if "c" in axes:
            c_size = int(self.data[0].shape[axes.index("c")])
            metadata.setdefault("channel_names", [f"Channel {i}" for i in range(c_size)])
            metadata.setdefault("channel_colors", ["FFFFFF"] * c_size)
            metadata["channel_colors"] = [
                parse_channel_color(c, i) for i, c in enumerate(metadata["channel_colors"])
            ]
        else:
            metadata.setdefault("channel_names", [])
            metadata.setdefault("channel_colors", [])

        if "t" in axes:
            metadata.setdefault("time_increment", 1.0)
            metadata.setdefault("time_increment_unit", "s")
        else:
            metadata.setdefault("time_increment", None)
            metadata.setdefault("time_increment_unit", None)
        metadata.setdefault("plane_files", None)
        return metadata
