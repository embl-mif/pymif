from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import dask.array as da

if TYPE_CHECKING:
    import napari

logger = logging.getLogger("zarrsmith")


class MicroscopeManager(ABC):
    """
    Abstract base class for managing microscope image datasets.

    Provides shared functionality for reading, writing, visualizing,
    and managing multiscale image data and metadata.

    Subclasses implement two hooks and inherit :meth:`read`:

    * :meth:`_parse_metadata` -- return the normalized metadata dictionary
      (see :func:`zarrsmith.utils.metadata.build_metadata`);
    * :meth:`_build_dask_array` -- return one lazy dask array per pyramid level.

    Managers can be used as context managers so that any open file handle is
    closed automatically::

        with mm.LuxendoManager("path/to/dataset") as lux:
            lux.to_zarr("out.zarr")
    """

    #: Chunk shape used when the constructor receives ``chunks=None``.
    DEFAULT_CHUNKS: Optional[Tuple[int, ...]] = (1, 1, 8, 4096, 4096)

    def __init__(self, path: Optional[str | Path] = None, chunks: Optional[Tuple[int, ...]] = None):
        """Initialize the common manager state.

        Subclasses populate :attr:`data` with one dask array per pyramid level
        and :attr:`metadata` with the normalized ZarrSmith metadata schema.
        ``_open_files`` stores any file handles that should be closed through
        :meth:`close`.

        Parameters
        ----------
        path : str or Path, optional
            Dataset location, stored as a :class:`~pathlib.Path` in :attr:`path`.
        chunks : tuple of int, optional
            Dask chunk shape. Falls back to :attr:`DEFAULT_CHUNKS` when ``None``.
        """
        self.data: List[da.Array] = []
        self.metadata: Dict[str, Any] = {}
        self._open_files: list = []
        self.path = Path(path) if path is not None else None
        self.chunks = chunks if chunks is not None else self.DEFAULT_CHUNKS

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def read(self) -> None:
        """Parse the metadata and build the lazy dask pyramid.

        Populates :attr:`metadata` and :attr:`data` in place; nothing is
        returned. Called automatically by the constructors.
        """
        self.metadata = self._parse_metadata()
        self.data = self._build_dask_array()

    @abstractmethod
    def _parse_metadata(self) -> Dict[str, Any]:
        """Return the metadata dictionary describing the dataset."""

    @abstractmethod
    def _build_dask_array(self) -> List[da.Array]:
        """Return one dask array per pyramid level, in the order given by ``metadata["axes"]``."""

    @staticmethod
    def _normalize_chunks(chunks, shape: Tuple[int, ...]):
        """Make ``chunks`` valid for an array of ``shape``.

        ``None``, ``"auto"``, non-iterables and tuples of the wrong length give
        ``"auto"``; otherwise every chunk is clipped to ``[1, axis size]``.
        """
        if chunks is None or chunks == "auto":
            return "auto"
        try:
            chunk_tuple = tuple(int(c) for c in chunks)
        except TypeError:
            return "auto"
        if len(chunk_tuple) != len(shape):
            return "auto"
        return tuple(max(1, min(c, int(s))) for c, s in zip(chunk_tuple, shape))

    def to_zarr(self, 
                path: str,
                **kwargs) -> None:
        """Write the current dataset to an OME-Zarr store.

        Parameters
        ----------
        path : str
            Output zarr path.
        **kwargs
            Keyword arguments forwarded to :class:`~zarrsmith.utils.ngff.ZarrWriteConfig`,
            such as ``ngff_version``, ``zarr_format``, ``compressor``,
            ``compressor_level``, ``overwrite``, ``chunks``, ``shards``,
            ``shard_target_mb``, ``shard_exclude_axes`` or ``drop_singleton``
            (default ``True``: singleton t/c/z axes are removed on write).
        """
        from ..utils.to_zarr import to_zarr as _to_zarr
        from ..utils.ngff import ZarrWriteConfig
        return _to_zarr(path, 
                      self.data, 
                      self.metadata, 
                      config=ZarrWriteConfig(**kwargs)
                      )

    def visualize(
        self,
        start_level: int = 0,
        stop_level: int = -1,
        in_memory: bool = False,
        viewer: "napari.Viewer | None" = None,
    ) -> Any:
        """Open the dataset in napari using the shared visualization helper.

        Parameters
        ----------
        start_level, stop_level : int
            Pyramid levels to expose. ``stop_level=-1`` means all available
            levels from ``start_level`` onward.
        in_memory : bool
            If ``True``, compute the selected levels before handing them to
            napari. Otherwise keep the dask-backed lazy representation.
        viewer : napari.Viewer | None
            Existing napari viewer to reuse. When omitted, a new viewer is
            created by the helper function.
        """
        from ..utils.visualize import visualize as _visualize
        return _visualize(
            self.data,
            self.metadata,
            start_level=start_level,
            stop_level=stop_level,
            in_memory=in_memory,
            viewer=viewer,
        )

    def build_pyramid(self, 
                      num_levels: Optional[int] = 3, 
                      downscale_factor: int | Sequence[int] | None = 2,
                      start_level: Optional[int] = 0,
                      ) -> None:
        """Build additional pyramid levels from the current base-resolution data.

        The resulting data and scale metadata replace ``self.data`` and
        ``self.metadata`` in-place. This is mainly useful for managers that
        initially expose only one resolution level.
        """
        from ..utils.pyramid import build_pyramid as _build_pyramid
        self.data, self.metadata = _build_pyramid(
            self.data, self.metadata, 
            num_levels=num_levels, 
            downscale_factor=2 if downscale_factor is None else downscale_factor,
            start_level = start_level,
        )

    def close(self) -> None:
        """Close all open resources, such as file handles."""
        for f in self._open_files:
            try:
                f.close()
            except Exception as e:
                logger.warning("Failed to close file: %s", e)
        self._open_files = []

    def __enter__(self) -> "MicroscopeManager":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def reorder_channels(self, new_order: List[int]) -> None:
        """
        Reorder the channel axis and update channel-related metadata.

        Parameters
        ----------
            new_order : List[int]
                A permutation of the channel indices.
        """
        from ..utils.dataset_ops import reorder_channels as _reorder

        self.data = _reorder(self.data, self.metadata, new_order)
        logger.info("Channels reordered to %s", list(new_order))

    def update_metadata(self, updates: Dict[str, Any]) -> None:
        """
        Safely update entries in the metadata dictionary with validation.

        Parameters
        ----------
            updates : Dict[str, Any]
                Dictionary of key-value updates.

                Supports:
                    - channel_names (list[str])
                    - channel_colors (list[str]): valid matplotlib colors or hex code
                    - scales (list[tuple])
                    - time_increment (float)
                    - time_increment_unit (str)
                    - units (tuple[str])
                    - data_type (``"intensity"`` or ``"label"``)

        Warnings
        ----------
            Unknown keys and channel updates that do not match the dataset
            are skipped with a warning; invalid values raise.
        """
        from ..utils.dataset_ops import apply_metadata_updates, known_updates

        apply_metadata_updates(self.data, self.metadata, known_updates(updates))

    def subset_dataset(self,
                    T: Optional[Sequence[int]] = None,
                    C: Optional[Sequence[int]] = None,
                    Z: Optional[Sequence[int]] = None,
                    Y: Optional[Sequence[int]] = None,
                    X: Optional[Sequence[int]] = None,
                    rebuild_pyramid: bool = True
                    ) -> None:
        """
        Subset the dataset by timepoints, channels, or spatial coordinates.

        Parameters
        ----------
        T, C, Z, Y, X : Optional[Sequence[int]]
            Optional sequences of indices for each axis.
            Must be uniformly spaced. For example:
            dataset.subset_dataset(T=np.arange(0, 10, 2), Z=[0,1,2])
        rebuild_pyramid : bool
            Rebuild the same number of pyramid levels (with the original
            per-axis downscale factors) after subsetting.

        Raises
        -------
        ValueError
            if index spacing is not uniform or out of bounds.
        """
        from ..utils.dataset_ops import subset_levels

        self.data, self.metadata = subset_levels(
            self.data, self.metadata,
            T=T, C=C, Z=Z, Y=Y, X=X,
            rebuild_pyramid=rebuild_pyramid,
        )
        logger.info("Dataset subset complete.")
