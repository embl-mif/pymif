import json
import logging
import re
import warnings
from typing import Any, Dict, Iterator, List, Optional, Tuple

import dask.array as da
import numpy as np

from .microscope_manager import MicroscopeManager
from ..utils.metadata import ChannelInfo, build_metadata, scale_for_level

logger = logging.getLogger("zarrsmith")

#: Tag of the per-frame stage Z position (µm) in the VSI metadata; slideio does not name it.
_FRAME_Z_TAG = 2014


def _open_scene(path: str, scene_index: int):
    import slideio

    return slideio.open_slide(path, "VSI").get_scene(scene_index)


class VsiSceneArray:
    """Lazy, read-only TCZYX view of one scene of an Olympus/Evident ``.vsi`` file.

    Every ``__getitem__`` call reads the requested block with slideio, so
    ``dask.array.from_array`` can wrap it directly. The slideio scene is opened
    on first use and dropped when pickled, so the object can be sent to worker
    processes.

    ``full_size`` is the ``(width, height)`` of the full-resolution scene. When
    ``shape`` is smaller in Y/X, the array is one of the lower-resolution
    levels stored in the file: blocks are requested in full-resolution
    coordinates and slideio reads them from the matching stored level.
    """

    def __init__(self, path: str, scene_index: int, shape: Tuple[int, ...], dtype,
                 full_size: Optional[Tuple[int, int]] = None):
        self.path = str(path)
        self.scene_index = int(scene_index)
        self.shape: Tuple[int, ...] = tuple(int(s) for s in shape)
        self.dtype = np.dtype(dtype)
        self.ndim = len(self.shape)
        self.full_size = tuple(full_size) if full_size else (self.shape[4], self.shape[3])
        self._scene = None

    @property
    def is_full_resolution(self) -> bool:
        return self.full_size == (self.shape[4], self.shape[3])

    @property
    def exact_factors(self) -> bool:
        """``True`` if the full resolution is an integer multiple of this level in Y and X.

        Only then do blocks read separately join without seams; other levels must
        be read one whole plane at a time (slideio rescales every block it reads).
        """
        (fw, fh), h, w = self.full_size, self.shape[3], self.shape[4]
        return fw % w == 0 and fh % h == 0

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_scene"] = None
        return state

    @property
    def scene(self):
        if self._scene is None:
            self._scene = _open_scene(self.path, self.scene_index)
        return self._scene

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (self.ndim - len(key))
        # Read the bounding box of the request, then apply steps / integer indices with numpy.
        ranges = []
        local = []
        for k, n in zip(key, self.shape):
            if isinstance(k, (int, np.integer)):
                k = int(k) + n if k < 0 else int(k)
                ranges.append((k, k + 1))
                local.append(0)
            elif isinstance(k, slice):
                start, stop, step = k.indices(n)
                stop = max(start, stop)
                ranges.append((start, stop))
                local.append(slice(0, stop - start, step))
            else:
                raise TypeError(f"VsiSceneArray only supports integer and slice indexing, got {k!r}.")

        (t0, t1), (c0, c1), (z0, z1), (y0, y1), (x0, x1) = ranges
        out = np.empty((t1 - t0, c1 - c0, z1 - z0, y1 - y0, x1 - x0), dtype=self.dtype)
        if out.size:
            for i, t in enumerate(range(t0, t1)):
                out[i] = self._read(t, c0, c1, z0, z1, y0, y1, x0, x1)
        return out[tuple(local)]

    def _read(self, t, c0, c1, z0, z1, y0, y1, x0, x1) -> np.ndarray:
        """Read one timepoint as CZYX. slideio returns (Z, Y, X, C) and drops singleton Z and C."""
        nc, nz, ny, nx = c1 - c0, z1 - z0, y1 - y0, x1 - x0
        if self.is_full_resolution:
            rect, size = (x0, y0, nx, ny), (0, 0)
        else:
            sx, sy = self.full_size[0] / self.shape[4], self.full_size[1] / self.shape[3]
            fx0, fx1, fy0, fy1 = round(x0 * sx), round(x1 * sx), round(y0 * sy), round(y1 * sy)
            rect, size = (fx0, fy0, fx1 - fx0, fy1 - fy0), (nx, ny)
        block = self.scene.read_block(
            rect=rect,
            size=size,
            channel_indices=list(range(c0, c1)),
            slices=(z0, z1),
            frames=(t, t + 1),
        )
        return np.asarray(block).reshape(nz, ny, nx, nc).transpose(3, 0, 1, 2)

    def __len__(self) -> int:
        return self.shape[0]

    def __repr__(self) -> str:
        return (f"VsiSceneArray({self.path!r}, scene_index={self.scene_index}, "
                f"shape={self.shape}, dtype={self.dtype}, full_size={self.full_size})")


class EvidentManager(MicroscopeManager):
    """
    A manager class for reading Olympus/Evident ``.vsi`` datasets (cellSens).

    A ``.vsi`` file only holds the metadata: the pixels live in the companion
    folder ``_<name>_/stack*/frame_t*.ets``, which must sit next to the ``.vsi``
    file. Tiled acquisitions are stored already stitched, so a 2 x 2 tile scan
    is read as one large image. The pixels are read lazily with slideio.

    A VSI file can hold several scenes (e.g. an overview and the acquired
    stack); one scene is loaded at a time and :meth:`read` can be called again
    with another ``scene_index``.

    cellSens stores lower-resolution levels for large (e.g. tiled) images.
    These are exposed as further pyramid levels, downsampled in Y and X only,
    which makes napari 3D rendering of the raw file fast. Single-tile
    acquisitions usually have only the full resolution.
    :meth:`build_pyramid` (used by ``zarrsmith convert``) always rebuilds the
    pyramid from the full resolution.
    """

    DEFAULT_CHUNKS = (1, 1, 8, 2048, 2048)

    #: Largest chunk (bytes) of a level read one whole plane at a time.
    MAX_PLANE_CHUNK_BYTES = 128 * 1024 ** 2

    def __init__(self,
                 path,
                 scene_index: Optional[int] = 0,
                 scene_name: Optional[str] = None,
                 chunks: Optional[Tuple[int, ...]] = None,
                 native_levels: bool = True,
                 ):
        """
        Initialize the EvidentManager.

        Parameters
        ----------
        path : str
            Path to the ``.vsi`` file.
        scene_index : int, optional
            Index of the scene to load. Ignored if ``scene_name`` is given.
        scene_name : str, optional
            Name of the scene to load (see :attr:`scenes`).
        chunks : Tuple[int, ...], optional
            Desired TCZYX chunk shape for the output Dask array. Default is :attr:`DEFAULT_CHUNKS`.
        native_levels : bool, optional
            Expose the lower-resolution levels stored in the file as extra
            pyramid levels. If ``False``, only the full resolution is read.
        """
        import slideio

        super().__init__(path, chunks)
        self.native_levels = native_levels
        self._slide = slideio.open_slide(str(self.path), "VSI")
        self.scenes = [self._slide.get_scene(i).name for i in range(self._slide.num_scenes)]
        self._raw_metadata = None

        if scene_name:  # None and "" both mean "select by index"
            if scene_name not in self.scenes:
                raise ValueError(f"Invalid scene {scene_name!r}: not in available scenes {self.scenes}.")
            scene_index = self.scenes.index(scene_name)

        logger.info("Scenes: %s. Rerun `read(scene_index)` to load another scene.", self.scenes)
        self.read(scene_index=scene_index)

    def read(self, scene_index: Optional[int] = None) -> None:
        """
        Read one scene of the VSI dataset and populate ``self.data`` and ``self.metadata``.

        Parameters
        ----------
        scene_index : int, optional
            Scene to load. Defaults to the scene currently selected.
        """
        if scene_index is None:
            scene_index = getattr(self, "scene_index", 0)
        if not 0 <= scene_index < len(self.scenes):
            raise ValueError(
                f"Invalid scene index {scene_index}, only {len(self.scenes)} scenes available: {self.scenes}"
            )
        self.scene_index = scene_index
        self.scene_name = self.scenes[scene_index]
        self._scene = self._slide.get_scene(scene_index)

        # Metadata describes the loaded array, so the data is built first.
        self.data = self._build_dask_array()
        self.metadata = self._parse_metadata()

    def _level_sizes(self) -> List[Tuple[int, int]]:
        """``(width, height)`` of the full resolution and of every stored lower-resolution level."""
        s = self._scene
        sizes = [tuple(s.size)]
        if self.native_levels:
            for level in range(1, getattr(s, "num_zoom_levels", 1)):
                size = s.get_zoom_level_info(level).size
                w, h = int(size.width), int(size.height)
                if 0 < w < sizes[-1][0] and 0 < h < sizes[-1][1]:
                    sizes.append((w, h))
        return sizes

    def _level_chunks(self, array: VsiSceneArray):
        """Chunks of one level: the requested ones, or whole planes when blocks would show seams."""
        chunks = self._normalize_chunks(self.chunks, array.shape)
        if array.exact_factors:
            return chunks
        t, c, z, y, x = array.shape
        nz = chunks[2] if chunks != "auto" else 8
        nz = max(1, min(nz, self.MAX_PLANE_CHUNK_BYTES // (y * x * array.dtype.itemsize), z))
        return (1, 1, nz, y, x)

    def _build_dask_array(self) -> List[da.Array]:
        """Return the selected scene as TCZYX dask arrays, one per stored resolution level."""
        s = self._scene
        dtypes = {np.dtype(s.get_channel_data_type(c)) for c in range(s.num_channels)}
        if len(dtypes) != 1:
            raise ValueError(f"Channels of scene {self.scene_name!r} have different data types: {dtypes}.")
        dtype = dtypes.pop()

        sizes = self._level_sizes()
        levels = []
        for width, height in sizes:
            shape = (s.num_t_frames, s.num_channels, s.num_z_slices, height, width)
            array = VsiSceneArray(self.path, self.scene_index, shape, dtype, full_size=sizes[0])
            array._scene = s
            levels.append(da.from_array(array, chunks=self._level_chunks(array), lock=False, asarray=False))
        return levels

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def raw_metadata(self) -> Dict[str, Any]:
        """The full VSI metadata tree, as decoded by slideio (nested ``name``/``tag``/``value`` nodes)."""
        if self._raw_metadata is None:
            self._raw_metadata = json.loads(self._slide.raw_metadata)
        return self._raw_metadata

    @staticmethod
    def _nodes(node: Dict[str, Any], name: Optional[str] = None, tag: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """All nodes below ``node`` (included) matching ``name`` and/or ``tag``."""
        if (name is None or node.get("name") == name) and (tag is None or node.get("tag") == tag):
            yield node
        if isinstance(node.get("value"), list):
            for child in node["value"]:
                yield from EvidentManager._nodes(child, name, tag)

    @staticmethod
    def _child_value(node: Dict[str, Any], name: str) -> Any:
        """Value of the first direct child called ``name``, or ``None``."""
        for child in node.get("value") or []:
            if isinstance(child, dict) and child.get("name") == name:
                return child.get("value")
        return None

    @staticmethod
    def _number(node: Dict[str, Any]) -> Optional[float]:
        """Numeric value of a VSI property node (stored in its ``Value`` child)."""
        value = EvidentManager._child_value(node, "Value")
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _scene_volume(self) -> Optional[Dict[str, Any]]:
        """Metadata subtree (``Multidimensional image volume``) describing the selected scene."""
        volumes = list(self._nodes(self.raw_metadata, "Multidimensional image volume"))
        for volume in volumes:
            for stack in self._nodes(volume, "Multidimensional stack properties"):
                if self._child_value(stack, "Stack name") == self.scene_name:
                    return volume
        return volumes[self.scene_index] if len(volumes) == len(self.scenes) else None

    def _channel_colors(self, volume: Optional[Dict[str, Any]]) -> Dict[str, str]:
        """Display color of each channel name, from the last entry of its display LUT."""
        colors = {}
        if volume is None:
            return colors
        for props in self._nodes(volume, "Channel properties"):
            name = self._child_value(props, "Channel name")
            lut = self._child_value(props, "Stack display LUT")
            lut = self._child_value({"value": lut}, "Value") if isinstance(lut, list) else None
            match = re.search(r"red\s*=\s*(\d+),\s*green\s*=\s*(\d+),\s*blue\s*=\s*(\d+)", str(lut or ""))
            if name and match:
                colors[name] = "".join(f"{int(v):02X}" for v in match.groups())
        return colors

    def _z_step(self, volume: Optional[Dict[str, Any]]) -> float:
        """Z step in micrometers.

        Taken from the scene calibration when present, otherwise from the
        stage Z position recorded for every frame (multi-tile acquisitions
        only store the latter).
        """
        s = self._scene
        if s.z_resolution and s.z_resolution > 0:
            return float(s.z_resolution) * 1e6
        if s.num_z_slices < 2:
            return 1.0
        if volume is not None:
            positions = [
                self._number(node)
                for frame in self._nodes(volume, "Frame properties")
                for node in self._nodes(frame, tag=_FRAME_Z_TAG)
            ]
            positions = np.array([p for p in positions if p is not None])
            if positions.size and np.ptp(positions) > 0:
                return round(float(np.ptp(positions)) / (s.num_z_slices - 1), 6)
        warnings.warn("VSI metadata has no z step; assuming 1 micrometer.", stacklevel=3)
        return 1.0

    def _parse_metadata(self) -> Dict[str, Any]:
        """
        Parse metadata of the currently selected scene.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing dataset shape, voxel sizes, channel info, and other metadata.
        """
        s = self._scene
        volume = self._scene_volume()

        rx, ry = s.resolution  # meters per pixel
        if not rx or not ry:
            warnings.warn("VSI metadata has no pixel size; assuming 1 micrometer.", stacklevel=2)
            rx = ry = 1e-6
        scales = (self._z_step(volume), float(ry) * 1e6, float(rx) * 1e6)

        colors = self._channel_colors(volume)
        channels = []
        for c in range(s.num_channels):
            name = s.get_channel_name(c) or f"ch{c}"
            channels.append(ChannelInfo(name, colors.get(name)))

        base_shape = self.data[0].shape[2:]
        interval = float(s.t_resolution or 0.0)
        return build_metadata(
            size=[level.shape for level in self.data],
            scales=[scale_for_level(scales, base_shape, level.shape[2:]) for level in self.data],
            units=("micrometer",) * 3,
            channels=channels,
            dtype=self.data[0].dtype,
            time_increment=interval if interval > 0 else 1.0,
            time_increment_unit="s",
        )
