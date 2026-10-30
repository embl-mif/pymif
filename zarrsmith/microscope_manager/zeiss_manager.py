import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

from bioio import BioImage

from .microscope_manager import MicroscopeManager
from ..utils.metadata import ChannelInfo, build_metadata
from ..utils.units import to_micrometers

logger = logging.getLogger("zarrsmith")


class ZeissManager(MicroscopeManager):
    """
    A manager class for reading and handling .czi datasets.

    This class lazily loads data into a dask array and parses associated .czi metadata.
    A CZI file can hold several scenes; one scene is loaded at a time and
    :meth:`read` can be called again with another ``scene_index``.
    """

    #: ``None`` keeps the native chunking of the first loaded scene.
    DEFAULT_CHUNKS = None

    def __init__(self,
                 path,
                 scene_index: int = 0,
                 scene_name: Optional[str] = None,
                 chunks: Optional[Tuple[int, ...]] = None,
                 ):
        """
        Initialize the ZeissManager.

        Parameters
        ----------
        path : str
            Path to the ``.czi`` file.
        scene_index : int, optional
            Index of the scene to load. Ignored if ``scene_name`` is given.
        scene_name : str, optional
            Name of the scene to load (see :attr:`scenes`).
        chunks : Tuple[int, ...], optional
            Desired chunk shape for the output Dask array. Default keeps the native chunking.
        """
        super().__init__(path, chunks)
        self._image = BioImage(str(self.path), reconstruct_mosaic=True, use_aicspylibczi=False)
        self.scenes = self._image.scenes

        if scene_name:  # None and "" both mean "select by index"
            if scene_name not in self.scenes:
                raise ValueError(f"Invalid scene {scene_name!r}: not in available scenes {self.scenes}.")
            scene_index = self.scenes.index(scene_name)

        logger.info("Scenes: %s. Rerun `read(scene_index)` to load another scene.", self.scenes)
        self.read(scene_index=scene_index)

    def read(self, scene_index: Optional[int] = None) -> None:
        """
        Read one scene of the Zeiss dataset and populate ``self.data`` and ``self.metadata``.

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
        self._image.set_scene(scene_index)

        # Metadata describes the loaded array, so the data is built first.
        self.data = self._build_dask_array()
        self.metadata = self._parse_metadata()

    def _build_dask_array(self) -> List[Any]:
        """Return the selected scene as a single-level TCZYX dask array."""
        array = self._image.get_image_dask_data("TCZYX")
        if self.chunks is None:
            self.chunks = array.chunksize
        return [array.rechunk(self.chunks)]

    @staticmethod
    def _distance(czi_metadata, axis: str, default: Optional[float] = None) -> float:
        """Voxel size along ``axis`` in micrometers (CZI stores meters)."""
        node = czi_metadata.find(f"./Metadata/Scaling/Items/Distance[@Id='{axis}']/Value")
        if node is None or not node.text:
            if default is None:
                raise ValueError(f"CZI metadata has no voxel size for axis {axis}.")
            warnings.warn(f"CZI metadata has no voxel size for axis {axis}; assuming {default}.", stacklevel=3)
            return default
        (um,), _ = to_micrometers((float(node.text),), ("meter",))
        return um

    def _parse_metadata(self) -> Dict[str, Any]:
        """
        Parse metadata of the currently selected scene.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing dataset shape, voxel sizes, channel info, and other metadata.
        """
        czi = self._image
        md = czi.metadata

        scales = (
            self._distance(md, "Z", default=1.0),
            self._distance(md, "Y"),
            self._distance(md, "X"),
        )

        interval = float(md.findtext(".//TimeSeriesSetup/Interval/TimeSpan/Value") or 1.0)
        time_unit = md.findtext(".//TimeSeriesSetup/Interval/TimeSpan/DefaultUnitFormat") or "s"

        colors = [ch.findtext("Color") for ch in md.findall(".//DisplaySetting/Channels/Channel")]
        channels = [
            ChannelInfo(str(name), colors[i] if i < len(colors) else None)
            for i, name in enumerate(czi.channel_names)
        ]

        return build_metadata(
            size=[self.data[0].shape],
            scales=[scales],
            units=("micrometer",) * 3,
            channels=channels,
            dtype=self.data[0].dtype,
            time_increment=interval if interval > 0 else 1.0,
            time_increment_unit=time_unit,
        )
