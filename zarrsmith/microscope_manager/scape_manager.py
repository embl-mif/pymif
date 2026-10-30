import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dask.array as da
from dask import delayed
from tifffile import TiffFile, imread

from .microscope_manager import MicroscopeManager
from ..utils.axes import to_tczyx
from ..utils.metadata import ChannelInfo, build_metadata
from ..utils.units import to_micrometers


class ScapeManager(MicroscopeManager):
    """
    MicroscopeManager for SCAPE microscope datasets where metadata is stored in an external
    .xlif file located in a 'Metadata' folder next to the OME-TIFF.
    """

    DEFAULT_CHUNKS = (1, 1, 8, 1024, 1024)

    #: Leica LUT names that matplotlib would spell differently (``green`` is dark green there).
    _LUT_COLORS = {
        "Cyan": "00FFFF", "Magenta": "FF00FF", "Yellow": "FFFF00", "Red": "FF0000",
        "Green": "00FF00", "Blue": "0000FF", "Gray": "808080", "Grey": "808080", "White": "FFFFFF",
    }

    #: ``DimID`` -> axis in Leica/LAS AF ``<DimensionDescription>`` elements.
    _DIM_IDS = {"x": 1, "y": 2, "z": 3, "t": 4}

    def __init__(
        self,
        path: Optional[str] = None,
        chunks: Optional[Tuple[int, ...]] = None,
        *,
        ome_tiff_path: Optional[str] = None,
    ):
        """Open a Leica SCAPE dataset described by an OME-TIFF and companion XLIF file.

        Parameters
        ----------
        path : str
            Path to the OME-TIFF file. A ``Metadata/<name>.xlif`` file is expected next to it.
        chunks : tuple of int, optional
            Dask chunk shape in TCZYX order. Defaults to :attr:`DEFAULT_CHUNKS`.
        ome_tiff_path : str, optional
            Backward-compatible keyword alias of ``path``.
        """
        if path is None:
            path = ome_tiff_path
        if path is None:
            raise TypeError("ScapeManager requires the path of the OME-TIFF file.")
        super().__init__(path, chunks)
        self.ome_tiff_path = self.path
        self.read()

    # ---------- Path resolution helpers ----------

    @staticmethod
    def _strip_ome_tiff_suffix(filename: str) -> str:
        """
        Convert typical Leica names like:
          'p1 (2).ome.tif'  -> 'p1 (2)'
          'p1 (2).ome.tiff' -> 'p1 (2)'
          'p1 (2).tif'      -> 'p1 (2)'
        """
        stripped = re.sub(r"\.(ome\.)?tiff?$", "", filename, flags=re.IGNORECASE)
        # Fallback: drop only the last suffix
        return stripped if stripped != filename else Path(filename).stem

    def _find_xlif_for_ome_tiff(self) -> Path:
        """
        Given an OME-TIFF path, look for ``<ome_dir>/Metadata/<base_name>.xlif``.

        If not found, fall back to a ``.xlif`` inside ``<ome_dir>/Metadata`` whose
        name contains the base name, and finally to the first ``.xlif`` there.
        """
        if not self.ome_tiff_path.is_file():
            raise FileNotFoundError(f"OME-TIFF file not found: {self.ome_tiff_path}")

        metadata_dir = self.ome_tiff_path.parent / "Metadata"
        if not metadata_dir.is_dir():
            raise FileNotFoundError(f"'Metadata' directory not found next to OME-TIFF: {metadata_dir}")

        base_name = self._strip_ome_tiff_suffix(self.ome_tiff_path.name)
        expected = metadata_dir / f"{base_name}.xlif"
        if expected.exists():
            return expected

        xlifs = sorted(metadata_dir.glob("*.xlif"))
        if not xlifs:
            raise FileNotFoundError(f"No .xlif files found in: {metadata_dir} (expected: {expected.name})")

        base_lower = base_name.lower()
        return next((x for x in xlifs if base_lower in x.name.lower()), xlifs[0])

    # ---------- Metadata parsing ----------

    def _parse_metadata(self) -> Dict[str, Any]:
        """Parse dimensional, physical, and channel metadata from the matched .xlif file."""
        xlif_path = self._find_xlif_for_ome_tiff()
        root = ET.parse(xlif_path).getroot()

        imgdesc = root.find(".//ImageDescription")
        if imgdesc is None:
            raise ValueError("<ImageDescription> element not found in .xlif file")
        dims_node = imgdesc.find("Dimensions")
        if dims_node is None:
            raise ValueError("<Dimensions> element not found in .xlif file")

        by_id = {int(d.attrib["DimID"]): d.attrib for d in dims_node.findall("DimensionDescription")}

        def dim(axis: str, default_unit: str, required: bool = True):
            """Return ``(n_elements, length, unit)`` of one axis."""
            attrs = by_id.get(self._DIM_IDS[axis])
            if attrs is None:
                if required:
                    raise ValueError(f"Dimension '{axis.upper()}' not found in {xlif_path.name}")
                return 1, 0.0, default_unit
            return (
                int(attrs["NumberOfElements"]),
                float(attrs["Length"]),
                attrs.get("Unit", default_unit),
            )

        size_x, len_x, unit_x = dim("x", "m")
        size_y, len_y, unit_y = dim("y", "m")
        size_z, len_z, unit_z = dim("z", "m", required=False)
        size_t, _, unit_t = dim("t", "s", required=False)

        channels_node = imgdesc.find("Channels")
        channel_descs = channels_node.findall("ChannelDescription") if channels_node is not None else []
        channels = []
        for i, ch in enumerate(channel_descs):
            lut = ch.attrib.get("LUTName", "").strip()
            channels.append(ChannelInfo(lut or f"Channel {i}", self._LUT_COLORS.get(lut, "FFFFFF")))
        if not channels:
            channels = [ChannelInfo("Channel 0", "FFFFFF")]

        # Voxel size = physical length / number of elements, converted to micrometers.
        scales, units = to_micrometers(
            (len_z / size_z if size_z else 1.0,
             len_y / size_y if size_y else 1.0,
             len_x / size_x if size_x else 1.0),
            (unit_z, unit_y, unit_x),
        )

        with TiffFile(str(self.ome_tiff_path)) as tf:
            dtype = tf.series[0].dtype

        return build_metadata(
            size=[(size_t, len(channels), size_z, size_y, size_x)],
            scales=[scales],
            units=units,
            channels=channels,
            dtype=dtype,
            # The XLIF only stores the total acquisition length, not a per-frame
            # interval, so the time increment is left at its default of 1.
            time_increment=1.0,
            time_increment_unit=unit_t,
            xlif_path=str(xlif_path),
            ome_tiff_path=str(self.ome_tiff_path.resolve()),
        )

    # ---------- Dask array construction ----------

    @staticmethod
    def _guess_axes(shape: Tuple[int, ...], n_channels: int) -> str:
        """Best-effort axes for TIFF series that do not declare them."""
        if len(shape) == 5:
            return "tczyx"
        if len(shape) == 4:
            if shape[1] == n_channels:
                return "zcyx"
            if shape[0] == n_channels:
                return "czyx"
            return "tzyx"
        if len(shape) == 3:
            return "zyx"
        if len(shape) == 2:
            return "yx"
        raise ValueError(f"Cannot infer the axes of a {len(shape)}-D TIFF series.")

    def _build_dask_array(self) -> List[da.Array]:
        """Build a TCZYX dask array from the provided OME-TIFF file."""
        ome_path = self.ome_tiff_path.resolve()

        with TiffFile(str(ome_path)) as tf:
            series = tf.series[0]
            tif_shape = series.shape
            tif_axes = getattr(series, "axes", None)

        arr = da.from_delayed(
            delayed(imread)(str(ome_path)),
            shape=tif_shape,
            dtype=self.metadata["dtype"],
        )
        axes = tif_axes or self._guess_axes(tif_shape, self.metadata["size"][0][1])
        return [to_tczyx(arr, axes).rechunk(self.chunks)]
