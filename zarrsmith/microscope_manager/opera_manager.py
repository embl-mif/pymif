from typing import Any, Dict, List, Optional, Tuple

import dask.array as da
import tifffile
import zarr

from .microscope_manager import MicroscopeManager
from ..utils.axes import to_tczyx
from ..utils.metadata import build_metadata, scale_for_level
from ..utils.ome import parse_ome_xml


class OperaManager(MicroscopeManager):
    """
    Manager for reading Opera (PerkinElmer Opera Phenix) pyramidal OME-TIFF datasets.

    This class reads and parses OME-XML metadata embedded in the TIFF file.
    """

    def __init__(self,
                 path: str,
                 chunks: Optional[Tuple[int, ...]] = None):
        """
        Initialize the OperaManager with the given file path.

        Parameters
        ----------
        path : str
            Path to the Opera pyramidal OME-TIFF file.
        chunks : tuple of int, optional
            Chunk sizes for Dask arrays in TCZYX order. Defaults to
            :attr:`DEFAULT_CHUNKS` ``(1, 1, 8, 4096, 4096)``.
        """
        super().__init__(path, chunks)
        self.read()

    def _parse_metadata(self) -> Dict[str, Any]:
        """
        Parse OME-XML metadata embedded in the pyramidal OME-TIFF file.

        Returns
        -------
        dict
            Metadata dictionary containing size, scales, units, channel info, dtype, and axes.
        """
        with tifffile.TiffFile(self.path) as tif:
            xml_string = tif.ome_metadata
        if not xml_string:
            raise ValueError(f"{self.path} does not contain OME-XML metadata.")

        px = parse_ome_xml(xml_string, color_format="auto")  # Opera writes 24-bit RGB colors
        return build_metadata(
            size=[(px.size_t, px.size_c, px.size_z, px.size_y, px.size_x)],
            scales=[px.scale_zyx],
            units=px.units_zyx,
            channels=px.channels,
            dtype=px.dtype,
            time_increment=px.time_increment,
            time_increment_unit=px.time_increment_unit,
        )

    def _build_dask_array(self) -> List[da.Array]:
        """
        Load pyramid levels from the pyramidal OME-TIFF and convert them to Dask arrays.

        The per-level ``size`` and ``scales`` entries of :attr:`metadata` are
        updated to match the levels found in the file.

        Returns
        -------
        list of dask.array.Array
            List of Dask arrays, each corresponding to a pyramid level, normalized to TCZYX axes order.
        """
        with tifffile.TiffFile(self.path) as tif:
            zgroup = zarr.open(tif.aszarr(), mode="r")

            if isinstance(zgroup, zarr.Array):
                pyramid = [(da.from_zarr(zgroup), tif.series[0].axes.lower())]
            else:
                pyramid = [
                    (da.from_zarr(zgroup[str(i)]), tif.series[0].levels[i].axes.lower())
                    for i in range(len(zgroup))
                ]

        base_scale = self.metadata["scales"][0]
        base_tc, base_zyx = self.metadata["size"][0][:2], self.metadata["size"][0][2:]

        data_levels, sizes, scales = [], [], []
        for arr, axes in pyramid:
            arr = to_tczyx(arr, axes)
            level_zyx = arr.shape[2:]
            scales.append(scale_for_level(base_scale, base_zyx, level_zyx))
            sizes.append(tuple(base_tc) + tuple(level_zyx))
            data_levels.append(arr.rechunk(chunks=self.chunks, method="tasks"))

        self.metadata["scales"] = scales
        self.metadata["size"] = sizes
        return data_levels
