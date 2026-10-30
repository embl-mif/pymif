import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

import dask.array as da
from dask import delayed
from tifffile import imread

from .microscope_manager import MicroscopeManager
from ..utils.metadata import build_metadata
from ..utils.ome import parse_ome_xml, xml_attr


class ViventisManager(MicroscopeManager):
    """
    Reader for Viventis microscope datasets with OME-TIFF and companion .ome XML files.

    This class lazily loads data into a dask array and parses associated OME-XML metadata.
    """

    def __init__(self,
                 path: str,
                 chunks: Optional[Tuple[int, ...]] = None):
        """
        Initialize the ViventisManager.

        Parameters
        ----------
        path : str
            Path to the folder containing the Viventis dataset (including `.ome` and `.tif` files).
        chunks : Tuple[int, ...], optional
            Desired chunk shape for the output Dask array. Defaults to
            :attr:`DEFAULT_CHUNKS` ``(1, 1, 8, 4096, 4096)``.
        """
        super().__init__(path, chunks)
        self.read()

    @staticmethod
    def _plane_files(tiffdata: List[ET.Element]) -> Dict[Tuple[int, int], str]:
        """Map ``(t, c)`` to the TIFF file holding that plane stack."""
        plane_files = {}
        for entry in tiffdata:
            t = xml_attr(entry, "FirstT", int, what="<TiffData>")
            c = xml_attr(entry, "FirstC", int, what="<TiffData>")
            uuid = entry.find(".//{*}UUID")
            if uuid is None:
                raise ValueError(f"<TiffData> for t={t}, c={c} has no <UUID FileName=...>.")
            plane_files[(t, c)] = xml_attr(uuid, "FileName", what="<UUID>")
        return plane_files

    def _parse_metadata(self) -> Dict[str, Any]:
        """
        Parse the companion `.ome` XML metadata file.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing extracted metadata such as size, scales, units, channel info,
            time increment, and TIFF file mapping (``plane_files``).
        """
        companion = next(self.path.glob("*.ome"), None)
        if companion is None:
            raise FileNotFoundError(f"No companion '*.ome' file found in {self.path}.")

        px = parse_ome_xml(ET.parse(companion).getroot())
        return build_metadata(
            size=[(px.size_t, px.size_c, px.size_z, px.size_y, px.size_x)],
            scales=[px.scale_zyx],
            units=px.units_zyx,
            channels=px.channels,
            dtype=px.dtype,
            time_increment=px.time_increment,
            time_increment_unit=px.time_increment_unit,
            plane_files=self._plane_files(px.tiffdata),
        )

    def _build_dask_array(self) -> List[da.Array]:
        """
        Lazily construct a dask array for the image data using tifffile and delayed loading.

        Returns
        -------
        List[da.Array]
            A list containing a single Dask array representing the full dataset (level 0).
        """
        t, c, z, y, x = self.metadata["size"][0]
        filenames = self.metadata["plane_files"]
        dtype = self.metadata["dtype"]

        missing = [(ti, ci) for ti in range(t) for ci in range(c) if (ti, ci) not in filenames]
        if missing:
            raise ValueError(f"No TIFF file listed for (t, c) = {missing[:5]}{'...' if len(missing) > 5 else ''}.")

        lazy_imread = delayed(imread)
        stack = da.stack(
            [
                [
                    da.from_delayed(lazy_imread(str(self.path / filenames[(ti, ci)])), shape=(z, y, x), dtype=dtype)
                    for ci in range(c)
                ]
                for ti in range(t)
            ],
            axis=0,
        ).rechunk(self.chunks)
        return [stack]  # level 0 only
