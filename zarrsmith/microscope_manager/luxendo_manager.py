import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dask.array as da
import h5py

from .microscope_manager import MicroscopeManager
from ..utils.metadata import ChannelInfo, build_metadata, scale_for_level


class LuxendoManager(MicroscopeManager):
    """
    Reader for Luxendo microscope data saved as multi-resolution HDF5 (.lux.h5) and XML metadata.

    This class parses Luxendo's XML configuration and builds a lazy Dask array pyramid for downstream processing.
    One ``*.lux.h5`` file holds one timepoint and channel (``tp-<t>`` / ``ch-<c>`` in the file name);
    HDF5 handles stay open until :meth:`close` (or leaving a ``with`` block).
    """


    def __init__(self,
                 path: str,
                 chunks: Optional[Tuple[int, ...]] = None):
        """
        Initialize the LuxendoManager.

        Parameters
        ----------
        path : str
            Path to the Luxendo dataset directory.
        chunks : Tuple[int, ...], optional
            Chunk shape for Dask arrays, by default :attr:`DEFAULT_CHUNKS` ``(1, 1, 8, 4096, 4096)``.
        """
        super().__init__(path, chunks)
        self._h5_handles: Dict[Path, h5py.File] = {}
        self.read()

    # ---------- File discovery ----------

    @classmethod
    def _tp_ch(cls, filename: str) -> Tuple[int, int]:
        """Extract ``(timepoint, channel)`` from a ``...tp-<t>...ch-<c>....lux.h5`` file name."""
        tp = re.search(r"tp-(\d+)", filename)
        ch = re.search(r"ch-(\d+)", filename)
        if tp is None or ch is None:
            raise ValueError(f"Cannot read timepoint/channel from file name {filename!r} (expected 'tp-<n>' and 'ch-<n>').")
        return int(tp.group(1)), int(ch.group(1))

    def _plane_files(self) -> Dict[Tuple[int, int], Path]:
        """Map ``(t, c)`` to the ``.lux.h5`` file holding it.

        The ``tp-``/``ch-`` numbers in file names are acquisition IDs, not
        indices (a single fused channel can be ``ch-4``), so they are renumbered
        to contiguous 0-based indices in sorted order.
        """
        raw = {self._tp_ch(f.name): f for f in sorted(self.path.glob("*.lux.h5"))}
        if not raw:
            raise FileNotFoundError(f"No '*.lux.h5' files found in {self.path}.")
        tps = sorted({tp for tp, _ in raw})
        chs = sorted({ch for _, ch in raw})
        return {(tps.index(tp), chs.index(ch)): f for (tp, ch), f in raw.items()}

    @staticmethod
    def _sorted_dataset_names(f: h5py.File) -> List[str]:
        """Return the "Data*" dataset names of an open file in natural scale order."""
        return sorted((k for k in f.keys() if k.startswith("Data")), key=lambda s: (len(s), s))

    @classmethod
    def get_available_datasets(cls, h5_file) -> List[str]:
        """
        Extract all dataset names from a .lux.h5 file.

        Parameters
        ----------
        h5_file : Path
            Path to a Luxendo HDF5 file.

        Returns
        -------
        List[str]
            Sorted list of dataset names (one per resolution level).
        """
        with h5py.File(h5_file, "r") as f:
            return cls._sorted_dataset_names(f)

    # ---------- Metadata ----------

    @staticmethod
    def _xyz_to_zyx(text: str, cast) -> Tuple:
        """Parse a space-separated ``x y z`` triplet and return it as ``(z, y, x)``."""
        x, y, z = (cast(v) for v in text.split())
        return z, y, x

    def _parse_metadata(self) -> Dict[str, Any]:
        """
        Parse XML metadata from the Luxendo dataset.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing dataset shape, voxel sizes, channel info, and other metadata.
        """
        xml_path = next(self.path.glob("*.xml"), None)
        if xml_path is None:
            raise FileNotFoundError(f"No '*.xml' metadata file found in {self.path}.")
        root = ET.parse(xml_path).getroot()

        timepoints = root.find(".//Timepoints")
        if timepoints is None:
            raise ValueError(f"No <Timepoints> element in {xml_path.name}.")
        size_t = int(timepoints.find("last").text) - int(timepoints.find("first").text) + 1

        setup = root.find(".//ViewSetup")  # all setups share size and voxel size
        if setup is None:
            raise ValueError(f"No <ViewSetup> element in {xml_path.name}.")
        base_scale = self._xyz_to_zyx(setup.find("voxelSize/size").text, float)
        unit = setup.findtext("voxelSize/unit") or "micrometer"

        channels = [ChannelInfo(ch.findtext("name") or f"Channel {i}") for i, ch in enumerate(root.findall(".//Channel"))]

        # Level shapes (and dtype) come from the first file of the dataset.
        ref_file = next(iter(sorted(self._plane_files().items())))[1]
        with h5py.File(ref_file, "r") as f:
            names = self._sorted_dataset_names(f)
            if not names:
                raise ValueError(f"No 'Data*' datasets in {ref_file.name}.")
            shapes = [f[n].shape for n in names]
            dtype = f[names[0]].dtype

        # Voxel size of each level from the actual shapes, so the physical extent
        # is preserved. Levels are not simply downsampled: Luxendo also resamples
        # z, so a level can have *more* planes than level 0 (e.g. 137 -> 192).
        scales = [scale_for_level(base_scale, shapes[0], shape) for shape in shapes]

        return build_metadata(
            size=[(size_t, len(channels)) + tuple(shape) for shape in shapes],
            scales=scales,
            units=(unit,) * 3,
            channels=channels,
            dtype=dtype,
            # The Luxendo XML does not record the interval between timepoints.
            time_increment=1.0,
            time_increment_unit="s",
        )

    # ---------- Dask array ----------

    def _read_h5_stack(self, h5_path: Path, dataset_name: str) -> da.Array:
        """
        Load a single resolution dataset lazily as a Dask array.

        Parameters
        ----------
        h5_path : Path
            Path to the .lux.h5 file.
        dataset_name : str
            Internal dataset name (e.g., "Data", "Data444", etc.)

        Returns
        -------
        dask.array.Array
            A lazy ``(z, y, x)`` Dask array backed by the HDF5 dataset.
        """
        f = self._h5_handles.get(h5_path)
        if f is None:  # one handle per file, shared by all pyramid levels
            f = self._h5_handles[h5_path] = h5py.File(h5_path, "r")
            self._open_files.append(f)
        return da.from_array(f[dataset_name], chunks=self.chunks[2:])

    def _build_dask_array(self) -> List[da.Array]:
        """
        Construct a multiscale image pyramid as Dask arrays.

        Returns
        -------
        List[da.Array]
            A list of Dask arrays representing each resolution level (from highest to lowest).
        """
        t, c = self.metadata["size"][0][:2]
        files = self._plane_files()
        missing = [(ti, ci) for ti in range(t) for ci in range(c) if (ti, ci) not in files]
        if missing:
            raise ValueError(
                f"Expected {t * c} '*.lux.h5' files (t={t}, c={c}) but {len(missing)} are missing, "
                f"e.g. (t, c) = {missing[:5]}."
            )

        names = self.get_available_datasets(files[(0, 0)])
        return [
            da.stack(
                [[self._read_h5_stack(files[(ti, ci)], name) for ci in range(c)] for ti in range(t)],
                axis=0,
            ).rechunk(self.chunks)  # T, C, Z, Y, X
            for name in names
        ]
