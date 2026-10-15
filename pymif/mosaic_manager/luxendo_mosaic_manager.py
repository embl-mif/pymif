from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

from tqdm.auto import tqdm

from .h5_array import H5Array
from .mosaic import MosaicManager
from .tile import Tile

logger = logging.getLogger("pymif")


class LuxendoMosaicManager(MosaicManager):
    """Raw Luxendo tiles (``*.lux.h5`` + ``*.json``) of one acquisition, before any fusion.

    Not to be confused with :class:`~pymif.microscope_manager.LuxendoManager`, which
    reads images already fused by the Luxendo software. This manager scans a raw
    acquisition folder such as ``2026-10-14_103957`` (tiles live in
    ``raw/<stack folder>/Cam_*.lux.h5``) and exposes every tile lazily:

    * ``channel``: the dye, with the illumination side removed
      (``autofluo-488-L`` and ``autofluo-488-R`` both give ``autofluo-488``), so
      both halves of a sample end up in the same fused channel;
    * ``group``: the illumination side, ``"L"`` or ``"R"`` (used by ``blend="midline"``);
    * ``attrs``: ``stack``, ``stack_description``, ``grid`` (``(ix, iy)``),
      ``channel_description`` and ``path``.

    Tile positions come from ``affine_to_sample`` in the JSON (the tile is centred
    on the stage translation in x/y; z is the plane position along the stacking
    direction). If a stack was re-acquired (``*_R_<n>.lux.h5``) only the latest
    retake is used. Tiles are :class:`~pymif.mosaic_manager.H5Array` objects, so no
    file stays open. Scanning reads only the JSON files (in parallel) and opens a
    single ``.h5`` file, which keeps loading fast on network drives.
    """

    @staticmethod
    def _retake(h5: Path) -> int:
        """``Cam_bottom_00000.lux.h5`` -> 0, ``Cam_bottom_00000_R_1.lux.h5`` -> 1."""
        m = re.search(r"_R_(\d+)\.lux\.h5$", h5.name)
        return int(m.group(1)) if m else 0

    def _scan(self) -> List[Tile]:
        # Over a network share every listing, stat and open is a slow round trip:
        # list only raw/<stack>/ (one listing per folder, in parallel) instead of
        # walking the whole acquisition, and open a single .h5 file (for the dtype).
        root = self.path / "raw" if (self.path / "raw").is_dir() else self.path
        folders = [root] + sorted(e.path for e in os.scandir(root) if e.is_dir())
        with ThreadPoolExecutor(max_workers=self.scan_workers) as pool:
            found = list(tqdm(pool.map(self._latest_h5, folders), total=len(folders),
                              desc="scanning tiles", unit="folder", disable=not self.progress))
        files = [h5 for h5 in found if h5 is not None]
        if not files:  # other layouts: search the whole tree
            files = [h5 for h5 in map(self._latest_h5, sorted({h.parent for h in self.path.rglob("*.lux.h5")}))
                     if h5 is not None]
        if not files:
            return []

        with ThreadPoolExecutor(max_workers=self.scan_workers) as pool:
            infos = list(tqdm(pool.map(self._read_info, files), total=len(files),
                              desc="reading tile metadata", unit="tile", disable=not self.progress))

        # All tiles of an acquisition share the dtype: read it (and check the shape) once.
        first = H5Array(files[0])
        expected = self._shape(infos[0])
        if first.shape != expected:
            raise ValueError(f"{files[0]}: data shape {first.shape} does not match the JSON image size {expected}.")
        return [self._tile_from(h5, info, first.dtype) for h5, info in zip(files, infos)]

    def __init__(self, path, scan_workers: int = 16, progress: bool = True):
        """
        Parameters
        ----------
        path
            Acquisition folder (containing ``raw/``), or the ``raw`` folder itself.
        scan_workers
            Folders listed / JSON files read in parallel while scanning (network latency).
        progress
            Show progress bars while scanning.
        """
        self.scan_workers = max(1, int(scan_workers))
        self.progress = progress
        super().__init__(path)

    @classmethod
    def _latest_h5(cls, folder) -> Optional[Path]:
        """Latest retake among the ``*.lux.h5`` files of ``folder`` that have a ``.json`` sibling."""
        names = set(os.listdir(folder))
        h5s = [n for n in names if n.endswith(".lux.h5") and n[: -len(".lux.h5")] + ".json" in names]
        if not h5s:  # e.g. only main_raw.lux.h5 (no image data)
            return None
        h5s.sort(key=lambda n: cls._retake(Path(n)))
        for skipped in h5s[:-1]:
            logger.info("Skipping %s (superseded by retake %s)", skipped, h5s[-1])
        return Path(folder) / h5s[-1]

    @staticmethod
    def _read_info(h5: Path) -> dict:
        return json.loads(h5.with_name(h5.name.replace(".lux.h5", ".json")).read_text())["processingInformation"]

    @staticmethod
    def _shape(info: dict) -> tuple:
        size = info["image_size_vx"]
        return (size["depth"], size["height"], size["width"])

    @classmethod
    def _tile_from(cls, h5: Path, info: Optional[dict] = None, dtype=None) -> Tile:
        """Tile of one ``.lux.h5`` file; the file is only opened when ``dtype`` is not given."""
        info = cls._read_info(h5) if info is None else info
        size, vox = info["image_size_vx"], info["voxel_size_um"]
        aff = info["affine_to_sample"][1]
        tx, ty, tz = aff["translation"]
        sz = aff["matrix"][2][2]  # um per plane, signed (stage z direction)
        desc = info.get("channel_description", info["channel"])
        side = re.search(r"[-_]([LR])$", desc)

        data = H5Array(h5, shape=cls._shape(info), dtype=dtype)

        grid = re.search(r"x(\d+)-y(\d+)", info["stack"])
        return Tile(
            data=data,
            origin=(
                tz / sz * vox["depth"],  # position of plane 0 along the stacking direction
                ty - (size["height"] - 1) / 2 * vox["height"],
                tx - (size["width"] - 1) / 2 * vox["width"],
            ),
            voxel_size=(vox["depth"], vox["height"], vox["width"]),
            channel=re.sub(r"[-_][LR]$", "", desc),
            group=side.group(1) if side else None,
            attrs={
                "stack": info["stack"],
                "stack_description": info.get("stack_description"),
                "grid": (int(grid.group(1)), int(grid.group(2))) if grid else None,
                "channel_description": desc,
                "path": str(h5),
            },
        )
