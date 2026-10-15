from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import List

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
    file stays open.
    """

    @staticmethod
    def _retake(h5: Path) -> int:
        """``Cam_bottom_00000.lux.h5`` -> 0, ``Cam_bottom_00000_R_1.lux.h5`` -> 1."""
        m = re.search(r"_R_(\d+)\.lux\.h5$", h5.name)
        return int(m.group(1)) if m else 0

    def _scan(self) -> List[Tile]:
        candidates = defaultdict(list)
        for h5 in self.path.rglob("*.lux.h5"):
            if h5.with_name(h5.name.replace(".lux.h5", ".json")).exists():  # skips e.g. main_raw.lux.h5
                candidates[h5.parent].append(h5)

        tiles: List[Tile] = []
        for folder in sorted(candidates):
            files = sorted(candidates[folder], key=self._retake)
            h5 = files[-1]
            for skipped in files[:-1]:
                logger.info("Skipping %s (superseded by retake %s)", skipped, h5.name)
            tiles.append(self._tile_from(h5))
        return tiles

    @staticmethod
    def _tile_from(h5: Path) -> Tile:
        info = json.loads(h5.with_name(h5.name.replace(".lux.h5", ".json")).read_text())["processingInformation"]
        size, vox = info["image_size_vx"], info["voxel_size_um"]
        aff = info["affine_to_sample"][1]
        tx, ty, tz = aff["translation"]
        sz = aff["matrix"][2][2]  # um per plane, signed (stage z direction)
        desc = info.get("channel_description", info["channel"])
        side = re.search(r"[-_]([LR])$", desc)

        data = H5Array(h5)
        expected = (size["depth"], size["height"], size["width"])
        if data.shape != expected:
            raise ValueError(f"{h5}: data shape {data.shape} does not match the JSON image size {expected}.")

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
