from __future__ import annotations

from typing import List

from .mosaic import MosaicManager
from .tile import Tile


class ZeissMosaicManager(MosaicManager):
    """Placeholder: raw Zeiss CZI tile (mosaic) tiles, before fusion.

    Not implemented yet. Implement :meth:`_scan` to return one
    :class:`~pymif.mosaic_manager.Tile` per acquired tile (lazy ``(z, y, x)`` data,
    stage position as ``origin``, voxel size, channel and optionally a ``group``);
    :meth:`~pymif.mosaic_manager.MosaicManager.fuse`, ``visualize`` and the rest then
    work unchanged. See :class:`~pymif.mosaic_manager.LuxendoMosaicManager` for a
    complete example. The existing :class:`~pymif.microscope_manager.ZeissManager` reads a
    single image and is unrelated.
    """

    def _scan(self) -> List[Tile]:
        raise NotImplementedError("ZeissMosaicManager is a placeholder; tile scanning is not implemented yet.")
