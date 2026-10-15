# Mosaics: raw tiles and fusion

Some acquisitions are saved as many separate **tiles** (stacks acquired at different stage positions) instead of one image. `pymif.mosaic_manager` reads such tiles lazily, shows them in napari at their position and **fuses** them into one OME-Zarr image.

A mosaic manager is different from a [microscope manager](readers.md): a `MicroscopeManager` is *one* image, while a `MosaicManager` is a set of positioned tiles that only becomes an image once it is fused. The fused result is an ordinary OME-Zarr dataset, so everything else in PyMIF works on it.

| Class | Reads | Status |
|---|---|---|
| `LuxendoMosaicManager` | raw Luxendo acquisitions (`raw/<stack>/Cam_*.lux.h5` + `.json`) | available |
| `OperaMosaicManager` | Opera tiles | placeholder |
| `ZeissMosaicManager` | Zeiss CZI tiles | placeholder |

`LuxendoMosaicManager` is for the **raw** tiles. Data already fused by the Luxendo software is read with `LuxendoManager`.

## Python

```python
from pymif.mosaic_manager import LuxendoMosaicManager

mosaic = LuxendoMosaicManager("path/to/2026-10-14_103957")   # acquisition folder
mosaic.channels          # ['autofluo-488', 'cFos-647']
mosaic.groups            # ['L', 'R']  (illumination side)
mosaic.bounding_box()    # ((zmin, ymin, xmin), (zmax, ymax, xmax)) in micrometer

# Check placement: one napari layer per tile, at its stage position
mosaic.visualize(tiles=mosaic.select(channel="cFos-647"))

# Fuse (3D by default) and get the result back as a ZarrManager
zm = mosaic.fuse("fused.ome.zarr", ds=(4, 8, 8), blend="midline", levels=3, workers=4)

# 2D maximum projection along z instead
mosaic.fuse("mip.ome.zarr", ds=(1, 8, 8), projection="max")
```

`visualize()` gives all tiles of a channel the same contrast, estimated from a small sample: a 64³-voxel block from the centre of up to 6 tiles, about 3 MB of reads per channel (`mosaic.contrast_limits()`). The layers of a channel are linked, so changing the contrast or gamma of one tile changes all of them. Pass `contrast_limits={"cFos-647": (100, 2000)}` to set them yourself.

`mosaic.tiles` is a list of `Tile` objects: a lazy `(z, y, x)` array (`data`), the position of its first voxel (`origin`), `voxel_size`, `channel`, an optional `group` and reader-specific `attrs`. `mosaic.tile(i)` returns one tile as an `ArrayManager`.

For Luxendo data, the two illumination sides of a dye (e.g. `autofluo-488-L` and `autofluo-488-R`) become **one channel** (`autofluo-488`). The side is kept as the tile `group`. If a stack was re-acquired (`*_R_<n>.lux.h5`), only the latest retake is used.

## Fusion options

| Option | Values | Meaning |
|---|---|---|
| `ds` | `(z, y, x)` integers | downsampling applied while reading (default `(1, 1, 1)`) |
| `projection` | `None`, `"max"`, `"sum"` | 3D volume (default) or a 2D projection along z |
| `blend` | `"max"`, `"sum"`, `"count"`, `"midline"` | how overlapping tiles are merged, see below |
| `downsample` | `"mean"`, `"stride"` | block average (default) or every n-th voxel (faster) |
| `channels` | list of names | channels to fuse, in output order (default: all) |
| `levels` | integer | pyramid levels written (default 1) |
| `workers` | integer | tiles read in parallel (default 1) |
| `out_dtype` | dtype | default: input dtype, `float32` for `sum`, `uint16` for `count` |
| `overwrite` | bool | replace an existing output |

Blend modes:

- **`max`**: keeps the brighter value.
- **`sum`**: adds the values, so overlaps show up as brighter bands.
- **`count`**: writes how many tiles cover each voxel. It reads no pixel data, so it is instant and shows exactly where tiles start and end.
- **`midline`**: uses `max` within each tile group. Where two groups overlap (for example the left and right illumination sides), each group is used on its own side of the middle of the overlap, along `midline_axis` (default `x`). Every side is then shown where its light sheet is sharpest.

`plan_fusion(mosaic.tiles, ...)` takes the same arguments and returns the output shape and size without reading anything:

```python
from pymif.mosaic_manager import plan_fusion

plan = plan_fusion(mosaic.tiles, ds=(4, 8, 8))
print(plan.summary())   # channels, output shape and axes, size in GB, voxel size
plan.sizes, plan.scales # per pyramid level
```

## How fusion works

The output is written one region at a time: one output chunk in `y`/`x`, over the full `z` range. For each region, only the tile voxels that fall inside it are read, downsampled, merged and written. Memory stays bounded and every full-resolution voxel is read once.

With `workers > 1`, the tiles of a region are read in **worker processes** when the tiles allow it (as Luxendo tiles do). h5py serialises reads within one process, so threads would not speed up reading HDF5 files. Merging and writing stay in the calling process. If worker processes cannot be started, fusion falls back to threads with a warning.

Raw tiles usually have no pyramid, so every fusion reads the full-resolution data, whatever `ds` is. On a network drive the read speed sets the run time.

## Napari

`Plugins > PyMIF > Mosaic Fusion` opens a widget with three boxes:

1. **Mosaic loading**: choose the microscope and the acquisition folder (browse, or drag and drop it, or any file inside it, from the file explorer), then *Load tiles*.
2. **Show tiles**: show the selected channels and groups as one layer per tile at its stage position. Every displayed plane reads data from all shown tiles, so show only a few channels or groups at a time.
3. **Fuse to OME-Zarr**: every option above, with a live estimate of the output size. The output path follows the parameters (for example `<acquisition>_3D_ds4-8-8.ome.zarr`) until you type your own; dropping a folder on the output field writes the result inside that folder. When fusion finishes, the result can be opened directly.

Loading and fusion run in the background, and their progress is shown in the log at the bottom.

## Adding a mosaic reader

Subclass `MosaicManager` and implement `_scan()`, which returns one `Tile` per acquired tile:

```python
from pymif.mosaic_manager import H5Array, MosaicManager, Tile


class MyMosaicManager(MosaicManager):
    def _scan(self):
        return [
            Tile(
                data=H5Array(path, "Data"),        # any lazy array with shape/dtype/slicing
                origin=(z0, y0, x0),               # position of voxel [0, 0, 0]
                voxel_size=(dz, dy, dx),
                channel="GFP",
                group=None,                        # e.g. illumination side, for blend="midline"
            )
            for path, (z0, y0, x0) in ...
        ]
```

`visualize()`, `fuse()` and the napari widget then work unchanged. Use picklable tile data (such as `H5Array`) so that fusion can read it in worker processes.
