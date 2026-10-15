# Mosaics: raw tiles and fusion

Some acquisitions are saved as many separate **tiles** (stacks acquired at different stage positions) instead of one image. `pymif.mosaic_manager` reads such tiles lazily, shows them in napari at their position and **fuses** them into one OME-Zarr image.

A mosaic manager is different from a [microscope manager](readers.md): a `MicroscopeManager` is *one* image, while a `MosaicManager` is a set of positioned tiles that only becomes an image once it is fused. The fused result is an ordinary OME-Zarr dataset, so everything else in PyMIF works on it.

| Class | Reads | Status |
|---|---|---|
| `LuxendoMosaicManager` | raw Luxendo acquisitions (`raw/<stack>/Cam_*.lux.h5` + `.json`) | available |
| `OperaMosaicManager` | Opera tiles | placeholder |
| `ZeissMosaicManager` | Zeiss CZI tiles | placeholder |

`LuxendoMosaicManager` is for the **raw** tiles. Data already fused by the Luxendo software is read with `LuxendoManager`.

```{note}
Mosaics are supported for a **single timepoint** so far. Loading a timelapse stops with an error that says so: several acquisitions of the same stack in one folder, or tiles whose metadata report different timepoints.
```

A runnable walk-through with synthetic data is in [examples/10_mosaic_fusion.ipynb](https://github.com/embl-mif/pymif/blob/main/examples/10_mosaic_fusion.ipynb).

## Load and inspect

```python
from pymif.mosaic_manager import LuxendoMosaicManager, plan_fusion

mosaic = LuxendoMosaicManager("path/to/2026-10-14_103957")   # acquisition folder (contains raw/)
mosaic.channels          # ['autofluo-488', 'cFos-647']
mosaic.groups            # ['L', 'R']  (illumination side)
mosaic.bounding_box()    # ((zmin, ymin, xmin), (zmax, ymax, xmax)) in micrometer

# Output shape and size, without reading any pixel data
print(plan_fusion(mosaic.tiles, ds=(4, 8, 8), blend="midline").summary())

# Check placement: one napari layer per tile, at its stage position
mosaic.visualize(tiles=mosaic.select(channel="cFos-647"))
```

Loading reads only the JSON files (in parallel) and a single `.h5` header, so it stays fast on network drives.

`mosaic.tiles` is a list of `Tile` objects: a lazy `(z, y, x)` array (`data`), the position of its first voxel (`origin`), `voxel_size`, `channel`, an optional `group` and reader-specific `attrs`. `mosaic.tile(i)` returns one tile as an `ArrayManager`.

For Luxendo data, the two illumination sides of a dye (e.g. `autofluo-488-L` and `autofluo-488-R`) become **one channel** (`autofluo-488`). The side is kept as the tile `group`. If a stack was re-acquired (`*_R_<n>.lux.h5`), only the latest retake is used.

`visualize()` gives all tiles of a channel the same contrast, estimated from a small sample: a 64³-voxel block from the centre of up to 6 tiles near the middle of each group, about 3 MB of reads per channel (`mosaic.contrast_limits()`). The layers of a channel are linked, so changing the contrast or gamma of one tile changes all of them. Pass `contrast_limits={"cFos-647": (100, 2000)}` to set them yourself.

## 3D fusion

```python
zm = mosaic.fuse(
    "acq_3D.ome.zarr",
    ds=(4, 8, 8),          # downsampling (z, y, x) applied while reading the tiles
    blend="midline",       # each illumination side on its own side of the overlap
    levels=3,              # pyramid levels
    workers=4,             # tiles read in parallel processes
)
zm.metadata["axes"]        # 'czyx'
```

## 2D maximum projection (MIP)

```python
mip = mosaic.fuse(
    "acq_MIP.ome.zarr",
    projection="max",      # or "sum"
    ds=(1, 8, 8),          # only y/x matter for a projection
    blend="midline",
    workers=4,
)
mip.metadata["axes"]       # 'cyx'
```

The projection is computed at full resolution along z and then downsampled in y/x, so the output is small even for a large acquisition. A `crop` in `z` limits the planes that are projected.

## Fusion options

| Option | Values | Meaning |
|---|---|---|
| `ds` | `(z, y, x)` integers | downsampling applied while reading: each output voxel is the mean of its block (default `(1, 1, 1)`) |
| `projection` | `None`, `"max"`, `"sum"` | 3D volume (default) or a 2D projection along z |
| `blend` | `"max"`, `"sum"`, `"midline"` | how overlapping tiles are merged, see below |
| `channels` | list of names | tile channels to fuse, in output order (default: all) |
| `channel_names`, `channel_colors` | lists | names and colours (hex or matplotlib names) of the output channels |
| `crop` | `{"z": slice, "y": slice, "x": slice}` | part of the output to write, in output voxels (after `ds`) |
| `skip` | tiles or tile ids | tiles to leave blank, see below |
| `levels`, `downscale_factor` | integer, integer or `(z, y, x)` | pyramid levels (default 1) and factor between them (default 2) |
| `workers` | integer | tiles read in parallel (default 1) |
| `out_dtype` | dtype | default: input dtype, `float32` for `sum` |
| `midline_axis` | `"x"`, `"y"`, `"z"` | axis along which `midline` separates the groups (default `x`) |
| `overwrite` | bool | replace an existing output |
| `temp_dir` | path | where the temporary fused level 0 goes (default: next to the output) |

Every other option of `to_zarr` is passed through unchanged, exactly as for `pymif convert`: `chunks` (in output axis order, `c, z, y, x` or `c, y, x`), `shards`, `shard_target_mb`, `shard_exclude_axes`, `zarr_format`, `ngff_version`, `compressor`, `compressor_level`, `drop_singleton`, `num_workers` (write threads) and `storage_options`. See [Advanced writing options](advanced.md). As with `pymif convert`, a single fused channel is written without its channel axis unless `drop_singleton=False`.

```python
mosaic.fuse(
    "acq_cfos.ome.zarr",
    ds=(2, 4, 4),
    channels=["cFos-647"], channel_names=["cfos"], channel_colors=["magenta"],
    crop={"z": slice(100, 400), "y": slice(0, 1500)},
    levels=4, downscale_factor=(1, 2, 2),
    chunks=(1, 32, 512, 512), shards="auto", shard_target_mb=512, zarr_format=3,
    workers=4, num_workers=8, temp_dir="D:/scratch",
)
```

Skipping tiles: `skip` lists tiles that are not read, for example tiles outside the sample or a failed stack. The output keeps the size and position it has with them, and their area stays 0 where no other tile covers it. Tiles are given as `Tile` objects or as ids: `"cFos-647/1-x00-y00"` (`channel/stack`, see `mosaic.tile_id(tile)`) for one tile, or `"1-x00-y00"` for that position in every channel.

```python
mosaic.fuse("acq_MIP.ome.zarr", projection="max", ds=(1, 8, 8), skip=["1-x00-y00", "cFos-647/2-x05-y03"])
```

Blend modes:

- **`max`**: keeps the brighter value.
- **`sum`**: adds the values, so overlaps show up as brighter bands.
- **`midline`**: uses `max` within each tile group. Where two groups overlap (for example the left and right illumination sides), the image switches sharply from one group to the other at the middle of their overlap, along `midline_axis`. Each side is then shown where its light sheet is sharpest.

`plan_fusion(mosaic.tiles, ...)` takes the fusion arguments (`ds`, `projection`, `blend`, `channels`, `crop`, `out_dtype`, `midline_axis`) and returns the level-0 geometry without reading anything: `plan.summary()`, `plan.shape`, `plan.scale`, `plan.nbytes`.

The fused output stores a **display window** for every channel (OMERO `window` metadata): the 0.5th to 99.9th percentile of the fused values, ignoring blank areas. Viewers that read it (napari-ome-zarr, Fiji, the *Open result* option of the napari widget) show the result with a usable contrast instead of the whole 0 to 65535 range. Read it with `pymif.mosaic_manager.display_windows(path)`.

## How fusion works

1. **Level 0** is fused one region at a time (256 × 256 output voxels in y/x for 3D, 1024 × 1024 for projections, over the full z range) into a temporary zarr. For each region, only the tile voxels that fall inside it are read, downsampled and merged. Memory stays bounded and every full-resolution voxel is read once.
2. The fused level 0, which is already downsampled and therefore small compared with the raw tiles, is written to the output with PyMIF's standard `to_zarr`. Pyramid levels, chunks, shards, compression and singleton axes therefore behave exactly as for any other conversion. The temporary store is then deleted.

With `workers > 1`, the tiles of a region are read in **worker processes** when the tiles allow it (as Luxendo tiles do). h5py serialises reads within one process, so threads would not speed up reading HDF5 files. Merging and writing stay in the calling process. If worker processes cannot be started, fusion falls back to threads with a warning.

Raw tiles usually have no pyramid, so every fusion reads the full-resolution data, whatever `ds` is. On a network drive the read speed sets the run time. Put `temp_dir` on a local disk when the output is on a network share.

## Command line

`pymif fuse` fuses one mosaic, or many from a CSV file with `--batch`. Every output option of `pymif convert` is available with the same flags. See [Command line](cli.md#fuse-a-mosaic).

```console
pymif fuse -i /path/to/2026-10-14_103957 -z acq_3D.ome.zarr -rds 4 8 8 -b midline -nl 3
pymif fuse -i /path/to/2026-10-14_103957 -z acq_MIP.ome.zarr -rds 1 8 8 -p max -b midline
```

## Napari

`Plugins > PyMIF > Mosaic Fusion` opens a widget with three boxes:

1. **Mosaic loading**: choose the microscope and the acquisition folder (browse, or drag and drop it, or any file inside it, from the file explorer), then *Load tiles*.
2. **Show tiles**: show the selected channels and groups as one layer per tile at its stage position, with one linked contrast per channel. Every displayed plane reads data from all shown tiles, so show only a few channels or groups at a time.
3. **Fuse to OME-Zarr**: the main fusion options, with a live estimate of the output size. *Tiles to fuse* can be set to *visible layers only* or *selected layers only*: tiles whose layer is hidden (eye icon off) or not selected in the layer list are then left blank, and *Blank in all channels* also blanks that position in the other channels. Tiles that are not shown in napari are fused normally. The output path follows the parameters (for example `<acquisition>_3D_ds4-8-8.ome.zarr`) until you type your own; dropping a folder on the output field writes the result inside that folder. When fusion finishes, the result can be opened directly.

4. **Batch CSV export**: *Append to batch CSV* adds the current settings, including the tiles left blank, as a row of a CSV file. Run the file later, for example on a workstation close to the data, with `pymif fuse --batch mosaic_batch.csv`.

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

`visualize()`, `fuse()`, `pymif fuse` and the napari widget then work unchanged. Use picklable tile data (such as `H5Array`) so that fusion can read it in worker processes.
