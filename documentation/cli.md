# Command line

The `pymif` command converts one dataset or a CSV-defined batch to OME-Zarr, and fuses mosaics.

| Command | Does |
|---|---|
| `pymif 2zarr` | convert one dataset |
| `pymif batch2zarr` | convert the datasets listed in a CSV file |
| `pymif mosaic2zarr` | fuse the raw tiles of one mosaic |
| `pymif batch-mosaic2zarr` | fuse the mosaics listed in a CSV file (alias `batchmosaic2zarr`) |

Get help on any of them:

```console
pymif -h
pymif 2zarr -h
pymif batch2zarr -h
pymif mosaic2zarr -h
pymif batch-mosaic2zarr -h
```

## Datasets

Single conversion:

```console
pymif 2zarr -i INPUT_PATH -m MICROSCOPE -z OUTPUT_ZARR
```

Batch conversion from a CSV manifest:

```console
pymif batch2zarr -i INPUT_FILE.csv
```

The output options (chunks, sharding, zarr format, singleton axes, workers) are described in [Advanced writing options](advanced.md).

## Mosaics

`mosaic2zarr` fuses the raw tiles of a mosaic (see [Mosaics](mosaic.md)) into one OME-Zarr image:

```console
# 3D, downsampled 4 (z) x 8 (y) x 8 (x), illumination sides cut at the middle of their overlap
pymif mosaic2zarr -i /path/to/2026-10-14_103957 -z acq_3D.ome.zarr -rds 4 8 8 -b midline -nl 3

# 2D maximum projection, downsampled 8 x 8 in y/x
pymif mosaic2zarr -i /path/to/2026-10-14_103957 -z acq_MIP.ome.zarr -rds 1 8 8 -p max -b midline

# where are the tiles? (overlaps are brighter)
pymif mosaic2zarr -i /path/to/2026-10-14_103957 -z acq_tiles.ome.zarr -rds 1 8 8 -p max -b sum
```

Every `2zarr` option is available with the same flag and meaning, applied to the fused output:

| Flag | Meaning for a mosaic |
|---|---|
| `-cs`, `--chunk_size` | output chunk size, in `C Z Y X` order (`C Y X` for projections) |
| `-ms`, `--max_size` | max output chunk size in MB (ignored with `-cs`) |
| `-si`, `--scene_index` | for Zeiss mosaics (not implemented yet) |
| `-cn`, `--channel_names` / `-cc`, `--channel_colors` | names and colours of the output channels |
| `-zf`, `--zarr_format` | 2 (NGFF 0.4) or 3 (NGFF 0.5) |
| `-df`, `--downscale_factor` / `-nl`, `--num_levels` | pyramid factor and number of levels (estimated if omitted) |
| `-s`, `--subset` | crop the fused output (`z`/`y`/`x` in output voxels, after `-rds`) and select channels (`c`), e.g. `"z=0:100;y=200:900;c=1"` |
| `-sh`, `-stm`, `-sea` | sharding, as for `2zarr` |
| `-ds`, `--drop_singleton` | drop a single channel axis (default true) |
| `-nw`, `--num_workers` | write threads |

The fusion options:

| Flag | Meaning |
|---|---|
| `-m`, `--microscope` | `luxendo`, `opera`, `zeiss`; auto-detected for raw Luxendo acquisitions |
| `-rds`, `--read_downsample` | downsampling while reading (block mean): one value or `Z Y X` (default 1) |
| `-p`, `--projection` | `none` (3D, default), `max` or `sum` |
| `-b`, `--blend` | `max` (default), `sum` or `midline` |
| `-ch`, `--channels` / `-g`, `--groups` | tile channels and groups (e.g. `L R`) to fuse |
| `-st`, `--skip_tiles` | tiles to leave blank: `channel/stack` ids such as `cFos-647/1-x00-y00`, or a stack such as `1-x00-y00` for every channel |
| `-ma`, `--midline_axis` | `x` (default), `y` or `z` |
| `-od`, `--out_dtype` | output dtype |
| `-rw`, `--read_workers` | tiles read in parallel processes (default 4) |
| `-td`, `--temp_dir` | folder for the temporary fused level 0 (a local disk is faster) |
| `-ow`, `--overwrite` | `true` to replace an existing output |

`batch-mosaic2zarr` reads a CSV file with the columns `input`, `microscope`, `output`, plus any of the `2zarr` columns (`chunk_size`, `max_size(MB)`, `scene_index`, `zarr_format`, `downscale_factor`, `subset`, `channel_colors`, `channel_names`, `num_levels`, `shards`, `shard_target_mb`, `shard_exclude_axes`, `drop_singleton`, `num_workers`) and of the fusion columns (`read_downsample`, `projection`, `blend`, `channels`, `groups`, `skip_tiles`, `midline_axis`, `out_dtype`, `read_workers`, `temp_dir`, `overwrite`). Leave a cell empty for the default; lists are space separated. The napari *Mosaic Fusion* widget writes such rows with *Append to batch CSV*.

```text
input,microscope,output,read_downsample,projection,blend,num_levels,shards
/path/to/acquisition_1,luxendo,/path/to/acq1_3D.ome.zarr,4 8 8,none,midline,3,auto
/path/to/acquisition_1,luxendo,/path/to/acq1_MIP.ome.zarr,1 8 8,max,midline,,none
```
