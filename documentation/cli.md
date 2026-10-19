# Command line

The `pymif` command has three subcommands:

| Command | Does |
|---|---|
| `pymif convert` | convert a dataset to OME-Zarr, or every dataset listed in a CSV file with `--batch` |
| `pymif fuse` | fuse the raw tiles of a mosaic into one OME-Zarr image, or every mosaic listed in a CSV file with `--batch` |
| `pymif chromatic-shift` | measure and correct the Z chromatic shift between the channels of an OME-Zarr image, or of every image listed in a CSV file with `--batch` |

`pymif -h` lists them, and `pymif convert -h`, `pymif fuse -h` and `pymif chromatic-shift -h` show every option, examples and the batch CSV columns. `pymif --version` prints the installed version.

Each command runs either one dataset (`-i INPUT -z OUTPUT`) or a CSV file (`--batch FILE.csv`), never both (`chromatic-shift` can also run with `-i` alone, to only measure). In a batch file, one row is one run and the columns take the place of the options.

## Convert a dataset

```console
pymif convert -i INPUT_PATH -m MICROSCOPE -z OUTPUT_ZARR
pymif convert --batch datasets.csv
```

`-m` is one of `luxendo`, `opera`, `viventis`, `zeiss`, `zarr`, `zarrv04`, `scape`; leave it out to detect it from the input path. The output options:

| Flag | Meaning |
|---|---|
| `-cs`, `--chunk_size` | chunk size, one value per axis of the dataset |
| `-ms`, `--max_size` | max chunk size in MB (ignored with `-cs`) |
| `-si`, `--scene_index` | scene of a `.czi` file |
| `-cn`, `--channel_names` / `-cc`, `--channel_colors` | names and colours (hex or matplotlib names) of the channels |
| `-zf`, `--zarr_format` | 2 (NGFF 0.4) or 3 (NGFF 0.5, default) |
| `-df`, `--downscale_factor` / `-nl`, `--num_levels` | pyramid factor and number of levels (estimated if omitted) |
| `-s`, `--subset` | crop or select before writing, e.g. `"y=10:100:2;x=20:80;c=0,2"` |
| `-sh`, `-stm`, `-sea` | sharding (see [Advanced writing options](advanced.md)) |
| `-ds`, `--drop_singleton` | drop size-1 `t`/`c`/`z` axes (default true) |
| `-nw`, `--num_workers` | write threads |

The batch CSV has the columns `input`, `microscope`, `output`, plus any of `chunk_size`, `max_size(MB)`, `scene_index`, `zarr_format`, `downscale_factor`, `subset`, `channel_colors`, `channel_names`, `num_levels`, `shards`, `shard_target_mb`, `shard_exclude_axes`, `drop_singleton`, `num_workers`. The napari converter writes such rows with *Append to batch CSV*.

## Fuse a mosaic

`fuse` fuses the raw tiles of a mosaic (see [Mosaics](mosaic.md)) into one OME-Zarr image:

```console
# 3D, downsampled 4 (z) x 8 (y) x 8 (x), smooth (sigmoid) transitions between tiles and illumination sides
pymif fuse -i /path/to/2026-10-14_103957 -z acq_3D.ome.zarr -rds 4 8 8 -nl 3

# 2D maximum projection, downsampled 8 x 8 in y/x, sharp cut between illumination sides
pymif fuse -i /path/to/2026-10-14_103957 -z acq_MIP.ome.zarr -rds 1 8 8 -p max -b midline

# where are the tiles? (overlaps are brighter)
pymif fuse -i /path/to/2026-10-14_103957 -z acq_tiles.ome.zarr -rds 1 8 8 -p max -b sum

# many mosaics from a CSV file
pymif fuse --batch mosaics.csv
```

The fusion options:

| Flag | Meaning |
|---|---|
| `-m`, `--microscope` | `luxendo`, `opera`, `zeiss`; auto-detected for raw Luxendo acquisitions and Opera Harmony exports (`Images/Index.xml`) |
| `-rds`, `--downsample` | downsampling while reading (block mean): one value or `Z Y X` (default 1) |
| `-p`, `--projection` | `none` (3D, default), `max` or `sum` |
| `-b`, `--blend` | `sigmoid` (default), `max`, `sum` or `midline` |
| `-ss`, `--sigmoid_steepness` | steepness of the `sigmoid` transition, higher is sharper (default 4) |
| `-ch`, `--channels` / `-g`, `--groups` | tile channels and groups (e.g. `L R`) to fuse |
| `-f`, `--fields` | the field to fuse (groups of touching tiles, `field0`, `field1`, ... listed when the mosaic is loaded); default: the first field. Tiles from several fields are not fused: an error is printed (see [Fields](mosaic.md#fields)) |
| `-t`, `--tiles` | only fuse these tiles, and fit the output to them: stacks such as `1-x00-y00` (every channel), or `channel/stack` ids |
| `-st`, `--skip_tiles` | tiles to leave blank (the output keeps its size): `channel/stack` ids such as `cFos-647/1-x00-y00`, or a stack such as `1-x00-y00` for every channel |
| `-nff`, `--no_flatfield` | Opera: do not apply Harmony's flat-field correction (applied by default, see [Opera](mosaic.md#opera-harmony-exports)) |
| `-ma`, `--midline_axis` | axis along which `sigmoid`/`midline` separate the tile groups: `x` (default), `y` or `z` |
| `-od`, `--out_dtype` | output dtype |
| `-rw`, `--read_workers` | tiles read in parallel processes (default 4); not to be confused with `-nw` (write threads) |
| `-td`, `--temp_dir` | folder for the temporary fused level 0 (a local disk is faster) |
| `-ow`, `--overwrite` | replace an existing output |

Every output option of `convert` is available too, applied to the fused output. Two read differently for a mosaic: `-cs` is given in `C Z Y X` order (`C Y X` for projections), and `-s/--subset` crops the fused output (`z`/`y`/`x` in output voxels, after `-rds`) and selects channels (`c`), e.g. `"z=0:100;y=200:900;c=1"`. `-si` is reserved for Zeiss mosaics (not implemented yet).

The batch CSV has the columns `input`, `microscope`, `output`, plus any of the output columns of `convert --batch` and of the fusion columns `downsample`, `projection`, `blend`, `sigmoid_steepness`, `channels`, `groups`, `fields`, `tiles`, `skip_tiles`, `midline_axis`, `out_dtype`, `read_workers`, `temp_dir`, `overwrite`. Leave a cell empty for the default; lists are space separated. A row asking for tiles from several fields is not fused; the other rows still run, and the rows that were not fused are listed at the end. The napari *Mosaic Fusion* widget writes such rows with *Append to batch CSV*.

```text
input,microscope,output,downsample,projection,blend,num_levels,shards
/path/to/acquisition_1,luxendo,/path/to/acq1_3D.ome.zarr,4 8 8,none,sigmoid,3,auto
/path/to/acquisition_1,luxendo,/path/to/acq1_MIP.ome.zarr,1 8 8,max,midline,,none
```

## Correct the chromatic shift

`chromatic-shift` measures the Z offset between the channels of an OME-Zarr image on the sample itself and writes a corrected copy with the same chunks, shards, compression and metadata as the input (see [Chromatic shift correction](chromatic-shift.md)):

```console
# measure, save the report and write the corrected image
pymif chromatic-shift -i brain.ome.zarr -z brain_csc.ome.zarr -r shift.json

# only measure
pymif chromatic-shift -i brain.ome.zarr -r shift.json

# apply known shifts (one per channel), or the shifts of an earlier report
pymif chromatic-shift -i brain.ome.zarr -z brain_csc.ome.zarr -sf 0 104.4
pymif chromatic-shift -i other.ome.zarr -z other_csc.ome.zarr -sf shift.json

# many images from a CSV file
pymif chromatic-shift --batch images.csv
```

| Flag | Meaning |
|---|---|
| `-i`, `--input_path` | input OME-Zarr image |
| `-z`, `--zarr_path` | corrected OME-Zarr image; leave out to only measure |
| `-r`, `--report` | save the estimate report (JSON) |
| `-sf`, `--shifts` | skip the estimate: one shift per channel in the Z unit, or a JSON report |
| `-rc`, `--reference_channel` | reference channel, as an index or a name (default 0) |
| `-mxs`, `--max_shift` | largest shift searched, in the Z unit (default: 10% of the stack depth) |
| `-tp`, `-cl`, `-rl`, `-nr`, `-cy`, `-mzn`, `-mc`, `-th` | timepoint, coarse level, refine level (`none` skips refinement), number of crops, crop width, `min_z_normal`, `min_correlation`, mask threshold |
| `-fw`, `--fail_on_warning` | exit with an error, writing only the report, when the estimate has warnings |
| `-e`, `--extent` | `union` (default), `same` or `intersection` |
| `-o`, `--order` | `1` (linear, default) or `0` (whole planes, values untouched) |
| `-ig`, `-il` | also write image subgroups / labels (`true` or `false`, default true) |
| `-ow`, `--overwrite` | replace an existing output |
| `-nw`, `--num_workers` | write threads |

In a pipeline (e.g. a Nextflow process), use `-r` to keep the report next to the output and `--fail_on_warning` so that a doubtful estimate (search limit reached, no usable crop) fails the step instead of producing a wrongly corrected image.

The batch CSV has the columns `input`, `output` (empty: only measure), plus any of `report`, `shifts`, `reference_channel`, `timepoint`, `max_shift`, `coarse_level`, `refine_level`, `n_regions`, `crop_yx`, `min_z_normal`, `min_correlation`, `threshold`, `extent`, `order`, `include_groups`, `include_labels`, `overwrite`, `num_workers`, `fail_on_warning`. Leave a cell empty for the default; shifts are space separated.

```text
input,output,report,reference_channel,max_shift
/path/to/a.ome.zarr,/path/to/a_csc.ome.zarr,/path/to/a_csc.json,cFos-647,150
/path/to/b.ome.zarr,,/path/to/b_shift.json,0,
```

## Old command names

The commands used to have other names. They still work, with their original options, and print a one-line notice with the new name:

| Old | New |
|---|---|
| `pymif 2zarr` | `pymif convert` |
| `pymif batch2zarr -i FILE.csv` | `pymif convert --batch FILE.csv` |
| `pymif mosaic2zarr` (`-ow true`) | `pymif fuse` (`--overwrite`) |
| `pymif batch-mosaic2zarr -i FILE.csv` | `pymif fuse --batch FILE.csv` |

Batch CSV files written before the rename (with a `read_downsample` column) still run.
