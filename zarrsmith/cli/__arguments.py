__all__ = ['_parse_arguments']

import argparse
import os
import re
import textwrap

from pymif import __version__
from pymif.utils.colors import parse_color as _parse_color


class MultilineDefaultsHelpFormatter(
    argparse.RawDescriptionHelpFormatter,
    argparse.ArgumentDefaultsHelpFormatter,
):
    """Formatter that preserves line breaks while still displaying defaults."""
    pass


# Valid type of's
def valid_input_path(x):
    """Validate an existing input file or directory argument."""
    if x is not None:
        if os.path.isdir(x) or os.path.isfile(x):
            return os.path.abspath(x)
        raise argparse.ArgumentTypeError(f'Input path {x} is not a valid directory')
    return None


def valid_output_path(x):
    """Validate that an output path does not already point to an existing file or directory."""
    if x is not None:
        if (not os.path.isdir(x)) and (not os.path.isfile(x)):
            return os.path.abspath(x)
        raise argparse.ArgumentTypeError(f'Output path {x} is not a valid directory')
    return None


def parse_color(value: str) -> str:
    """A CLI color (6-digit hex code, ``#`` optional, or a matplotlib color name) as ``RRGGBB``;
    see :func:`pymif.utils.colors.parse_color`."""
    try:
        return _parse_color(value)
    except TypeError as error:
        raise argparse.ArgumentTypeError(str(error)) from None

def parse_downscale_factor(value: str):
    """Parse a scalar or anisotropic downscale factor from a CSV cell.

    Examples
    --------
    ``2`` -> 2
    ``1 2 2`` -> (1, 2, 2)
    ``1,2,2`` -> (1, 2, 2)
    """
    if value is None:
        return 2

    text = str(value).strip()
    if text in {'', '-1', 'none', 'null', 'nan'}:
        return 2

    parts = [p for p in re.split(r'[\s,;]+', text) if p]
    if not parts:
        return 2

    try:
        factors = [int(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid downscale factor '{value}'. Use a scalar like '2' or a sequence like '1 2 2'."
        ) from exc

    if any(f <= 0 for f in factors):
        raise argparse.ArgumentTypeError(
            f"Invalid downscale factor '{value}'. All values must be positive integers."
        )

    return factors[0] if len(factors) == 1 else tuple(factors)


def parse_bool(value) -> bool:
    """Parse a --drop_singleton value or batch CSV cell into a bool."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1"}:
        return True
    if text in {"false", "f", "no", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Invalid boolean value '{value}'. Use true or false."
    )


def parse_shards_spec(value):
    """Parse a --shards value or batch CSV cell into "auto" or a shard shape.

    Examples
    --------
    ``auto`` -> ``"auto"``
    ``1 1 8 2160 4096`` -> ``(1, 1, 8, 2160, 4096)``
    ``1,1,8,2160,4096`` -> ``(1, 1, 8, 2160, 4096)``
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        tokens = [str(v).strip() for v in value]
        if len(tokens) == 1 and tokens[0].lower() in {"none", "null", "nan", "-1", ""}:
            return None
    else:
        text = str(value).strip()
        if text == "" or text.lower() in {"none", "null", "nan", "-1"}:
            return None
        tokens = [p for p in re.split(r'[\s,;]+', text) if p]

    if len(tokens) == 1 and tokens[0].lower() == "auto":
        return "auto"

    try:
        return tuple(int(t) for t in tokens)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid shards value '{value}'. Use 'auto' or a shard shape matching "
            "the dataset's axes, e.g. '1 1 8 2160 4096'."
        ) from exc


def parse_shard_exclude_axes_spec(value):
    """Parse a --shard_exclude_axes value or batch CSV cell into an axis tuple.

    Examples
    --------
    ``t c`` -> ``('t', 'c')``
    ``none`` -> ``()`` (allow every axis to be merged by shards="auto")
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        tokens = [str(v).strip().lower() for v in value]
    else:
        text = str(value).strip()
        if text == "":
            return None
        tokens = [p for p in re.split(r'[\s,;]+', text) if p]
        tokens = [t.lower() for t in tokens]

    if len(tokens) == 1 and tokens[0] in {"none", "null", "nan", "-1"}:
        return ()

    valid = {"t", "c", "z", "y", "x"}
    invalid = [t for t in tokens if t not in valid]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Invalid shard_exclude_axes {invalid!r}. Must be a subset of t, c, z, y, x, "
            "or 'none' to allow every axis to merge."
        )
    return tuple(tokens)


def _parse_subset_selection(selection: str, axis: str):
    """Parse a per-axis selection from a CLI subset specification."""
    text = selection.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()

    if text == "" or text.lower() in {"all", "none", "null", "nan", "*"}:
        return slice(None)

    if ":" in text:
        parts = [part.strip() for part in text.split(":")]
        if len(parts) > 3:
            raise argparse.ArgumentTypeError(
                f"Invalid subset selection '{selection}' for axis '{axis}'. "
                "Use 'start:stop[:step]', a comma-separated index list, or a single integer."
            )

        def _maybe_int(value: str):
            return None if value == "" else int(value)

        try:
            start = _maybe_int(parts[0])
            stop = _maybe_int(parts[1]) if len(parts) > 1 else None
            step = _maybe_int(parts[2]) if len(parts) > 2 else None
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid subset selection '{selection}' for axis '{axis}'. "
                "Slice bounds must be integers."
            ) from exc

        if step == 0:
            raise argparse.ArgumentTypeError(
                f"Invalid subset selection '{selection}' for axis '{axis}'. Step cannot be 0."
            )
        return slice(start, stop, step)

    if "," in text:
        try:
            return [int(part.strip()) for part in text.split(",") if part.strip()]
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid subset selection '{selection}' for axis '{axis}'. "
                "Use integers separated by commas."
            ) from exc

    try:
        return int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid subset selection '{selection}' for axis '{axis}'. "
            "Use a single integer, a comma-separated list, or a slice like '0:10:2'."
        ) from exc


def parse_subset_spec(value: str):
    """Parse a compact axis subset string into kwargs for subset_dataset().

    Examples
    --------
    ``y=10:100:2;x=20:80`` -> ``{"Y": slice(10, 100, 2), "X": slice(20, 80, None)}``
    ``z=0:10;c=0,2`` -> ``{"Z": slice(0, 10, None), "C": [0, 2]}``
    """
    if value is None:
        return None

    text = str(value).strip()
    if text == "" or text.lower() in {"none", "null", "nan", "-1"}:
        return None

    subset = {}
    for raw_spec in re.split(r"[;\n]+", text):
        spec = raw_spec.strip()
        if not spec:
            continue
        if "=" not in spec:
            raise argparse.ArgumentTypeError(
                f"Invalid subset specification '{value}'. "
                "Use axis=selection pairs separated by semicolons, e.g. 'y=10:100:2;x=20:80'."
            )
        axis, selection = spec.split("=", 1)
        axis = axis.strip().lower()
        if axis not in {"t", "c", "z", "y", "x"}:
            raise argparse.ArgumentTypeError(
                f"Invalid axis '{axis}' in subset specification '{value}'. "
                "Supported axes are t, c, z, y, and x."
            )
        subset[axis.upper()] = _parse_subset_selection(selection, axis)

    return subset

def _add_zarr_arguments(parser):
    """OME-Zarr output options shared by ``convert`` and ``fuse``."""
    parser.add_argument(
        '-cs', '--chunk_size',
        required= False,
        nargs= '+',
        help= 'Chunk size in TCZYX format, or whatever axes are present in the dataset.',
        type= int
    )
    parser.add_argument(
        '-ms', '--max_size',
        required= False,
        help= 'Max chunk size in MB. Ignored if --chunk_size is provided.',
        type= int
    )
    parser.add_argument(
        '-si', '--scene_index',
        required= False,
        help= 'Scene index for .czi and .vsi files.',
        type= int
    )
    parser.add_argument(
        '-cn', '--channel_names',
        required= False,
        nargs= '+',
        help= 'Name of channels. Example: -cn bf gfp rfp'
    )
    parser.add_argument(
        '-cc', '--channel_colors',
        required= False,
        type= parse_color,
        nargs= '+',
        help= 'Color(s) of channel(s). It needs to be hex or matplotlib color name. Example: -cc 0000FF cyan 00ff00'
    )
    parser.add_argument(
        '-zf', '--zarr_format',
        required=False,
        choices=[2, 3],
        type=int,
        help='Output zarr format. Zarr v2 maps to NGFF 0.4 and Zarr v3 maps to NGFF 0.5.',
    )
    parser.add_argument(
        '-df', '--downscale_factor',
        required=False,
        nargs='+',
        type=int,
        metavar='FACTOR',
        help='Pyramid downsampling factor. Use one value for isotropic downsampling or three values for anisotropic Z Y X factors, e.g. -df 1 2 2.',
    )
    parser.add_argument(
        '-nl', '--num_levels',
        required=False,
        type=int,
        metavar='FACTOR',
        help='Number of pyramidal levels.',
    )
    parser.add_argument(
        '-s', '--subset',
        required=False,
        type=str,
        help='Subset the input before pyramid generation. Format: axis=selection pairs separated by semicolons, e.g. "y=10:100:2;x=20:80". Use integers, comma-separated indices, or slices like 0:10:2.',
    )
    parser.add_argument(
        '-sh', '--shards',
        required=False,
        nargs='+',
        help=(
            'Zarr v3 sharding for the output. Use "auto" to size shards automatically, '
            'a shard shape matching the dataset axes, e.g. "-sh 1 1 8 2160 4096", or '
            '"none" to disable sharding. With --zarr_format 2 sharding is not supported '
            'and is set to none with a warning. Default: auto.'
        ),
    )
    parser.add_argument(
        '-stm', '--shard_target_mb',
        required=False,
        type=float,
        help='Target uncompressed shard size in MB, used when --shards auto. Default: 1024.',
    )
    parser.add_argument(
        '-sea', '--shard_exclude_axes',
        required=False,
        nargs='+',
        help=(
            'Axes that --shards auto never merges chunks along, e.g. "-sea t c". '
            'Use "none" to allow every axis to merge. Default: t c.'
        ),
    )
    parser.add_argument(
        '-ds', '--drop_singleton',
        required=False,
        type=parse_bool,
        help='Remove singleton t/c/z axes from the output ("true" or "false"). Default: true.',
    )
    parser.add_argument(
        '-nw', '--num_workers',
        required=False,
        type=int,
        help='Number of threads used to write the zarr in parallel. Default: half of the available cores, capped at 8.',
    )


def _add_mosaic_arguments(parser, legacy: bool = False):
    """Fusion options of ``fuse`` (``legacy``: the old ``mosaic2zarr`` spelling of --overwrite)."""
    parser.add_argument(
        '-rds', '--read_downsample' if legacy else '--downsample', dest='read_downsample', nargs='+', type=int,
        metavar='FACTOR',
        help='Downsampling applied while reading the tiles (block mean): one value or Z Y X, e.g. -rds 4 8 8. '
             'Default: 1 (full resolution).',
    )
    parser.add_argument(
        '-p', '--projection', choices=['none', 'max', 'sum'], default='none',
        help='"none" writes a 3D volume; "max"/"sum" write a 2D projection along z.',
    )
    parser.add_argument(
        '-b', '--blend', choices=['sigmoid', 'max', 'sum', 'midline'], default='sigmoid',
        help='How overlapping tiles are merged: sigmoid (default: smooth transition across every overlap, '
             'between tiles and between tile groups), max, sum (overlaps brighter), or midline (each tile group, '
             'e.g. illumination side, up to the middle of the overlap, with a sharp switch there).',
    )
    parser.add_argument('-ss', '--sigmoid_steepness', type=float,
                        help='Steepness of the blend=sigmoid transition: higher is sharper. Default: 4 '
                             '(the switch takes the central ~27%% of the overlap).')
    parser.add_argument('-ch', '--channels', nargs='+', help='Tile channels to fuse, in output order. Default: all.')
    parser.add_argument('-g', '--groups', nargs='+', help='Tile groups to fuse (e.g. L R). Default: all.')
    parser.add_argument(
        '-f', '--fields', nargs='+',
        help='Field to fuse (the output fits it): fields are groups of touching tiles, named field0, field1, '
             '... by position and listed when the mosaic is loaded. Tiles from several fields are not fused '
             '(an error is printed). Default: the first field (field0), unless --tiles is given.',
    )
    parser.add_argument(
        '-t', '--tiles', nargs='+',
        help='Only fuse these tiles (the output fits them): stacks such as 1-x00-y00 (every channel), or '
             '"channel/stack" ids such as cFos-647/1-x00-y00. Default: all.',
    )
    parser.add_argument(
        '-st', '--skip_tiles', nargs='+',
        help='Tiles to leave blank (not read; the output keeps its full size): "channel/stack" ids such as '
             'cFos-647/1-x00-y00, or a stack such as 1-x00-y00 for every channel.',
    )
    parser.add_argument('-ma', '--midline_axis', choices=['x', 'y', 'z'], default='x',
                        help='Axis along which blend=midline/sigmoid separates the groups.')
    parser.add_argument('-od', '--out_dtype', help='Output dtype, e.g. uint16 or float32. Default: input dtype '
                        '(float32 for blend=sum).')
    parser.add_argument('-rw', '--read_workers', type=int, default=4,
                        help='Tiles read in parallel (worker processes). Not to be confused with -nw (write threads).')
    parser.add_argument('-td', '--temp_dir', help='Folder for the temporary fused level 0. Default: next to the output.')
    parser.add_argument('-nff', '--no_flatfield', dest='flatfield', action='store_false',
                        help='Opera: do not apply the flat-field correction stored by Harmony '
                             '(applied by default, as in the images stitched by Harmony).')
    if legacy:
        parser.add_argument('-ow', '--overwrite', type=parse_bool, default=False,
                            help='Replace an existing output ("true" or "false").')
    else:
        parser.add_argument('-ow', '--overwrite', action='store_true', help='Replace an existing output.')


def _adapt_zarr_help_for_mosaic(parser):
    """Reword the shared output options where they mean something specific for a mosaic."""
    helps = {
        '--chunk_size': 'Output chunk size in C Z Y X (C Y X for projections) order.',
        '--max_size': 'Max output chunk size in MB. Ignored if --chunk_size is provided.',
        '--scene_index': 'Scene index, for Zeiss mosaics (not implemented yet).',
        '--channel_names': 'Names of the output channels. Example: -cn autofluo cfos',
        '--subset': 'Crop the fused output and/or select channels: axis=selection pairs separated by semicolons, '
                    'e.g. "z=0:100;y=200:900;c=1". z/y/x are in output voxels (after --downsample); '
                    'slices may not have a step.',
    }
    for action in parser._actions:
        for flag, text in helps.items():
            if flag in action.option_strings:
                action.help = text


CONVERT_HELP = """\
    Convert a dataset to OME-Zarr, or every dataset listed in a CSV file (--batch).

    Examples:
      pymif convert -i /path/to/dataset -z out.ome.zarr
      pymif convert -i /path/to/file.czi -m zeiss -si 1 -z out.ome.zarr -nl 4 -sh auto
      pymif convert --batch datasets.csv

    The --batch CSV file has the columns input | microscope | output, plus any of
    chunk_size | max_size(MB) | scene_index | zarr_format | downscale_factor | subset | channel_colors |
    channel_names | num_levels | shards | shard_target_mb | shard_exclude_axes | drop_singleton | num_workers
    with the same meaning as the options below, e.g.
      input            | microscope | output          | chunk_size    | zarr_format | downscale_factor | subset             | num_levels | shards
      /path/to/input_1 | opera      | /path/to/zarr_1 | 1 1 2 512 512 | 3           | 1 2 2            | y=10:100:2;x=20:80 | 3          | auto
      /path/to/input_2 | viventis   | /path/to/zarr_2 |               | 2           | 2                | z=0:10;c=0,2       |            |
    Leave a cell empty for the default. channel_colors can be hex codes or matplotlib color names.
    shards: "auto" (default when empty), an explicit shard shape, or "none". Not supported with zarr_format 2.
    shard_target_mb only applies to shards=auto (default 1024). shard_exclude_axes defaults to "t c"
    ("none" lets every axis merge). drop_singleton (true/false, default true) removes singleton t/c/z axes.
    num_workers is the number of write threads (default: half of the available cores, capped at 8).
"""

FUSE_HELP = """\
    Fuse the raw tiles of a mosaic (a raw Luxendo acquisition, an Opera Harmony export) into one
    OME-Zarr image, 3D or a 2D projection (-p max), or every mosaic listed in a CSV file (--batch).
    The output options are those of `pymif convert`; -s/--subset crops the fused output
    (z/y/x in output voxels, after --downsample) and selects channels (c).

    Examples:
      pymif fuse -i /path/to/2026-10-14_103957 -z out_3D.ome.zarr -rds 4 8 8 -nl 3
      pymif fuse -i /path/to/2026-10-14_103957 -z out_MIP.ome.zarr -rds 1 8 8 -p max -b midline
      pymif fuse -i /path/to/2026-10-14_103957 -z field1_MIP.ome.zarr -rds 1 8 8 -p max -f field1 -st 3-x01-y00
      pymif fuse -i "/path/to/plate-Measurement 1" -z A3_MIP.ome.zarr -p max -f A3_field1
      pymif fuse --batch mosaics.csv

    Tiles that touch or overlap form a field (field0, field1, ... ordered by position; for Opera plates
    one set per well: A3_field0, A3_field1, ...); the fields are listed when the mosaic is loaded. Only
    one field is fused at a time: the first one by default, another with -f. Tiles from separate fields
    are never fused together: an error is printed and the fusion is not done (in a batch, the other
    rows still run). Opera tiles are flat-field corrected as in Harmony, unless -nff is given.

    The --batch CSV file has the columns input | microscope | output, plus any of the output columns of
    `pymif convert --batch` (chunk_size | max_size(MB) | zarr_format | downscale_factor | subset | channel_colors |
    channel_names | num_levels | shards | shard_target_mb | shard_exclude_axes | drop_singleton | num_workers)
    and of the fusion columns
    downsample | projection | blend | sigmoid_steepness | channels | groups | fields | tiles | skip_tiles | midline_axis |
    out_dtype | read_workers | temp_dir | overwrite | flatfield (true/false, Opera only, default true), e.g.
      input                  | microscope | output                 | downsample | projection | blend   | num_levels
      /path/to/acquisition_1 | luxendo    | /path/to/acq1_3D.zarr  | 4 8 8      | none       | sigmoid | 3
      /path/to/acquisition_1 | luxendo    | /path/to/acq1_MIP.zarr | 1 8 8      | max        | midline |
    Leave a cell empty for the default. Lists (channels, groups, fields, tiles, skip_tiles, channel_names...) are space
    separated.
    The napari "Mosaic Fusion" widget writes such files with "Append to batch CSV".
"""

CHROMATIC_SHIFT_HELP = """\
    Measure the axial (Z) chromatic shift between the channels of an OME-Zarr image on the sample
    itself (its lower and upper surfaces) and write a corrected copy with the same chunks, shards,
    compression and metadata as the input. Only a Z translation per channel is corrected.

    Examples:
      pymif chromatic-shift -i brain.ome.zarr -z brain_csc.ome.zarr -r shift.json
      pymif chromatic-shift -i brain.ome.zarr -r shift.json                  (measure only)
      pymif chromatic-shift -i brain.ome.zarr -z brain_csc.ome.zarr -sf 0 104.4
      pymif chromatic-shift -i other.ome.zarr -z other_csc.ome.zarr -sf shift.json
      pymif chromatic-shift -i brain.ome.zarr -z brain_csc.ome.zarr -r shift.json --fail_on_warning
      pymif chromatic-shift --batch images.csv

    Shifts are corrections in the Z unit (usually micrometer): positive values move a channel
    towards higher Z. The reference channel gets 0.

    The --batch CSV file has the columns input | output (empty: measure only), plus any of
    report | shifts | reference_channel | timepoint | max_shift | coarse_level | refine_level | n_regions |
    crop_yx | min_z_normal | min_correlation | threshold | extent | order | include_groups | include_labels |
    overwrite | num_workers | fail_on_warning, with the same meaning as the options below, e.g.
      input                | output                   | report               | reference_channel | max_shift
      /path/to/a.ome.zarr  | /path/to/a_csc.ome.zarr  | /path/to/a_csc.json  | cFos-647          | 150
      /path/to/b.ome.zarr  | /path/to/b_csc.ome.zarr  |                      | 0                 |
    Leave a cell empty for the default; shifts are space separated.
"""


def parse_refine_level(value):
    """``--refine_level`` value: a level index, or ``none`` to skip refinement."""
    text = str(value).strip().lower()
    if text in {"none", "null"}:
        return "none"
    try:
        return int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid refine level '{value}'. Use a level index or 'none'.") from exc


def _add_chromatic_shift_arguments(parser):
    # Options whose default lives in the library are left out of the namespace when not
    # given (SUPPRESS), so the help shows the real default instead of "(default: None)".
    io = parser.add_argument_group('Input and output (or --batch)')
    io.add_argument('-i', '--input_path', type=valid_input_path, help='Path to the input OME-Zarr image.')
    io.add_argument('-z', '--zarr_path', type=os.path.abspath,
                    help='Path to the corrected .ome.zarr. Leave out to only measure the shift.')
    io.add_argument('-r', '--report', type=os.path.abspath,
                    help='Save the estimate report (JSON) to this file.')
    io.add_argument('--batch', metavar='CSV', type=valid_input_path,
                    help='Run every row of this CSV file instead (see the description above).')

    est = parser.add_argument_group('Estimate')
    est.add_argument('-sf', '--shifts', default=argparse.SUPPRESS, nargs='+',
                     help='Skip the estimate and apply these shifts: one value per channel, or a JSON report '
                          'from an earlier run.')
    est.add_argument('-rc', '--reference_channel', default='0',
                     help='Reference channel, as an index or a channel name.')
    est.add_argument('-tp', '--timepoint', default=argparse.SUPPRESS, type=int, help='Timepoint used for the estimate. Default: 0.')
    est.add_argument('-mxs', '--max_shift', default=argparse.SUPPRESS, type=float,
                     help='Largest shift searched, in the Z unit. Default: 10%% of the stack depth.')
    est.add_argument('-cl', '--coarse_level', default=argparse.SUPPRESS, type=int,
                     help='Pyramid level of the coarse estimate. Default: the coarsest level with at least '
                          '16 Z planes and 32 pixels in Y and X.')
    est.add_argument('-rl', '--refine_level', default=argparse.SUPPRESS, type=parse_refine_level,
                     help='Pyramid level of the refinement crops, or "none" to skip refinement. Default: 0.')
    est.add_argument('-nr', '--n_regions', default=argparse.SUPPRESS, type=int, help='Number of refinement crops. Default: 32.')
    est.add_argument('-cy', '--crop_yx', default=argparse.SUPPRESS, type=int, help='Crop width in Y and X, in pixels. Default: 64.')
    est.add_argument('-mzn', '--min_z_normal', default=argparse.SUPPRESS, type=float,
                     help='How flat a surface must be to count as Z-facing (1 = flat). Default: 0.7.')
    est.add_argument('-mc', '--min_correlation', default=argparse.SUPPRESS, type=float,
                     help='Crops with a lower correlation peak are left out. Default: 0.2.')
    est.add_argument('-th', '--threshold', default=argparse.SUPPRESS, type=float,
                     help='Sample mask threshold on the 0-1 rescaled coarse image. Default: Otsu.')
    est.add_argument('-fw', '--fail_on_warning', action='store_true',
                     help='Exit with an error, without writing the corrected image, when the estimate has '
                          'warnings (e.g. the search limit was reached).')

    out = parser.add_argument_group('Corrected image')
    out.add_argument('-e', '--extent', choices=['union', 'same', 'intersection'], default='union',
                     help='union grows Z so no channel loses data, same keeps the input size, intersection '
                          'crops to the planes every channel covers.')
    out.add_argument('-o', '--order', type=int, choices=[0, 1], default=1,
                     help='1 interpolates linearly; 0 rounds the shifts to whole planes (voxel values untouched).')
    out.add_argument('-ig', '--include_groups', type=parse_bool, default=True,
                     help='Also shift image subgroups with the same channels ("true" or "false").')
    out.add_argument('-il', '--include_labels', type=parse_bool, default=True,
                     help='Also write labels, padded or cropped to the new Z extent ("true" or "false").')
    out.add_argument('-ow', '--overwrite', action='store_true', help='Replace an existing output.')
    out.add_argument('-nw', '--num_workers', default=argparse.SUPPRESS, type=int,
                     help='Number of threads used to write the zarr. Default: half of the available cores, '
                          'capped at 8.')


#: Old command names, kept working: old name -> (mode, new spelling)
RENAMED = {
    '2zarr': ('convert', 'pymif convert'),
    'batch2zarr': ('convert_batch', 'pymif convert --batch'),
    'mosaic2zarr': ('fuse', 'pymif fuse'),
    'batch-mosaic2zarr': ('fuse_batch', 'pymif fuse --batch'),
    'batchmosaic2zarr': ('fuse_batch', 'pymif fuse --batch'),
}


def _add_convert_microscope(parser):
    parser.add_argument(
        '-m', '--microscope',
        required=False,
        default=None,
        help='Microscope used for imaging. Leave empty to auto-detect from the input path.',
        choices=['luxendo', 'opera', 'viventis', 'zeiss', 'evident', 'zarrv04', 'zarr', 'scape'],
        type=str,
    )


def _add_fuse_microscope(parser):
    parser.add_argument(
        '-m', '--microscope',
        required=False,
        default=None,
        choices=['luxendo', 'opera', 'zeiss'],
        type=str,
        help='Microscope of the mosaic. Leave empty to auto-detect (raw Luxendo acquisitions, Opera Harmony '
             'exports with Images/Index.xml). Zeiss mosaics are not implemented yet.',
    )


def _add_input_output(parser, *, required: bool, input_help: str, output_type):
    group = parser.add_argument_group('Input and output' + ('' if required else ' (or --batch)'))
    group.add_argument('-i', '--input_path', required=required, type=valid_input_path, help=input_help)
    group.add_argument('-z', '--zarr_path', required=required, type=output_type, help='Path to the output .ome.zarr.')
    if not required:
        group.add_argument('--batch', metavar='CSV', type=valid_input_path,
                           help='Run every row of this CSV file instead (see the description above).')


def _add_batch_file(parser):
    group = parser.add_argument_group('Required Named arguments.')
    group.add_argument('-i', '--input_file', required=True, type=valid_input_path, help='Path to the batch .csv file.')


def _parse_arguments(argv=None):
    """Build and parse the PyMIF command line interface arguments."""

    parser = argparse.ArgumentParser(
        description="""\
            Welcome fellow MIF users!
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    subparsers = parser.add_subparsers(
        title='Commands',
        description="Run `pymif <command> -h` for the options of a command.",
        dest='command',
        metavar='{convert,fuse,chromatic-shift}',
        required=True,
    )

    #####################################################################################
    # convert: one dataset, or a CSV batch
    convert = subparsers.add_parser(
        'convert',
        help='Convert a dataset to OME-Zarr (or a CSV batch of datasets with --batch).',
        description=textwrap.dedent(CONVERT_HELP),
        formatter_class=MultilineDefaultsHelpFormatter,
    )
    convert.set_defaults(mode='convert', renamed_to=None)
    _add_convert_microscope(convert)
    _add_zarr_arguments(convert)
    _add_input_output(convert, required=False, input_help='Path to the input dataset (file or folder).',
                      output_type=valid_output_path)

    #####################################################################################
    # fuse: one mosaic, or a CSV batch
    fuse = subparsers.add_parser(
        'fuse',
        help='Fuse the raw tiles of a mosaic into one OME-Zarr image (or a CSV batch with --batch).',
        description=textwrap.dedent(FUSE_HELP),
        formatter_class=MultilineDefaultsHelpFormatter,
    )
    fuse.set_defaults(mode='fuse', renamed_to=None)
    _add_fuse_microscope(fuse)
    _add_mosaic_arguments(fuse)
    _add_zarr_arguments(fuse)
    _adapt_zarr_help_for_mosaic(fuse)
    _add_input_output(fuse, required=False, input_help='Acquisition folder (for Luxendo: the folder containing raw/; '
                                                       'for Opera: the Harmony export containing Images/).',
                      output_type=os.path.abspath)

    #####################################################################################
    # chromatic-shift: one OME-Zarr image, or a CSV batch
    chromatic = subparsers.add_parser(
        'chromatic-shift',
        help='Measure and correct the Z chromatic shift between channels of an OME-Zarr image '
             '(or a CSV batch with --batch).',
        description=textwrap.dedent(CHROMATIC_SHIFT_HELP),
        formatter_class=MultilineDefaultsHelpFormatter,
    )
    chromatic.set_defaults(mode='chromatic_shift', renamed_to=None)
    _add_chromatic_shift_arguments(chromatic)

    #####################################################################################
    # Old names (not listed in the help), with their original arguments
    for old, (mode, new) in RENAMED.items():
        if old == 'batchmosaic2zarr':
            continue  # alias of batch-mosaic2zarr
        aliases = ['batchmosaic2zarr'] if old == 'batch-mosaic2zarr' else []
        p = subparsers.add_parser(old, aliases=aliases, description=f'Old name of `{new}`.',
                                  formatter_class=MultilineDefaultsHelpFormatter)
        p.set_defaults(mode=mode, renamed_to=new)
        if mode == 'convert':
            _add_convert_microscope(p)
            _add_zarr_arguments(p)
            _add_input_output(p, required=True, input_help='Path to input file.', output_type=valid_output_path)
        elif mode == 'fuse':
            _add_fuse_microscope(p)
            _add_mosaic_arguments(p, legacy=True)
            _add_zarr_arguments(p)
            _adapt_zarr_help_for_mosaic(p)
            _add_input_output(p, required=True, input_help='Acquisition folder.', output_type=os.path.abspath)
        else:
            _add_batch_file(p)

    args = parser.parse_args(argv)

    # convert / fuse: either -i and -z, or --batch
    if args.mode in ('convert', 'fuse') and args.renamed_to is None:
        command = subparsers.choices[args.mode]
        if args.batch is not None:
            if args.input_path is not None or args.zarr_path is not None:
                command.error('--batch cannot be combined with -i/-z: the CSV file lists inputs and outputs.')
            args.mode += '_batch'
            args.input_file = args.batch
        elif args.input_path is None or args.zarr_path is None:
            command.error('the following arguments are required: -i/--input_path and -z/--zarr_path (or --batch CSV)')

    # chromatic-shift: -i (with an optional -z), or --batch
    if args.mode == 'chromatic_shift':
        command = subparsers.choices['chromatic-shift']
        if args.batch is not None:
            if args.input_path is not None or args.zarr_path is not None or args.report is not None:
                command.error('--batch cannot be combined with -i/-z/-r: the CSV file lists them.')
            args.mode = 'chromatic_shift_batch'
            args.input_file = args.batch
        elif args.input_path is None:
            command.error('the following arguments are required: -i/--input_path (or --batch CSV)')
    return args
