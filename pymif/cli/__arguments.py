__all__ = ['_parse_arguments']

import argparse
import os
import re
import textwrap
from matplotlib.colors import cnames

from pymif import __version__


class MultilineDefaultsHelpFormatter(
    argparse.RawDescriptionHelpFormatter,
    argparse.ArgumentDefaultsHelpFormatter,
):
    """Formatter that preserves line breaks while still displaying defaults."""
    pass


HEX_PATTERN = re.compile(r'^#?[0-9a-fA-F]{6}$')


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
    """Parse a CLI color input:

    - Accept 6-digit hex codes (# optional)

    - Accept color names from matplotlib.colors.cnames

    - Raise a meaningful error if invalid
    """

    v = value.strip()

    # --- 1) Hex code (with or without #) ---
    if HEX_PATTERN.match(v):
        return v.replace("#", "").upper()

    # --- 2) Matplotlib named color ---
    lower = v.lower()
    if lower in cnames:
        # cnames returns a hex string with '#', e.g. "#ff00ff"
        return cnames[lower].replace("#", "").upper()

    # --- 3) Fail: report detailed reason ---
    raise argparse.ArgumentTypeError(
        f"Invalid color '{value}'. "
        f"Must be:\n"
        f"  • A 6-digit hex code (e.g. FF00FF or #ff00ff), OR\n"
        f"  • A valid color name from matplotlib ({', '.join(list(cnames.keys())[:10])}, ...)"
    )

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
    """OME-Zarr output options shared by ``2zarr`` and ``mosaic2zarr``."""
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
        help= 'Scene index for .czi files.',
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


def _add_mosaic_arguments(parser):
    """Fusion options of ``mosaic2zarr``."""
    parser.add_argument(
        '-rds', '--read_downsample', nargs='+', type=int, metavar='FACTOR',
        help='Downsampling applied while reading the tiles (block mean): one value or Z Y X, e.g. -rds 4 8 8. Default: 1 (full resolution).',
    )
    parser.add_argument(
        '-p', '--projection', choices=['none', 'max', 'sum'], default='none',
        help='"none" writes a 3D volume; "max"/"sum" write a 2D projection along z.',
    )
    parser.add_argument(
        '-b', '--blend', choices=['max', 'sum', 'midline'], default='max',
        help='How overlapping tiles are merged: max, sum (overlaps brighter), or midline (each tile group, '
             'e.g. illumination side, up to the middle of the overlap, with a sharp switch there).',
    )
    parser.add_argument('-ch', '--channels', nargs='+', help='Tile channels to fuse, in output order. Default: all.')
    parser.add_argument('-g', '--groups', nargs='+', help='Tile groups to fuse (e.g. L R). Default: all.')
    parser.add_argument(
        '-st', '--skip_tiles', nargs='+',
        help='Tiles to leave blank (not read; the output keeps its full size): "channel/stack" ids such as '
             'cFos-647/1-x00-y00, or a stack such as 1-x00-y00 for every channel.',
    )
    parser.add_argument('-ma', '--midline_axis', choices=['x', 'y', 'z'], default='x',
                        help='Axis along which blend=midline separates the groups.')
    parser.add_argument('-od', '--out_dtype', help='Output dtype, e.g. uint16 or float32. Default: input dtype '
                        '(float32 for blend=sum).')
    parser.add_argument('-rw', '--read_workers', type=int, default=4,
                        help='Tiles read in parallel (worker processes).')
    parser.add_argument('-td', '--temp_dir', help='Folder for the temporary fused level 0. Default: next to the output.')
    parser.add_argument('-ow', '--overwrite', type=parse_bool, default=False,
                        help='Replace an existing output ("true" or "false").')


def _adapt_zarr_help_for_mosaic(parser):
    """Reword the shared 2zarr options where they mean something specific for a mosaic."""
    helps = {
        '--chunk_size': 'Output chunk size in C Z Y X (C Y X for projections) order.',
        '--max_size': 'Max output chunk size in MB. Ignored if --chunk_size is provided.',
        '--scene_index': 'Scene index, for Zeiss mosaics (not implemented yet).',
        '--channel_names': 'Names of the output channels. Example: -cn autofluo cfos',
        '--subset': 'Crop the fused output and/or select channels: axis=selection pairs separated by semicolons, '
                    'e.g. "z=0:100;y=200:900;c=1". z/y/x are in output voxels (after --read_downsample); '
                    'slices may not have a step.',
    }
    for action in parser._actions:
        for flag, text in helps.items():
            if flag in action.option_strings:
                action.help = text


def _parse_arguments():
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

    # Sub-parsers
    subparsers = parser.add_subparsers(
        title='Runmodes',
        description= """\
            PyMIF has several runmodes, each with different and specific arguments.
            Please consult each runmode's help manual before running any of them.
            Enjoy PyMIF!
        """,
        required=True,
    )

    #####################################################################################
    # Single convert 2 zarr parser
    single_convert_parser = subparsers.add_parser(
        '2zarr',
        help= 'Convert to zarr format a single image.',
        formatter_class= argparse.ArgumentDefaultsHelpFormatter
    )
    single_convert_parser.add_argument(
        '--runmode',
        help= argparse.SUPPRESS,
        default= 0,
        type= int
    )

    # Optional args
    single_convert_parser.add_argument(
        '-m', '--microscope',
        required=False,
        default=None,
        help='Microscope used for imaging. Leave empty to auto-detect from the input path.',
        choices=['luxendo', 'opera', 'viventis', 'opera', 'zeiss', 'zarrv04', 'zarr', 'scape'],
        type=str,
    )
    _add_zarr_arguments(single_convert_parser)

    # Required args
    requiredNamed = single_convert_parser.add_argument_group('Required Named arguments.')
    requiredNamed.add_argument(
        '-i', '--input_path',
        required= True,
        help= 'Path to input file.',
        type= valid_input_path
    )
    requiredNamed.add_argument(
        '-z', '--zarr_path',
        required= True,
        help= 'Path to output zarr.',
        type= valid_output_path
    )

    #####################################################################################
    # Batch convert 2 zarr parser
    long_block = """\
        Convert to zarr format a batch of images.
        The INPUT_FILE is a .csv file of the form:
        input              | microscope  | output           | chunk_size    | max_size(MB) | scene_index | zarr_format | downscale_factor | subset                         | channel_colors | channel_names | num_levels | shards            | shard_target_mb | shard_exclude_axes | drop_singleton | num_workers
        /path/to/input_1   | opera       | /path/to/zarr_1  | 1 1 2 512 512 | 100          | 0           | 3           | 1 2 2            | y=10:100:2;x=20:80             | lime white     | gfp bf        | 3          | auto              | 256             | t c                | true           | 4
        /path/to/input_2   | viventis    | /path/to/zarr_2  |               | 100          |             | 2           | 2                | z=0:10;c=0,2                   | 000FF FF00FF   |               |            |                   |                 |                    |                |
        ...
        /path/to/input_n   | viventis    | /path/to/zarr_n  | 1 1 2 512 512 |              | 0           | 3           | 2                |                                |                |               | 2          | 1 1 8 2160 4096   |                 | none               | false          |
        channel_colors can be hex code or valid matplotlib colors.
        shards: "auto" (default when empty), an explicit shard shape matching the dataset axes, or "none" to disable. Not supported with zarr_format 2 (set to none with a warning).
        shard_target_mb only applies to shards=auto (default 1024). shard_exclude_axes defaults to "t c"; use "none" to allow every axis to merge.
        drop_singleton (true/false, default true) removes singleton t/c/z axes from the output.
        num_workers is the number of write threads (default: half of the available cores, capped at 8).
    """
    batch_convert_parser = subparsers.add_parser(
        'batch2zarr',
        help= 'Convert a batch of images to zarr (short one-line help). A dedicated instructions file needs to be generated beforehand. Proceed to the batch2zarr help manual for the example use.',
        description= textwrap.dedent(long_block),
        formatter_class= MultilineDefaultsHelpFormatter
    )
    batch_convert_parser.add_argument(
        '--runmode',
        help= argparse.SUPPRESS,
        default= 1,
        type= int
    )
    
    # Required args
    requiredNamed = batch_convert_parser.add_argument_group('Required Named arguments.')
    requiredNamed.add_argument(
        '-i', '--input_file',
        required= True,
        help= 'Path to input file.',
        type= valid_input_path
    )

    #####################################################################################
    # Mosaic fusion 2 zarr parser
    mosaic_parser = subparsers.add_parser(
        'mosaic2zarr',
        help='Fuse the raw tiles of a mosaic (e.g. a raw Luxendo acquisition) into one OME-Zarr image.',
        description=textwrap.dedent("""\
            Fuse the raw tiles of a mosaic into one OME-Zarr image (3D, or a 2D projection with -p max).
            Every 2zarr option is available with the same meaning; -s/--subset crops the fused output
            (z/y/x in output voxels, after --read_downsample) and selects channels (c).
            Examples:
              pymif mosaic2zarr -i /path/to/2026-10-14_103957 -z out_3D.ome.zarr -rds 4 8 8 -b midline -nl 3
              pymif mosaic2zarr -i /path/to/2026-10-14_103957 -z out_MIP.ome.zarr -rds 1 8 8 -p max
        """),
        formatter_class=MultilineDefaultsHelpFormatter,
    )
    mosaic_parser.add_argument('--runmode', help=argparse.SUPPRESS, default=2, type=int)
    mosaic_parser.add_argument(
        '-m', '--microscope',
        required=False,
        default=None,
        choices=['luxendo', 'opera', 'zeiss'],
        type=str,
        help='Microscope of the mosaic. Leave empty to auto-detect (raw Luxendo acquisitions). '
             'Opera and Zeiss mosaics are not implemented yet.',
    )
    _add_mosaic_arguments(mosaic_parser)
    _add_zarr_arguments(mosaic_parser)
    _adapt_zarr_help_for_mosaic(mosaic_parser)
    requiredNamed = mosaic_parser.add_argument_group('Required Named arguments.')
    requiredNamed.add_argument(
        '-i', '--input_path', required=True, type=valid_input_path,
        help='Acquisition folder (for Luxendo: the folder containing raw/).',
    )
    requiredNamed.add_argument(
        '-z', '--zarr_path', required=True, type=os.path.abspath,
        help='Path to the output .ome.zarr (use --overwrite true to replace an existing one).',
    )

    #####################################################################################
    # Batch mosaic fusion parser
    mosaic_block = """\
        Fuse a batch of mosaics to zarr. The INPUT_FILE is a .csv file with the columns
        input | microscope | output, plus any of the 2zarr columns
        chunk_size | max_size(MB) | scene_index | zarr_format | downscale_factor | subset | channel_colors |
        channel_names | num_levels | shards | shard_target_mb | shard_exclude_axes | drop_singleton | num_workers
        and any of the fusion columns
        read_downsample | projection | blend | channels | groups | skip_tiles |
        midline_axis | out_dtype | read_workers | temp_dir | overwrite
        Example:
        input                    | microscope | output                 | read_downsample | projection | blend   | num_levels | shards
        /path/to/acquisition_1   | luxendo    | /path/to/acq1_3D.zarr  | 4 8 8           | none       | midline | 3          | auto
        /path/to/acquisition_1   | luxendo    | /path/to/acq1_MIP.zarr | 1 8 8           | max        | midline |            | none
        Leave a cell empty for the default. Lists (channels, groups, channel_names...) are space separated.
    """
    batch_mosaic_parser = subparsers.add_parser(
        'batch-mosaic2zarr',
        aliases=['batchmosaic2zarr'],
        help='Fuse a batch of mosaics to zarr from a CSV file. See the batch-mosaic2zarr help manual for the columns.',
        description=textwrap.dedent(mosaic_block),
        formatter_class=MultilineDefaultsHelpFormatter,
    )
    batch_mosaic_parser.add_argument('--runmode', help=argparse.SUPPRESS, default=3, type=int)
    requiredNamed = batch_mosaic_parser.add_argument_group('Required Named arguments.')
    requiredNamed.add_argument('-i', '--input_file', required=True, type=valid_input_path,
                               help='Path to the batch .csv file.')

    #####################################################################################
    # Possible other runmodes

    #####################################################################################
    args = parser.parse_args()
    return args