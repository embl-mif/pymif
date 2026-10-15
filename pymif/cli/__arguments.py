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
                        help='Tiles read in parallel (worker processes). Not to be confused with -nw (write threads).')
    parser.add_argument('-td', '--temp_dir', help='Folder for the temporary fused level 0. Default: next to the output.')
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
    Fuse the raw tiles of a mosaic (e.g. a raw Luxendo acquisition) into one OME-Zarr image,
    3D or a 2D projection (-p max), or every mosaic listed in a CSV file (--batch).
    The output options are those of `pymif convert`; -s/--subset crops the fused output
    (z/y/x in output voxels, after --downsample) and selects channels (c).

    Examples:
      pymif fuse -i /path/to/2026-10-14_103957 -z out_3D.ome.zarr -rds 4 8 8 -b midline -nl 3
      pymif fuse -i /path/to/2026-10-14_103957 -z out_MIP.ome.zarr -rds 1 8 8 -p max -b midline
      pymif fuse --batch mosaics.csv

    The --batch CSV file has the columns input | microscope | output, plus any of the output columns of
    `pymif convert --batch` (chunk_size | max_size(MB) | zarr_format | downscale_factor | subset | channel_colors |
    channel_names | num_levels | shards | shard_target_mb | shard_exclude_axes | drop_singleton | num_workers)
    and of the fusion columns
    downsample | projection | blend | channels | groups | skip_tiles | midline_axis | out_dtype | read_workers |
    temp_dir | overwrite, e.g.
      input                  | microscope | output                 | downsample | projection | blend   | num_levels
      /path/to/acquisition_1 | luxendo    | /path/to/acq1_3D.zarr  | 4 8 8      | none       | midline | 3
      /path/to/acquisition_1 | luxendo    | /path/to/acq1_MIP.zarr | 1 8 8      | max        | midline |
    Leave a cell empty for the default. Lists (channels, groups, skip_tiles, channel_names...) are space separated.
    The napari "Mosaic Fusion" widget writes such files with "Append to batch CSV".
"""

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
        choices=['luxendo', 'opera', 'viventis', 'zeiss', 'zarrv04', 'zarr', 'scape'],
        type=str,
    )


def _add_fuse_microscope(parser):
    parser.add_argument(
        '-m', '--microscope',
        required=False,
        default=None,
        choices=['luxendo', 'opera', 'zeiss'],
        type=str,
        help='Microscope of the mosaic. Leave empty to auto-detect (raw Luxendo acquisitions). '
             'Opera and Zeiss mosaics are not implemented yet.',
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
        metavar='{convert,fuse}',
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
    _add_input_output(fuse, required=False, input_help='Acquisition folder (for Luxendo: the folder containing raw/).',
                      output_type=os.path.abspath)

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
    return args
