# Contributing and extending PyMIF

## Package layout

| Package | What it contains |
|---|---|
| `pymif.microscope_manager` | **microscope managers**: one reader class per acquisition system, each exposing *one image* (`MicroscopeManager` subclasses), plus `ZarrManager` for reading, editing and correcting OME-Zarr stores |
| `pymif.mosaic_manager` | **mosaic managers**: readers for acquisitions saved as separate positioned *tiles* (`MosaicManager` subclasses), the `Tile` model and the fusion engine (`fuse_tiles`, `plan_fusion`) |
| `pymif.utils` | helpers shared by both: metadata, axes, units, colours, pyramids, OME-Zarr writing, chromatic shift correction (see [Layout of `pymif.utils`](#layout-of-pymifutils)) |
| `pymif.cli` | the `pymif convert` and `pymif fuse` commands |
| `pymif.napari` | the Converter and Mosaic Fusion widgets |

## Two kinds of readers

PyMIF has two extension points, and a new acquisition system needs one or both:

- A **microscope manager** reads a dataset that already is **one image** (one array per pyramid level, possibly with several timepoints and channels). It is what `pymif convert` and the Converter widget use.
- A **mosaic manager** reads an acquisition saved as **many tiles** at different stage positions. It only becomes an image once the tiles are fused, which `pymif fuse` and the Mosaic Fusion widget do. The fused result is an ordinary OME-Zarr image, read back with `ZarrManager`.

Some systems need both: Luxendo data fused by the vendor software is read by `LuxendoManager`, the raw tiles by `LuxendoMosaicManager`.

## Adding a microscope manager

### Implement the reader

Subclass `MicroscopeManager` (exported from `pymif.microscope_manager`) and implement two hooks. The base class provides `read()`, which calls them and stores the results in `self.metadata` and `self.data`:

```python
from pymif.microscope_manager import MicroscopeManager
from pymif.utils.metadata import ChannelInfo, build_metadata

class MyManager(MicroscopeManager):
    DEFAULT_CHUNKS = (1, 1, 8, 2048, 2048)  # used when chunks=None

    def __init__(self, path, chunks=None):
        super().__init__(path, chunks)  # sets self.path (Path) and self.chunks
        self.read()

    def _parse_metadata(self) -> dict:
        # Parse the vendor metadata and return the normalized schema.
        return build_metadata(
            size=[(t, c, z, y, x)],           # one shape per pyramid level
            scales=[(sz, sy, sx)],            # one voxel size per pyramid level (z, y, x)
            units=("micrometer",) * 3,
            channels=[ChannelInfo("DAPI", "0000FF"), ChannelInfo("GFP", "#00ff00")],
            dtype="uint16",
            time_increment=60.0,
            time_increment_unit="s",
        )

    def _build_dask_array(self) -> list:
        # Return one lazy dask array per pyramid level, in TCZYX order.
        ...
```

`build_metadata()` normalizes the units and colours for you, and `pymif.utils` has helpers that the built-in readers share: `parse_channel_color` (vendor colour formats to `RRGGBB`), `normalize_unit` / `to_micrometers`, `to_tczyx` (reorder or add axes), `parse_ome_xml` (OME-XML `<Pixels>` and channels) and `scale_for_level`.

The metadata must follow the PyMIF schema used across the repository:

```python
{
  "size": [... per pyramid level ...],
  "chunksize": [... per pyramid level ...],
  "scales": [... per pyramid level ...],
  "units": (...),
  "axes": "tczyx",
  "channel_names": [...],
  "channel_colors": [...],
  "time_increment": ...,
  "time_increment_unit": ...,
  "dtype": ...,
}
```

If a manager needs the data before it can describe it (like `ArrayManager`) or reads several datasets at once (like `ZarrManager`), it can override `read()` instead; it should still return `None` and fill `self.data` and `self.metadata`. File handles appended to `self._open_files` are closed by `close()` and when the manager is used as a context manager.

Once that contract is respected, the new manager gets the common PyMIF tooling: `build_pyramid()`, `to_zarr()`, `visualize()`, `reorder_channels()`, `update_metadata()` and `subset_dataset()`.

### Register it

1. Export the class from `pymif/microscope_manager/__init__.py`.
2. Command line: add a key to `_manager_name_map()` in `pymif/cli/pymif.py` and to the `--microscope` choices in `_add_convert_microscope()` (`pymif/cli/__arguments.py`). If the format can be recognised from the path, add a rule to `_resolve_zarr_manager()` so `-m` can be left out.
3. Napari: add the key to `dataset_reader()` and to the `file_format` choices of the Converter widget, both in `pymif/napari/_convert_widget.py`.
4. Documentation: list it in [Readers and conventions](readers.md).

### Test it

A reader can be tested without real data by writing a small synthetic dataset in the vendor format to `tmp_path`, or by subclassing with hooks that return in-memory arrays (see `tests/test_manager_refactor.py`). Check at least the shape, axes, scales, channel names and a round trip through `to_zarr()` and `ZarrManager`.

## Adding a mosaic manager

### Implement the reader

Subclass `MosaicManager` (exported from `pymif.mosaic_manager`) and implement a single hook, `_scan()`, which returns one `Tile` per acquired tile. The base constructor calls it, stops with an error when it returns no tile, and refuses mosaics with more than one timepoint (not supported yet).

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
                group=None,                        # e.g. illumination side, for blend="sigmoid"/"midline"
            )
            for path, (z0, y0, x0) in ...
        ]
```

A `Tile` describes one stack and where it sits in the sample:

| Field | Meaning |
|---|---|
| `data` | lazy `(z, y, x)` array: anything with `shape`, `dtype` and numpy-style slicing (`h5py.Dataset`, dask, numpy, `H5Array`). Only read when a fusion needs it |
| `origin`, `voxel_size` | `(z, y, x)` position of the first voxel and voxel spacing, in `units`. Placement is translation only, and all tiles of a mosaic share units and voxel size |
| `channel` | dye name; tiles with the same name are fused into one output channel |
| `group` | optional label of tiles that are fused among themselves first, such as the `L`/`R` illumination side; `blend="sigmoid"` and `blend="midline"` then join the groups across their overlap |
| `timepoint` | timepoint index (`0` for a single timepoint) |
| `attrs` | free-form reader metadata (stack name, grid position, ...) |

Things to get right:

- **Keep loading light.** `_scan()` should read metadata only, never pixel data. Pass `shape` and `dtype` to `H5Array` when the metadata knows them, so no file has to be opened: this matters on network drives (`LuxendoMosaicManager` reads its JSON files in parallel and a single `.h5` header).
- **Make tile data picklable for parallel fusion.** `fuse_tiles(workers > 1)` reads tiles in worker processes when the data allows it. `H5Array` opens the file on every read, holds no handle and sets `picklable = True`; use it, or the same pattern, for file-backed tiles. Other data falls back to threads.
- **Close what you open.** Append open file handles to `self._open_files`; `close()` and the context manager close them.
- **Name tiles stably.** `tile_id()` (`<channel>/<stack>`) is used by `skip`, the CSV batch `tiles` column and the napari tile map. Set `attrs["stack"]` to the vendor's stack name; without it the id falls back to the tile's index, which changes with the scan order.

`visualize()`, `fuse()`, `plan_fusion()`, `pymif fuse` and the napari widget then work unchanged.

### Register it

1. Export the class from `pymif/mosaic_manager/__init__.py`.
2. Command line: add a key to `_mosaic_manager_name_map()` in `pymif/cli/pymif.py` and to the `--microscope` choices in `_add_fuse_microscope()` (`pymif/cli/__arguments.py`). Add an auto-detection rule to `_resolve_mosaic_manager()` if the folder layout is recognisable.
3. Napari: add it to `MOSAIC_READERS` in `pymif/napari/_mosaic_widget.py`.
4. Documentation: add it to the reader table in [Mosaics](mosaic.md).

`OperaMosaicManager` and `ZeissMosaicManager` are registered placeholders whose `_scan()` raises `NotImplementedError`; implementing `_scan()` is all that is left for them.

### Test it

Fusion can be tested with in-memory tiles: a `MosaicManager` subclass whose `_scan()` returns `Tile`s holding small numpy arrays at known origins (see `tests/test_mosaic_fusion.py`). For the reader itself, write a few tiny tiles in the vendor format to `tmp_path` and check positions, channels, groups and `tile_id()`s (see `tests/test_luxendo_mosaic.py`).

## Layout of `pymif.utils`

Shared helpers live in one place each, so readers, writers and the napari viewer behave the same way:

| Module | What it owns |
| --- | --- |
| `colors` | `parse_channel_color` / `parse_color` / `ome_color_to_hex` / `hex_to_rgb` and the default channel palette |
| `units` | `normalize_unit`, `to_micrometers` |
| `axes` | axis validation, `to_tczyx`, index selections |
| `chunks` | `shape_tuple`, `get_chunks`, `rechunk_to_shape` |
| `downsampling` | factor math and `downsample_nearest` (NumPy or Dask) |
| `metadata`, `ome` | `build_metadata`, `ChannelInfo`, `scale_for_level`, `parse_ome_xml` |
| `dataset_ops` | validation and in-place edits shared by `MicroscopeManager` and `ZarrManager` (`subset_levels`, `reorder_channels`, `apply_metadata_updates`) |
| `ngff` | `ZarrWriteConfig`, array creation (`_array_create_kwargs`), sharding, multiscales metadata (`_build_multiscales`) and the pyramid writers |
| `chromatic_shift` | Z chromatic shift estimation (`estimate_chromatic_shift`) and the writer of corrected stores that copies the input's array layout (`write_chromatic_shift_corrected`); see [Chromatic shift correction](chromatic-shift.md) |

## Documentation

PyMIF uses **Sphinx + MyST + AutoAPI**, and the `documentation/` folder is the Sphinx source:

- the root `README.md` is the landing page. `documentation/index.md` pulls it in with a MyST `include`, so it is never copied;
- every other page is a Markdown file in `documentation/`, readable on GitHub and rendered on the documentation site;
- API pages are generated from the Python docstrings, so documenting classes, methods and helpers in the source code is the best way to improve the API reference.

To add a page, create `documentation/<page>.md`, add it to the `toctree` in `documentation/index.md` and link it from the README table. Build the site locally with:

```console
pip install -e .[docs]
sphinx-build documentation _build
```

The API entry points to document and keep stable are:

- `MicroscopeManager`, the vendor reader classes and `ZarrManager` in `pymif.microscope_manager`
- `MosaicManager`, `Tile`, `H5Array`, `fuse_tiles` and `plan_fusion` in `pymif.mosaic_manager`
- the zarr-writing helpers in `pymif.utils`
- the CLI entry points in `pymif.cli`
- the napari widgets in `pymif.napari`
