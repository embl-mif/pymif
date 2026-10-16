# Contributing and extending PyMIF

## Documentation

PyMIF uses **Sphinx + MyST + AutoAPI**, and the `documentation/` folder is the Sphinx source:

- the root `README.md` is the landing page. `documentation/index.md` pulls it in with a MyST `include`, so it is never copied;
- every other page is a Markdown file in `documentation/`, readable on GitHub and rendered on the documentation site;
- API pages are generated from the Python docstrings, so documenting classes, methods and helpers in the source code is the best way to improve the API reference.

To add a page, create `documentation/<page>.md`, add it to the `toctree` in `documentation/index.md` and link it from the README. Build the site locally with:

```console
pip install -e .[docs]
sphinx-build documentation _build
```

The most important API entry points to document and keep stable are:

- `MicroscopeManager`
- vendor reader classes in `pymif.microscope_manager`
- `ZarrManager`
- zarr-writing helpers in `pymif.utils`
- CLI entry points in `pymif.cli`
- mosaic readers and `fuse_tiles` in `pymif.mosaic_manager`
- napari widgets in `pymif.napari`

## Adding a microscope reader

New microscope support is typically added by subclassing `MicroscopeManager` (exported from `pymif.microscope_manager`) and implementing two hooks. The base class provides `read()`, which calls them and stores the results in `self.metadata` and `self.data`:

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

`build_metadata()` normalizes the units and colors for you, and the `pymif.utils` package has helpers that the built-in readers share: `parse_channel_color` (vendor color formats to `RRGGBB`), `normalize_unit` / `to_micrometers`, `to_tczyx` (reorder or add axes), `parse_ome_xml` (OME-XML `<Pixels>` and channels) and `scale_for_level`.

The metadata must follow the PyMIF schema used across the repository, including:

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

If a manager needs the data before it can describe it (like `ArrayManager`) or reads several datasets at once (like `ZarrManager`), it can override `read()` instead; it should still return `None` and fill `self.data` and `self.metadata`.

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
| `chromatic_shift` | axial chromatic shift estimation (`estimate_chromatic_shift`) and the structure-preserving corrected-store writer (`write_chromatic_shift_corrected`) |
| `ngff` | `ZarrWriteConfig`, array creation (`_array_create_kwargs`), sharding, multiscales metadata (`_build_multiscales`) and the pyramid writers |

Once that contract is respected, the new manager automatically benefits from the common PyMIF tooling such as `build_pyramid()`, `to_zarr()`, `visualize()`, `reorder_channels()`, `update_metadata()`, and `subset_dataset()`.
