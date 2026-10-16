# Python API

Typical Python usage of the managers and of the zarr writing helpers.

## Read, build a pyramid and export

```python
import pymif.microscope_manager as mm

# Read a source dataset
source = mm.ViventisManager("path/to/Position_1")

# Build a pyramid in memory
source.build_pyramid(num_levels=3)

# Export to OME-Zarr (default: NGFF v0.5 / zarr v3)
source.to_zarr("output.zarr")

# Re-open the written zarr dataset
z = mm.ZarrManager("output.zarr")
viewer = z.visualize(start_level=0, in_memory=False)
```

## Create an empty zarr dataset from metadata

```python
import pymif.microscope_manager as mm

z = mm.ZarrManager(
    "empty.zarr",
    mode="a",
    metadata={
        "size": [(1, 2, 16, 256, 256)],
        "chunksize": [(1, 1, 16, 128, 128)],
        "scales": [(2.0, 0.5, 0.5)],
        "units": ("micrometer", "micrometer", "micrometer"),
        "axes": "tczyx",
        "channel_names": ["GFP", "RFP"],
        "channel_colors": ["00FF00", "FF0000"],
        "time_increment": 1.0,
        "time_increment_unit": "second",
        "dtype": "uint16",
    },
)
```

## Axis-aware ZarrManager metadata

Most microscope-specific managers still normalize data to legacy `tczyx`, but `ZarrManager` and `ArrayManager` can now write and read any unique subset of the image axes `t`, `c`, `z`, `y`, and `x`. 

> [!NOTE]
> The axes string must have one label per array dimension and may not contain non-image axes such as tiles, ROIs, scenes, or wells.

For example, a two-dimensional YX intensity image can be written as:

```python
import dask.array as da
import numpy as np
import pymif.microscope_manager as mm

img = np.zeros((512, 512), dtype=np.uint16)
levels = [da.from_array(img, chunks=(256, 256))]
metadata = {
    "axes": "yx",
    "size": [(512, 512)],
    "chunksize": [(256, 256)],
    "scales": [(0.5, 0.5)],       # follows the spatial axes in axes order
    "units": ("micrometer", "micrometer"),
    "dtype": "uint16",
    "data_type": "intensity",    # or "label"
}

mm.ArrayManager(levels, metadata).to_zarr("yx_image.zarr")
```

Use `data_type="label"` for segmentation, mask, or annotation data. Label arrays must use an integer dtype. For NGFF v0.5/Zarr v3, PyMIF stores the semantic type under `attrs["ome"]["data_type"]`, writes array-level `dimension_names`, writes `multiscales[0]["type"] = "label"`, and adds `image-label` metadata. For NGFF v0.4/Zarr v2, the equivalent metadata is stored directly on the group attributes.

Legacy calls that create labels from an intensity image metadata dictionary still work:

```python
z = mm.ZarrManager("image.zarr", mode="a")
z.create_empty_group("nuclei", image_metadata, is_label=True)
```

When `image_metadata["axes"] == "tczyx"`, this legacy `is_label=True` form creates a `tzyx` label group by dropping the intensity channel axis. New callers that really want channelled label data can pass explicit label metadata instead:

```python
label_metadata = {**image_metadata, "data_type": "label"}
z.create_empty_group("classes", label_metadata, data_type="label")
```

## Update a region in an existing zarr image

```python
import numpy as np
import pymif.microscope_manager as mm

z = mm.ZarrManager("output.zarr", mode="a")
patch = np.full((1, 1, 2, 64, 64), 999, dtype=np.uint16)

z.write_image_region(
    patch,
    t=slice(0, 1),
    c=slice(0, 1),
    z=slice(10, 12),
    y=slice(100, 164),
    x=slice(100, 164),
    level=0,
)
```

## Correct the chromatic shift between channels

`ZarrManager` can measure the axial (Z) offset between the channels of a light-sheet dataset on the sample itself and write a corrected copy with the same structure as the input:

```python
z = mm.ZarrManager("input.zarr")
out = z.correct_chromatic_shift("corrected.zarr", reference_channel=0)
```

See [Chromatic shift correction](chromatic-shift.md) for how the shift is measured, the options and how to check the result.
