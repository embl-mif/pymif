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

## Correct the axial chromatic shift between channels

Light-sheet channels are often offset along Z, and the offset changes whenever the system is realigned. `ZarrManager` estimates it from the sample itself: the sample's lower and upper surfaces are visible in every channel (through background and autofluorescence), so each channel's Z-derivative is correlated with the reference channel around those surfaces, first on a coarse level and then on small crops of a finer level.

```python
import json
from pathlib import Path
import pymif.microscope_manager as mm

z = mm.ZarrManager("input.zarr")

# Shifts are corrections in the Z unit (usually micrometers), one per channel.
report = z.estimate_chromatic_shift(reference_channel=0)   # searches up to 10% of the stack depth
print(report["shifts"], report["spread"], report["surface_shifts"], report["warnings"])
Path("chromatic_shift.json").write_text(json.dumps(report, indent=2))

# Write a corrected copy. Chunks, shards, codecs, pixel sizes and channel
# metadata are copied from the input; only the Z size changes.
out = z.apply_chromatic_shift("corrected.zarr", report, extent="union", order=1)

# Or both steps in one call:
out = z.correct_chromatic_shift("corrected.zarr", reference_channel=0)
```

- `extent="union"` grows Z so no channel loses data, `"same"` keeps the input size and `"intersection"` crops to the planes covered by every channel. Padding is rounded to whole planes of the coarsest level so all pyramid levels stay aligned; where input plane 0 ended up is stored in the output root attribute `pymif_chromatic_shift` (`z_offset`), together with the applied shifts.
- `order=1` interpolates linearly; `order=0` rounds the shifts to whole planes and leaves voxel values untouched.
- Image subgroups with the same channels are shifted like the raw image; labels are padded or cropped to the new extent, following the reference channel.
- The estimate needs Z-facing sample surfaces inside the stack. If the sample fills the whole stack in Z it fails with a clear error; shifts measured on another dataset can then be passed to `apply_chromatic_shift` directly.
