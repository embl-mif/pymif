# PyMIF notebooks

This folder contains a user-friendly notebook collection for the current PyMIF API.

Recommended order:

1. `00_installation_and_mental_model.ipynb`
2. `01_read_microscope_datasets_templates.ipynb`
3. `02_array_manager_from_numpy_dask.ipynb`
4. `03_create_ome_zarr_with_metadata.ipynb`
5. `04_read_zarr_inspect_groups_labels.ipynb`
6. `05_add_groups_and_write_image_regions.ipynb`
7. `06_labels_append_inside_existing_zarr.ipynb`
8. `07_create_new_zarr_dataset_with_labels.ipynb`
9. `08_metadata_subset_pyramid_and_channels.ipynb`
10. `09_legacy_options_v04_and_troubleshooting.ipynb`
11. `10_mosaic_fusion.ipynb`: fuse the raw tiles of a mosaic into a 3D volume or a 2D MIP

The notebooks write small synthetic examples to `pymif_tutorial_output/` in the working directory where you run them.

## Main beginner rules

- Prefer `ZarrManager` for reading both modern and legacy OME-Zarr stores.
- Use `ZarrV04Manager` only when you explicitly want the v0.4 compatibility wrapper.
- Always set `metadata["axes"]` explicitly when creating data from arrays.
- Use `data_type="intensity"` for raw or processed image data.
- Use `data_type="label"` and an integer dtype for labels.
- For segmentation labels, prefer explicit no-channel axes such as `"tzyx"`.
- Use `mode="r+"` when adding groups or labels to an existing Zarr.
- PyMIF reports progress through the standard `logging` module (silent by default). Run `logging.basicConfig(format="%(message)s")` and `logging.getLogger("pymif").setLevel(logging.INFO)` to see it; the notebooks that need it do this in their setup cell.
- `to_zarr()` shows a progress bar while writing; pass `progress=False` to silence it.
- Vendor readers are context managers (`with mm.LuxendoManager(path) as m: ...`) and always report channel colors as `RRGGBB` and units as NGFF names such as `micrometer`.

## Optional dependencies

Most notebooks require PyMIF plus its normal Dask/Zarr/image dependencies. Visualization cells require napari and a GUI-capable Python session.
