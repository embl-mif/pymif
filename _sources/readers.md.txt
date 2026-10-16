# Readers and conventions

The microscope managers in `pymif.microscope_manager` read a dataset and normalize it to one shared API.

The main reader classes currently exposed by `pymif.microscope_manager` are:

- `ArrayManager` — wrap an in-memory NumPy or Dask array using PyMIF metadata conventions.
- `LuxendoManager` — Luxendo XML + HDF5 datasets.
- `OperaManager` — Opera Phenix / Opera PE OME-TIFF style datasets.
- `ScapeManager` — Leica SCAPE OME-TIFF + XLIF datasets.
- `ViventisManager` — Viventis LS1 datasets.
- `ZeissManager` — Zeiss CZI datasets.
- `ZarrManager` — NGFF v0.4/v0.5 OME-Zarr datasets.
- `ZarrV04Manager` — compatibility reader for older v0.4-style datasets.

All file-based readers share the same conventions:

- The dataset location is the first argument (`path`; for `ScapeManager` the OME-TIFF file) and `chunks` is an optional TCZYX tuple. When omitted, `chunks` defaults to the manager's `DEFAULT_CHUNKS` (`(1, 1, 8, 4096, 4096)` for most readers, `(1, 1, 8, 1024, 1024)` for `ScapeManager`, the native chunking for `ZeissManager`).
- The dataset is read in the constructor, so `.data` (one lazy dask array per pyramid level) and `.metadata` are available immediately. `read()` returns `None`.
- Managers can be used as context managers, which closes any open file handle (HDF5 files for `LuxendoManager`) on exit:

  ```python
  with mm.LuxendoManager("path/to/luxendo_folder") as lux:
      lux.to_zarr("output.zarr")
  ```
- The metadata uses one spelling everywhere: channel colors are `RRGGBB` hex strings (no `#`), spatial units are NGFF names such as `"micrometer"`, and time units are e.g. `"second"`.
- Progress messages (opened scenes, subsets, metadata updates, the Zarr tree printed when opening a store) go through the standard `logging` module under the `"pymif"` logger and are silent by default. Show them with:

  ```python
  import logging
  logging.basicConfig(format="%(message)s")
  logging.getLogger("pymif").setLevel(logging.INFO)
  ```

To support a new acquisition system, see [Adding a microscope manager](contributing.md#adding-a-microscope-manager); for acquisitions saved as separate tiles, see [Mosaics](mosaic.md).
