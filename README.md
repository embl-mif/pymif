<h1>
  <img src="https://raw.githubusercontent.com/embl-mif/zarrsmith/main/zarrsmith/napari/resources/zarrsmith-icon.svg" alt="ZarrSmith logo" width="120" align="left">
  ZarrSmith — microscopy I/O, OME-Zarr forging, and NGFF utilities
  <br clear="left">
</h1>
<!-- The logo and the links to the documentation are absolute URLs so that they also work on PyPI.
documentation/index.md repeats this header in Markdown (Sphinx needs a Markdown title and does not
rewrite HTML paths) and includes this README from here. -->

![Python](https://img.shields.io/badge/python-3.12-green)
[![License](https://img.shields.io/github/license/embl-mif/zarrsmith.svg)](https://github.com/embl-mif/zarrsmith/blob/main/LICENSE)
[![Tests](https://github.com/embl-mif/zarrsmith/actions/workflows/tests.yml/badge.svg)](https://github.com/embl-mif/zarrsmith/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/embl-mif/zarrsmith/graph/badge.svg)](https://codecov.io/gh/embl-mif/zarrsmith)
[![Documentation](https://github.com/embl-mif/zarrsmith/actions/workflows/documentation.yml/badge.svg)](https://github.com/embl-mif/zarrsmith/actions/workflows/documentation.yml)
[![Docker](https://github.com/embl-mif/zarrsmith/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/embl-mif/zarrsmith/actions/workflows/docker-publish.yml)
[![GitHub release](https://img.shields.io/github/v/release/embl-mif/zarrsmith.svg)](https://github.com/embl-mif/zarrsmith/releases)
[![GitHub stars](https://img.shields.io/github/stars/embl-mif/zarrsmith.svg?style=social)](https://github.com/embl-mif/zarrsmith/stargazers)

**ZarrSmith** is a Python package for reading microscopy datasets from multiple acquisition systems, building multiscale pyramids, writing **OME-NGFF / OME-Zarr**, and working with those datasets from Python, the command line and napari.

The rendered documentation is at **[embl-mif.github.io/zarrsmith](https://embl-mif.github.io/zarrsmith/)**.

> [!NOTE]
> ZarrSmith was called **PyMIF** up to version 0.5.5. From 0.6.0, import `zarrsmith` and run the `zarrsmith` command instead of `pymif`. The Docker images of PyMIF stay available at `ghcr.io/embl-mif/pymif:<version>`; newer images are at `ghcr.io/embl-mif/zarrsmith`.

---

## What ZarrSmith does

- **Microscope managers** read Luxendo, Opera, SCAPE, Viventis, Zeiss, Evident (Olympus VSI) and OME-Zarr datasets into one lazy (Dask) API with a shared metadata schema. → [Readers and conventions](https://embl-mif.github.io/zarrsmith/readers.html)
- **OME-Zarr writing**: full export, empty datasets and groups, image and label region updates, any subset of the `tczyx` axes. → [Python API](https://embl-mif.github.io/zarrsmith/python-api.html)
- **Pyramids and subsetting**: multiscale generation, cropping, channel reordering, metadata updates. → [Python API](https://embl-mif.github.io/zarrsmith/python-api.html)
- **Mosaics**: raw tiles shown at their stage position and fused into one OME-Zarr image (3D or projection). → [Mosaics](https://embl-mif.github.io/zarrsmith/mosaic.html)
- **Chromatic shift correction**: the Z offset between channels measured on the sample itself, and a corrected copy with the same layout as the input. → [Chromatic shift correction](https://embl-mif.github.io/zarrsmith/chromatic-shift.html)
- **Command line**: one-off and CSV-batch conversion, mosaic fusion and chromatic shift correction. → [Command line](https://embl-mif.github.io/zarrsmith/cli.html)
- **Napari widgets**: conversion with ROI selection, mosaic fusion, exploration of segmented objects (features plot linked with the viewer). → [Napari plugin](https://embl-mif.github.io/zarrsmith/napari-plugin.html)
- **Output tuning**: sharding, singleton axes, parallel writing, progress bars. → [Advanced writing options](https://embl-mif.github.io/zarrsmith/advanced.html)

## Installation

```console
conda create -n zarrsmith python=3.12
conda activate zarrsmith
git clone https://github.com/embl-mif/zarrsmith.git
cd zarrsmith
pip install .            # or: pip install -e .[napari] for development with the napari widgets
```

Versioning, releases and development installs: [Installation and versioning](https://embl-mif.github.io/zarrsmith/installation.html).

## Quick start

```python
import zarrsmith.microscope_manager as mm

source = mm.ViventisManager("path/to/Position_1")   # any microscope manager
source.build_pyramid(num_levels=3)
source.to_zarr("output.zarr")                       # NGFF v0.5 / Zarr v3

z = mm.ZarrManager("output.zarr")
z.visualize()                                       # open in napari
```

```console
zarrsmith convert -i INPUT_PATH -m MICROSCOPE -z OUTPUT_ZARR
zarrsmith fuse -i ACQUISITION_FOLDER -z OUTPUT_ZARR -rds 4 8 8   # fuse raw tiles
zarrsmith chromatic-shift -i INPUT_ZARR -z OUTPUT_ZARR -r shift.json   # align channels in Z
```

## Documentation

**Getting started**

| Page | Contents |
|---|---|
| [Installation and versioning](https://embl-mif.github.io/zarrsmith/installation.html) | environments, extras, release labels |
| [Readers and conventions](https://embl-mif.github.io/zarrsmith/readers.html) | supported microscopes, shared reader behaviour, logging |

**Working with data**

| Page | Contents |
|---|---|
| [Python API](https://embl-mif.github.io/zarrsmith/python-api.html) | export, empty datasets, axis-aware metadata, labels, region writes |
| [Mosaics](https://embl-mif.github.io/zarrsmith/mosaic.html) | raw tiles, placement check, fusion options |
| [Chromatic shift correction](https://embl-mif.github.io/zarrsmith/chromatic-shift.html) | measuring the Z offset between channels, the corrected dataset, checking the result, `zarrsmith chromatic-shift` |
| [Advanced writing options](https://embl-mif.github.io/zarrsmith/advanced.html) | sharding, singleton axes, parallel writing, progress bar |

**Interfaces**

| Page | Contents |
|---|---|
| [Command line](https://embl-mif.github.io/zarrsmith/cli.html) | `zarrsmith convert`, `zarrsmith fuse`, `zarrsmith chromatic-shift` (one dataset or a CSV batch) |
| [Napari plugin](https://embl-mif.github.io/zarrsmith/napari-plugin.html) | converter, mosaic fusion and objects explorer widgets |

**Reference**

| Page | Contents |
|---|---|
| [API reference](https://embl-mif.github.io/zarrsmith/autoapi/zarrsmith/index.html) | generated from the docstrings |
| [Contributing](https://embl-mif.github.io/zarrsmith/contributing.html) | package layout, adding a microscope manager or a mosaic manager, `zarrsmith.utils`, documentation |

## Help

For questions or bug reports, open an
[issue](https://github.com/embl-mif/zarrsmith/issues) on GitHub, or contact the current maintainers.

> [!IMPORTANT]
> If ZarrSmith contributes to your work, please acknowledge ZarrSmith and the EMBL Mesoscopic Imaging Facility in your publications.
