# PyMIF — microscopy I/O, OME-Zarr conversion, and NGFF utilities

![Python](https://img.shields.io/badge/python-3.12-green)
[![License](https://img.shields.io/github/license/embl-mif/pymif.svg)](https://github.com/embl-mif/pymif/blob/main/LICENSE)
[![Tests](https://github.com/embl-mif/pymif/actions/workflows/tests.yml/badge.svg)](https://github.com/embl-mif/pymif/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/embl-mif/pymif/graph/badge.svg)](https://codecov.io/gh/embl-mif/pymif)
[![Documentation](https://github.com/embl-mif/pymif/actions/workflows/documentation.yml/badge.svg)](https://github.com/embl-mif/pymif/actions/workflows/documentation.yml)
[![Docker](https://github.com/embl-mif/pymif/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/embl-mif/pymif/actions/workflows/docker-publish.yml)
[![GitHub release](https://img.shields.io/github/v/release/embl-mif/pymif.svg)](https://github.com/embl-mif/pymif/releases)
[![GitHub stars](https://img.shields.io/github/stars/embl-mif/pymif.svg?style=social)](https://github.com/embl-mif/pymif/stargazers)

**PyMIF** is a Python package for reading microscopy datasets from multiple acquisition systems, building multiscale pyramids, writing **OME-NGFF / OME-Zarr**, and working with those datasets from Python, the command line and napari.

It is developed for users of the [Mesoscopic Imaging Facility (MIF)](https://www.embl.org/groups/mesoscopic-imaging-facility/). The rendered documentation is at **[embl-mif.github.io/pymif](https://embl-mif.github.io/pymif/)**.

> [!NOTE]
> PyMIF writes **NGFF v0.5 / Zarr v3** by default. **NGFF v0.4 / Zarr v2** datasets remain supported through `ZarrManager` and `ZarrV04Manager`.

![Demo](documentation/demo.gif)

*Demonstration of PyMIF usage. Data: near newborn mouse embryo (~1.5 cm long). Fluorescence signal: methylene blue + autofluorescence. Sample processed and imaged by Montserrat Coll at the Mesoscopic Imaging Facility. Video speed: 2.5× real speed.*

---

## What PyMIF does

- **Microscope managers** read Luxendo, Opera, SCAPE, Viventis, Zeiss and OME-Zarr datasets into one lazy (Dask) API with a shared metadata schema. → [Readers and conventions](documentation/readers.md)
- **OME-Zarr writing**: full export, empty datasets and groups, image and label region updates, any subset of the `tczyx` axes. → [Python API](documentation/python-api.md)
- **Pyramids and subsetting**: multiscale generation, cropping, channel reordering, metadata updates. → [Python API](documentation/python-api.md)
- **Mosaics**: raw tiles shown at their stage position and fused into one OME-Zarr image (3D or projection). → [Mosaics](documentation/mosaic.md)
- **Command line**: one-off and CSV-batch conversion and mosaic fusion. → [Command line](documentation/cli.md)
- **Napari widgets**: conversion with ROI selection, mosaic fusion, overview generation. → [Napari plugin](documentation/napari-plugin.md)
- **Output tuning**: sharding, singleton axes, parallel writing, progress bars. → [Advanced writing options](documentation/advanced.md)

## Installation

```console
conda create -n pymif python=3.12
conda activate pymif
git clone https://github.com/embl-mif/pymif.git
cd pymif
pip install .            # or: pip install -e .[napari] for development with the napari widgets
```

Versioning, releases and development installs: [Installation and versioning](documentation/installation.md).

## Quick start

```python
import pymif.microscope_manager as mm

source = mm.ViventisManager("path/to/Position_1")   # any microscope manager
source.build_pyramid(num_levels=3)
source.to_zarr("output.zarr")                       # NGFF v0.5 / Zarr v3

z = mm.ZarrManager("output.zarr")
z.visualize()                                       # open in napari
```

```console
pymif convert -i INPUT_PATH -m MICROSCOPE -z OUTPUT_ZARR
pymif fuse -i ACQUISITION_FOLDER -z OUTPUT_ZARR -rds 4 8 8   # fuse raw tiles
```

## Documentation

| Page | Contents |
|---|---|
| [Installation and versioning](documentation/installation.md) | environments, extras, release labels |
| [Readers and conventions](documentation/readers.md) | supported microscopes, shared reader behaviour, logging |
| [Python API](documentation/python-api.md) | export, empty datasets, axis-aware metadata, labels, region writes |
| [Mosaics](documentation/mosaic.md) | raw tiles, placement check, fusion options |
| [Command line](documentation/cli.md) | `pymif convert`, `pymif fuse` (one dataset or a CSV batch) |
| [Napari plugin](documentation/napari-plugin.md) | converter, mosaic fusion and overview widgets |
| [Advanced writing options](documentation/advanced.md) | sharding, singleton axes, parallel writing, progress bar |
| [Contributing](documentation/contributing.md) | documentation layout, adding a reader, `pymif.utils` |

The [API reference](https://embl-mif.github.io/pymif/autoapi/pymif/index.html) is generated from the docstrings.
