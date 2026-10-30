<!-- The header of README.md, in Markdown. The logo is the html_logo of conf.py, which Sphinx copies to _static/. -->
<img src="_static/zarrsmith-icon.svg" alt="zarrsmith logo" width="120" align="left">

# ZarrSmith — microscopy I/O, OME-Zarr forging, and NGFF utilities

<br clear="left">

```{include} ../README.md
:start-after: rewrite HTML paths) and includes this README from here. -->
:relative-docs: documentation/
:relative-images:
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Getting started

installation
readers
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Working with data

python-api
mosaic
chromatic-shift
advanced
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Interfaces

cli
napari-plugin
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Reference

api_reference
contributing
```
