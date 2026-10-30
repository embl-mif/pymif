# Installation and versioning

A clean conda environment is recommended:

```console
conda create -n zarrsmith python=3.12
conda activate zarrsmith
```

Then install the latest release from PyPI (add `[napari]` for the napari widgets):

```console
pip install zarrsmith
pip install "zarrsmith[napari]"
```

Or install from the repository:

```console
git clone https://github.com/embl-mif/zarrsmith.git
cd zarrsmith
pip install .
```

For development work:

```console
pip install -e .
```

To use the napari widgets as well:

```console
pip install -e .[napari]
```

The optional extras are `napari` (the widgets), `notebooks` (Jupyter, to run the example notebooks), `dev` (tests and packaging), `docs` (the documentation build) and `all` (all of them):

```console
pip install -e .[napari,notebooks]
```

## Versioning

ZarrSmith uses semantic versions (`MAJOR.MINOR.PATCH`) derived from git tags with [setuptools-scm](https://setuptools-scm.readthedocs.io/). The version is never edited by hand. Releases are **opt-in**: merging a PR into `main` does nothing unless the PR carries a release label, so you can merge many feature PRs and release once, from the PR that should trigger it:

| PR label | Result |
|---|---|
| `release:major` | new `vX.0.0` tag, GitHub Release, PyPI release and Docker image |
| `release:minor` | new `vX.Y.0` tag, GitHub Release, PyPI release and Docker image |
| `release:patch` | new `vX.Y.Z` tag, GitHub Release, PyPI release and Docker image |
| none | no release |

The label must be on the PR when it is merged (create the three labels once in the repository's Labels page). The generated release notes cover everything merged since the previous release.

Direct pushes to `main` do not create a release; a Docker image can also be built by hand by pushing a `v*` tag or running the Docker workflow manually. Installs from untagged commits get a dev version such as `0.3.2.dev3+g1a2b3c4`. Check the installed version with `zarrsmith --version` or `zarrsmith.__version__`.
