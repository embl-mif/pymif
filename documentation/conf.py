# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# The sources are the Markdown pages of documentation/ (also readable on GitHub);
# index.md pulls in the root README.md, so nothing is copied.
import os
import re
import sys
from importlib.metadata import version as get_version, PackageNotFoundError
sys.path.insert(0, os.path.abspath(".."))

project = 'PyMIF'
copyright = '2023, European Molecular Biology Laboratory (EMBL)'
author = 'Nicola Gritti'
try:
    release = get_version("pymif")
except PackageNotFoundError:
    # The version comes from git tags at install time; without an install it is unknown.
    release = "unknown"

version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['myst_parser', 'autoapi.extension', 'sphinx.ext.linkcode']

myst_enable_extensions = [
    "colon_fence",
]
# Anchors for headings up to level 3, so links such as page.md#sharding work.
myst_heading_anchors = 3

autoapi_dirs = [
    '../pymif',
    ]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

def linkcode_resolve(domain, info):
    
    if domain != 'py':
        return None
    if not info['module']:
        return None
    filename = info['module'].replace('.', '/')
    return "https://github.com/embl-mif/pymif/blob/main/%s.py" % filename

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '*.txt']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = []
# One copy of the logo, shipped with the package for the napari widgets.
html_logo = '../pymif/napari/resources/zarrsmith-icon.svg'

# -- GitHub alerts -----------------------------------------------------------
# The pages use GitHub alerts (> [!NOTE] ...), which MyST renders as a plain quote;
# turn them into the matching admonition, also in the included README.md.
GITHUB_ALERT = re.compile(r"^> \[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\][ \t]*\n((?:>.*(?:\n|$))*)", re.MULTILINE)


def github_alerts_to_admonitions(text):
    def admonition(match):
        body = re.sub(r"^> ?", "", match.group(2), flags=re.MULTILINE).rstrip("\n")
        return f"```{{{match.group(1).lower()}}}\n{body}\n```\n"
    return GITHUB_ALERT.sub(admonition, text)


def _convert_source(app, docname, source):
    source[0] = github_alerts_to_admonitions(source[0])


def _convert_include(app, relative_path, parent_docname, content):
    content[0] = github_alerts_to_admonitions(content[0])


def setup(app):
    app.connect("source-read", _convert_source)
    app.connect("include-read", _convert_include)
