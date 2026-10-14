# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# this is a trick to make sphinx find the modules in the parent directory
import os
import sys
from importlib.metadata import version as get_version, PackageNotFoundError
sys.path.insert(0, os.path.abspath("../../"))

project = 'PyMIF'
copyright = '2025, Nicola Gritti'
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
    # "admonition",
]

autoapi_dirs = [
    '../../pymif/pymif',
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
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
