# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
import os
import sys

sys.path.insert(0, os.path.abspath('../src'))

project = 'LayeredCompModel'
copyright = '2026, John Kossa'
author = 'John Kossa'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'autoapi.extension',
    'myst_parser',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autodoc',
]

templates_path = ['_templates']
exclude_patterns = []

autoapi_type = 'python'
autoapi_dirs = ['../src']
autoapi_options = ['members', 'show-inheritance', 'special-members', 'undoc-members']

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "html_admonition",
    "html_image",
    "colon_fence",
    "smartquotes",
    "replacements",
    "strikethrough",
    "substitution",
]

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']