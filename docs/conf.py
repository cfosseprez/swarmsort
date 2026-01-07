# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the package root to the Python path
sys.path.insert(0, os.path.abspath('../src'))  # adjust if your package lives elsewhere

# -- Project information -----------------------------------------------------
project = 'SwarmSort'
copyright = '2025, Charles Claude D Fosseprez'
author = 'Charles Claude D Fosseprez'
release = '0.1'
version = '0.1'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',          # pull docstrings from code
    'sphinx.ext.autosummary',      # generate summary tables
    'sphinx.ext.viewcode',         # add links to source
    'sphinx.ext.napoleon',         # support Google/NumPy docstrings
    'sphinx.ext.intersphinx',      # link to other projects
    'sphinx.ext.mathjax',          # math rendering
    'sphinx_copybutton',           # copy button for code blocks
    'sphinx_autodoc_typehints',    # show type hints
    'myst_parser',                 # markdown support
]

templates_path = ['_templates']
exclude_patterns = []

# -- HTML output -------------------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = '_static/logo.jpg'  # optional, adjust if you have a logo
html_favicon = '_static/favicon.ico'

html_theme_options = {
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 3,
    'includehidden': True,
    'titles_only': False
}
html_show_sourcelink = True
html_css_files = []

# -- Napoleon configuration --------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False

# -- Autodoc configuration ---------------------------------------------------
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

# -- Intersphinx mapping -----------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

# -- MyST parser settings ----------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
]

# -- Suppress warnings -------------------------------------------------------
suppress_warnings = ['image.not_readable']

# -- Autosummary configuration -----------------------------------------------
autosummary_generate = True

# -- Link check --------------------------------------------------------------
linkcheck_ignore = [
    r'http://localhost:\d+/',
]

# -- Master document ---------------------------------------------------------
master_doc = 'index'
htmlhelp_basename = 'SwarmSortDoc'
