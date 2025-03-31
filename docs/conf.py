import sphinx_rtd_theme
import sys
import os

print(sys.executable)

# to be able to import hostphot
sys.path.insert(0, os.path.abspath("../src"))

with open("../src/hostphot/_version.py") as version_file:
    for line in version_file:
        if "__version__" in line:
            __version__ = line.split()[-1].replace('"', "")

# -- Project information -----------------------------------------------------

project = "HostPhot"
copyright = "2022, Tomás E. Müller Bravo"
author = "Tomás E. Müller Bravo"

# The short X.Y version.
version = __version__
# The full version, including alpha/beta/rc tags
release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "nbsphinx",
    # "sphinx.ext.pngmath",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_logo = "hostphot_logo.png"
html_theme_options = {
    "logo_only": False,
#    "display_version": True,
}
#html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

html_context = dict(
    display_github=True,
    github_user="temuller",
    github_repo="hostphot",
    github_version="master",
    conf_py_path="/docs/",
)

# Name of the file to be used as the master document.
master_doc = "index"
