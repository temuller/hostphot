.. HostPhot documentation master file, created by
   sphinx-quickstart on Mon Jul  1 13:44:03 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to HostPhot's documentation!
====================================

Global and local photometry of galaxies hosting supernovae or other transients.

HostPhot is being actively developed in `a public repository on GitHub
<https://github.com/temuller/hostphot>`_ so if you have any trouble, `open an issue
<https://github.com/temuller/hostphot/issues>`_ there.

.. image:: https://img.shields.io/badge/GitHub-temuller%2Fpiscola-blue.svg?style=flat
    :target: https://github.com/temuller/hostphot
.. image:: https://readthedocs.org/projects/piscola/badge/?version=latest&style=flat
    :target: https://hostphot.readthedocs.io/en/latest/?badge=latest
.. image:: http://img.shields.io/badge/license-MIT-blue.svg?style=flat
    :target: https://github.com/temuller/hostphot/blob/master/LICENSE
.. image:: https://github.com/temuller/hostphot/actions/workflows/main.yml/badge.svg
    :target: https://github.com/temuller/hostphot/actions/workflows/main.yml
.. image:: https://img.shields.io/badge/Python-3.8%2B-blue
    :target: Python Version
.. image:: https://img.shields.io/pypi/v/hostphot?label=PyPI&logo=pypi&logoColor=white
    :target: https://pypi.org/project/hostphot/
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.6469981.svg
    :target: https://doi.org/10.5281/zenodo.6469981


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   setup/installation.rst

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/work_dir.rst
   examples/cutouts_example.rst
   examples/preprocessing.rst
   examples/photometry_example.rst
   examples/interactive_aperture.rst

.. toctree::
   :maxdepth: 2
   :caption: API User Guide

   api/hostphot.rst
   api/cutouts.rst
   api/objects_detect.rst
   api/rgb_images.rst
   api/image_cleaning.rst
   api/image_masking.rst
   api/coadd.rst
   api/dust.rst
   api/local_photometry.rst
   api/global_photometry.rst
   api/interactive_aperture.rst
   api/utils.rst


.. toctree::
   :maxdepth: 1
   :caption: About the Code

   about/details.rst


Citing HostPhot
---------------

If you make use of HostPhot, please cite:

.. code-block:: tex

	@software{hostphot,
	author       = {Tomás E. Müller Bravo and
		          Lluís Galbany},
	title        = {temuller/hostphot: HostPhot v2},
	month        = may,
	year         = 2022,
	publisher    = {Zenodo},
	version      = {v2.0.0},
	doi          = {10.5281/zenodo.6586565},
	url          = {https://doi.org/10.5281/zenodo.6586565}
	}

License & Attribution
---------------------

Copyright 2022 Tomás E. Müller Bravo.

HostPhot is being developed by `Tomás E. Müller Bravo <https://temuller.github.io/>`_ in a
`public GitHub repository <https://github.com/hostphot/piscola>`_.
The source code is made available under the terms of the MIT license.
