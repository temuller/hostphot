
.. _installation:

Installation
========================

HostPhot is easy to install and use.

Using pip
########################

The recommended option is by using `pip <https://pip.pypa.io/en/stable/>`_:

.. code::

	pip install hostphot

From source
########################

To install the code from source, do the following:

.. code::

	git clone https://github.com/temuller/hostphot.git
	cd hostphot
	python -m pip install .


Requirements
########################

HostPhot has the following requirements:

```code
numpy
pandas
matplotlib
astropy
reproject
photutils
astroquery
extinction
sfdmap
pyvo
sep
ipywidgets (optional: for interactive aperture)
ipympl (optional: for interactive aperture)
pytest (optional: for testing the code)
```
