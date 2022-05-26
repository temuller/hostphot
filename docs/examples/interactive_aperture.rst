.. _inter_ap:

Interactive Aperture
====================

HostPhot also allows the user to interactively set the aperture used for photometry:

.. code:: python

	from hostphot.interactive_aperture import InteractiveAperture

	obj = InteractiveAperture('SN2009Y')

.. image:: static/inter_ap.png

The calculated photometry can then be exported into a ``csv`` file:

.. code:: python

	obj.export_photometry()
