.. _cutouts_example:

Cutouts
=======

This module allows you to download image cutouts from :code:`PS1`, :code:`DES`, :code:`SDSS`, :code:`GALEX`, :code:`2MASS` and :code:`WISE` (and maybe other surveys in the future). For this, the user can use :func:`download_images` and use the coordinates of an object:

.. code:: python

	import hostphot
	print('HostPhot version:', hostphot.__version__)
	
.. code:: python

	HostPhot version: 2.6.2

.. code:: python

	from hostphot.cutouts import download_images

	name = 'SN2004eo'
	ra, dec =  308.22579, 9.92853
	host_ra, host_dec = 308.2092, 9.92755  # coords of host galaxy 
	z = 0.0157  # redshift
	survey = 'PS1'
	download_images(name, host_ra, host_dec, survey=survey)

A new directory is created with the name ``name`` under the working directory (see :ref:`Working Directory <work_dir>`). The downloaded fits images will have the format ``<survey>_<filter>.fits``. If the filters are not specified, images in all the available filters (survey dependent) are downloaded. It is recommended to use the coordinates of the host galaxy to have it in the centre of the image. Sometimes, the transient can be very far from the host galaxy, so having separate images for the object and its host might be better.

Let's check the downloaded image:

.. code:: python

	from hostphot.utils import plot_fits

	plot_fits('images/SN2004eo/PS1_g.fits')

.. image:: static/SN2004eo.png
