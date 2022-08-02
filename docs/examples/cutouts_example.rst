.. _cutouts_example:

Cutouts
=======

This module allows you to download image cutouts from :code:`PS1`, :code:`DES`, :code:`SDSS`, :code:`GALEX`, :code:`2MASS` and :code:`WISE` (and maybe other surveys in the future). For this, the user can use :func:`download_images` and use the coordinates of an object:

.. code:: python

	from hostphot.cutouts import download_images

	name = 'SN2004eo'
	host_ra, host_dec = 308.2092, 9.92755  # coords of host galaxy of SN2004eo
	survey = 'PS1'
	download_images(name, host_ra, host_dec, survey=survey)

A new directory is created with the name ``name`` under the working directory (see :ref:`Working Directory <work_dir>`). The downloaded fits images will have the format ``<survey>_<filter>.fits``. If the filters are not specified, images in all the available filters (survey dependent) are downloaded. It is recommended to use the coordinates of the host galaxy to have it in the centre of the image. Sometimes, the transient can be very far from the host galaxy, so having separate images for the object and its host might be better.

Let's check the downloaded image:

.. code:: python

	import numpy as np
	import matplotlib.pyplot as plt
	from astropy.io import fits

	img = fits.open('images/SN2004eo/PS1_g.fits')

	data = img[0].data
	m, s = np.nanmean(data), np.nanstd(data)

	fig, ax = plt.subplots(figsize=(8, 8))
	im = ax.imshow(data, interpolation='nearest',
		       cmap='gray',
		       vmin=m-s, vmax=m+s,
		       origin='lower')
	plt.show()

.. image:: static/SN2004eo.png
