.. _cutouts_example:

Cutouts
=======

This module allows you to download image cutouts from :code:`PS1`, :code:`DES` and :code:`SDSS`. For this, you can use :func:`get_PS1_images()`, :func:`get_DES_images()` and :func:`get_SDSS_images()`, respectively. For example:

.. code:: python

	from hostphot.cutouts import get_PS1_images

	ra, dec = 30, 100
	size = 400  # in pixels
	filters = 'grizy'

	fits_images = get_PS1_images(ra, dec, size, filters)

where :code:`fits_images` is a list with the fits images in the given filters.

You can also use :func:`download_multiband_images()` for multiple images:

.. code:: python

	from hostphot.cutouts import download_multiband_images

	download_multiband_images(sn_name, ra, dec, size,
		                        work_dir, filters,
		                          overwrite, survey)

where :code:`work_dir` is where all the images will be downloaded. A Subdirectory inside :code:`work_dir` will be created with the SN name as the directory name.
