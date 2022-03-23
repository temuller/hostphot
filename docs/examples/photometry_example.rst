.. _photometry_example:

Photometry
==========

Local Photometry
~~~~~~~~~~~~~~~~

Local photometry can be obtained for the downloaded images. For this, use :func:`extract_local_photometry()` for a single image:


.. code:: python

	from hostphot.local_photometry import extract_local_photometry

	fits_file = 'path/to/local/fits_file'
	ra, dec = 30, 100
	z = 0.01  # redshift
	ap_radius = 4  # aperture for the photometry in kpc
	survey = 'PS1'

	extract_local_photometry(fits_file, ra, dec, z, ap_radius, survey)

which returns ``mag`` and ``mag_err``. You can also use :func:`multi_local_photometry()` for multiple images:


.. code:: python

	from hostphot.local_photometry import multi_local_photometry

	multi_local_photometry(name_list, ra_list, dec_list, z_list,
		                     ap_radius, work_dir, filters,
		                       survey, correct_extinction)

where :code:`work_dir` should be the same as used in :func:`download_multiband_images()` and :code:`name_list` should contain the names of the SNe used in :func:`download_multiband_images()` as well. This produces a pandas DataFrame as an output where, e.g., column ``g`` is the g-band magnitude and ``g_err`` its uncertainty.


Global Photometry
~~~~~~~~~~~~~~~~~

Global photometry can be obtained in a similar way to local photometry. Use :func:`extract_global_photometry()` for a single image:

.. code:: python

	from hostphot.global_photometry import extract_global_photometry

	survey = 'PS1'

	extract_global_photometry(fits_file, host_ra, host_ra, survey=survey)

which returns ``mag`` and ``mag_err``. You can also use :func:`multi_global_photometry()` for multiple images:


.. code:: python

	from hostphot.global_photometry import multi_global_photometry

	survey = 'PS1'
	correct_extinction = True

	multi_global_photometry(name_list, host_ra_list, host_dec_list, work_dir, filters,
		                       survey=survey, correct_extinction=correct_extinction)
