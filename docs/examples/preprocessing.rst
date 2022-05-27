.. _preprocessing:

Image Pre-processing
====================

A set of pre-processing steps can be performed to ensure an accurate photometry calculation. Note that all the steps are performed following the structure given by the working directory (see :ref:`Working Directory <work_dir>`).

Coadding
~~~~~~~~

The user can perform image coadding in a single line:

.. code:: python

	from hostphot.coadd import coadd_images

	coadd_filters = 'riz'
	survey = 'PS1'
	coadd_images('SN2004eo', coadd_filters, survey)

This creates a new fits image under the object's directory, in this case, with the name ``PS1_riz.fits``. Coadding images is useful for common aperture photometry (see below). For the coadd, HostPhot makes use of `reproject <https://reproject.readthedocs.io/en/stable/index.html>`_.


Image Masking
~~~~~~~~~~~~~

Some low-redshift galaxy can have foreground stars "sitting" on top of them. HostPhot can remove these first detecting them with pseudo-sigma clipping (using `sep <https://github.com/kbarbary/sep/>`_) and cross-matching the sources with a catalog of stars (using `astroquery MAST <https://astroquery.readthedocs.io/en/latest/mast/mast.html>`_), and then masking them using a 2D Gaussian kernel. The coadded image created above can be masked and the parameters of the mask can be extracted to be used on single filter images of the same object: 


.. code:: python

	from hostphot.image_masking import create_mask

	host_ra, host_dec = 308.2092, 9.92755  # coods of host galaxy of SN2004eo
	coadd_mask_params = create_mask(name, host_ra, host_dec, 
                                         filt=coadd_filters, survey=survey, 
                                         extract_params=True)

	for filt in 'grizy':
            create_mask(name, host_ra, host_dec, filt, survey=survey, 
			common_params=coadd_mask_params)


Note that the host-galaxy coordinates need to be provided so HostPhot knows which object not to mask out. The steps above create masked fits images, in this case, with the names ``masked_PS1_<filter>.fits``. See below an exmaple of this mask applied:

.. image:: static/mask.png

The sigma clipping step can be fine tuned by setting the parameter of ``threshold`` (how many sigmas above the background noise the sources are detected). Note that the images can be background subtracted by setting ``bkg_sub=True``, although this is not needed for the ``DES`` and ``PS1`` images downloaded by HostPhot.
