.. _photometry_example:

Photometry
==========

Local Photometry
~~~~~~~~~~~~~~~~

Local photometry can be obtained for the downloaded images. For this, a circular aperture is used, assuming a cosmology (``H0=70`` and ``Om0=0.3`` by default):


.. code:: python

	import hostphot.local_photometry as lp

	ap_radii = [1, 2, 3, 4]  # in units of kpc
	results = lp.multi_band_phot(name='SN2004eo', ra=308.22579, dec=9.92853, z=0.0157, 
			   survey='PS1', ap_radii=ap_radii, use_mask=True, save_plots=True)


``results`` is a dictionary with the photometry (magnitudes) of the filters used. Note that the coordinates are at the position of the object (``SN2004eo``). The cosmology can be changed with :func:`lp.choose_cosmology()`. Setting ``use_mask=True`` tells HostPhot to used the masked images previously created (see :ref:`Image Pre-processing <preprocessing>`) and setting ``save_plots=True`` provides output plots with the images and the apertures used.

Image with the aperture:

.. image:: static/local.png

Global Photometry
~~~~~~~~~~~~~~~~~

Global photometry works in a relatively similar way:

.. code:: python

	import hostphot.global_photometry as gp

	host_ra, host_dec = 308.2092, 9.92755  # coods of host galaxy of SN2004eo
	results = gp.multi_band_phot(name='SN2004eo', host_ra, host_dec, survey='PS1',
				    use_mask=True, common_aperture=True, coadd_filters='riz',
				    optimze_kronrad=True, save_plots=True)

Setting ``common_aperture=True`` tells HostPhot to used the same aperture for all the filters, obtrained from the coadded image (``coadd_filters='riz'``; see :ref:`Image Pre-processing <preprocessing>`) and setting ``optimze_kronrad=True`` provides provides a more reliable aperture than used using the default parameters commonly used by SExtractor as the aperture is increased until the change in flux is neglegible (this can be changed with ``eps``). The rest of the parameters are the same as before.

Image with the aperture:

.. image:: static/global.png
