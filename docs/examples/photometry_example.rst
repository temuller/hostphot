.. _photometry_example:

Photometry
==========

Local Photometry
~~~~~~~~~~~~~~~~

Local photometry can be obtained for the downloaded images. For this, a circular aperture is used (multiple apertures can be set), assuming a cosmology (``H0=70`` and ``Om0=0.3`` by default; hence a redshift is needed):


.. code:: python

	import hostphot.local_photometry as lp

	name = 'SN2004eo'
	ra, dec = 308.22579, 9.92853  # SN coords
	z = 0.0157  # redshift
	ap_radii = [1, 2, 3, 4]  # in units of kpc
	results = lp.multi_band_phot(name=name, ra=ra, dec=dec, z=z, survey='PS1', 
				ap_radii=ap_radii, use_mask=True, save_plots=True)


``results`` is a dictionary with the photometry (magnitudes) of the filters used. Note that the coordinates are at the position of the object (``SN2004eo``). The cosmology can be changed with :func:`lp.choose_cosmology()`. Setting ``use_mask=True`` tells HostPhot to used the masked images previously created (see :ref:`Image Pre-processing <preprocessing>`) and setting ``save_plots=True`` provides output plots with the images and the apertures used, which are saved under the object's directory.

Image with local aperture:

.. image:: static/local.png

Global Photometry
~~~~~~~~~~~~~~~~~

Global photometry relies on `sep <https://github.com/kbarbary/sep/>`_ and uses `Kron fluxes`. It works in a relatively similar way to the local photometry:

.. code:: python

	import hostphot.global_photometry as gp

	host_ra, host_dec = 308.2092, 9.92755  # coords of host galaxy of SN2004eo
	results = gp.multi_band_phot(name, host_ra, host_dec, survey='PS1',
				    use_mask=True, common_aperture=True, coadd_filters='riz',
				    optimize_kronrad=True, save_plots=True)

Setting ``common_aperture=True`` tells HostPhot to used the same aperture for all the filters, obtained from the coadded image (``coadd_filters='riz'``; see :ref:`Image Pre-processing <preprocessing>`) and setting ``optimize_kronrad=True`` provides a more reliable aperture than using the default parameters commonly used by SExtractor as the aperture is increased until the change in flux is neglegible (this can be changed with ``eps``). The rest of the parameters are the same as before.

Image with global aperture:

.. image:: static/global.png
