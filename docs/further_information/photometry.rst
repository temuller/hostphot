.. _information_photometry:

Photometry
==========

To obtain calibrated photometry in magnitudes, the zero-point (ZP) is essential and usually all that is needed apart from the counts ("flux") measured from the images:

	:math:`m = -2.5*log_{10}(\text{counts}) + ZP`

However, the error propagation can be a lot more problematic for some surveys.

**Note:** out of the implemented surveys, only WISE and 2MASS images need background subtraction. The images from the other surveys are already background subtracted.

For the global photometry, HostPhot uses :func:`sep.sum_ellipse()` to meaused the counts of the host galaxy, where the value of the parameter ``gain`` depends on each survey+image and the parameter ``err`` is taken to be the global RMS of the background of the image, calculated with :func:`sep.Background()`.

For the local photometry, HostPhot uses :func:`photutils.aperture_photometry()` to meaused the counts of the host galaxy, where the value of the parameter ``error`` is calculated with :func:`astropy.stats.sigma_clipped_stats` (using ``sigma=3``) and :func:`photutils.utils.calc_total_error` (using the exposure time of each image). From the output of :func:`photutils.aperture_photometry()` the counts/flux is given by ``aperture_sum`` while the error is given by ``aperture_sum_err``.

In addition to the `magnitude` uncertainty calculated from the aperture photometry above (:math:`\sigma_{\text{ap}}`), other uncertainties are added in quadrature (see below).

Note that the global and local photometry are calculates in a similar way as in `Wiseman et al., 2020 <https://ui.adsabs.harvard.edu/abs/2020MNRAS.495.4040W/abstract>`_ and `Kelsey et al., 2021  <https://ui.adsabs.harvard.edu/abs/2021MNRAS.501.4861K/abstract>`_, respectively.


PS1
~~~

* **ZP**
  
  The PS1 images are rescaled to a ZP of :math:`25 + 2.5*log_{10}(\text{exposure time})`, where the exposure time is given by the ``EXPTIME`` keyword in the images (see the `PS1 Photometric Calibration <https://outerspace.stsci.edu/display/PANSTARRS/PS1+Stack+images#PS1Stackimages-Photometriccalibration>`_).
  
* **Error Propagation**

  The readnoise and Poisson noise are propagated using the following formula in magnitude space:
  
  :math:`\sigma_{\text{noise}} = 2.5/ln(10) * sqrt(A_{\text{ap}} * (\text{readnoise}**2) + flux / \text{gain}) / flux`
  
  where :math:`A_{\text{ap}}` is the area of the aperture used (:math:`\pi*a*b` for an ellipse) and :math:`flux` are the counts measured inside that area. The readnoise is obtained from the ``HIERARCH CELL.READNOISE`` keyword in the image's header, while the gain is obtained from the ``HIERARCH CELL.GAIN`` keyword.
  
  Thus, :math:`\sigma = sqrt(\sigma_{\text{ap}}^2 + \sigma_{\text{noise}}^2)`.
  


DES
~~~

* **ZP**
  
  DES coadd images use a global ZP set to :math:`30` (see the `DES DR1 Processing <https://des.ncsa.illinois.edu/releases/dr1/dr1-docs/processing>`_).
  
* **Error Propagation**

  The errors are propagated in the same way as for PS1. However, there is an additional component coming from the calibration of the photometric system (see the `DES DR1 Quality website <https://des.ncsa.illinois.edu/releases/dr1/dr1-docs/quality>`_). The calibration uncertainties (:math:`\sigma_{\text{cal}}`) are 2.6, 2.9, 3.4, 2.5, and 4.5 mmag for the :math:`g`, :math:`r`, :math:`i`, :math:`z`, and :math:`Y` bands, respectively.
  
  Thus, :math:`\sigma = sqrt(\sigma_{\text{ap}}^2 + \sigma_{\text{noise}}^2 + \sigma_{\text{cal}}^2)`.


SDSS
~~~~

* **ZP**
  
  Given that the units of the SDSS images are in nanomaggies, the ZP is equal to :math:`22.5` (see `https://www.sdss.org/dr13/help/glossary/#nanomaggie <https://www.sdss.org/dr13/help/glossary/#nanomaggie>`_).
  
* **Error Propagation**

  Additional noise is propagated with the following formula in magnitude space:
  
  :math:`\sigma_{\text{noise}} = 2.5/ln(10) * sqrt(\text{dark variance} + flux / \text{gain}) / flux`
  
  where the values of gain and dark variance are obtained from `https://data.sdss.org/datamodel/files/BOSS_PHOTOOBJ/frames/RERUN/RUN/CAMCOL/frame.html <https://data.sdss.org/datamodel/files/BOSS_PHOTOOBJ/frames/RERUN/RUN/CAMCOL/frame.html>`_ and they are assumed to be the largest available values, for a conservative approach.

  Thus, :math:`\sigma = sqrt(\sigma_{\text{ap}}^2 + \sigma_{\text{noise}}^2)`.


GALEX
~~~~~

* **ZP**
  
  GALEX images have different ZPs for the two filters: :math:`18.82` and :math:`20.08` for :math:`FUV` and :math:`NUV`, respectively (see `https://asd.gsfc.nasa.gov/archive/galex/FAQ/counts_background.html <https://asd.gsfc.nasa.gov/archive/galex/FAQ/counts_background.html>`_).
  
* **Error Propagation**

  The formulas from the GALEX website are used (in magnitude space):
  
  :math:`\sigma_{\text{rep}} (FUV) = -2.5*\Big(log_{10}(\text{CPS}) - log_{10}\big(\text{CPS} + sqrt(\text{CPS} * t_{\text{exp}} + (0.050 * \text{CPS} * t_{\text{exp}} )^2) / t_{\text{exp}} \big) \Big)`
  :math:`\sigma_{\text{rep}} (NUV) = -2.5*\Big(log_{10}(\text{CPS}) - log_{10}\big(\text{CPS} + sqrt(\text{CPS} * t_{\text{exp}} + (0.027 * \text{CPS} * t_{\text{exp}} )^2) / t_{\text{exp}} \big) \Big)`
    
  where CPS is counts per second and :math:`t_{\text{exp}}` is the exposure time. The later is obtained from the images obtained with `astroquery.mast.Observations <https://astroquery.readthedocs.io/en/latest/mast/mast.html>`_ and save in the ``EXPTIME`` keyword.
  
  Thus, :math:`\sigma = sqrt(\sigma_{\text{ap}}^2 + \sigma_{\text{rep}}^2)`.


2MASS
~~~~~

* **ZP**
  
  Each 2MASS image has its own ZP (``MAGZP`` keyword in the header, as described in `Section 4.3 <https://irsa.ipac.caltech.edu/data/2MASS/docs/releases/allsky/doc/sec4_3.html>`_ from the 2MASS website).
  
* **Error Propagation**

  To calculate the coadd noise we follow the equations described in `https://wise2.ipac.caltech.edu/staff/jarrett/2mass/3chan/noise/ <https://wise2.ipac.caltech.edu/staff/jarrett/2mass/3chan/noise/>`_:
  
  :math:`\sigma_{\text{noise}} = 1.0857/\text{SNR}`
  :math:`\text{SNR} = S / sqrt\big( (S/G*N_c) + n_c*(2*k_z*\sigma_c)^2 + (n_c*0.024*\sigma_c)^2 \big)`
    
  where :math:`S` is the integrated aperture flux, :math:`G` is the gain (typically 10), :math:`N_c` is the number of coadds per pixel (assumed to be 6), :math:`n_c` is the number of coadd pixels in the aperture (equal to :math:`4*n_f`), :math:`k_z` is the kernel smoothing factor (:math:`\sim1.7`) and :math:`\sigma_c` is the coadd noise (assumed to be approximately the global RMS of the image's background). :math:`n_f` is the number of frame pixels in the aperture and is assumed to be equal to the aperture area in pixel units.
  
  Thus, :math:`\sigma = sqrt(\sigma_{\text{ap}}^2 + \sigma_{\text{noise}}^2)`.


WISE
~~~~

* **Correct flux**

  To properly calculate the flux of the WISE images, and aperture correction factor (:math:`f_{\text{apcorr}}`) is applied, as described in `Section 2.3 <https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec2_3f.html>`_ of the WISE website. This is assumed to be :math:`1.0` as HostPhot does not use PSF fitting.

* **ZP**
  
  The WISE images also have their own ZP in their headers (``MAGZP`` keyword in the header, as described in Section 2.3 of the WISE website, link above).
  
* **Error Propagation**

  The ZP comes with an associated uncertainty (:math:`\sigma_{ZP}`, ``MAGZPUNC`` keyword in the header).
  The source uncertainty is:
  
  :math:`\sigma_{\text{src}} = sqrt\big(f_{\text{apcorr}}^2 * F_{\text{src}} * (\Sigma\sigma_i^2 + k*(N_A^2/N_B) * \sigma^2_{\bar{B}/pix}) + \sigma_{\text{conf}}^2 \big)`,
  
  where :math:`F_{\text{corr}}` is the correlated noise correction factor for flux variance , :math:`N_A` and :math:`N_B` are the number of pixels in the source aperture and annulus (both assume to be equal to the aperture area in pixel units), respectively, :math:`\sigma_i` is the flux uncertainty for pixel :math:`i` from uncertainty map (assumed to be approximately the error on the aperture sum above), :math:`\sigma^2_{\bar{B}/pix}` is the variance in sky-background annulus (assumed to be equal to the global RMS of the image's background), and :math:`\sigma_{\text{conf}}^2` is the confusion noise-variance on scale (assumed to be approximately the error on the aperture sum above).
  
  Thus, :math:`\sigma = sqrt\big(\sigma_{ZP}^2 + 1.179*(\sigma_{\text{src}}^2 / F_{\text{src}}^2) \big)`, as described in the link above, were :math:`F_{\text{src}}` is the integrated aperture flux of the source (e.g. galaxy).


unWISE
~~~~~~

* **ZP**
  
  unWISE images are rescaled to have ZPs of :math:`22.5`, as SDSS.
  
* **Error Propagation**

  This is assumed to be the same as for WISE.
  

Legacy Survey
~~~~~~~~~~~~~

* **ZP**
  
  Legacy Survey images use a global ZP set to :math:`22.5` (see the `Legacy Survey website <https://www.legacysurvey.org/dr9/description/>`_).
  
* **Error Propagation**

  The errors are propagated in the same way as for DES and PS1. The gain, exposure time and readnoise are assumed to be similar to those of DES: :math:`30` :math:`e`/ADU, :math:`900` s and :math:`7` :math:`e`/pixel, respectively.
  
  Thus, :math:`\sigma = sqrt(\sigma_{\text{ap}}^2 + \sigma_{\text{noise}}^2)`.
  
  
Spitzer
~~~~~~~

* **ZP**
  
  Spitzer images include their own ZP in their headers. They include both VEGA and AB ZPs, although the latter the used. This is found in the ``ZPAB`` keyword, although it is renamed to ``MAGZP`` to follow HostPhot convention. For more information, check the `calibration of IRAC by Gillian Wilson <https://faculty.ucr.edu/~gillianw/cal.html>_`
  
* **Error Propagation**

  The errors are propagated in the same way as for DES and PS1, where the gain and readnoise come from the `IRAC <https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/iracinstrumenthandbook/IRAC_Instrument_Handbook.pdf>`_ and `MIPS <https://irsa.ipac.caltech.edu/data/SPITZER/docs/mips/mipsinstrumenthandbook/MIPS_Instrument_Handbook.pdf>`_ instrument manuals (see tables 2.3 and 2.4).
  
  Thus, :math:`\sigma = sqrt(\sigma_{\text{ap}}^2 + \sigma_{\text{noise}}^2)`.
  
  
VISTA
~~~~~

* **ZP**
  
  VISTA images include their own ZP in their headers: ``MAGZPT`` keyword, although this is renamed to ``MAGZP`` to follow HostPhot convention.
  
* **Error Propagation**

  The errors are propagated in the same way as for DES and PS1, with an additional component coming from the ZP calibration, found in the header (``MAGZRR`` keyword).
  
  Thus, :math:`\sigma = sqrt(\sigma_{\text{ap}}^2 + \sigma_{\text{noise}}^2)`.
