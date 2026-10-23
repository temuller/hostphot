.. _information_photometry:

Photometry
==========

To obtain calibrated photometry in magnitudes, the zero-point (ZP) is essential and usually all that is needed apart from the counts ("flux") measured from the images:

	:math:`m = -2.5*log_{10}(\text{counts}) + ZP`

However, the error propagation can be a lot more problematic for some surveys.

**Note:** out of the included surveys, WISE, 2MASS, VISTA, SkyMapper, UKIDSS, Spitzer, JWST and Herschel images need background subtraction (background subtraction of these surveys is performed by default by HostPhot; it can be changed with the ``bkg_sub`` parameter). Spitzer (SEIP) mosaics only have a model of the zodiacal light subtracted, the sky is not subtracted from JWST ``i2d`` images, and the Herschel SPIRE extended-emission maps include an absolute sky level (PACS maps can have small offsets). For GALEX, the background-subtracted intensity maps of the GALEX pipeline (``intbgsub``) are downloaded. The images from the other surveys are already background subtracted by their respective pipelines, although small residuals can remain (e.g. SDSS, HSC and S-PLUS), which can affect faint sources and large apertures (``bkg_sub=True`` can be used in those cases).

For the global photometry, HostPhot uses :func:`sep.sum_ellipse()` to meaused the counts of the host galaxy, where the value of the parameter ``gain`` depends on each survey+image and the parameter ``err`` is taken to be the global RMS of the background of the image, calculated with :func:`sep.Background()`.

For the local photometry, HostPhot uses :func:`photutils.aperture_photometry()` to meaused the counts of the host galaxy, where the value of the parameter ``error`` is calculated with :func:`astropy.stats.sigma_clipped_stats` (using ``sigma=3``) and :func:`photutils.utils.calc_total_error` (using the exposure time of each image). From the output of :func:`photutils.aperture_photometry()` the counts/flux is given by ``aperture_sum`` while the error is given by ``aperture_sum_err`` (:math:`\sigma_{\text{ap}}`). Other uncertainties are added in quadrature (see below for the respective surveys).

Note that the global and local photometry are calculates in a similar way as in `Wiseman et al. (2020) <https://ui.adsabs.harvard.edu/abs/2020MNRAS.495.4040W/abstract>`_ and `Kelsey et al. (2021)  <https://ui.adsabs.harvard.edu/abs/2021MNRAS.501.4861K/abstract>`_, respectively.

Any offsets added to the zeropoints are instead propagated into the fluxes, such that the zeropoints remain unchaged.


PS1
~~~

* **ZP**
  
  The PS1 images have a fixed ZP of :math:`25`. The flux for PS1 needs to be rescaled by the exposure time, given by the ``EXPTIME`` keyword in the images (see the `PS1 Photometric Calibration <https://outerspace.stsci.edu/display/PANSTARRS/PS1+Stack+images#PS1Stackimages-Photometriccalibration>`_).
  
* **Error Propagation**

  The aperture uncertainty (:math:`\sigma_{\text{ap}}`) includes the background noise (estimated from the image, which includes the readnoise) and the Poisson noise of the source. As the images are coadds, their noise is correlated between pixels, which is not captured by the pixel-wise uncertainties: a noise floor is therefore applied, given by the scatter of the fluxes in blank apertures around the source (:math:`\sigma_{\text{blank}}`). These apertures do not overlap with detected sources (including the host galaxy), and their fluxes are measured relative to a smooth local background, so that they are not affected by the faint wings of the galaxy. The aperture uncertainty is then :math:`\sigma_{\text{ap}} = max(\sigma_{\text{ap}}, \sigma_{\text{blank}})`. In addition, a systematic error floor is added to each filter, (:math:`\sigma_g, \sigma_r, \sigma_i, \sigma_z, \sigma_y`) = (14, 14, 15, 15, 18) mmag, as described in `Magnier et al. (2020) <https://ui.adsabs.harvard.edu/abs/2020ApJS..251....6M/abstract>`_.
  
  Thus, :math:`\sigma = sqrt(\sigma_{\text{ap}}^2 + \sigma_{\text{floor}}^2)`.
  


DES
~~~

* **ZP**
  
  DES coadd images use a global ZP set to :math:`30` (see the `DES DR1 Processing <https://des.ncsa.illinois.edu/releases/dr1/dr1-docs/processing>`_).
  
* **Error Propagation**

  The errors are propagated in the same way as for PS1 (including the blank-aperture noise floor). However, there is an additional component coming from the calibration of the photometric system (see the `DES DR1 Quality website <https://des.ncsa.illinois.edu/releases/dr1/dr1-docs/quality>`_). There are statistical uncertainties on the shifts applied to DES photometry to place it in the AB system (:math:`\sigma_{\text{stat, shift}}`), which are 2.6, 2.9, 3.4, 2.5, and 4.5 mmag for the :math:`g`, :math:`r`, :math:`i`, :math:`z`, and :math:`Y` bands, respectively. In addition, there are median coadd zeropoint statistical uncertainties (:math:`\sigma_{\text{stat, zp}}`): 5, 4, 5, 6, and 5 mmag for the :math:`g`, :math:`r`, :math:`i`, :math:`z`, and :math:`Y` bands, respectively.
  
  Thus, :math:`\sigma = sqrt(\sigma_{\text{ap}}^2 + \sigma_{\text{stat, shift}}^2 + \sigma_{\text{stat, zp}}^2)`.


SDSS
~~~~

* **ZP**
  
  Given that the units of the SDSS images are in nanomaggies, the ZP is equal to :math:`22.5` (see `https://www.sdss.org/dr13/help/glossary/#nanomaggie <https://www.sdss.org/dr13/help/glossary/#nanomaggie>`_). However, SDSS magnitudes are not exactly in AB system, as described in `https://www.sdss4.org/dr12/algorithms/fluxcal/#SDSStoAB <https://www.sdss4.org/dr12/algorithms/fluxcal/#SDSStoAB>`_. Therefore, offsets need to be applied to :math:`u` and :math:`z` bands: :math:`u_{\rm AB} = u_{\rm SDSS} - 0.04` and :math:`z_{\rm AB} = z_{\rm SDSS} + 0.02`, which are instead propagated into the fluxes to keep a constant zeropoint for all bands.
  
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
    
  where CPS is counts per second and :math:`t_{\text{exp}}` is the exposure time. The later is obtained from the images obtained with `astroquery.mast.Observations <https://astroquery.readthedocs.io/en/latest/mast/mast.html>`_ and saved in the ``EXPTIME`` keyword.
  
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
  
  :math:`\sigma_{\text{src}} = sqrt\big(f_{\text{apcorr}}^2 * F_{\text{corr}} * (\Sigma\sigma_i^2 + k*(N_A^2/N_B) * \sigma^2_{\bar{B}/pix}) + \sigma_{\text{conf}}^2 \big)`,
  
  where :math:`F_{\text{corr}} = N_p (s_{\text{in}}/s_{\text{out}})^2` is the correlated noise correction factor for flux variance, with :math:`N_p` the effective number of noise-pixels of the PRF (:math:`13.772`, :math:`17.636`, :math:`35.476` and :math:`24.462` for *W1*-*W4*, from `Section IV.6.c.i <https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_6ci.html>`_) and :math:`s_{\text{in}}/s_{\text{out}}` the ratio of the native to Atlas Image pixel scales (:math:`2` for *W1*-*W3* and :math:`4` for *W4*), limited to :math:`N_A` for small apertures; :math:`N_A` is the number of pixels in the source aperture and :math:`N_B` the number of pixels used for the background (the whole image); :math:`\sigma_i` is the flux uncertainty for pixel :math:`i` (assumed to be the global RMS of the image's background, i.e. background-dominated noise), so that :math:`\Sigma\sigma_i^2 = N_A \sigma_i^2`; :math:`\sigma^2_{\bar{B}/pix}` is the variance of the background (the global RMS of the image's background squared) and :math:`k = \pi/2`; the confusion noise (:math:`\sigma_{\text{conf}}`) is not included.
  
  Thus, :math:`\sigma = sqrt\big(\sigma_{ZP}^2 + 1.179*(\sigma_{\text{src}}^2 / F_{\text{src}}^2) \big)`, as described in the link above, were :math:`F_{\text{src}}` is the integrated aperture flux of the source (e.g. galaxy).


unWISE
~~~~~~

* **ZP**
  
  unWISE images are rescaled to have ZPs of :math:`22.5`, as explained in `Lang (2014) <https://iopscience.iop.org/article/10.1088/0004-6256/147/5/108>`_. Note that to improve the agreement between unWISE and AllWISE fluxes, it is recommend subtracting :math:`4` mmag from unWISE *W1* and :math:`32` mmag from unWISE *W2* fluxes, as descrived in the `unWISE catalogs website <https://catalog.unwise.me/catalogs.html#absolute>`_ (performed internally by HostPhot).

* **Background**

  The unWISE pipeline already subtracts the sky (a constant level from each input frame and from each coadd tile; `Lang 2014 <https://iopscience.iop.org/article/10.1088/0004-6256/147/5/108>`_), so HostPhot does not subtract a background from unWISE images by default (it can be done with ``bkg_sub=True``). In tests on high-latitude fields, the residual sky was small (:math:`\sim\pm1` nanomaggies per pixel), while subtracting a background degraded the agreement with the AllWISE aperture photometry of stars (the background mesh is only a few times larger than the WISE PSF).
  
* **Error Propagation**

  The unWISE images are downloaded together with their inverse-variance maps (`Lang 2014 <https://iopscience.iop.org/article/10.1088/0004-6256/147/5/108>`_), which are used to calculate the flux uncertainties in the same way as for the Legacy Survey, including a noise floor from the scatter of blank (source-free) apertures around the source, which accounts for correlated noise. The ZP uncertainties (:math:`\sigma_{ZP}`) are those of the WISE Atlas Images: :math:`6`, :math:`7`, :math:`15` and :math:`12` mmag for *W1*-*W4*, respectively (Table 1 of `Section 2.3 <https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec2_3f.html>`_ of the WISE website).

  Thus, :math:`\sigma = sqrt(\sigma_{\text{invvar}}^2 + \sigma_{\text{ZP}}^2)`, where :math:`\sigma_{\text{invvar}}` is the largest between the propagated inverse-variance and blank-aperture uncertainties.
  

Legacy Survey
~~~~~~~~~~~~~

* **ZP**
  
  Legacy Survey images use a global ZP set to :math:`22.5` (see the `Legacy Survey website <https://www.legacysurvey.org/dr9/description/>`_).
  
* **Error Propagation**

  This surveys has two main sources of unertainties: one coming from the photometric calibration of DR10 (see `Zhou et al. (2023) <https://ui.adsabs.harvard.edu/abs/2023RNAAS...7..105Z/abstract>`_), which are 5.0, 3.9, 4.3 and 5.5 mmag for the :math:`g`, :math:`r`, :math:`i` and :math:`z` bands, respectively, and the other coming from the inverse-variance maps included with the images, with a noise floor given by the scatter of blank apertures (with the detected sources, including the host galaxy, masked) around the source.
  
  Thus, :math:`\sigma = sqrt(\sigma_{\text{ap}}^2 + \sigma_{\text{ZP}}^2 + \sigma_{\text{invvar}}^2)`.
  
  
Spitzer
~~~~~~~

* **ZP**
  
  Spitzer images include their own ZP in their headers. They include both VEGA and AB ZPs, although the latter is used. This is found in the ``ZPAB`` keyword, although it is renamed to ``MAGZP`` to follow HostPhot convention. For more information, check the `calibration of IRAC by Gillian Wilson <https://faculty.ucr.edu/~gillianw/cal.html>`_
  
* **Error Propagation**

  The errors are propagated in the same way as for PS1 (including the blank-aperture noise floor, as the mosaics have correlated noise), with an additional component coming from the absolute flux calibration (:math:`\sigma_{\text{cal}}`): :math:`3\%` for IRAC (Section 4.3 of the `IRAC Instrument Handbook <https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/iracinstrumenthandbook/14/>`_) and :math:`4\%` for MIPS :math:`24 \mu \text{m}` (`MIPS Instrument Handbook <https://irsa.ipac.caltech.edu/data/SPITZER/docs/mips/mipsinstrumenthandbook/42/>`_), i.e. :math:`\sim0.033` and :math:`\sim0.043` mag, respectively. All the uncertainties are calculated in the units of the images (MJy/sr).
  
  Thus, :math:`\sigma = sqrt(\sigma_{\text{ap}}^2 + \sigma_{\text{cal}}^2)`.

* **Extended-source correction (IRAC)**

  The IRAC calibration is tied to point sources, so the photometry of extended sources (galaxies) is corrected for the light scattered across the IRAC focal plane, following Section 8.2.2 of the `IRAC Instrument Handbook <https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/iracinstrumenthandbook/46/>`_: :math:`F_{\text{true}} = F_{\text{measured}} \times (A e^{-r^B} + C)`, where :math:`r` is the (effective, for elliptical apertures) aperture radius in arcsec, and the coefficients (:math:`A`, :math:`B`, :math:`C`) are (:math:`0.82`, :math:`0.370`, :math:`0.910`), (:math:`1.16`, :math:`0.433`, :math:`0.94`), (:math:`1.49`, :math:`0.207`, :math:`0.66`) and (:math:`1.37`, :math:`0.330`, :math:`0.740`) for IRAC channels 1 to 4, respectively. For example, for a :math:`30` arcsec radius the fluxes are corrected by factors of :math:`0.93`, :math:`0.96`, :math:`0.86` and :math:`0.80`, respectively. As recommended by the handbook, the correction is not applied to apertures with radii smaller than :math:`8` arcsec. The corrections are good to :math:`10\%`, which is propagated as an uncertainty of :math:`10\%` of the size of the correction (:math:`0.1 \times |1 - \text{correction}|`). Note that these corrections were derived from elliptical galaxies, and larger departures can be expected for low surface brightness galaxies. The correction is applied to all the apertures with radii of at least :math:`8` arcsec, including those centred on point-like sources (e.g. stars, or compact nuclei in local photometry), for which it is not appropriate: their fluxes are then too faint (e.g. by :math:`0.03`, :math:`0.00`, :math:`0.07` and :math:`0.14` mag in channels 1 to 4 for a :math:`12.2` arcsec radius, as found when comparing with the SEIP Source List photometry of stars). For these sources, the correction can be undone by dividing the fluxes by the factor given by ``hostphot.photometry.photometry_utils.irac_extended_source_correction(filt, radius)``.

* **Extended sources (MIPS)**

  No extended-source correction is applied to MIPS :math:`24 \mu \text{m}`, as extended-source measurements are consistent with the point-source calibration (Section 4.1.3 of the `MIPS Instrument Handbook <https://irsa.ipac.caltech.edu/data/SPITZER/docs/mips/mipsinstrumenthandbook/42/>`_). Point-source aperture corrections are not applied either, as for the other surveys (see the HST section). However, note that the MIPS :math:`24 \mu \text{m}` PSF is large (FWHM of :math:`\sim6` arcsec, with strong Airy rings): the point-source aperture corrections are :math:`\sim1.6`-:math:`2.0` for a :math:`7` arcsec radius aperture and :math:`\sim1.16` for :math:`13` arcsec (Table 4.13 of `Section 4.3.4 <https://irsa.ipac.caltech.edu/data/SPITZER/docs/mips/mipsinstrumenthandbook/50/>`_). Therefore, the photometry of small apertures (radius :math:`\lesssim13` arcsec, i.e. :math:`\sim2` times the FWHM) is dominated by the PSF rather than by the light distribution of the galaxy, and apertures enclosing a whole (compact) galaxy can miss a fraction (:math:`\sim5`-:math:`15\%`) of its flux, scattered by the PSF wings. It is recommended to use apertures with radii of at least :math:`\sim13` arcsec for MIPS, and larger for compact galaxies.
  
  
VISTA
~~~~~

* **ZP**
  
  VISTA images include their own ZP in their headers: ``MAGZPT`` keyword (see the `CASU VISTA website <http://casu.ast.cam.ac.uk/surveys-projects/vista/technical/photometric-properties>`_). Atmospheric extinction correction needs to be applied to the VISTA images in order to obtain an "effective" zeropoint (private communication with Nicholas Cross and VSA support).
  The atmospheric extinction is calculated as:
  
  :math:`extinction = c_{\text{ext}} \times (airmass - 1)`, 
  
  where the extinction coefficient :math:`c_{\text{ext}}` is 0.05 (``EXTINCT`` keyword in the header) and the airmass is the average between the values at the start and end of the observations (taken from the header as well). Thus, the effective zeropoint is:
  
  :math:`ZP_{\text{eff}} = MAGZPT - extinction` 
  
  and is stored in the ``MAGZP`` to follow HostPhot convention. In addition, the flux is rescaled by the exposure time, in the same way as for the PS1 images.

  The images are tiles, for which no further photometric correction of the astrometric distortion is necessary (see the `CASU tile generation website <http://casu.ast.cam.ac.uk/surveys-projects/vista/technical/tiles>`_).

* **Comparison with 2MASS and UKIDSS**

  For stars in a field covered by 2MASS, VISTA and UKIDSS, HostPhot photometry in :math:`6` arcsec radius apertures agrees with the 2MASS Point Source Catalog (transformed into each photometric system using the CASU and `Hodgkin et al. (2009) <https://ui.adsabs.harvard.edu/abs/2009MNRAS.394..675H/abstract>`_ colour equations) to within :math:`\sim0.01` mag for 2MASS, :math:`\sim0.02`-:math:`0.03` mag for UKIDSS and :math:`\sim0.04` mag for VISTA (HostPhot being brighter). The difference for VISTA is consistent with the broader PSF wings of the tiles, which combine up to 96 detector images with different PSFs: a fixed aperture collects more flux than the (point-source, aperture-corrected) calibration assumes, and the difference increases with the aperture size. This mainly affects point-like sources; for extended sources (galaxies), the light scattered by the PSF out of the aperture is largely compensated by the light scattered into it. Note that the agreement with the VHS catalogue (Petrosian magnitudes) is worse (:math:`\sim0.04`-:math:`0.10` mag), as these do not capture the full flux of the stars.
  
* **Error Propagation**

  The errors are propagated in the same way as for PS1 (including the blank-aperture noise floor), with an additional component coming from the ZP calibration (:math:`\sigma_{\text{ZP}}`), found in the header of the images (``MAGZRR`` keyword), combined with the systematic uncertainty of the calibration from 2MASS: :math:`2\%` in :math:`YJHK_s` and :math:`3\%` in :math:`Z` (`González-Fernández et al. 2018 <https://ui.adsabs.harvard.edu/abs/2018MNRAS.474.5459G/abstract>`_), i.e. :math:`\sigma_{\text{ZP}} = \sqrt{MAGZRR^2 + \sigma_{\text{cal}}^2}`. Note that the Poisson noise of the sources is calculated from the image counts (ADU) without the detector gain (:math:`\sim4.2\,e^-/\text{ADU}`) or the number of combined exposures, so it is overestimated, which is conservative but negligible in practice (e.g. :math:`\lesssim0.003` mag for stars).
  
  Thus, :math:`\sigma = sqrt(\sigma_{\text{ap}}^2 + \sigma_{\text{ZP}}^2)`.
  
  
HST (WFC3, ACS/WFC and NICMOS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **ZP**
  
  HST zeropoints can be calculated using the `PHOTFLAM` and `PHOTPLAM` keywords from the image' header, as explained in `https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration <https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration>`_ and `https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration <https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration>`_: 
  
  :math:`ZP_{\text{AB}} = -2.5\log(PHOTFLAM) - 5\log(PHOTPLAM) - 2.408`.
  
  This is saved in the header under the ``MAGZP`` keyword. 
  
  The value of `PHOTFLAM` depends on the detector (`PHOTFLAM1` and `PHOTFLAM2` for the WFC3/UVIS detectors UVIS1 and UVIS2), but the images are already calibrated to a single value (`PHOTFLAM`), and the same applies to `PHOTPLAM`.

* **Aperture correction**

  No aperture (encircled energy) correction is applied. These corrections (see `EE-UVIS <https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration/uvis-encircled-energy>`_ and `EE-IR <https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration/ir-encircled-energy>`_) are meant for point sources, while HostPhot targets extended sources (galaxies): for these, the light scattered by the PSF out of the aperture is largely compensated by the light scattered into it from the rest of the galaxy. In addition, the HST PSF (FWHM of :math:`\sim0.07` and :math:`\sim0.15` arcsec for WFC3/UVIS and WFC3/IR, respectively) is much smaller than the usual apertures, so the effect is negligible. This is also consistent with the other surveys, for which no aperture correction is applied either.
  
* **Error Propagation**

  The aperture uncertainty (:math:`\sigma_{\text{ap}}`) includes the background and Poisson noise, with an additional component coming from the ZP calibration (`ERR_PHOTFLAM`; :math:`\sigma_{\text{ZP}}`), taken from the tables found in the photometric calibration websites of the instruments (see `UVIS photometric calibration <https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration/uvis-photometric-calibration>`_ `IR photometric calibration <https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration/ir-photometric-calibration>`_), which is of the order of a few percent at most (2% is assumed for the WFC3/UVIS quad filters, which are not in the tables). For ACS/WFC, the absolute flux calibration is accurate to :math:`\sim1\%` (`Bohlin 2016 <https://ui.adsabs.harvard.edu/abs/2016AJ....152...60B/abstract>`_), and for NICMOS to :math:`\sim5\%` (`NICMOS Data Handbook <https://www.stsci.edu/documents/dhb/web/c20_nicdataanal.fm2.html>`_; note that NICMOS images are in counts (DN) per second, converted into electrons with the ``ADCGAIN`` keyword for the Poisson noise, and that NICMOS has a count-rate non-linearity, not corrected by HostPhot). These are used as :math:`\sigma_{\text{ZP}}`.
  
  Thus, :math:`\sigma = sqrt(\sigma_{\text{ap}}^2 + \sigma_{\text{ZP}})`.


SkyMapper
~~~~~~~~~

* **ZP**
  
  SkyMapper images include their own ZP in their headers: ``ZPAPPROX`` keyword (see the `survey website forum <https://skymapper.anu.edu.au/forum/forum/using-the-tools-2/topic/photometric-calibration-magnitudes-etc-4/>`_), although this is renamed to ``MAGZP`` to follow HostPhot convention.
  
* **Error Propagation**

  The errors are propagated in the same way as for PS1 (including the blank-aperture noise floor), with an additional component coming from the ZP calibration (:math:`\sigma_{\text{ZP}}`), found in the header of the images (``ZPTERR`` keyword).
  
  Thus, :math:`\sigma = sqrt(\sigma_{\text{ap}}^2 + \sigma_{\text{ZP}}^2)`.


SPLUS
~~~~~

* **ZP**
  
  S-PLUS images have their ZP tabulated for the different filters in the `DR3_zero-points.cat <https://splus.cloud/files/documentation/DR3/iDR3_zps.cat>`_ file (field dependent values) found in the `SPLUS DR2/3 website <https://splus.cloud/documentation/dr2_3>`_: these are added to the image header under the ``MAGZP`` keyword to follow HostPhot convention.
  
* **Error Propagation**

  The errors are propagated in the same way as for PS1 (including the blank-aperture noise floor), with an additional component coming from the ZP calibration (:math:`\sigma_{\text{ZP}}`), following Section 4.4 of `Almeida-Fernandes et al. (2022) <https://ui.adsabs.harvard.edu/abs/2022MNRAS.511.4590A/abstract>`_: :math:`25` mmag for :math:`U` and :math:`F395` filters, :math:`15` mmag for :math:`F378` filter and :math:`10` mmag for the rest.
  
  Thus, :math:`\sigma = sqrt(\sigma_{\text{ap}}^2 + \sigma_{\text{ZP}}^2)`.


UKIDSS
~~~~~~

* **ZP**
  
  UKIDSS images include their own ZP in their headers and an "effective" zeropoint has to be calculated (stored in the ``MAGZP`` keyword in the header), correcting for atmospheric extinction, in the same way as for the VISTA images. In addition, the flux is rescaled by the exposure time, in the same way as for the PS1 images.
  
* **Error Propagation**

  The errors are propagated in the same way as for PS1 (including the blank-aperture noise floor), with an additional component coming from the ZP calibration (:math:`\sigma_{\text{ZP}}`), found in the header of the images (``MAGZRR`` keyword), combined with the systematic uncertainty of the calibration from 2MASS: :math:`1.5\%` in :math:`JHK` and :math:`2\%` in :math:`ZY` (`Hodgkin et al. 2009 <https://ui.adsabs.harvard.edu/abs/2009MNRAS.394..675H/abstract>`_), i.e. :math:`\sigma_{\text{ZP}} = \sqrt{MAGZRR^2 + \sigma_{\text{cal}}^2}`.
  
  Thus, :math:`\sigma = sqrt(\sigma_{\text{ap}}^2 + \sigma_{\text{ZP}}^2)`.

HSC
~~~

* **ZP**
  
  HSC-SSP PDR3 coadds are calibrated to a global ZP of :math:`27.0` (AB), obtained from the ``FLUXMAG0`` keyword of the images (:math:`ZP = 2.5\log(\text{FLUXMAG0})`) and stored in the header under the ``MAGZP`` keyword to follow HostPhot convention.
  
* **Error Propagation**

  The uncertainties come from the variance maps included with the images, propagated in the same way as the inverse-variance maps of the Legacy Survey, and from the photometric calibration of PDR3, assumed to be :math:`10` mmag (see `Aihara et al. 2022 <https://ui.adsabs.harvard.edu/abs/2022PASJ...74..247A/abstract>`_).
  
  Thus, :math:`\sigma = sqrt(\sigma_{\text{var}}^2 + \sigma_{\text{ZP}}^2)`.

  HostPhot aperture photometry has been checked against the HSC-SSP PDR3 Wide catalog: for apertures of 1 arcsec radius (``apertureflux_20``, i.e. 2 arcsec diameter), the magnitudes agree within :math:`\sim0.005` mag for point sources and :math:`\sim0.01` mag for galaxies in the :math:`grizy` bands.
