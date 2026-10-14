# Changelog

All notable changes to HostPhot are listed here.

## v4.0.0

* Adding HSC (Hyper Suprime-Cam SSP PDR3) support: broadband (g, r, r2, i, i2, z, Y) and narrowband (NB387, NB816, NB921, NB1010) filters via HTTP Basic Auth cutout API. Set the `HSC_SSP_USERNAME` and `HSC_SSP_PASSWORD` environment variables (e.g. in a `.env` file) to use.
* HSC aperture photometry agrees within ~0.01 mag with the HSC-SSP PDR3 catalog
* The `.env` file is now searched for from the current working directory (previously from the HostPhot installation directory)
* Fixing JWST zeropoint uncertainty: 1% of the flux (~0.011 mag) instead of 1% of the zeropoint in magnitudes (~0.27 mag). JWST magnitude and flux errors from previous versions were overestimated
* Fixing the aperture area used for the uncertainties of the global photometry: it now includes the Kron radius and its scale (it was underestimated by a factor of ~10-100). Global photometry errors of 2MASS increase (e.g. from ~0.003 to ~0.02-0.04 mag for bright galaxies); optical surveys are barely affected
* Fixing the WISE noise model, following the WISE Explanatory Supplement: the correlated-noise factor now uses the PRF noise-pixels of each band instead of the aperture area, so the noise scales with the square root of the aperture area (it previously scaled linearly). WISE errors from previous versions were overestimated (e.g. 0.05-0.6 mag down to ~0.01-0.02 mag for a bright host), especially for large apertures and W4
* unWISE uncertainties now use the inverse-variance maps of the unWISE coadds (downloaded together with the images), with a blank-aperture noise floor, in the same way as Legacy Survey and HSC. unWISE errors from previous versions were overestimated (e.g. 0.05-1.9 mag down to ~0.006-0.03 mag for a bright host). The unWISE W3 zeropoint uncertainty is also corrected (0.015 mag). Previously downloaded unWISE images need to be downloaded again to include the inverse-variance maps
* Fixing the blank-aperture noise floor (used by Legacy Survey, HSC and unWISE): detected sources, including the host galaxy, are now masked before placing the blank apertures. Previously, the apertures could land on the host galaxy, overestimating the errors of local photometry with small apertures (e.g. ~0.45 mag instead of ~0.005 mag for a 1 kpc aperture in Legacy Survey)
* No aperture (encircled energy) correction is applied to HST photometry. This correction is meant for point sources, while for extended sources (galaxies) the light scattered by the PSF out of the aperture is largely compensated by the light scattered into it, and the HST PSF is much smaller than the usual apertures, so this is fine for HostPhot's use case (consistent with the other surveys). The previous implementation was effectively inactive for local photometry due to a unit mismatch (pixels vs arcsec), so those results do not change. For global photometry, it was wrongly applied (using the unscaled Kron ellipse area), so HST global magnitudes are now ~0.04 mag brighter (e.g. F275W)
* New error model for PanSTARRS, DES, SkyMapper, SPLUS, VISTA and UKIDSS: the readnoise and Poisson noise were counted twice (and with inconsistent units for some surveys), and the correlated noise of the coadds was not accounted for. The aperture uncertainty (background and Poisson noise) now has a blank-aperture noise floor instead, as for Legacy Survey, HSC and unWISE, which agrees with the noise measured in blank apertures to within ~10-30% (previously, the errors were underestimated by up to ~2x for PanSTARRS and overestimated by ~2-3x for DES, VISTA, UKIDSS and SPLUS). The uncertainties of bright galaxies, dominated by the calibration, barely change
* Fixing Spitzer uncertainties: the flux uncertainty was returned in different units (DN/s) from the flux (MJy/sr), and the readnoise and Poisson terms used inconsistent units, which overestimated the errors by large factors (e.g. ~10-1000x for IRAC and ~30-70x for MIPS). Spitzer now uses the same error model as PanSTARRS (with the blank-aperture noise floor), plus the absolute calibration uncertainty (3% for IRAC and 4% for MIPS 24 micron), which was previously not included
* Adding the IRAC extended-source aperture correction (Section 8.2.2 of the IRAC Instrument Handbook) for apertures with radii of at least 8 arcsec, as the IRAC calibration is tied to point sources. IRAC magnitudes of galaxies from previous versions were too bright, e.g. by ~0.05-0.1 mag in channels 1 and 2, and ~0.15-0.3 mag in channels 3 and 4, for apertures with radii of 30-100 arcsec. The uncertainty of the correction (10% of its size) is included in the errors
* Documenting that no extended-source correction is needed for MIPS 24 micron (consistent with the point-source calibration), and recommending apertures with radii of at least ~13 arcsec for MIPS due to its large PSF
* Validating the NIR photometry with stars: HostPhot agrees with the 2MASS Point Source Catalog (transformed into each photometric system) to within ~0.01 mag for 2MASS, ~0.02-0.03 mag for UKIDSS and ~0.04 mag for VISTA (6 arcsec radius apertures). The VISTA difference is consistent with the broader PSF wings of the VISTA tiles (see the documentation). The 2MASS/VISTA/UKIDSS comparison tests now check the agreement (previously they did not assert anything)
* Improving the blank-aperture noise floor: the blank-aperture fluxes are measured relative to a smooth background (with sources masked), so that the faint, undetected wings of the galaxy do not inflate the uncertainties. For large apertures on galaxies filling most of the image, the floor can still be conservative
* Fixing PanSTARRS gain, which was not read from the image header (a value of 1 was used instead). This has a negligible effect on the uncertainties (<4%)
* Fixing several download issues:
  * Herschel maps were read after deleting the temporary files containing them
  * Spitzer cutouts were larger than requested (e.g. 5 arcmin instead of 3 arcmin), as the size was sent in pixels instead of arcsec
  * VISTA no longer crashes when there is no coverage
  * 2MASS no longer crashes when an image is missing for a filter
  * filters given as a single string with multi-character names (e.g. `filters="FUV"`) were split into characters, saving images with wrong names; single strings are now handled for all surveys (e.g. `"FUV"`, `"Ks"` or `"grizy"`)
  * PanSTARRS images could be saved with the wrong filter name if the filters were not given in `grizy` order or if an image was missing
  * HST: no crash with `overwrite=False`, and a clear error if no filters are given (also for JWST)
  * SED plotting no longer crashes for HST, and Legacy Survey filters default to DECam if no version is given (a duplicated `extract_filter` function was removed)
* Apertures partially without data (NaN pixels, e.g. image edges or chip gaps, or outside the image) are now handled consistently for all surveys: if up to 5% of the aperture has no data, the flux is measured from the remaining pixels with a warning; otherwise, NaN is returned with a warning. Previously, NaN was returned silently for most surveys, while Legacy Survey, HSC and unWISE silently returned fluxes measured from the remaining pixels only (i.e. underestimated). For the global (Kron) photometry, the tolerance applies to the estimated fraction of the flux without data (from the mean surface brightness at the same elliptical radius) instead of the area, as the outer parts of Kron apertures contain little flux (e.g. 20% of the area but ~2% of the flux)
* The files with the input parameters (e.g. `input_cutouts.csv`) could not be read with pandas when the saved parameters changed between versions (e.g. the timestamp column). They are now rewritten with a consistent header in that case, which also repairs existing files the next time they are updated
* Fixing the orientation of apertures transferred between images (common-aperture photometry and masks created from a reference filter/survey): the position angle returned by photutils (in degrees) was used as radians, so the transferred ellipses were wrongly oriented. Checked against the exact transformation for PanSTARRS, DES, VISTA and UKIDSS (including the flipped surveys). Common-aperture magnitudes and reference-based masks from previous versions were affected
* Fixing SDSS uncertainties: SDSS images are in nanomaggies, but the Poisson noise of the sources used the exposure time as gain (instead of gain/NMGY, ~1000 e-/nanomaggy), and an extra dark-current term was added with inconsistent units (and counted twice, as it is already part of the background noise). SDSS errors from previous versions were overestimated by ~3-10x for faint sources (e.g. 1.3 instead of 0.36 mag for a 3-sigma source). SDSS now also includes the blank-aperture noise floor
* Fixing 2MASS uncertainties: the number of pixels in the aperture was counted 4 times (the aperture area is already in 1-arcsec coadd pixels), overestimating the background noise by 2x. 2MASS errors from previous versions were overestimated by ~2-2.5x (agreeing now within ~10-30% with the scatter of blank apertures). The blank-aperture noise floor and a 1% calibration uncertainty are now also included
* Fixing S-PLUS broad-band filters (u, g, r, i, z): they were never downloaded (silently), as the filter names are in upper case in the image service, and the i band was also missing from the filters configuration. A warning is now shown when no image is found for a requested S-PLUS filter, and the weight maps are never downloaded instead of the images. HostPhot S-PLUS photometry agrees with the S-PLUS catalogue within ~0.01 mag for stars
* Fixing crashes of the global photometry: NaN pixels close to the galaxy (e.g. PanSTARRS images) made the Kron radius NaN (they are now masked in the background estimation, source detection and Kron radius), a galaxy not detected in an image now raises a clear error (or returns NaN with `raise_exception=False`) instead of a `TypeError`, and the source-detection pixel buffer is scaled with the image size (large galaxies in high-resolution images)
* Background subtraction is now applied by default to Spitzer and JWST images (the SEIP mosaics only have a model of the zodiacal light subtracted and the JWST i2d images are not sky-subtracted), and the background-subtracted GALEX images (`intbgsub`, using the GALEX pipeline sky model) are now downloaded instead of the intensity maps with sky. Previously, the fluxes of faint sources and large apertures were overestimated, e.g. by ~2-3x (IRAC2) and ~50-80x (IRAC4) for stars, and by up to several magnitudes for GALEX NUV. HostPhot now agrees within ~5% with the Spitzer SEIP and GALEX catalogues. Previously downloaded GALEX images need to be downloaded again (`overwrite=True`)
* New background estimation for the surveys that need background subtraction (2MASS, WISE, VISTA, SkyMapper, UKIDSS, Spitzer and JWST, or when `bkg_sub=True`): the sources, including the host galaxy (with an enlarged Kron ellipse), are masked, and the background is estimated with a mesh of 30 arcsec (instead of an unmasked mesh of 64 pixels, i.e. 13-88 arcsec depending on the survey). Previously, part of the galaxy light was subtracted as background, so the magnitudes of large apertures and global photometry were too faint, e.g. by ~0.1-0.2 mag at 60 arcsec and ~0.3 mag for the Kron magnitudes of nearby galaxies in SkyMapper, and by ~0.1-0.2 mag in UKIDSS and VISTA (checked against Legacy Survey and 2MASS). Very large images (e.g. JWST) are binned for the background estimation. A warning is now raised when most of the image is masked (e.g. the galaxy fills the image), recommending a larger image size
* Fixing Legacy Survey downloads of a single filter (e.g. `filters="r"`), which saved a single row of pixels instead of the image
* VISTA downloads no longer hang indefinitely when the VISTA Science Archive does not respond (a clear error is raised after 5 minutes)
* Fixing the blank-aperture noise floor, which was underestimated for apertures of radius >~5 arcsec (e.g. by ~1.5-2.5x for PanSTARRS and DES at 10 arcsec): the blank apertures were placed using a 1-sigma source mask, which also masks positive noise fluctuations (especially with correlated noise), so only quiet regions were sampled. The floor now uses the same 3-sigma source mask as the background estimation, including an enlarged ellipse around the host galaxy (smaller if it covers most of the image). The quoted errors now agree within ~30-40% with the scatter of blank apertures (PanSTARRS, DES, Legacy Survey, UKIDSS, 2MASS)
* Speeding up the photometry (results unchanged): the filters configuration file is no longer re-read hundreds of times per call, the dust maps are loaded once, and the Kron-aperture optimisation computes the error map once and uses `sep` for the (exact) elliptical sums. E.g., for five PanSTARRS images, local photometry went from ~12 s to ~1.5 s, and global photometry from ~26 s to ~3 s
* Renaming `extract_legacy_aperture_flux` and `extract_legacy_kron_flux` to `extract_invvar_aperture_flux` and `extract_invvar_kron_flux`, as these are used by multiple surveys

## v3.2.3

* Normalise RA to [0, 360) in `download_images` (solves issue with WISE)

## v3.2.2

* Fixed hardcoded image size for 2MASS

## v3.2.1

* very minor bug fix

## v3.2.0

* 2MASS, VISTA and UKIDSS photometry agrees within ~0.1 mag of catalog sources (still some small discrepancies)
* Fixing Legacy Survey erros - now using the invvar maps for proper values
* Adding DLR distance to the aperture output
* Speeding up some downloads of image cutouts

## v3.1.2

* Fixing download of GALEX images (issue introduced in v3.1.0 when trying to avoid timeout error - [#13](https://github.com/temuller/hostphot/issues/13)).
* Also storing the coordinates of detected objects in RA,DEC and the ellipse parameters a/b in degrees.

## v3.1.1

* Fixing minor bug when calculating dust extinction for Legacy Survey (Issue [#12](https://github.com/temuller/hostphot/issues/12))
* Saving input for aperture extraction

## v3.1.0

* Using DES DR2 instead of DR1
* Avoid timeout errors when downloading image cutouts for some surveys.
* Adapting global aperture even if the survey is the same as the reference (mainly to avoid issues with JWST).
* Fixing/adding features from different issues, mainly for HST/JWST ([#8](https://github.com/temuller/hostphot/issues/8), [#9](https://github.com/temuller/hostphot/issues/9), [#10](https://github.com/temuller/hostphot/issues/10), [#11](https://github.com/temuller/hostphot/issues/11) - thanks Lisa!).

## v3.0.0

* Major internal restructuring of the code:
  * Convert the cutouts into a module with each survey as a different submodule
  * Storage of input parameters (with timestamps)
  * Storage of output parameters in a more friendly format
  * Restructuring of output files
* Fix the VISTA SCIENCE ARCHIVE url for downloading VISTA images.
* Fix issue with VISTA image size when downloading cuouts
* The download of VISTA images now automatically chooses the survey with overlap for the given coordinates, prioritising the survey with deeper coverage.
* The user experience is kept approximately consistent with previous versions, but there are some changes
* Download of HST images is now working!
* Adding download of JWST images!

___
## v2.10.0

* Adding JWST! (few NIRCam filters for now - thanks to Lisa Kelsey for this addition)
* SkyMapper upgrade from DR2 to DR4 (note that there is still a problem with the photometric calibration of extended sources)
* Fix HST functions, which includes working with Drizzlepac images
* Change path handling package (os --> pathlib)
* Interactive aperture and download of RGB images have been removed and are no longer available
* Close some fits images that were previously kept opened

## v2.9.0

* Fixing issue when applying/propagating common-mask parameters (e.g. from a coadd) to other images
* Updated error propagation for Legacy Survey with inverse-variance maps (using it as a weight mask now)
* Bumping up the DR version of VIKING and VIDEO for VISTA

## v2.8.0

* Fix SED plotting with negative or too large values/errors, and improve plotting aesthetics
* Fix problem downloading some 2MASS images giving internal server error (no image overlap)
* Adding UKIDSS, SPLUS and SkyMapper surveys. WARNINGS: SkyMapper photometry (DR2) is not very accurate with extended sources!
* The marker showing the galaxy position on images has changed colour for better visualization
* Adding functions to easily load previously saved parameteres from pickle files
* Adding missing extinction correction for flux (not magnitudes)
* Including exposure time into flux for PS1, VISTA and UKIDSS surveys
* Adding SDSS ZP offsets into flux
* Output zeropoint together with the photometry
* Updating LegacySurvey errors, taken from inverse-variance maps and ZP uncertainty
* Adding missing offsets to unWISE W1 and W2 bands into fluxes
* Kron radius is no longer "optimized", just the scale, which doesn't have any effect in practice for the user

## v2.7.1

* Adding SED plotting
* Fixed plotting angle of masked object

## v2.7.0

* MAJOR BUG: Convertion factor between different pixel scales fixed (it was previously inverted). This affects the common aperture photometry between different surveys!
* Moving font family to `_constant.py` file

## v2.6.2

* Prettier plots (now using `aplpy`), but also more informative!
* Scale of the apertures for the masks is now a parameter (`r`)
* Raise exception is now `True` by default when calculating photometry

## v2.6.1

* 2MASS cutouts improved (picking largest image)

## v2.6.0

* HST (WFC3 only - pseudo-trial) included
* 2MASS cutouts fixed (it now downloads the image closest to the given coordinates)

## v2.5.1

* Using sfdmap2 instead of sfdmap to avoid issues with numpy version (requires Python>=3.9)

## v2.5.0

* Systematic error floor added to PS1 photometry
* Added missing uncertainties in the error budget of DES (~5 mmag)
* Flux/counts have been added to output photometry
* GALEX now downloads images with largest exposure time by default
* Fixed image realignment/orientation between different surveys when using common apertures (for masking and global photometry)
* Added option to output mask parameters, and also aperture parameters for global photometry
* Option added to set a distant threshold to identify the host galaxy in the image (by default use the nearest object)
* Slight change in the column names of the local photometry output file
* Offsets in SDSS zeropoints to place it in the AB system
* overwrite set to True by default when downloading image cutouts
* Other minor bugs fixed

## v2.4.0

* Fixed and improved GALEX cutouts
* Better galaxy identification
* Better output plots for masked images
* Option to return NaN or raise an exception when getting photometry

## v2.3.2

* Improved download of GALEX images (download even if they seem to be just background noise)
