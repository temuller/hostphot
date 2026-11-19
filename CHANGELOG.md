# Changelog

All notable changes to HostPhot are listed here.

## v4.0.0

A major validation release: most surveys were checked against catalogues, other surveys and injected galaxies, which led to many fixes of the photometry, the uncertainties and the downloads. See *Migrating from v3 to v4* in the documentation for a summary, and the photometry documentation for the validation details.

### Changes that affect existing results

* The Milky-Way extinction correction is no longer applied by default (`correct_extinction=False`); when enabled, the flux uncertainties are also corrected (the signal-to-noise ratio is unchanged)
* Kron apertures: the standard Kron scale of 2.5 (as SExtractor's MAG_AUTO) is now the default (`optimize_kronrad=False`). The optimisation of the scale did not converge with noise or residual sky, so the apertures could grow up to 10 Kron radii, with fluxes biased by the sky (e.g. 0.1-0.2 mag brighter than the 2MASS XSC). The default magnitudes are therefore fainter (typically by 0.02-0.2 mag for bright galaxies). When enabled, the optimisation stops when the flux gained is consistent with the noise of the added ring, up to a scale of 6
* `create_mask`: the default detection threshold is now 10 sigma (previously 15), the same as the photometry, so that fainter neighbours are also masked (at 15 sigma, only ~30-70% of the neighbours brighter than r = 20 near the host were masked)
* Host identification (photometry and masking): the nearest detected object is only used if the galaxy coordinates are within its extent; otherwise, an error suggests a lower `threshold`. Previously, another object could be used silently (e.g. a star 53 arcsec away in 2MASS Ks)
* A non-positive flux now gives a NaN magnitude and magnitude uncertainty, with a warning (the flux and its uncertainty are kept, e.g. for upper limits)
* Apertures partially without data (NaN or zero-filled regions, image edges, chip gaps) are handled in the same way for all surveys: up to 5% of the aperture (local) or of the flux (global, estimated assuming a symmetric galaxy, and added to the flux and uncertainty) is allowed, with the `coverage` flag; otherwise, NaN with a warning. Previously, NaN was returned silently for most surveys, while Legacy Survey, HSC and unWISE silently returned underestimated fluxes
* HST: no aperture (encircled-energy) correction is applied (meant for point sources; it was inactive for local photometry and wrongly applied to global photometry, which is now ~0.04 mag brighter)
* Re-downloading is needed for: GALEX (background-subtracted `intbgsub` images), HSC (coadds without local sky subtraction, with the mask plane) and unWISE (with inverse-variance maps); the download and masking flags also need images downloaded or masked with v4.0.0
* API: `common_aperture=True` without `ref_filt`, and using reference parameters before they have been saved, now raise an error (previously, independent apertures were used silently); the `dlr` column of the aperture parameters is renamed `sep_arcsec`; `extract_legacy_aperture_flux` and `extract_legacy_kron_flux` are renamed `extract_invvar_aperture_flux` and `extract_invvar_kron_flux`; the `.env` file is searched for from the current working directory

### New features

* Curve-of-growth (CoG) aperture for the global photometry, `aperture_method="cog"` (the default, `"kron"`, is unchanged): the profile is measured in elliptical annuli out to where it is no longer significant (or flattens, rises or is contaminated in the outskirts), and the flux beyond it is estimated with a Sérsic fit out to 8 effective radii. With a common aperture, the tail of the reference is applied to all the filters. For 111 galaxies injected in a PanSTARRS field: median difference -0.019 mag (NMAD 0.035), against +0.136 (0.150) for Kron 2.5 and +0.002 (0.041) for the optimised Kron aperture
* Masking with a model of the host galaxy, `create_mask(method="host_model")` (the default, `"ellipses"`, is unchanged): the host is modelled with a Sérsic profile, and only Gaia stars and sources much brighter than the host where they lie are masked, with the masked pixels filled with the model plus noise. The masked fraction of the host light goes from ~4-23% to ~0-2% (72 galaxies of 6 types), and from 16-18% to ~2% for large spirals; the residual sources reaching the centre of the host (its bulge or bar, which the single Sérsic profile can underestimate) are not masked unless they are Gaia stars (e.g. the bar of a barred spiral was masked depending on the cutout size)
* Quality flags: boolean output columns (`{filt}_{flag}_flag`, with warnings) for less reliable measurements: `bkg`, `coverage`, `psf`, `err`, `aperture`, `calib`, `fallback_image`, `mask_fallback` and `mask_nogaia` (see `hostphot.flags`)
* New output columns, the same for all the surveys: AB magnitudes (`{filt}_AB`), flux densities in µJy and their uncertainties (`{filt}_flux_uJy`, `{filt}_flux_uJy_err`), 3-sigma upper limits (`{filt}_AB_ulim`) and, for the local photometry, the PSF FWHM (`{filt}_psf_fwhm`, with a warning for apertures smaller than the PSF). The SED plots use them (upper limits for non-detections)
* Provenance: the saved images include the HostPhot version, download date, survey, survey version (the one actually used, e.g. the default data release or the VISTA survey and GALEX project chosen for the position), size and source file (`HP*` keywords), and the photometry outputs include the HostPhot version, an identifier of each image used and its download date
* The saved aperture and mask parameters include the ellipses on the sky (`ra`, `dec`, `a_arcsec`, `b_arcsec`, `pa_deg`), which are used to transfer them to other filters and surveys: exact for any orientation of the images (no `flip` correction) and without the reference image, which previously had to be on disk unchanged (re-trimmed to rebuild its WCS). Results are unchanged (common-aperture magnitudes within 3e-5 mag); parameters saved by previous versions are still transferred through the reference image
* Gaia DR3 queries of the masking are cached per object (in memory and in `gaia_dr3_stars.ecsv`), with the Gaia partner data centres (ARI, AIP, GAVO) as fallback when the ESA archive fails
* Network robustness: all the archive queries and downloads have a timeout (300 s by default; `hostphot.utils.set_network_timeout()`), and failed queries and downloads are retried (with a wait) and reported with warnings; the status pages printed by astroquery when the ESA archives (Gaia, HST, JWST, Herschel) are down are no longer shown (e.g. on `import hostphot`)

### New surveys and data

* HSC (SSP PDR3; broad and narrow bands; needs `HSC_SSP_USERNAME` and `HSC_SSP_PASSWORD`, e.g. in a `.env` file), using the coadds without local sky subtraction (`coadd/bg`; the standard coadds remove the extended light of galaxies, e.g. ~2 mag for the Kron magnitude of NGC 799) and the mask plane (apertures with `NO_DATA` pixels return NaN). Stars agree within ~0.01 mag with the PDR3 catalogue
* KiDS (DR5; u, g, r, i), from the ESO archive, with the DR5 zeropoint corrections (`DMAG`) and systematic uncertainties. Stars agree with SDSS within ~0.01-0.04 mag
* UHS (DR3; J, H, K), processed as UKIDSS. Stars agree within ~0.01 mag with the UHS catalogue
* Euclid (DR1, the default, and Q1 MER mosaics; VIS, Y, J, H), from IRSA. VIS Kron magnitudes agree within ~0.01-0.02 mag with the MER catalogues; DR1 and Q1 differ by up to ~0.1 mag (recalibration)
* Herschel: PACS 70 microns (`PACS70`) and local photometry (in Jy/pixel, with blank-aperture uncertainties); background subtraction by default (the SPIRE maps include an absolute sky level, e.g. fluxes ~2x too large in a 7 arcmin aperture for NGC 628). Fluxes agree with KINGFISH (Dale et al. 2012) within ~1-19%
* HST: ACS/WFC and NICMOS (NIC1-3), in addition to WFC3 (previously, their filters passed the validation but failed later)
* Spitzer: the Spitzer Heritage Archive post-BCD mosaics are used when no SEIP mosaic covers the cutout (e.g. warm-mission IRAC; doubles the IRAC 3.6 micron coverage of a sample of nearby galaxies), with the `fallback_image` flag; IRAC extended-source aperture correction (IRAC Instrument Handbook, Section 8.2.2) for radii of at least 8 arcsec (galaxy magnitudes were ~0.05-0.3 mag too bright)

### Uncertainties

* New error model for PanSTARRS, DES, SkyMapper, S-PLUS, VISTA, UKIDSS, SDSS, 2MASS, Spitzer, HST, JWST, GALEX and WISE: the aperture uncertainty (background and Poisson noise) has a blank-aperture noise floor for correlated noise, as for Legacy Survey, HSC and unWISE (inverse-variance maps). Previously, the readnoise and Poisson noise were counted twice for several surveys, with inconsistent units for some (e.g. Spitzer errors overestimated ~10-1000x, SDSS ~3-10x for faint sources, 2MASS ~2x, DES/VISTA/UKIDSS/S-PLUS ~2-3x; PanSTARRS underestimated up to ~2x)
* Blank-aperture noise floor: the detected sources (including the host) are masked with the same 3-sigma mask as the background (a 1-sigma mask sampled only quiet regions, underestimating the floor ~1.5-2.5x at 10 arcsec, and unmasked hosts overestimated it, e.g. 0.45 instead of 0.005 mag); the fluxes are measured relative to a smooth background, with distant apertures used if the local ones are contaminated by faint galaxy light (`err` flag); for apertures larger than the blank sky allows, it is extrapolated from smaller apertures (previously skipped, e.g. 0.004 instead of 0.05 mag); for NICMOS, it is measured on the images (pedestal residuals); it is no longer skipped silently when the pixels without data surround the image (e.g. rotated maps); Herschel global photometry uses the same estimate (it had no uncertainty when the blank apertures did not fit)
* Global photometry of extended galaxies: the uncertainty of the sky level within the aperture (its variations and residual offset, from the blank apertures measured on the image itself) is added, and, for the CoG aperture, the uncertainty of the flux beyond the aperture (at least 80% of the Sérsic tail, or the flux in the outermost annuli without a tail). For injected galaxies, the CoG uncertainties were underestimated ~2-5x and are now consistent (normalised scatter ~1.0-1.1); the CoG magnitudes of 72 galaxies in Legacy Survey, PanSTARRS and SDSS now agree within their uncertainties. The uncertainties of large galaxies are larger (e.g. ~0.06-0.09 mag instead of ~0.02 mag)
* WISE: the correlated-noise factor of the WISE formula uses the PRF noise-pixels (errors were overestimated, e.g. 0.05-0.6 mag instead of 0.01-0.02 mag), and the floor-based uncertainty is used if larger (W1/W2 CoG errors underestimated ~3-8x, ~1.6-3.7x now with masks from PanSTARRS; W3/W4 consistent). unWISE uses the inverse-variance maps (errors were overestimated up to ~1.9 mag)
* GALEX: calibration uncertainty (5% FUV, 2.7% NUV) instead of a double-counted Poisson term. In shallow images (e.g. AIS), the background RMS collapses to ~0 (few sky photons per pixel), so every photon was a detection (FUV fluxes ~1-2 mag too bright when measured on the GALEX images alone) and the errors were ~0: the RMS is now estimated from the variance in 16x16-pixel boxes when it collapses (other surveys are not affected). With the masks and aperture from PanSTARRS, the GALEX magnitudes are unchanged and the CoG uncertainties are consistent
* Calibration terms: SDSS (0.01 mag in gri, 0.02 in uz), 2MASS (1%), Spitzer (3% IRAC, 4% MIPS), UKIDSS and VISTA (2MASS calibration, 1.5-3%), KiDS (DR5, 0.03-0.05 mag), corrected S-PLUS (10-25 mmag), JWST (1% of the flux instead of ~0.27 mag) and unWISE W3 (0.015 mag)
* Poisson noise of the sources: correct gains for SDSS (gain/NMGY), PanSTARRS (header), Spitzer (SEIP `GAIN`), JWST (effective gain from the pipeline variance, `EFFGAIN`), NICMOS (`ADCGAIN`); none for Herschel (it stopped the Kron optimisation too early, e.g. PACS fluxes 0.32-0.69 of KINGFISH)
* The aperture area used for the uncertainties of the global photometry includes the Kron radius and scale (it was ~10-100x too small; e.g. 2MASS errors ~0.003 -> ~0.02-0.04 mag); 2MASS uses the larger of its two error estimates for negative fluxes

### Photometry fixes

* Apertures transferred between images (common apertures and reference masks): the position angle from photutils (degrees) was used as radians, and the WCS of the untrimmed image was used for parameters measured on trimmed images (shifts of several arcsec), so transferred apertures and masks were misplaced or misoriented (the orientation is also fixed in v3.2.4)
* UKIDSS and UHS: the fluxes are divided by the exposure time of a single exposure (stacks with `NEXP > 1` were 2.5 log10(NEXP) mag too faint, e.g. 0.75 mag for UHS H), and the pixel scale is taken from the images (micro-stepped images had apertures of half the requested radius)
* The pixel scale of the images is used for VISTA, UHS, HST and Herschel
* `extract_aperture()` subtracts the background by default for the surveys that need it (`bkg_sub=None`, as the photometry and the masking); it never subtracted it (`bkg_sub=False`), so reference apertures measured with it on, e.g., 2MASS, WISE, UKIDSS or UHS images were biased by the sky level, or the galaxy was not detected
* Background subtraction by default for Spitzer and JWST (fluxes of faint sources and large apertures were overestimated, e.g. ~50-80x for IRAC4 stars), and for Herschel; GALEX uses the background-subtracted images of the pipeline (`intbgsub`; NUV fluxes were up to several magnitudes too bright). HostPhot agrees within ~5% with the Spitzer SEIP and GALEX catalogues
* Global photometry crashes fixed (NaN pixels near the galaxy, undetected galaxies, large galaxies in high-resolution images); Herschel global photometry checks the aperture coverage and applies the extinction correction
* `remove_nan` only trims the rows and columns without data at the edges (it removed valid data when the empty regions were inside the image)
* HST WFC3/UVIS: zeropoint uncertainties for all the apertures (others failed) and the quad filters (2% assumed); JWST filter transmission functions found (the extinction correction and SED plots failed)
* SED plots: fixed for HST and JWST, flux axis in µJy, and the local radius is matched numerically
* Image plots: the axes of rotated images (e.g. UHS, with RA along the y axis) are labelled with the coordinate along each of them (RA and Dec labels were swapped)
* The input-parameter files are rewritten with a consistent header when their columns change (they could not be read), and the global photometry no longer overwrites `input_aperture_parameters.csv` for every filter
* The photometry no longer leaves the image files open

### Masking and background

* New background estimation for the surveys that need background subtraction (2MASS, WISE, VISTA, SkyMapper, UKIDSS, UHS, Spitzer, JWST, Herschel, or `bkg_sub=True`): the sources and an enlarged ellipse around the host are masked, with a 30 arcsec mesh. Previously, part of the galaxy light was subtracted (e.g. Kron magnitudes ~0.3 mag too faint in SkyMapper, ~0.1-0.2 mag in UKIDSS and VISTA). The `bkg` flag marks apertures where the background follows the outskirts of the galaxy
* `sep` background: the mesh size avoids 1-pixel strips at the edges (unstable values, e.g. +17 sigma), and the source-mask dilation is limited to 10 pixels when the FWHM cannot be measured (most of HST/JWST cutouts were masked)
* Masking with `crossmatch=True`: Gaia DR3 sources with significant proper motion or parallax (Gaia EDR3 and the TESS Input Catalog included HII regions of nearby galaxies as stars), matched with proper angular separations (cos(dec) and the RA wrap were ignored)
* Masking with reference parameters and `save_mask_params=True` failed, and saved reference parameters could not be loaded again

### Downloads and image selection

* Images without data at the target (zero- or NaN-filled cutouts, footprint edges) are not saved, with a warning, and a warning is given when no images are found; large regions of exact zeros are treated as missing data in the photometry
* The images are chosen by their coverage of the requested cutout and then by depth, downloading them one by one until one covers it: 2MASS, UKIDSS (deep stacks first; previously, e.g. 1815 frames were downloaded for UDS), VISTA (well-calibrated tiles, MAGZRR <= 0.05 mag, and the latest VIDEO mosaics; tiles with MAGZRR of 0.5-0.9 mag were used), Spitzer, unWISE, HST (deepest first, including mosaics and zipped products), SDSS (frames containing the cutout; trimmed strips were returned), S-PLUS (tiles with coverage; empty regions gave magnitudes ~6 mag too faint), GALEX (tiles with data at the position) and Herschel (final maps of the photometry observations covering the target, ranked by depth; previously, e.g. 20 observations including spectroscopy were downloaded for NGC 628)
* DES: positions in the outer parts of the tiles in RA (~1 - cos(dec) of the area, e.g. ~43% at dec = -55 deg) found no image (silently)
* S-PLUS: the broad-band filters were never downloaded (upper-case names), and the i band was missing
* Legacy Survey: single-filter downloads saved one row of pixels, and positions outside the footprint raised an exception
* Filters given as a single string (e.g. `"FUV"`, `"Ks"`, `"grizy"`) were split into characters; PanSTARRS images could be saved with the wrong filter name; unWISE NEOWISE versions modified the list of filters
* UKIDSS: H2 images could be selected for H, all the bands were downloaded, and images without the image extension made all the filters fail
* VISTA: the `version` parameter was ignored, and downloads crashed without coverage or could hang
* HST and JWST: downloads failed when copying the files (table extensions, empty primary HDUs), and the filters are now required; HST candidates are the images whose footprint contains the position
* Herschel: maps were read after deleting their temporary files, file names were matched against the user's paths, corrupt tarballs broke later downloads, PACS 70-micron maps could be returned as PACS100, the maps were not cut to the requested size, and the pixel scales of the configuration file were corrected
* Spitzer cutouts were larger than requested (the size was sent in pixels); 2MASS crashed when a filter was missing; HSC downloaded all 11 filters by default

### Maintenance

* Speed-ups with unchanged results: e.g. for five PanSTARRS images, local photometry ~12 s -> ~1.5 s and global photometry ~26 s -> ~3 s (configuration and dust maps read once, `sep` elliptical sums, vectorised blank apertures); plotting ~2x faster for the masking and local photometry and ~4x for the global photometry (identical figures: cheaper layout pass for the tight bounding box, no extra redraw after saving, global figure no longer drawn twice); Spitzer SHA fallback reads the mosaic headers to download fewer files
* Packaging: `requests`, `scipy` and `pyyaml` declared as dependencies, test dependencies in the `test` extra, photutils 4 and Python 3.13 supported, release checklist in `RELEASING.md`
* Tests: the archive tests are marked `network` and run weekly in a separate workflow; the tests run on every push use local data and mocked archives (offline coverage ~84%); the NIR comparison tests now check the agreement with the 2MASS Point Source Catalogue (within ~0.01-0.04 mag)
* Documentation: *Migrating from v3 to v4* page, notes on the K (WFCAM) and Ks (VISTA, 2MASS) filters and on MIPS 24 micron (no extended-source correction; apertures of at least ~13 arcsec)

## v3.2.4

* Fixing the orientation of common apertures with photutils >= 3.0 (issue #19): the angle returned by photutils (in degrees since photutils 3.0) is now converted into radians

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
