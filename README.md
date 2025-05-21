<p align="center">
	<img src="docs/hostphot_logo.png" alt="drawing" width="300"/>
</p>

Global and local photometry of galaxies hosting supernovae or other transients

[![repo](https://img.shields.io/badge/GitHub-temuller%2Fhostphot-blue.svg?style=flat)](https://github.com/temuller/hostphot)
[![Documentation Status](https://readthedocs.org/projects/hostphot/badge/?version=latest)](https://hostphot.readthedocs.io/en/latest/?badge=latest)
[![license](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/temuller/hostphot/blob/master/LICENSE)
[![Tests and Publish](https://github.com/temuller/hostphot/actions/workflows/main.yml/badge.svg)](https://github.com/temuller/hostphot/actions/workflows/main.yml)
![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)
[![PyPI](https://img.shields.io/pypi/v/hostphot?label=PyPI&logo=pypi&logoColor=white)](https://pypi.org/project/hostphot/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04508/status.svg)](https://doi.org/10.21105/joss.04508)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6992139.svg)](https://doi.org/10.5281/zenodo.6992139)
[![Coverage](https://raw.githubusercontent.com/temuller/hostphot/main/coverage.svg)](https://raw.githubusercontent.com/temuller/hostphot/main/coverage.svg)


Read the full documentation at [hostphot.readthedocs.io](https://hostphot.readthedocs.io/en/latest/). It is recommended to read the **Further Information** section to understand how HostPhot works.
___
## Conda environment

It is recommended to create an environment before installing HostPhot:

```code
conda create -n hostphot pip
conda activate hostphot
pip install hostphot
```

### Requirements

HostPhot requirements can be found in the `requirements.txt` file.

### Tests

To run the tests, go to the parent directory and run the following command:

```code
pytest -v
```

## Modules

### Cutouts

This module allows you to download image cutouts from different surveys (e.g. `PanSTARRS`):

```python
from hostphot.cutouts import download_images

name = 'SN2004eo'
host_ra, host_dec = 308.2092, 9.92755  # coords of host galaxy of SN2004eo
survey = 'PanSTARRS'
download_images(name, host_ra, host_dec, survey=survey)
```

### Image Pre-processing

Coadds can be created and stars can be masked out of the images:

```python
from hostphot.coadd import coadd_images

coadd_filters = 'riz'
coadd_images(name, filters=coadd_filters, survey=survey)  # creates a new fits file
```

```python
from hostphot.processing.masking import create_mask

# one can extract the mask parameters from the coadd
# this also creates new fits files
coadd_mask_params = create_mask(name, host_ra, host_dec,
                                filt=coadd_filters, survey=survey,
                                extract_params=True)  

for filt in 'grizy':
    create_mask(name, host_ra, host_dec, filt, survey=survey,
                common_params=coadd_mask_params)
```

If the user is not happy with the result of the masking, there are a few parameters that can be adjusted. For instance, `threshold` sets the threshold used by `sep` for detecting objects. Lowering it will allow the detection of fainter objects. `sigma` is the width of the gaussian used for convolving the image and masking the detected objects. If `crossmatch` is set to `True`, the detected objects are cross-matched with the Gaia catalog and only those in common are kept. This is useful for very nearby host galaxies (e.g. that of SN 2011fe) so the structures of the galaxy are not maked out, artificially lowering its flux.

### Local Photometry

Local photometry can be obtained for multiple circular apertures:


```python
import hostphot.photometry.local_photometry as lp

ap_radii = [3, 4]  # aperture radii in units of kpc
ra, dec =  308.22579, 9.92853 # coords of SN2004eo
z = 0.0157  # redshift

results = lp.multi_band_phot(name, ra, dec, z,
                             survey=survey, ap_radii=ap_radii, 
                             use_mask=True, correct_extinction=True,
                             save_plots=True)
```

If the results return `NaN` values, this means that the flux is below the detection limit for the given survey.

### Global Photometry

Global photometry can be obtained in a similar way to local photometry, using common aperture:

```python
import hostphot.photometry.global_photometry as gp

results = gp.multi_band_phot(name, host_ra, host_dec, 
                             survey=survey, ra=ra, dec=dec,
                             use_mask=True, correct_extinction=True,
                             common_aperture=True, coadd_filters='riz', 
                             save_plots=True)
```

By default, HostPhot corrects for Milky Way extinction using the recalibrated dust maps
by Schlafly & Finkbeiner (2011) and the extinction law from Fitzpatrick (1999).

### Surveys List

This is the list of surveys in HostPhot:

  * DES
  * PS1
  * SDSS
  * GALEX
  * 2MASS
  * WISE
  * unWISE
  * Legacy Survey
  * Spitzer (SEIP)
  * VISTA (VHS, VIDEO, VIKING)
  * HST (WFC3 only)
  * SkyMapper (not well calibrated for extended sources)
  * SPLUS
  * UKIDSS
  * JWST (NIRCam only)

## Contributing

To contribute, either open an issue or send a pull request (prefered option). You can also contact me directly (check my profile: https://github.com/temuller).

### Adding other surveys

If you wish a survey to be added to HostPhot, there are a couple of ways of doing it. 1) You can do a pull request, following the same structure as used for the surveys that are already implemented, or 2) open an issue asking for a survey to be added. Either way, there are a fews things needed to add a survey: where to download the images from (e.g., using `astroquery`), zero-points to convert the images's flux/counts values into magnitudes, the magnitude system (e.g. AB, Vega), the pixel scaling of the images (in units of arcsec/pixel), the filters transmission functions and any other piece of information necessary to properly estimate magnitudes, errors, etc. If you open an issue asking for a survey to be added, please include all this information. For more information, please check the [Adding New Surveys](https://hostphot.readthedocs.io/en/latest/further_information/adding_surveys.html) section of the documentation.


## Citing HostPhot

If you make use of HostPhot, please cite the following [paper](https://joss.theoj.org/papers/10.21105/joss.04508):

```code
@article{Müller-Bravo2022, 
  author = {Tomás E. Müller-Bravo and Lluís Galbany},
  title = {HostPhot: global and local photometry of galaxies hosting supernovae or other transients},
  doi = {10.21105/joss.04508}, 
  url = {https://doi.org/10.21105/joss.04508}, 
  year = {2022}, 
  publisher = {The Open Journal}, 
  volume = {7}, 
  number = {76}, 
  pages = {4508},  
  journal = {Journal of Open Source Software} 
} 
```

## What's new!
v3.0.0
* Major internal restructuring of the code
    * Convert the cutouts into a module with each survey as a different submodule
    * Storage of input parameters (with timestamps)
    * Storage of output parameters in a more friendly format
    * Restructuring of output files
* Fix the VISTA SCIENCE ARCHIVE url for downloading VISTA images
* Fix issue with VISTA image size when downloading cuouts
* The download of VISTA images now automatically chooses the survey with overlap for the given coordinates, prioritising the survey with deeper coverage.
* The user experience is kept approximately consistent with previous versions, but there are some changes
* Download of HST images is now working!
* Adding download of JWST images!
___
v2.10.0
* Adding JWST! (few NIRCam filters for now - thanks to Lisa Kelsey for this addition)
* SkyMapper upgrade from DR2 to DR4 (note that there is still a problem with the photometric calibration of extended sources)
* Fix HST functions, which includes working with Drizzlepac images
* Change path handling package (os --> pathlib)
* Interactive aperture and download of RGB images have been removed and are no longer available
* Close some fits images that were previously kept opened

v2.9.0
* Fixing issue when applying/propagating common-mask parameters (e.g. from a coadd) to other images
* Updated error propagation for Legacy Survey with inverse-variance maps (using it as a weight mask now)
* Bumping up the DR version of VIKING and VIDEO for VISTA

v2.8.0
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

v2.7.1
* Adding SED plotting
* Fixed plotting angle of masked object 

v2.7.0
* MAJOR BUG: Convertion factor between different pixel scales fixed (it was previously inverted). This affects the common aperture photometry between different surveys!
* Moving font family to `_constant.py` file

v2.6.2
* Prettier plots (now using `aplpy`), but also more informative!
* Scale of the apertures for the masks is now a parameter (`r`) 
* Raise exception is now `True` by default when calculating photometry

v2.6.1
* 2MASS cutouts improved (picking largest image)

v2.6.0:
* HST (WFC3 only - pseudo-trial) included
* 2MASS cutouts fixed (it now downloads the image closest to the given coordinates)

v2.5.1:
* Using sfdmap2 instead of sfdmap to avoid issues with numpy version (requires Python>=3.9)

v2.5.0:
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

v2.4.0:
* Fixed and improved GALEX cutouts
* Better galaxy identification
* Better output plots for masked images
* Option to return NaN or raise an exception when getting photometry

v2.3.2:
* Improved download of GALEX images (download even if they seem to be just background noise)


## Acknowledgements

I thank Yuchen LIU for helping me adding Spitzer as part of AstroHackWeek 2022.
