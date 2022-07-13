<p align="center">
	<img src="docs/hostphot_logo.png" alt="drawing" width="300"/>
</p>

Global and local photometry of galaxies hosting supernovae or other transients

[![repo](https://img.shields.io/badge/GitHub-temuller%2Fhostphot-blue.svg?style=flat)](https://github.com/temuller/hostphot)
[![documentation status](https://readthedocs.org/projects/hostphot/badge/?version=latest&style=flat)](https://hostphot.readthedocs.io/en/latest/?badge=latest)
[![license](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/temuller/hostphot/blob/master/LICENSE)
[![Tests and Publish](https://github.com/temuller/hostphot/actions/workflows/main.yml/badge.svg)](https://github.com/temuller/hostphot/actions/workflows/main.yml)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)
[![PyPI](https://img.shields.io/pypi/v/hostphot?label=PyPI&logo=pypi&logoColor=white)](https://pypi.org/project/hostphot/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6469981.svg)](https://doi.org/10.5281/zenodo.6469981)


Read the full documentation at [hostphot.readthedocs.io](https://hostphot.readthedocs.io/en/latest/).
___
## Conda environment

It is recommended to create an environment before installing HostPhot:

```code
conda create -n hostphot pip
conda activate hostphot
pip install hostphot
```

### Tests

To run the tests, go to the parent directory and run the following command:

```code
pytest -v
```

## Modules

### Cutouts

This module allows you to download image cutouts from `PS1`, `DES` and `SDSS`:

```python
from hostphot.cutouts import download_images

name = 'SN2004eo'
host_ra, host_dec = 308.2092, 9.92755  # coords of host galaxy of SN2004eo
survey = 'PS1'
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
from hostphot.image_masking import create_mask

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
import hostphot.local_photometry as lp

ap_radii = [3, 4]  # aperture radii in units of kpc
ra, dec =  308.22579, 9.92853 # coords of SN2004eo
z = 0.0157  # redshift

results = lp.multi_band_phot(name, ra, dec, z,
                             survey=survey, ap_radii=ap_radii, 
                             use_mask=True, save_plots=True)
```

If the results return `NaN` values, this means that the flux is below the detection limit for the given survey.

### Global Photometry

Global photometry can be obtained in a similar way to local photometry, using common aperture:

```python
import hostphot.global_photometry as gp

results = gp.multi_band_phot(name, host_ra, host_dec, 
                             survey=survey, ra=ra, dec=dec,
                             use_mask=True, common_aperture=True, 
                             coadd_filters='riz', save_plots=True)
```

## Contributing

To contribute, either open an issue or send a pull request (prefered option). You can also contact me directly.

### Adding other surveys

If you wish to add a survey not implemented in HostPhot, there are a few things needed: where to download the images from (optional), zero-points to convert the images's flux values into magnitudes (e.g. AB magnitudes), the scaling of the images (pixels/arcsec), the filters transmission functions and any other piece of information necessary to properly estimate magnitudes, errors, etc. 


## Citing HostPhot

If you make use of HostPhot, please cite:

```code
@software{hostphot,
  author       = {Tom\'as E. M\"uller-Bravo and
                  Llu\'is Galbany},
  title        = {temuller/hostphot: HostPhot v2},
  month        = may,
  year         = 2022,
  publisher    = {Zenodo},
  version      = {v2.0.0},
  doi          = {10.5281/zenodo.6586565},
  url          = {https://doi.org/10.5281/zenodo.6586565}
}
```
