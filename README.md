# HostPhot

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

## Modules

### Cutouts

This module allows you to download image cutouts from `PS1`, `DES` and `SDSS`:

```python
from hostphot.cutouts import download_images
download_images(name='SN2004eo', ra=308.22579, dec=9.92853, survey='PS1')
```

### Image Pre-processing

Coadds can be created and stars can be masked out of the images:

```python
from hostphot.coadd import coadd_images

coadd_filters = 'riz'
survey = 'PS1'
coadd_images('SN2004eo', coadd_filters, survey)   # creates a new fits file
```

```python
from hostphot.image_masking import create_mask

host_ra, host_dec = 308.2092, 9.92755  # coods of host galaxy of SN2004eo
coadd_mask_params = create_mask(name, host_ra, host_dec,
                                 filt=coadd_filters, survey=survey,
                                 extract_params=True)  # we can extract the mask parameters from the coadd

for filt in 'grizy':
    create_mask(name, host_ra, host_dec, filt, survey=survey,
                common_params=coadd_mask_params)
```

### Local Photometry

Local photometry can be obtained for multiple circular apertures:


```python
import hostphot.local_photometry as lp

ap_radii = [1, 2, 3, 4]  # in units of kpc
results = lp.multi_band_phot(name='SN2004eo', ra=308.22579, dec=9.92853, z=0.0157,
                   survey='PS1', ap_radii=ap_radii, use_mask=True, save_plots=True)
```

### Global Photometry

Global photometry can be obtained in a similar way to local photometry, using common aperture:

```python
import hostphot.global_photometry as gp

host_ra, host_dec = 308.2092, 9.92755  # coods of host galaxy of SN2004eo
results = gp.multi_band_phot(name='SN2004eo', host_ra, host_dec, survey='PS1',
                            use_mask=True, common_aperture=True, coadd_filters='riz',
                            optimze_kronrad=True, save_plots=True)

```

## Citing HostPhot

If you make use of HostPhot, please cite:

```code
@software{hostphot,
  author       = {Tom\'as E. M\"uller-Bravo and
                  Llu\'is Galbany},
  title        = {HostPhot},
  month        = apr,
  year         = 2022,
  publisher    = {Zenodo},
  version      = {v1.0.1},
  doi          = {10.5281/zenodo.6469981},
  url          = {https://doi.org/10.5281/zenodo.6469981}
}
```
