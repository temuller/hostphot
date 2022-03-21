# HostPhot

## Conda environment

It is recommended to create an environment for every new project:

```code
conda create -n hostphot pip
conda activate hostphot
pip install hostphot
```

## Modules

### Cutouts

This module allows you to download image cutouts from `PS1`, `DES` and `SDSS`. For this, you can use `get_PS1_images()`, `get_DES_images()` and `get_SDSS_images()`, respectively. For example:

```python
from hostphot.cutouts import get_PS1_images

ra, dec = 30, 100
size = 400  # in pixels
filters = 'grizy'

fits_images = get_PS1_images(ra, dec, size, filters)
```

where `fits_images` is a list with the fits images in the given filters.

You can also use `download_multiband_images()` for multiple images:

```python
from hostphot.cutouts import download_multiband_images

download_multiband_images(sn_name, ra, dec, size,
                                work_dir, filters,
                                  overwrite, survey)
```

where `work_dir` is where all the images will be downloaded. A Subdirectory inside `work_dir` will be created with the SN name as the directory name.


### Local Photometry

Local photometry can be obtained for the downloaded images. For this, use `extract_local_photometry()` for a single image:


```python
from hostphot.local_photometry import extract_local_photometry

fits_file = 'path/to/local/fits_file'
ra, dec = 30, 100
z = 0.01  # redshift
ap_radius = 4  # aperture for the photometry in kpc
survey = 'PS1'

extract_local_photometry(fits_file, ra, dec, z, ap_radius, survey)
```

which returns `mag` and `mag_err`. You can also use `multi_local_photometry()` for multiple images:


```python
from hostphot.local_photometry import multi_local_photometry

multi_local_photometry(name_list, ra_list, dec_list, z_list,
                             ap_radius, work_dir, filters,
                               survey, correct_extinction)
```

where `work_dir` should be the same as used in `download_multiband_images()` and `name_list` should contain the names of the SNe used in `download_multiband_images()` as well. This produces a pandas DataFrame as an output where, e.g., column `g` is the g-band magnitude and `g_err` its uncertainty.


### Global Photometry

Global photometry can be obtained in a similar way to local photometry. Use `extract_global_photometry()` for a single image:

```python
from hostphot.global_photometry import extract_global_photometry

survey = 'PS1'

extract_global_photometry(fits_file, host_ra, host_ra, survey=survey)
```

which returns `mag` and `mag_err`. You can also use `multi_global_photometry()` for multiple images:


```python
from hostphot.global_photometry import multi_global_photometry

survey = 'PS1'
correct_extinction = True

multi_global_photometry(name_list, host_ra_list, host_dec_list, work_dir, filters,
                               survey=survey, correct_extinction=correct_extinction)
```
