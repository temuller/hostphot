import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from pyvo.dal import sia
import astropy.units as u
from astropy.io import fits

import hostphot
hostphot_path = Path(hostphot.__path__[0])
from hostphot.utils import open_fits_from_url
from hostphot.surveys_utils import get_survey_filters, check_filters_validity

def get_SPLUS_urls(ra: float, dec: float, fov: float | u.Quantity = 3, 
                    filters: Optional[str | list] = None) -> list[str] | None:
    """Obtains the URLs of the S-Plus images.
    
    The available filters are: 'F378', 'F395', 'F410', 'F430', 
    'F515', 'F660', 'F861', 'G', 'I', 'R', 'U', 'Z'.

    Parameters
    ----------
    ra: Right ascension in degrees.
    dec: Declination in degrees.
    fov: Field of view in degrees.
    filters: Filters to use. If ``None``, uses all available filters.

    Returns
    -------
    url_list: List of URLs with S-Plus images.
    """
    splus_url = "https://datalab.noirlab.edu/sia/splus_dr1"
    svc = sia.SIAService(splus_url)
    imgs_table = svc.search(
        (ra, dec), (fov / np.cos(dec * np.pi / 180), fov), verbosity=2
    )
    if len(imgs_table) == 0:
        print(("Warning: empty table returned for " f"ra={ra}, dec={dec}"))
        return None

    imgs_df = pd.DataFrame(imgs_table)
    imgs_df = imgs_df[imgs_df.access_format.str.endswith('fits')]
    if len(imgs_df) == 0:
        return None
    url_list = []
    for filt in filters:
        filt_df = imgs_df[imgs_df.obs_bandpass==filt]
        if len(filt_df) == 0:
            url_list.append(None)
        else:
            fits_url = filt_df.access_url.values[0]  # first image
            url_list.append(fits_url)    
    return url_list

def get_SPLUS_images(ra: float, dec: float, size: float | u.Quantity = 3, 
                    filters: Optional[str] = None) -> fits.HDUList | None:
    """Gets S-Plus fits images for the given coordinates and
    filters.

    The available filters are: 'F378', 'F395', 'F410', 'F430', 
    'F515', 'F660', 'F861', 'G', 'I', 'R', 'U', 'Z'.

    Parameters
    ----------
    ra: Right Ascension in degrees.
    dec: Declination in degrees.
    size: Image size. If a float is given, the units are assumed to be arcmin.
    filters: Filters to use. If ``None``, uses all available filters.

    Returns
    -------
    hdu_list: List of fits images.
    """
    survey = "SPLUS"
    if filters is None:
        filters = get_survey_filters(survey)
    check_filters_validity(filters, survey)
    if isinstance(size, (float, int)):
        fov = (size * u.arcmin).to(u.degree).value
    else:
        fov = size.to(u.degree).value

    url_list = get_SPLUS_urls(ra, dec, fov, filters)
    if url_list is None:
        return None
    hdu_list = []
    for url in url_list:
        if url is None:
            hdu_list.append(None)
        else:
            hdu = open_fits_from_url(url)
            # add zeropoint
            # file from https://splus.cloud/documentation/dr2_3
            zps_file = hostphot_path.joinpath('filters', 'SPLUS', 'iDR3_zps.cat')
            zps_df = pd.read_csv(zps_file, sep='\\s+')
            field = hdu[0].header['OBJECT'].replace('_', '-')
            field_df = zps_df[zps_df['#field']==field]
            img_filt = hdu[0].header['FILTER']
            zp = field_df[img_filt].values[0]
            hdu[0].header['MAGZP'] = zp
            # initial EXPTIME is normalised, so it doesn't help
            hdu[0].header['EXPTIME'] = hdu[0].header['TEXPOSED']
            hdu_list.append(hdu)
    return hdu_list