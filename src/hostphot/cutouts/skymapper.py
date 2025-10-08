import numpy as np
import pandas as pd

import astropy.units as u
from astropy.io import fits

from pyvo.dal import sia
from hostphot.utils import open_fits_from_url
from hostphot.surveys_utils import get_survey_filters, check_filters_validity

def get_SkyMapper_urls(ra: float, dec: float, fov: float | u.Quantity = 3, 
                    filters: str = "uvgriz") -> list[str] | None:
    """Obtains the URLs of the SkyMapper images.
    
    The images closest to the given coordinates are retrieved 

    Parameters
    ----------
    ra: Right ascension in degrees.
    dec: Declination in degrees.
    fov: Field of view in degrees.
    filters: Filters to use. If ``None``, uses ``uvgriz``.

    Returns
    -------
    url_list: List of URLs with SkyMapper images.
    """
    sm_url = "https://api.skymapper.nci.org.au/public/siap/dr4/query"
    svc = sia.SIAService(sm_url)
    imgs_table = svc.search(
        (ra, dec), (fov / np.cos(dec * np.pi / 180), fov), verbosity=2
    )
    if len(imgs_table) == 0:
        print(("Warning: empty table returned for " f"ra={ra}, dec={dec}"))
        return None
    imgs_df = pd.DataFrame(imgs_table)
    imgs_df = imgs_df[imgs_df.format.str.endswith('fits')]
    if len(imgs_df) == 0:
        return None

    url_list = []
    for filt in filters:
        filt_df = imgs_df[imgs_df.band==filt]
        # obtain the images with largest exposure times and file size
        # and lowest mean FWHM, as suggested by Christopher Onken, from the SkyMapper Team
        filt_df = filt_df[filt_df.exptime==filt_df.exptime.max()]
        filt_df = filt_df[filt_df.filesize==filt_df.filesize.max()]
        filt_df = filt_df[filt_df.mean_fwhm==filt_df.mean_fwhm.min()]
        if len(filt_df) == 0:
            url_list.append(None)
        else:
            fits_url = filt_df.get_fits.values[0]
            url_list.append(fits_url)    
    return url_list

def get_SkyMapper_images(ra: float, dec: float, size: float | u.Quantity = 3, 
                    filters: str = "uvgriz") -> fits.HDUList | None:
    """Gets SkyMapper DR4 fits images for the given coordinates and
    filters.

    Parameters
    ----------
    ra: Right Ascension in degrees.
    dec: Declination in degrees.
    size: Image size. If a float is given, the units are assumed to be arcmin.
    filters: Filters to use. If ``None``, uses ``uvgriz``.

    Returns
    -------
    hdu_list: List of fits images.
    """
    # initial checks
    survey = "SkyMapper"
    if filters is None:
        filters = get_survey_filters(survey)
    check_filters_validity(filters, survey)
    if isinstance(size, (float, int)):
        fov = (size * u.arcmin).to(u.degree).value
    else:
        fov = size.to(u.degree).value
    # download images
    url_list = get_SkyMapper_urls(ra, dec, fov, filters)
    if url_list is None:
        return None
    hdu_list = []
    for url in url_list:
        if url is None:
            hdu_list.append(None)
        else:
            hdu = open_fits_from_url(url)
            # add zeropoint
            hdu[0].header['MAGZP'] = hdu[0].header['ZPAPPROX']
            hdu_list.append(hdu)
    return hdu_list