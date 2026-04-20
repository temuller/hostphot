from astropy import wcs
import astropy.units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astroquery.ukidss import Ukidss

from hostphot.utils import open_fits_from_url, open_fits_from_urls
from hostphot.surveys_utils import get_survey_filters, check_filters_validity, survey_pixel_scale

import warnings
from astropy.utils.exceptions import AstropyWarning

def get_UKIDSS_images(ra: float, dec: float, size: float=3, filters: str='ZYJHK') -> list[fits.ImageHDU]:
    """Gets UKIDSS fits images for the given coordinates and
    filters.

    Parameters
    ----------
    ra: Right Ascension in degrees.
    dec: Declination in degrees.
    size: Image size. If a float is given, the units are assumed to be arcmin.
    filters: Filters to use. If ``None``, uses ``ZYJHK``.

    Returns
    -------
    fits_files: list
        List of fits images.
    """
    survey = "UKIDSS"
    if filters is None:
        filters = get_survey_filters(survey)
    check_filters_validity(filters, survey)
    
    database = 'UKIDSSDR11PLUS'
    # programme = 'LAS'  # ['UDS', 'GCS', 'GPS', 'DXS', 'LAS']
    
    survey_pixel_scale(survey, "H")  # same scale for all pixels, except J for LAS
    if isinstance(size, (float, int)):
        size = (size * u.arcmin)

    u_obj = Ukidss(database=database)
    pos = SkyCoord(ra, dec, unit="deg")
    urls = u_obj.get_image_list(pos, waveband='all', frame_type="stack", image_width=size)

    # select URLs for each filter
    url_list = []
    for filt in filters:
        found_url = None
        for url in urls:
            if f'band={filt}' in url:
                found_url = url
                break
        url_list.append(found_url)

    hdu_list_all = open_fits_from_urls(url_list)

    hdu_list = []
    for hdu in hdu_list_all:
        if hdu is not None:
            # update first extension with data and WCS
            hdu[0].data = hdu[1].data
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", AstropyWarning)
                img_wcs = wcs.WCS(hdu[1].header)
                hdu[0].header.update(img_wcs.to_header())
            # add some of the keywords to the PHU
            hdu[0].header['MAGZRR'] = hdu[1].header['MAGZRR']
            hdu[0].header['GAIN'] = hdu[1].header['GAIN']
            hdu[0].header['READNOIS'] = hdu[1].header['READNOIS']
            # calculate effective ZP (considering atmospheric extinction)
            # calculate extinction first
            airmass = (hdu[0].header['AMSTART'] + hdu[0].header['AMEND'])/2
            ext_coeff = hdu[1].header['EXTINCT']
            extinction = ext_coeff*(airmass - 1)
            # calculate effective ZP
            zp = hdu[1].header['MAGZPT']
            hdu[0].header['MAGZP'] = zp - extinction
        hdu_list.append(hdu)
    return hdu_list