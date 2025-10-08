import astropy.units as u
from astropy.io import fits
from typing import Optional

from hostphot.utils import open_fits_from_url
from hostphot.surveys_utils import get_survey_filters, check_filters_validity, survey_pixel_scale

def get_LegacySurvey_images(ra: float, dec: float, size: float | u.Quantity = 3, 
                        filters: Optional[str] = None, version: str = "dr10") -> list[list[fits.HDUList]]:
    """Gets Legacy Survey fits images for the given coordinates and
    filters.

    Parameters
    ----------
    ra: Right Ascension in degrees.
    dec: Declination in degrees.
    size: Image size. If a float is given, the units are assumed to be arcmin.
    filters: Filters to use. If ``None``, uses ``grz``.
    version: Data release version. E.g. ``dr10``, ``dr9``, ``dr8``, etc..

    Returns
    -------
    fits_files: List of fits images.
    """
    survey = "LegacySurvey"
    if filters is None:
        filters = get_survey_filters(survey)
    check_filters_validity(filters, survey)
    if isinstance(filters, list):
        filters = "".join(filter for filter in filters)

    pixel_scale = survey_pixel_scale(survey, "g")  # same pixel scale for all filters
    if isinstance(size, (float, int)):
        size_arcsec = (size * u.arcmin).to(u.arcsec).value
    else:
        size_arcsec = size.to(u.arcsec).value
    size_pixels = int(size_arcsec / pixel_scale)

    if version is None:
        version = "dr10"  # latest data release

    base_url = "https://www.legacysurvey.org/viewer/fits-cutout?"
    params = f"ra={ra}&dec={dec}&layer=ls-{version}&pixscale={pixel_scale}&bands={filters}&size={size_pixels}&invvar"
    url = base_url + params

    master_hdu = open_fits_from_url(url)
    master_header = master_hdu[0].header
    master_header_invvar = master_hdu[1].header

    hdu_list = []
    for i, filt in enumerate(filters):
        data = master_hdu[0].data[i]
        header = master_header.copy()
        header.append(('BAND', filt, ' Band - added by HostPhot'), end=True)
        hdu = fits.PrimaryHDU(data=data, header=header)
        
        header_invvar = master_header_invvar.copy()
        header_invvar.append(('BAND', filt, ' Band - added by HostPhot'), end=True)
        data_invvar = master_hdu[1].data[i]
        hdu_invvar = fits.ImageHDU(data=data_invvar, 
                                header=header_invvar)
        hdu_list.append(fits.HDUList([hdu, hdu_invvar]))
    master_hdu.close()
    return hdu_list