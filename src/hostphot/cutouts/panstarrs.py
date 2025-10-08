import numpy as np
from typing import Optional

from astropy.io import fits
from astropy import units as u
from astropy.table import Table

from hostphot.utils import open_fits_from_url
from hostphot.surveys_utils import get_survey_filters, check_filters_validity, survey_pixel_scale

def query_ps1(ra: float, dec: float, size: float | u.Quantity = 3, filters: Optional[str] = None) -> Table:
    """Query 'ps1filenames.py' service to get a list of images

    Parameters
    ----------
    ra: Right Ascension in degrees.
    dec: Declination in degrees.
    size: Image size. If a float is given, the units are assumed to be arcmin.
    filters: Filters to use. If ``None``, uses ``grizy``.

    Returns
    -------
    table: Astropy table with the results.
    """
    # check filters
    survey = "PanSTARRS"
    check_filters_validity(filters, survey)
    if filters is None:
        filters = get_survey_filters(survey)
    if isinstance(filters, list):
        filters = "".join(filt for filt in filters)
    # get size in pixels (all filters have the same size)
    pixel_scale = survey_pixel_scale(survey, "g")
    if isinstance(size, (float, int)):
        size_arcsec = (size * u.arcmin).to(u.arcsec).value
    else:
        size_arcsec = size.to(u.arcsec).value
    size_pixels = int(size_arcsec / pixel_scale)
    # download table
    service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    url = (
        f"{service}?ra={ra}&dec={dec}&size={size_pixels}&format=fits&"
        f"filters={filters}"
    )

    return Table.read(url, format="ascii")

def get_PS1_urls(ra: float, dec: float, size: float | u.Quantity = 3, filters: Optional[str] = None) -> list[str]:
    """Get URLs for images obtained with :func:`query_ps1()`.

    Parameters
    ----------
    ra: Right Ascension in degrees.
    dec: Declination in degrees.
    size: Image size. If a float is given, the units are assumed to be arcmin.
    filters: Filters to use. If ``None``, uses ``grizy``.

    Returns
    -------
    url_list: List of URLs for the fits images.
    """
    # check filters
    survey = "PanSTARRS"
    check_filters_validity(filters, survey)
    if filters is None:
        filters = get_survey_filters(survey)
    # get size in pixels (all filters have the same size)
    pixel_scale = survey_pixel_scale(survey, "g")
    if isinstance(size, (float, int)):
        size_arcsec = (size * u.arcmin).to(u.arcsec).value
    else:
        size_arcsec = size.to(u.arcsec).value
    size_pixels = int(size_arcsec / pixel_scale)
    # get PS1 table with ULRs
    table = query_ps1(ra, dec, size=size, filters=filters)
    # sort filters from blue to red
    flist = ["grizy".find(x) for x in table["filter"]]
    table = table[np.argsort(flist)]
    # get list of URLs for the images
    url = (
        "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
        f"ra={ra}&dec={dec}&size={size_pixels}&format=fits"
    )
    base_url = url + "&red="
    url_list = []
    for filename in table["filename"]:
        url_list.append(base_url + filename)
    return url_list

def get_PanSTARRS_images(ra: float, dec: float, size: float | u.Quantity = 3, filters: Optional[str] = None) -> fits.HDUList:
    """Gets PanSTARRS fits images for the given coordinates and
    filters.

    Parameters
    ----------
    ra: Right Ascension in degrees.
    dec: Declination in degrees.
    size: Image size. If a float is given, the units are assumed to be arcmin.
    filters: Filters to use. If ``None``, uses ``grizy``.

    Returns
    -------
    fits_files: List of fits images.
    """
    # check filters
    survey = "PanSTARRS"
    if filters is None:
        filters = get_survey_filters(survey)
    check_filters_validity(filters, survey)
    # get URLs
    fits_url = get_PS1_urls(ra, dec, size, filters)
    # download FITS images
    hdu_list = []
    for url, filt in zip(fits_url, filters):
        hdu = open_fits_from_url(url)
        hdu_list.append(hdu)
    return hdu_list
