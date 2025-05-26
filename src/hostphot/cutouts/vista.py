import re
import urllib
from typing import Optional

from astropy import wcs
import astropy.units as u
from astropy.io import fits

from hostphot.surveys_utils import get_survey_filters, check_filters_validity
from hostphot.moc.maps import contains_coords

import warnings
from astropy.utils.exceptions import AstropyWarning


def get_VISTA_images(ra: float, dec: float, size: float | u.Quantity = 3, 
                        filters: Optional[str] = None) -> list[fits.ImageHDU]:
    """Gets VISTA fits images for the given coordinates and
    filters.

    Note: the different surveys included in VISTA cover different
    parts of the sky and do not necessarily contain the same filters.
    
    The available surveys are "VIDEO", "VIKING", "VHS", and chosen in
    that order, depending on the coverage.

    Parameters
    ----------
    ra: float
        Right Ascension in degrees.
    dec: float
        Declination in degrees.
    size: float or ~astropy.units.Quantity, default ``3``
        Image size. If a float is given, the units are assumed to be arcmin.
    filters: str, default ``None``
        Filters to use. If ``None``, uses ``Z, Y, J, H, Ks``.
    version: str, default ``VHS``
        Survey to use: ``VHS``, ``VIDEO`` or ``VIKING``.

    Returns
    -------
    fits_files: list
        List of fits images.
    """
    survey = "VISTA"
    if filters is None:
        filters = get_survey_filters(survey)
    check_filters_validity(filters, survey)
    if not isinstance(size, (float, int)):
        size = size.to(u.arcmin).value

    for version in ["VIDEO", "VIKING", "VHS"]:
        overlap = contains_coords(ra, dec, version)
        if overlap is True:
            break
    if overlap is False:
        return [False] * len(filters)
    
    # Latest data releases
    database_dict = {
        "VHS": "VHSDR6",
        "VIDEO": "VIDEODR6",
        "VIKING": "VIKINGDR5",
    }
    database = database_dict[version]

    base_url = "http://vsa.roe.ac.uk:8080/vdfs/GetImage?archive=VSA&"
    survey_dict = {
        "database": database,
        "ra": ra,
        "dec": dec,
        "sys": "J",
        "filterID": "all",
        "xsize": size,  # in arcmin
        "ysize": size,  # in arcmin
        "obsType": "object",
        "frameType": "tilestack",
    }
    survey_url = "&".join([f"{key}={val}" for key, val in survey_dict.items()])
    url = base_url + survey_url
    results = urllib.request.urlopen(url).read()
    links = re.findall('href="(http://.*?)"', results.decode("utf-8"))
    
    # find url for each filter (None if not found)
    urls_dict = {filt: [] for filt in filters}
    for filt in filters:
        for link in links:
            url = link.replace("getImage", "getFImage", 1)
            if f"band={filt}" in url:
                urls_dict[filt].append(url)
    
    hdu_list = []
    for filt, url_list in urls_dict.items():
        if len(url_list) > 0:
            exptime = 0
            # select image with longest exposure
            for url in url_list:
                hdu_ = fits.open(url)
                if hdu_[1].header["EXPTIME"] > exptime:
                    exptime = hdu_[1].header["EXPTIME"]
                    hdu = hdu_
                else:
                    hdu_.close()
            hdu[0].data = hdu[1].data
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", AstropyWarning)
                img_wcs = wcs.WCS(hdu[1].header)
                hdu[0].header.update(img_wcs.to_header())
            # add some keywords to the PHU
            hdu[0].header['EXPTIME'] = hdu[1].header['EXPTIME']
            hdu[0].header['MAGZRR'] = hdu[1].header['MAGZRR']
            # calculate effective ZP (considering atmospheric extinction)
            # calculate extinction first
            airmass = (hdu[1].header['HIERARCH ESO TEL AIRM START'] + hdu[1].header['HIERARCH ESO TEL AIRM END'])/2
            ext_coeff = hdu[1].header['EXTINCT']
            extinction = ext_coeff*(airmass - 1)
            # calculate effective ZP
            zp = hdu[1].header['MAGZPT']
            hdu[0].header['MAGZP'] = zp - extinction
            hdu_list.append(hdu)
        else:
            hdu_list.append(None)
    return hdu_list