
import re
import requests
from typing import Optional

import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord

from hostphot.utils import open_fits_from_url
from hostphot.surveys_utils import get_survey_filters, check_filters_validity, survey_pixel_scale

    
def get_2MASS_images(
    ra: float, dec: float, size: float | u.Quantity = 3, filters: Optional[str] = None
) -> fits.HDUList:
    """Downloads a set of 2MASS fits images for a given set
    of coordinates and filters.

    Parameters
    ----------
    ra: Right ascension in degrees.
    dec: Declination in degrees.
    size: Image size. If a float is given, the units are assumed to be arcmin.
    filters:  Filters to use. If ``None``, uses ``FUV, NUV``.

    Return
    ------
    hdu_list: List with fits images for the given filters. ``None`` is returned if no image is found.
    """
    survey = "2MASS"
    if filters is None:
        filters = get_survey_filters(survey)
    check_filters_validity(filters, survey)

    if isinstance(size, (float, int)):
        size_degree = (size * u.arcmin).to(u.degree)
    else:
        size_degree = size.to(u.degree)
    size_degree = size_degree.value
    coords = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame="icrs")
    
    # this code was taken from chatgpt and combined with that from Blast
    # Base URL for 2MASS cutout service
    base_url = "https://irsa.ipac.caltech.edu/cgi-bin/2MASS/IM/nph-im_sia"
    # Query parameters
    params = {
        "POS": f"{ra},{dec}",
        #"SIZE": size_degree, 
        "SIZE": 0.01, 
        "FORMAT": "image/fits"
    }
    # Make the request to the 2MASS server
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    
    hdu_list = []
    for filter in filters:
        if filter == "Ks":
            filter = "K"
        fits_image = None
        for line in response.content.decode("utf-8").split("<TD><![CDATA["):
            if re.match(f"https://irsa.*{filter.lower()}i.*fits", line.split("]]>")[0]):
                fitsurl = line.split("]]")[0]
                fits_image = open_fits_from_url(fitsurl)
                wcs = WCS(fits_image[0].header)
                if coords.contained_by(wcs):
                    break
        else:
            fits_image = None
        hdu_list.append(fits_image)

    for filt, hdu in zip(filters, hdu_list):
        pixel_scale = survey_pixel_scale(survey, filt)
        size_pixels = int((size_degree * u.deg).to(u.arcsec).value / pixel_scale)
    
        wcs = WCS(hdu[0].header)
        cutout = Cutout2D(hdu[0].data, coords, size_pixels, wcs=wcs)
        hdu[0].data = cutout.data
        hdu[0].header.update(cutout.wcs.to_header())
        
    return hdu_list
