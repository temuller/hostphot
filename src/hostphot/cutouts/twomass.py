import pyvo
import numpy as np
from typing import Optional

import astropy.units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord

from hostphot.surveys_utils import get_survey_filters, check_filters_validity

import warnings
from astropy.utils.exceptions import AstropyWarning


def _get_2MASS_images(
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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        coords = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame="icrs")
        twomass_services = pyvo.regsearch(servicetype="image", keywords=["2mass"])
        table = twomass_services[0].search(pos=coords, size=size_degree)
        twomass_df = table.to_table().to_pandas()
        twomass_df = twomass_df[twomass_df.format == "image/fits"]

        # for more info: https://irsa.ipac.caltech.edu/ibe/docs/twomass/allsky/allsky/#main
        base_url = "https://irsa.ipac.caltech.edu/ibe/data/twomass/allsky/allsky"

        hdu_list = []
        for filt in filters:
            if filt == "Ks":
                filt = "K"
            band_df = twomass_df[twomass_df.band == filt]
            if len(band_df) == 0:
                # no data for this band:
                hdu_list.append(None)
                continue

            # pick the largest images which is also the most "square" one
            # this works better than picking the image closest to the given coordinates
            # don't know why
            sizes = []
            tmp_hdu_list = []
            for i in range(len(band_df)):
                fname = band_df.download.values[i].split("=")[-1]
                hemisphere = band_df.hem.values[i]
                ordate = band_df.date.values[i]
                scanno = band_df.scan.values[i]
                # add leading zeros for scanno bellow 100
                n_zeros = 3 - len(str(scanno))
                scanno = n_zeros * "0" + str(scanno)

                tile_url = f"/{ordate}{hemisphere}/s{scanno}"
                fits_url = f"/image/{fname}.gz"
                params_url = f"center={ra},{dec}&size={size_degree}degree&gzip=0"  # center and size of the image
                url = base_url + tile_url + fits_url + "?" + params_url
                try:
                    hdu = fits.open(url)
                    ny, nx = hdu[0].data.shape
                    sizes.append(nx * ny)
                    tmp_hdu_list.append(hdu)
                except:
                    # some images might give 500 Internal Server Error
                    # because the cutout does not overlap the image
                    pass
            if len(tmp_hdu_list) == 0:
                hdu_list.append(None)
            else:
                # pick largest image, which usually is the best
                i = np.argmax(sizes)
                hdu_list.append(tmp_hdu_list[i])
    return hdu_list

import requests
import re
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy import units as u
from hostphot.surveys_utils import survey_pixel_scale
    
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
    filters = [filt if filt!="Ks" else "K" for filt in filters]

    if isinstance(size, (float, int)):
        size_degree = (size * u.arcmin).to(u.degree)
    else:
        size_degree = size.to(u.degree)
    size_degree = size_degree.value
    
    pixel_scale = survey_pixel_scale(survey)
    size_pixels = int((size_degree * u.deg).to(u.arcsec).value / pixel_scale)
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
        fits_image = None
        for line in response.content.decode("utf-8").split("<TD><![CDATA["):
            if re.match(f"https://irsa.*{filter.lower()}i.*fits", line.split("]]>")[0]):
                fitsurl = line.split("]]")[0]
                fits_image = fits.open(fitsurl, cache=None)
                wcs = WCS(fits_image[0].header)
                if coords.contained_by(wcs):
                    break
        else:
            fits_image = None
        hdu_list.append(fits_image)

    for hdu in hdu_list:
        wcs = WCS(hdu[0].header)
        cutout = Cutout2D(hdu[0].data, coords, size_pixels, wcs=wcs)
        hdu[0].data = cutout.data
        hdu[0].header.update(cutout.wcs.to_header())
        
    return hdu_list
