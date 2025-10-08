import tarfile
import requests
import pandas as pd
from pathlib import Path
from typing import Optional

import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astroquery.skyview import SkyView
from astropy.coordinates import SkyCoord

from hostphot.utils import open_fits_from_url
from hostphot.surveys_utils import get_survey_filters, check_filters_validity, survey_pixel_scale

import warnings
from astropy.utils.exceptions import AstropyWarning


def get_WISE_images(ra: float, dec: float, size: float | u.Quantity = 3, 
                    filters: Optional[str] = None) -> list[fits.ImageHDU]:
    """Downloads a set of WISE fits images for a given set
    of coordinates and filters.

    Parameters
    ----------
    ra: Right ascension in degrees.
    dec: Declination in degrees.
    size: Image size. If a float is given, the units are assumed to be arcmin.
    filters: Filters to use. If ``None``, uses all WISE filters.

    Return
    ------
    hdu_list: List with fits images for the given filters. ``None`` is returned if no image is found.
    """
    survey = "WISE"
    if filters is None:
        filters = get_survey_filters(survey)
    check_filters_validity(filters, survey)

    if isinstance(size, (float, int)):
        size_arcsec = (size * u.arcmin).to(u.arcsec).value
    else:
        size_arcsec = size.to(u.arcsec).value
    pixel_scale = survey_pixel_scale(survey, "W1")  # same pixel scale for all filters
    image_size = size_arcsec / pixel_scale
    coords = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame="icrs")

    base_url = "https://irsa.ipac.caltech.edu/SIA?COLLECTION=wise_allwise"
    params_url = f"&POS=circle+{ra}+{dec}+0.002777&RESPONSEFORMAT=CSV&FORMAT=image/fits"
    response = requests.get(base_url + params_url)
    # transform the response into a dataframe
    split_text = response.text.split("\n")
    url_dict = {key:[] for key in split_text[0].split(",")}
    for row in split_text[1:]:
        if row == "":
            continue
        values = row.split(",")
        for key, value in zip(url_dict.keys(), values):
            url_dict[key].append(value)
    url_df = pd.DataFrame(url_dict)

    hdu_list = []
    for filt in filters:
        filt_df = url_df[url_df.energy_bandpassname==filt]
        if len(filt_df) == 0:
            hdu_list.append(None)
        else:
            img_url = filt_df.access_url.values[0]
            t_expt = filt_df.t_exptime.values[0]
            hdu = open_fits_from_url(img_url)
            hdu[0].header["EXPTIME"] = t_expt
            # create cutout
            wcs = WCS(hdu[0].header)
            cutout = Cutout2D(hdu[0].data, coords, image_size, wcs=wcs)
            hdu[0].data = cutout.data
            hdu[0].header.update(cutout.wcs.to_header())
            hdu_list.append(hdu)
    return hdu_list

def get_unWISE_images(ra: float, dec: float, size: float | u.Quantity = 3, 
                    filters: Optional[str] = None, version: str = "allwise") -> list[fits.ImageHDU]:
    """Downloads a set of unWISE fits images for a given set
    of coordinates and filters.

    Parameters
    ----------
    ra: Right ascension in degrees.
    dec: Declination in degrees.
    size: Image size. If a float is given, the units are assumed to be arcmin.
    filters: Filters to use. If ``None``, uses all WISE filters.
    version:  Version of the unWISE images. Either ``allwise`` or ``neo{i}`` for {i} = 1 to 7.

    Return
    ------
    hdu_list: List with fits images for the given filters. ``None`` is returned if no image is found.
    """
    survey = "unWISE"
    if version is None:
        version = "allwise"
    else:
        # check validity of the version used
        neo_versions = [f"neo{i}" for i in range(1, 8)]
        all_versions = ["allwise"] + neo_versions
        assert (
            version in all_versions
        ), f"Not a valid version ({version}): {all_versions}"
    if filters is None:
        filters = get_survey_filters(survey)
    check_filters_validity(filters, survey)

    if "neo" in version:
        # these only have W1 and W2 data
        if "W3" in filters:
            filters.remove("W3")
        if "W4" in filters:
            filters.remove("W4")

    if isinstance(size, (float, int)):
        size_arcsec = (size * u.arcmin).to(u.arcsec)
    else:
        size_arcsec = size.to(u.arcsec)
    pixel_scale = survey_pixel_scale(survey, "W1")  # same pixel scale for all filters
    size_pixels = int(size_arcsec.value / pixel_scale)
    assert size_pixels <= 1024, "Maximum cutout size for unWISE is 1024 pixels"

    bands = "".join(filt[-1] for filt in filters)  # e.g. 1234
    # for more info: http://unwise.me/imgsearch/
    base_url = "http://unwise.me/cutout_fits?"
    params_url = (
        f"version={version}&ra={ra}&dec={dec}&size={size_pixels}&bands={bands}"
    )
    master_url = base_url + params_url

    response = requests.get(master_url, stream=True)
    target_file = Path(f"unWISE_images_{ra}_{dec}.tar.gz")  # current directory
    if response.status_code == 200:
        with open(target_file, "wb") as f:
            f.write(response.raw.read())

    hdu_list = []
    with tarfile.open(target_file) as tar_file:
        files_list = tar_file.getnames()
        for fits_file in files_list:
            for filt in filters:
                if f"{filt.lower()}-img-m.fits" in fits_file:
                    tar_file.extraction_filter = (lambda member, path: member)
                    tar_file.extract(fits_file, ".")
                    hdu = fits.open(fits_file)
                    hdu_list.append(hdu)
                    Path(fits_file).unlink()  # remove file
    # remove the tarfile
    if target_file.is_file() is True:
        target_file.unlink()
    return hdu_list

def get_used_image(header: fits.header.Header) -> str:
    """Obtains the name of the image downloaded by SkyView.

    Parameters
    ----------
    header: fits header
        Header of an image.

    Returns
    -------
    used_image: str
        Name of the image.
    """
    image_line = header["HISTORY"][-3]
    used_image = image_line.split("/")[-1].split("-")[0]
    return used_image