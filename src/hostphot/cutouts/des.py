import numpy as np
import pandas as pd
from typing import Optional

from pyvo.dal import sia
from astropy.io import fits
from astropy import units as u

from hostphot.utils import open_fits_from_url
from hostphot.surveys_utils import get_survey_filters, check_filters_validity


def get_DES_urls(ra: float, dec: float, fov: float, filters: str="grizY") -> tuple[list[str], list[str]]:
    """Obtains the URLs of the DES images+weights with the
    largest exposure times in the given filters.

    Parameters
    ----------
    ra: Right Ascension in degrees.
    dec: Declination in degrees.
    fov: Field of view in degrees.
    filters: Filters to uses.

    Returns
    -------
    url_list:  List of URLs with DES images.
    url_w_list: List of URLs with DES images weights.
    """
    # check filters
    if filters is None:
        filters = get_survey_filters("DES")
    check_filters_validity(filters, "DES")
    # get field-of-view table
    des_access_url = "https://datalab.noirlab.edu/sia/des_dr2"
    svc = sia.SIAService(des_access_url)
    imgs_table = svc.search(
        (ra, dec), (fov / np.cos(dec * np.pi / 180), fov), verbosity=2
    )
    if len(imgs_table) == 0:
        print(("Warning: empty table returned for " f"ra={ra}, dec={dec}"))
        return None, None
    # conditions to select images of interest
    imgs_df = pd.DataFrame(imgs_table)
    cond_stack = imgs_df["proctype"] == "Stack"
    cond_image = imgs_df["prodtype"] == "image"
    cond_weight = imgs_df["prodtype"] == "weight"

    url_list = []
    url_w_list = []
    for filt in filters:
        # gather images of interes
        cond_filt = imgs_df["obs_bandpass"] == filt
        selected_images = imgs_df[cond_filt & (cond_stack & cond_image)]
        selected_weights = imgs_df[cond_filt & (cond_stack & cond_weight)]
        if len(selected_images) > 0:
            # pick image with the longest exposure time
            max_exptime = np.argmax(selected_images["exptime"])
            row = selected_images.iloc[max_exptime]
            url = row["access_url"]  # get the download URL
            if len(selected_weights) == 0:
                url_w = None
            else:
                max_exptime = np.argmax(selected_weights["exptime"])
                row_w = selected_weights.iloc[max_exptime]
                url_w = row_w["access_url"]
        else:
            url = url_w = None
        url_list.append(url)
        url_w_list.append(url_w)
    return url_list, url_w_list

def get_DES_images(ra: float, dec: float, size: float | u.Quantity = 3, filters: Optional[str] = None) -> list[fits.HDUList] | None:
    """Gets DES fits images for the given coordinates and
    filters.

    Parameters
    ----------
    ra: Right Ascension in degrees.
    dec: Declination in degrees.
    size: Image size. If a float is given, the units are assumed to be arcmin.
    filters: Filters to use. If ``None``, uses ``grizY``.

    Returns
    -------
    hdu_list: List of fits images. ``None`` if the images are not found.
    """
    # check filters
    survey = "DES"
    if filters is None:
        filters = get_survey_filters(survey)
    check_filters_validity(filters, survey)
    # get image size in degrees
    if isinstance(size, (float, int)):
        fov = (size * u.arcmin).to(u.degree).value
    else:
        fov = size.to(u.degree).value
    # get URLs
    url_list, url_w_list = get_DES_urls(ra, dec, fov, filters)
    if url_list is None:
        return None
    # download images
    hdu_list = []
    for url, url_w in zip(url_list, url_w_list):
        # combine image+weights on a single fits file
        #image_hdu = fits.open(url, timeout=120)
        image_hdu = open_fits_from_url(url)
        hdu = fits.PrimaryHDU(image_hdu[0].data, header=image_hdu[0].header)
        if url_w is None:
            hdu_sublist = fits.HDUList([hdu])
        else:
            print(url_w)
            #weight_hdu = fits.open(url_w, timeout=120)
            weight_hdu = open_fits_from_url(url_w)
            hdu_err = fits.ImageHDU(
                weight_hdu[0].data, header=weight_hdu[0].header
            )
            hdu_sublist = fits.HDUList([hdu, hdu_err])
            weight_hdu.close()
        hdu_list.append(hdu_sublist)
        image_hdu.close()
    return hdu_list