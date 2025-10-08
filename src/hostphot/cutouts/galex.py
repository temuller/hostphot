import numpy as np
from typing import Optional

from astropy.wcs import WCS
import astropy.units as u
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astroquery.mast import Observations

from hostphot.utils import open_fits_from_url
from hostphot.surveys_utils import get_survey_filters, check_filters_validity, survey_pixel_scale

import warnings
from astropy.utils.exceptions import AstropyWarning

def get_GALEX_images(ra: float, dec: float, size: float | u.Quantity = 3, 
                    filters: Optional[str] = None, version: Optional[str] = None) -> list[fits.ImageHDU]:
    """Downloads a set of GALEX fits images for a given set
    of coordinates and filters.

    Parameters
    ----------
    ra: Right ascension in degrees.
    dec: Declination in degrees.
    size: Image size. If a float is given, the units are assumed to be arcmin.
    filters: Filters to use. If ``None``, uses ``FUV, NUV``.
    version: Version of GALEX images. Either Deep (``DIS``), Medium (``MIS``) or
             All-Sky Imaging Survey (``AIS``), or Nearby Galaxy Survey (``NGS``) or
             Guest Investigator Survey (``GII``). If ``None``, take the image with the
             longest exposure time.

    Returns
    -------
    hdu_list: List with fits images for the given filters. ``None`` is returned if no image is found.
    """
    survey = "GALEX"
    if filters is None:
        filters = get_survey_filters(survey)
    if isinstance(filters, str):
        filters = [filters]
    check_filters_validity(filters, survey)
    # get size in pixels
    if isinstance(size, (float, int)):
        size_arcsec = (size * u.arcmin).to(u.arcsec)
    else:
        size_arcsec = size.to(u.arcsec)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        coords = SkyCoord(
            ra=ra, dec=dec, unit=(u.degree, u.degree), frame="icrs"
        )
    obs_table = Observations.query_criteria(
        coordinates=coords, radius=size_arcsec, obs_collection=["GALEX"]
    )
    obs_df_ = obs_table.to_pandas()

    if version is None:
        # starts from the survey with the deepest images first
        obs_df_ = obs_df_.sort_values("t_exptime", ascending=False)
        projects = obs_df_.project.unique()
    else:
        # only use the survey requested by the user
        projects = [version]

    hdu_dict = {"NUV": None, "FUV": None}
    for project in projects:
        obs_df = obs_df_[obs_df_.project == project]

        for filt in filters:
            if hdu_dict[filt] is not None:
                # if the image was already found, skip this filter
                continue

            # get only "intensity" images
            filt_extension = {
                "NUV": "-nd-int.fits.gz",
                "FUV": "-fd-int.fits.gz",
            }
            # get unique image sectors
            files = []
            for file in obs_df.dataURL.values:
                sector_info = file.split("-")[:-2]
                file = (
                    "-".join(string for string in sector_info)
                    + filt_extension[filt]
                )
                if file not in files:
                    files.append(file)

            # download the FITS images
            hdu_list = []
            for file in files:
                try:
                    hdu = open_fits_from_url(file)
                    hdu_list.append(hdu)
                except:
                    pass

            # calculate the separation of the galaxy to the image center
            separations = []
            for hdu in hdu_list:
                ra_img = float(hdu[0].header["RA_CENT"])
                dec_img = float(hdu[0].header["DEC_CENT"])
                coords_img = SkyCoord(
                    ra=ra_img,
                    dec=dec_img,
                    unit=(u.degree, u.degree),
                    frame="icrs",
                )
                separation = coords.separation(coords_img).value
                separations.append(separation)

            # get the image with the galaxy closest to the center
            if len(separations) != 0:
                id_file = np.argmin(separations)
                hdu = hdu_list[id_file]
            else:
                hdu = None

            hdu_dict[filt] = hdu

    hdu_list = []
    for filt in filters:
        pixel_scale = survey_pixel_scale(survey, filt)
        size_pixels = int(size_arcsec.value / pixel_scale)
    
        hdu = hdu_dict[filt]
        if hdu is None:
            hdu_list.append(None)
            continue  # no image in this filter
        # trim data to requested size
        img_wcs = WCS(hdu[0].header)
        pos = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)

        trimmed_data = Cutout2D(hdu[0].data, pos, size_pixels, img_wcs)
        hdu[0].data = trimmed_data.data
        hdu[0].header.update(trimmed_data.wcs.to_header())
        hdu_list.append(hdu)
    return hdu_list