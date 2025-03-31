import numpy as np
from typing import Optional

from astropy.wcs import WCS
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astroquery.sdss import SDSS

from hostphot.surveys_utils import get_survey_filters, check_filters_validity, survey_pixel_scale

import warnings
from astropy.utils.exceptions import AstropyWarning

def get_SDSS_images(ra: float, dec: float, size: float | u.Quantity = 3, 
                    filters: Optional[str] = None, version: Optional[str] = None) -> fits.HDUList | None:
    """Downloads a set of SDSS fits images for a given set
    of coordinates and filters using SkyView.

    Parameters
    ----------
    ra: Right Ascension in degrees.
    dec: Declination in degrees.
    size: Image size. If a float is given, the units are assumed to be arcmin.
    filters: Filters to use. If ``None``, uses ``ugriz``.
    version: Data release version to use. If not given, use the latest one (``dr17``).

    Returns
    -------
    hdu_list: List with fits images for the given filters. ``None`` is returned if no image is found.
    """
    # check filters
    survey = "SDSS"
    if filters is None:
        filters = get_survey_filters(survey)
    check_filters_validity(filters, survey)
    # check data release version
    if version is None:
        version = "dr17"
    versions = [f"dr{i}" for i in range(12, 17 + 1)]
    assert (
        version in versions
    ), f"The given version ({version}) is not a valid data release: {versions}"
    dr = int(version.replace("dr", ""))
    # get size in pixels
    if isinstance(size, (float, int)):
        size_arcsec = (size * u.arcmin).to(u.arcsec)
    else:
        size_arcsec = size.to(u.arcsec)
    pixel_scale = survey_pixel_scale(survey, "g")  # same pixel scale for all filters
    size_pixels = int(size_arcsec.value / pixel_scale)
    # get SDSS ids near the given coordinates
    coords = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)
    for radius in np.arange(1, 60, 1):
        ids = SDSS.query_region(
            coords, radius=radius * u.arcsec, data_release=dr
        )
        if ids is not None:
            if 'ra' in ids.colnames and 'dec' in ids.colnames:
                break
            else:
                ids = None
    if ids is None:
        return None
    # get the pointing closest to the given coordinates
    coords_imgs = SkyCoord(
        ra=ids["ra"].value,
        dec=ids["dec"].value,
        unit=(u.degree, u.degree),
        frame="icrs",
    )
    separation = coords.separation(coords_imgs).value
    pointing_id = np.argmin(separation)
    ids2remove = list(np.arange(len(separation)))
    del ids2remove[pointing_id]
    ids.remove_rows(ids2remove)
    # download images
    hdu_list = SDSS.get_images(matches=ids, band=filters, data_release=dr)
    # SDSS images are large so need to be trimmed
    for hdu in hdu_list:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AstropyWarning)
            img_wcs = WCS(hdu[0].header)
        trimmed_data = Cutout2D(hdu[0].data, coords, size_pixels, img_wcs)
        hdu[0].data = trimmed_data.data
        hdu[0].header.update(trimmed_data.wcs.to_header())
    return hdu_list