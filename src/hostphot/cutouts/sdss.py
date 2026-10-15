import numpy as np
from astropy.table import Table
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

def _best_frame(ra: float, dec: float, size_arcmin: float, dr: int) -> Table | None:
    """Chooses the SDSS frame (run, rerun, camcol and field) containing the given position where
    a cutout of the given size fits best, using the Field table.

    Among the frames where the whole cutout fits (margin to the frame edges of at least half the
    cutout size), the one with the best quality and largest margin is chosen; if the cutout does not
    fit in any frame, the one with the largest margin is chosen.

    Parameters
    ----------
    ra: Right Ascension in degrees.
    dec: Declination in degrees.
    size_arcmin: Cutout size in arcmin.
    dr: Data release.

    Returns
    -------
    frame: Table with the run, rerun, camcol and field of the frame, or ``None`` if not found.
    """
    query = (
        "SELECT run, rerun, camcol, field, raMin, raMax, decMin, decMax, quality FROM Field "
        f"WHERE raMin < {ra} AND raMax > {ra} AND decMin < {dec} AND decMax > {dec}"
    )
    try:
        frames = SDSS.query_sql(query, data_release=dr)
    except Exception:
        return None
    if frames is None or len(frames) == 0:
        return None
    cos_dec = np.cos(np.radians(dec))
    margins = 60 * np.min(
        [
            (ra - frames["raMin"]) * cos_dec,
            (frames["raMax"] - ra) * cos_dec,
            dec - frames["decMin"],
            frames["decMax"] - dec,
        ],
        axis=0,
    )
    fits_frame = margins >= size_arcmin / 2
    if fits_frame.any():
        # frames where the cutout fits, then the best quality (3: good, 2: acceptable),
        # then the largest margin
        order = np.lexsort((-margins, -np.asarray(frames["quality"]), ~fits_frame))
    else:
        order = np.argsort(-margins)
    best = frames[order[0]]
    return Table({key: [best[key]] for key in ["run", "rerun", "camcol", "field"]})


def _closest_object_frame(coords: SkyCoord, dr: int) -> Table | None:
    """Returns the frame of the SDSS object closest to the given coordinates."""
    ids = None
    for radius in np.arange(1, 60, 1):
        ids = SDSS.query_region(coords, radius=radius * u.arcsec, data_release=dr)
        if ids is not None:
            if "ra" in ids.colnames and "dec" in ids.colnames:
                break
            else:
                ids = None
    if ids is None:
        return None
    coords_imgs = SkyCoord(ra=ids["ra"].value, dec=ids["dec"].value, unit=(u.degree, u.degree), frame="icrs")
    pointing_id = int(np.argmin(coords.separation(coords_imgs).value))
    return ids[pointing_id : pointing_id + 1]


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
    coords = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)
    # choose the frame (among the overlapping ones) where the cutout fits best
    ids = _best_frame(ra, dec, size_arcsec.value / 60, dr)
    if ids is None:
        # fall back to the frame of the object closest to the given coordinates
        ids = _closest_object_frame(coords, dr)
    if ids is None:
        return None
    # download images
    hdu_list = SDSS.get_images(matches=ids, band=filters, data_release=dr)
    # SDSS images are large so need to be trimmed
    for hdu in hdu_list:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AstropyWarning)
            img_wcs = WCS(hdu[0].header)
        trimmed_data = Cutout2D(hdu[0].data, coords, size_pixels, img_wcs)
        if trimmed_data.data.shape != (size_pixels, size_pixels):
            warnings.warn(
                f"The SDSS cutout was trimmed by the edge of the frame ({trimmed_data.data.shape[1]}x"
                f"{trimmed_data.data.shape[0]} pixels instead of {size_pixels}x{size_pixels}); "
                "SDSS frames are 13.5 x 9.8 arcmin."
            )
        hdu[0].data = trimmed_data.data
        hdu[0].header.update(trimmed_data.wcs.to_header())
    return hdu_list
