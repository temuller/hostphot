import io
import time
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import requests
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.exceptions import AstropyWarning

from hostphot.utils import open_fits_from_urls, select_image
from hostphot.surveys_utils import get_survey_filters, check_filters_validity

# ESO archive: ObsCore table (TAP) and cutout service (SODA)
_TAP_URL = "https://archive.eso.org/tap_obs/sync"
_SODA_URL = "https://dataportal.eso.org/dataPortal/soda/sync"
# KiDS DR5 co-added tiles (Astro-WISE), the only KiDS images in the ESO archive (DR5 supersedes the previous releases)
_COLLECTION = "KIDS"
_ESO_FILTERS = {"u": "u_SDSS", "g": "g_SDSS", "r": "r_SDSS", "i": "i_SDSS"}


def query_kids_tiles(ra: float, dec: float) -> pd.DataFrame:
    """Queries the KiDS tiles (co-added images) containing the given position in the ESO archive.

    Parameters
    ----------
    ra: Right Ascension in degrees.
    dec: Declination in degrees.

    Returns
    -------
    tiles: Table with the ESO product IDs (``dp_id``), filters and exposure times of the tiles.
    """
    query = (
        "select dp_id, filter, t_exptime from ivoa.ObsCore "
        f"where obs_collection='{_COLLECTION}' and dataproduct_type='image' "
        f"and CONTAINS(POINT('ICRS', {ra}, {dec}), s_region)=1"
    )
    params = {"QUERY": query, "LANG": "ADQL", "REQUEST": "doQuery", "FORMAT": "csv"}
    for attempt in range(3):
        # retries, as the archive can fail temporarily
        try:
            response = requests.get(_TAP_URL, params=params, timeout=120)
            response.raise_for_status()
            break
        except requests.exceptions.RequestException:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    return pd.read_csv(io.StringIO(response.text))


def get_KiDS_images(
    ra: float, dec: float, size: float | u.Quantity = 3, filters: Optional[str] = None
) -> list[fits.HDUList | None]:
    """Gets KiDS (DR5) image cutouts for the given position and filters from the ESO archive.

    The pixel values are calibrated with the nightly zeropoints (``PHOTZP``), corrected with the
    offsets of the stellar-locus regression and Gaia recalibration of DR5 (``DMAG``; Wright et al.
    2024): the AB zeropoint ``PHOTZP + DMAG`` is stored in the ``MAGZP`` keyword. When several tiles
    cover the position (e.g. the two i-band passes, or overlapping tiles), the one covering the whole
    cutout is used (the deepest one if there are several).

    Parameters
    ----------
    ra: Right Ascension in degrees.
    dec: Declination in degrees.
    size: Image size. If a float is given, the units are assumed to be arcmin.
    filters: Filters to use. If ``None``, uses ``ugri``.

    Returns
    -------
    hdu_list: List of FITS images (``None`` for the filters without coverage).
    """
    survey = "KiDS"
    if filters is None:
        filters = get_survey_filters(survey)
    check_filters_validity(filters, survey)

    if isinstance(size, (float, int)):
        size_arcsec = (size * u.arcmin).to(u.arcsec).value
    else:
        size_arcsec = size.to(u.arcsec).value

    tiles = query_kids_tiles(ra, dec)
    if len(tiles) == 0:
        warnings.warn(f"No KiDS images found at RA={ra}, Dec={dec} (outside the KiDS footprint).")
        return [None] * len(filters)

    # square cutouts (the bounding box of a circle with a radius of half the size)
    radius = size_arcsec / 3600 / 2
    candidates = {filt: tiles[tiles["filter"] == _ESO_FILTERS[filt]] for filt in filters}
    urls = [
        f"{_SODA_URL}?ID=ivo://eso.org/ID?{dp_id}&CIRCLE={ra} {dec} {radius}"
        for filt in filters for dp_id in candidates[filt]["dp_id"]
    ]
    url_to_hdu = dict(zip(urls, open_fits_from_urls(urls, max_workers=4, attempts=3)))

    hdu_list = []
    for filt in filters:
        hdus, images = [], []
        for _, tile in candidates[filt].iterrows():
            url = f"{_SODA_URL}?ID=ivo://eso.org/ID?{tile['dp_id']}&CIRCLE={ra} {dec} {radius}"
            hdu = url_to_hdu.get(url)
            if hdu is None or hdu[0].data is None:
                continue
            hdus.append(hdu)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", AstropyWarning)  # (deprecated keywords fixed by astropy)
                img_wcs = WCS(hdu[0].header)
            images.append((hdu[0].data, img_wcs, float(tile["t_exptime"])))
        index = select_image(images, ra, dec, size_arcsec, label=f"{survey} {filt}")
        if index is None:
            hdu_list.append(None)
            continue
        hdu = hdus[index]
        header = hdu[0].header
        if "DMAG" in header:
            header["MAGZP"] = (header["PHOTZP"] + header["DMAG"], "AB zeropoint (PHOTZP + DMAG)")
        else:
            warnings.warn(f"{survey} {filt}: no DR5 zeropoint correction (DMAG) in the header; using PHOTZP only.")
            header["MAGZP"] = (header["PHOTZP"], "AB zeropoint (PHOTZP)")
        hdu_list.append(fits.HDUList([fits.PrimaryHDU(data=hdu[0].data, header=header)]))

    return hdu_list
