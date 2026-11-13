import time
import warnings
from io import BytesIO
from typing import Optional

import numpy as np
import requests
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.io.votable import parse_single_table

from hostphot.utils import network_timeout, open_fits_from_urls, select_image
from hostphot.surveys_utils import get_survey_filters, check_filters_validity

# IRSA Simple Image Access (SIA) service, with the Euclid MER mosaics
_SIA_URL = "https://irsa.ipac.caltech.edu/SIA"
# IRSA cutout service (the mosaics are in irsacloud, which does not accept cutout parameters in the file URLs)
_CUTOUT_URL = "https://irsacloud.ipac.caltech.edu/ibe/cutout"
# collections of the MER background-subtracted mosaics for each data release
_COLLECTIONS = {
    "q1": ["euclid_DpdMerBksMosaic"],
    "dr1": ["euclid_dr1_wide_DpdMerBksMosaic", "euclid_dr1_deep_DpdMerBksMosaic"],
}
_DEFAULT_VERSION = "dr1"
# Euclid instruments (the MER tiles also include ground-based images resampled to the same grid)
_INSTRUMENTS = {"VIS": "VIS", "Y": "NISP", "J": "NISP", "H": "NISP"}
# RMS value of the masked pixels in the MER mosaics (1e16; with zeros in the images)
_MASKED_RMS = 1e10


def query_euclid_mosaics(ra: float, dec: float, version: Optional[str] = None) -> list[dict]:
    """Queries the Euclid MER mosaics (science and RMS maps) containing the given position.

    Parameters
    ----------
    ra: Right Ascension in degrees.
    dec: Declination in degrees.
    version: Data release: ``dr1`` (the default; wide and deep surveys) or ``q1``.

    Returns
    -------
    mosaics: List of dictionaries with the tile, filter and links to the science and RMS mosaics.
    """
    version = (version or _DEFAULT_VERSION).lower()
    if version not in _COLLECTIONS:
        raise ValueError(f"Euclid version '{version}' not available; use one of {list(_COLLECTIONS)}.")
    mosaics = {}
    for collection in _COLLECTIONS[version]:
        params = {
            "COLLECTION": collection,
            "POS": f"CIRCLE {ra} {dec} 0.0001",
            "RESPONSEFORMAT": "VOTABLE",
        }
        for attempt in range(4):
            # retries, as IRSA can reset the connections or time out temporarily
            try:
                response = requests.get(_SIA_URL, params=params, timeout=network_timeout())
                response.raise_for_status()
                break
            except requests.exceptions.RequestException:
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
        table = parse_single_table(BytesIO(response.content)).to_table(use_names_over_ids=True)

        for row in table:
            filt = str(row["energy_bandpassname"])
            if _INSTRUMENTS.get(filt) != str(row["instrument_name"]):
                continue  # ground-based (external) images or other filters
            subtype = str(row["dataproduct_subtype"])
            if subtype not in ["science", "noise"]:
                continue  # flag maps, PSFs and background models
            tile = str(row["obs_id"]).split("_")[0]
            mosaic = mosaics.setdefault((collection, tile, filt), {"tile": tile, "filter": filt})
            mosaic["science" if subtype == "science" else "rms"] = str(row["access_url"])
    return [mosaic for mosaic in mosaics.values() if "science" in mosaic and "rms" in mosaic]


def _cutout_url(url: str, ra: float, dec: float, size_deg: float) -> str:
    """URL of a cutout of a MER mosaic, from the IRSA cutout service.

    Parameters
    ----------
    url: URL of the mosaic (``.../ibe/data/euclid/...``).
    ra: Right Ascension in degrees.
    dec: Declination in degrees.
    size_deg: Cutout size in degrees.

    Returns
    -------
    cutout_url: URL of the cutout.
    """
    path = url.split("/ibe/data/", 1)[-1]
    return f"{_CUTOUT_URL}?ra={ra}&dec={dec}&size={size_deg}&path={path}"


def get_Euclid_images(
    ra: float,
    dec: float,
    size: float | u.Quantity = 3,
    filters: Optional[str | list] = None,
    version: Optional[str] = None,
) -> list[fits.HDUList | None]:
    """Gets Euclid MER mosaic cutouts (background subtracted) for the given position and filters.

    The images are in ADU/s (VIS) and electrons/s (NISP), with the AB zeropoint in the ``MAGZP``
    keyword. The RMS map (including the Poisson noise of the sources) is stored as a variance
    map in the ``VARIANCE`` extension. When several tiles cover the position, the one covering
    the whole cutout is used (the deepest one, from its RMS map, if there are several).

    Parameters
    ----------
    ra: Right Ascension in degrees.
    dec: Declination in degrees.
    size: Image size. If a float is given, the units are assumed to be arcmin.
    filters: Filters to use. If ``None``, uses ``VIS, Y, J, H``.
    version: Data release: ``dr1`` (the default; wide and deep surveys) or ``q1``.

    Returns
    -------
    hdu_list: List of FITS images (``None`` for the filters without coverage).
    """
    survey = "Euclid"
    if filters is None:
        filters = get_survey_filters(survey)
    elif isinstance(filters, str):
        filters = [filters]
    check_filters_validity(filters, survey)

    if isinstance(size, (float, int)):
        size_arcsec = (size * u.arcmin).to(u.arcsec).value
    else:
        size_arcsec = size.to(u.arcsec).value

    mosaics = [mosaic for mosaic in query_euclid_mosaics(ra, dec, version) if mosaic["filter"] in filters]
    if len(mosaics) == 0:
        warnings.warn(f"No Euclid images found at RA={ra}, Dec={dec} (outside the footprint of the release).")
        return [None] * len(filters)

    # cutouts of the science and RMS mosaics (IRSA cutout service; sizes in degrees)
    urls = [_cutout_url(mosaic[key], ra, dec, size_arcsec / 3600) for mosaic in mosaics for key in ["science", "rms"]]
    # (IRSA can reset the connections with several parallel requests)
    hdus = open_fits_from_urls(urls, max_workers=4, attempts=4)
    for i, mosaic in enumerate(mosaics):
        mosaic["science_hdu"], mosaic["rms_hdu"] = hdus[2 * i], hdus[2 * i + 1]

    hdu_list = []
    for filt in filters:
        candidates = [
            mosaic for mosaic in mosaics
            if mosaic["filter"] == filt and mosaic["science_hdu"] is not None and mosaic["rms_hdu"] is not None
        ]
        images = []
        for mosaic in candidates:
            data = mosaic["science_hdu"][0].data
            rms = mosaic["rms_hdu"][0].data
            valid = np.isfinite(rms) & (rms > 0) & (rms < _MASKED_RMS)
            depth = 1 / np.median(rms[valid]) ** 2 if valid.any() else None  # inverse variance
            images.append((data, WCS(mosaic["science_hdu"][0].header), depth))
        index = select_image(images, ra, dec, size_arcsec, label=f"{survey} {filt}")
        if index is None:
            hdu_list.append(None)
            continue

        mosaic = candidates[index]
        header = mosaic["science_hdu"][0].header.copy()
        header["MAGZP"] = (header["MAGZERO"], "AB zeropoint (MAGZERO)")
        header["TILEID"] = (mosaic["tile"], "Euclid MER tile")
        rms = mosaic["rms_hdu"][0].data.astype(float)
        # pixels without valid data (masked, with zeros in the image and RMS = 1e16, or with zero or
        # non-finite RMS) are set to NaN
        data = mosaic["science_hdu"][0].data.astype(float)
        invalid = ~(np.isfinite(rms) & (rms > 0) & (rms < _MASKED_RMS))
        data[invalid] = np.nan
        rms[invalid] = np.nan
        hdu = fits.HDUList([
            fits.PrimaryHDU(data=data, header=header),
            fits.ImageHDU(data=rms**2, header=WCS(mosaic["rms_hdu"][0].header).to_header(), name="VARIANCE"),
        ])
        hdu_list.append(hdu)

    for mosaic in mosaics:
        for key in ["science_hdu", "rms_hdu"]:
            if mosaic[key] is not None:
                mosaic[key].close()
    return hdu_list
