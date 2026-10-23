import re
import socket
import urllib
from typing import Optional

from astropy import wcs
import astropy.units as u
from astropy.io import fits

import numpy as np

from hostphot.utils import open_fits_from_url, open_fits_from_urls, select_image, url_band, image_coverage, FULL_COVERAGE
from hostphot.surveys_utils import get_survey_filters, check_filters_validity
from hostphot.moc.maps import contains_coords

import warnings
from astropy.utils.exceptions import AstropyWarning


def get_VISTA_images(ra: float, dec: float, size: float | u.Quantity = 3, 
                        filters: Optional[str] = None, version: Optional[str] = None) -> list[fits.ImageHDU]:
    """Gets VISTA fits images for the given coordinates and
    filters.

    Note: the different surveys included in VISTA cover different
    parts of the sky and do not necessarily contain the same filters.
    
    The available surveys are "VIDEO", "VIKING", "VHS". By default, the first one
    (in that order, i.e. the deepest) covering the position is used.

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
    version: str, default ``None``
        Survey to use: ``VHS``, ``VIDEO`` or ``VIKING``. If ``None``, the deepest survey
        covering the position is used.

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

    versions = ["VIDEO", "VIKING", "VHS"]
    if version is None:
        # the deepest survey covering the position
        for version in versions:
            overlap = contains_coords(ra, dec, version)
            if overlap is True:
                break
        if overlap is False:
            return [None] * len(filters)
    else:
        version = version.upper()
        if version not in versions:
            raise ValueError(f"Not a valid VISTA version ({version}): {versions}")
        if contains_coords(ra, dec, version) is False:
            warnings.warn(f"The position is not covered by VISTA {version}.")
            return [None] * len(filters)
    
    # Latest data releases
    database_dict = {
        "VHS": "VHSDR6",
        "VIDEO": "VIDEODR6",
        "VIKING": "VIKINGDR5",
    }
    database = database_dict[version]

    size_arcsec = size * 60
    hdu_list = []
    tile_links = None
    if version == "VIDEO":
        # deep mosaics (stacks of all the VIDEO observations)
        mosaic_links = _query_links(database, ra, dec, size, "mosaicdeep")
    else:
        mosaic_links = []
    for filt in filters:
        hdu = None
        mosaic_urls = _latest_version(_filter_urls(mosaic_links, filt))
        if len(mosaic_urls) > 0:
            # a few mosaics per filter: the deepest one covering the whole cutout is used
            mosaics = [h for h in open_fits_from_urls(mosaic_urls) if _has_image(h)]
            images = [(h[1].data, _image_wcs(h), _total_exptime(h)) for h in mosaics]
            index = select_image(images, ra, dec, size_arcsec, label=f"{survey} {filt} (deep mosaics)")
            if index is not None:
                coverage = image_coverage(images[index][0], images[index][1], ra, dec, size_arcsec)
                if coverage >= FULL_COVERAGE:
                    hdu = mosaics[index]
        if hdu is None:
            # individual tiles (stacks of one observation)
            if tile_links is None:
                tile_links = _query_links(database, ra, dec, size, "tilestack")
            hdu = _select_tile(_filter_urls(tile_links, filt), ra, dec, size_arcsec, f"{survey} {filt}")
        hdu_list.append(None if hdu is None else _prepare_image(hdu))
    return hdu_list


# maximum number of tiles downloaded per filter while looking for one covering the cutout
MAX_DOWNLOADS = 10
# maximum zeropoint uncertainty (MAGZRR, in mag) of a tile to be considered well calibrated
MAX_MAGZRR = 0.05


def _query_links(database: str, ra: float, dec: float, size: float, frame_type: str) -> list[str]:
    """Queries the VISTA Science Archive for the images (of the given frame type) at a position."""
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
        "frameType": frame_type,
    }
    url = base_url + "&".join([f"{key}={val}" for key, val in survey_dict.items()])
    try:
        # the archive can be unresponsive, so a timeout is used (in seconds)
        results = urllib.request.urlopen(url, timeout=300).read()
    except (socket.timeout, TimeoutError) as exc:
        raise TimeoutError(
            "The VISTA Science Archive (VSA) did not respond within 5 minutes. Please, try again later."
        ) from exc
    links = re.findall('href="(http://.*?)"', results.decode("utf-8"))
    return [link.replace("getImage", "getFImage", 1) for link in links]


def _filter_urls(links: list[str], filt: str) -> list[str]:
    """URLs of a filter (exact match, e.g. not a narrow band with the same prefix), without duplicates."""
    return list(dict.fromkeys(link for link in links if url_band(link) == filt))


def _latest_version(urls: list[str]) -> list[str]:
    """Keeps the URLs of the latest processing version of the mosaics (e.g. ``.../mosaics/20210908_v10/...``),
    as the archive also returns superseded versions; all the URLs are kept if the version is not found."""
    versions = [re.search(r"/mosaics/[^/]*_v(\d+)/", url) for url in urls]
    if len(urls) == 0 or any(version is None for version in versions):
        return urls
    numbers = [int(version.group(1)) for version in versions]
    return [url for url, number in zip(urls, numbers) if number == max(numbers)]


def _has_image(hdu: Optional[fits.HDUList]) -> bool:
    return hdu is not None and len(hdu) > 1 and hdu[1].data is not None and np.ndim(hdu[1].data) == 2


def _image_wcs(hdu: fits.HDUList) -> wcs.WCS:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        return wcs.WCS(hdu[1].header)


def _keyword(hdu: fits.HDUList, key: str, default=None):
    """Value of a keyword from the image extension or, if not there, from the primary header
    (e.g. the exposure time and airmass of the deep mosaics are in the primary header)."""
    return hdu[1].header.get(key, hdu[0].header.get(key, default))


def _total_exptime(hdu: fits.HDUList) -> float:
    """Total exposure time of a stack: ``TEXPTIME`` (deep mosaics) or DIT x NDIT x NJITTER (tiles)."""
    if _keyword(hdu, "TEXPTIME") is not None:
        return float(_keyword(hdu, "TEXPTIME"))
    dit = _keyword(hdu, "HIERARCH ESO DET DIT", _keyword(hdu, "EXPTIME", np.nan))
    ndit = _keyword(hdu, "HIERARCH ESO DET NDIT", 1)
    return float(dit) * float(ndit) * float(_keyword(hdu, "NJITTER", 1))


def _select_tile(urls: list[str], ra: float, dec: float, size_arcsec: float, label: str):
    """Downloads the tiles one by one until a well-calibrated one (MAGZRR <= MAX_MAGZRR) covers the
    whole cutout; otherwise, the best one (largest coverage, then smallest MAGZRR) is used, with a warning."""
    best, best_key = None, None
    for url in urls[:MAX_DOWNLOADS]:
        hdu = open_fits_from_urls([url])[0]
        if not _has_image(hdu):
            continue
        coverage = image_coverage(hdu[1].data, _image_wcs(hdu), ra, dec, size_arcsec)
        magzrr = float(_keyword(hdu, "MAGZRR", np.inf))
        full = coverage >= FULL_COVERAGE
        good = magzrr <= MAX_MAGZRR
        key = (full and good, round(coverage, 3), -magzrr)
        if best_key is None or key > best_key:
            best, best_key = hdu, key
        if full and good:
            break
    if best is None:
        return None
    coverage, magzrr = best_key[1], -best_key[2]
    if coverage < FULL_COVERAGE:
        warnings.warn(
            f"{label}: none of the available images covers the whole cutout; using the one with the "
            f"largest coverage ({100 * coverage:.0f}% of the requested area)."
        )
    if magzrr > MAX_MAGZRR:
        warnings.warn(
            f"{label}: no well-calibrated image was found; using one with a zeropoint uncertainty of "
            f"{magzrr:.3f} mag (MAGZRR)."
        )
    return best


def _prepare_image(hdu: fits.HDUList) -> fits.HDUList:
    """Returns the image (in the primary HDU) with the WCS and calibration keywords used by HostPhot."""
    header = hdu[0].header.copy()
    # keywords of the image extension (e.g. CASUVERS, SATURATE, DATE-OBS), for reference
    structural = ("SIMPLE", "BITPIX", "NAXIS", "EXTEND", "XTENSION", "PCOUNT", "GCOUNT", "EXTNAME",
                  "CHECKSUM", "DATASUM", "BSCALE", "BZERO", "COMMENT", "HISTORY", "")
    for card in hdu[1].header.cards:
        if not card.keyword.startswith(structural) and card.keyword not in header:
            try:
                header[card.keyword] = (card.value, card.comment)
            except (ValueError, KeyError):
                pass
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        header.update(_image_wcs(hdu).to_header())
    # exposure time (to which the counts are normalised; 1 for the deep mosaics) and calibration
    header["EXPTIME"] = float(_keyword(hdu, "EXPTIME"))
    header["MAGZRR"] = float(_keyword(hdu, "MAGZRR"))
    header["TEXPTIME"] = (_total_exptime(hdu), "Total exposure time of the stack (s)")
    # effective ZP considering the atmospheric extinction (already included in the deep mosaics,
    # with EXTINCT = 0)
    airmass = (
        float(_keyword(hdu, "HIERARCH ESO TEL AIRM START", 1.0)) + float(_keyword(hdu, "HIERARCH ESO TEL AIRM END", 1.0))
    ) / 2
    extinction = float(_keyword(hdu, "EXTINCT", 0.0)) * (airmass - 1)
    header["MAGZP"] = float(_keyword(hdu, "MAGZPT")) - extinction
    # only the science image is kept
    return fits.HDUList([fits.PrimaryHDU(data=hdu[1].data, header=header)])
