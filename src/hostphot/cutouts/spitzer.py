import io
import warnings
import requests
import numpy as np
from typing import Optional

import astropy.units as u
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.io.votable import parse_single_table

from astropy.wcs import WCS

from hostphot.utils import open_fits_from_url, open_fits_from_urls, select_image, image_coverage, FULL_COVERAGE
from hostphot.surveys_utils import get_survey_filters, check_filters_validity

# Spitzer Heritage Archive (SHA): post-BCD mosaics of each observation (AOR), including the warm mission
# (IRAC 3.6 and 4.5 microns, 2009-2020), which is not included in the SEIP super mosaics (cryogenic mission only)
_SIA_URL = "https://irsa.ipac.caltech.edu/SIA"
_SHA_BANDS = {"IRAC.1": "IRAC1", "IRAC.2": "IRAC2", "IRAC.3": "IRAC3", "IRAC.4": "IRAC4", "MIPS.1": "MIPS24"}
# maximum number of SHA mosaics downloaded per filter while looking for one covering the cutout
MAX_SHA_DOWNLOADS = 10


def query_sha_mosaics(ra: float, dec: float, filters: list[str]) -> dict[str, list[str]]:
    """Queries the post-BCD mosaics (``*_maic.fits``) of the Spitzer Heritage Archive containing the given
    position, sorted by their exposure time per frame (longest first).

    Parameters
    ----------
    ra: Right Ascension in degrees.
    dec: Declination in degrees.
    filters: HostPhot Spitzer filters (e.g. ``IRAC.1``).

    Returns
    -------
    urls: Links to the mosaics of each filter.
    """
    params = {"COLLECTION": "spitzer_sha", "POS": f"CIRCLE {ra} {dec} 0.0005", "RESPONSEFORMAT": "VOTABLE"}
    response = requests.get(_SIA_URL, params=params, timeout=300)
    response.raise_for_status()
    table = parse_single_table(io.BytesIO(response.content)).to_table(use_names_over_ids=True).to_pandas()
    urls = {filt: [] for filt in filters}
    if len(table) == 0:
        return urls
    table = table[(table["dataproduct_subtype"] == "science") & table["access_url"].str.endswith("_maic.fits")]
    table = table.sort_values("t_exptime", ascending=False, kind="stable")
    for filt in filters:
        band_urls = table.loc[table["energy_bandpassname"] == _SHA_BANDS[filt], "access_url"]
        urls[filt] = list(dict.fromkeys(band_urls))
    return urls


def _sha_cutout(hdu: fits.HDUList, ra: float, dec: float, size_arcsec: float) -> fits.HDUList:
    """Cuts an SHA mosaic to the requested size and adds the keywords used by HostPhot, as for the SEIP
    mosaics: the AB zeropoint (``ZPAB`` and ``MAGZP``, for MJy/sr) and the gain in e-/(MJy/sr) (``GAIN``)."""
    header = hdu[0].header
    img_wcs = WCS(header)
    pixel_scale = np.sqrt(np.abs(np.linalg.det(img_wcs.pixel_scale_matrix))) * 3600
    npix = int(round(size_arcsec / pixel_scale))
    cutout = Cutout2D(hdu[0].data, SkyCoord(ra, dec, unit="deg"), (npix, npix), wcs=img_wcs, mode="trim")

    new_header = header.copy()
    for key in list(new_header.keys()):
        if key.startswith(("CD", "PC", "CDELT", "CROTA", "CRPIX", "CRVAL", "PV")):
            del new_header[key]
    new_header.update(cutout.wcs.to_header())
    # AB zeropoint of the images in MJy/sr: 1 MJy/sr is 1e6 Jy/sr times the pixel area
    pixel_area_sr = (pixel_scale / 206264.806) ** 2
    new_header["ZPAB"] = (-2.5 * np.log10(1e6 * pixel_area_sr) + 8.90, "Magnitude Zero-Point on AB System")
    new_header["MAGZP"] = new_header["ZPAB"]
    # gain of the mosaic in e-/(MJy/sr), as for SEIP (GAIN = EXPGAIN * EXPTIME / EFCONV); EXPTIME is that of a
    # single frame, so the Poisson noise of the sources is overestimated where several frames overlap
    new_header["EXPGAIN"] = (header["GAIN"], "e-/DN for input exposures")
    new_header["EFCONV"] = (header["FLUXCONV"], "(MJy/sr)/(DN/s) for input exposures")
    new_header["GAIN"] = (header["GAIN"] * header["EXPTIME"] / header["FLUXCONV"], "Conversion in e-/(MJy/sr)")
    new_header["ARCHIVE"] = ("SHA", "Spitzer Heritage Archive post-BCD mosaic")
    return fits.HDUList([fits.PrimaryHDU(data=cutout.data, header=new_header)])


def get_SHA_image(ra: float, dec: float, size_arcsec: float, urls: list[str]) -> tuple[fits.HDUList | None, float]:
    """Gets the SHA mosaic covering the cutout (the first one, in the given order, covering the whole cutout;
    otherwise, the one with the largest coverage), cut to the requested size.

    Parameters
    ----------
    ra: Right Ascension in degrees.
    dec: Declination in degrees.
    size_arcsec: Image size in arcsec.
    urls: Links to the SHA mosaics (see :func:`query_sha_mosaics`).

    Returns
    -------
    hdu: Cutout of the selected mosaic (``None`` if none covers the position).
    coverage: Fraction of the cutout covered by the mosaic.
    """
    best, best_coverage = None, 0.0
    # the mosaics are downloaded one by one (~10 MB each), until one covers the whole cutout
    for url in urls[:MAX_SHA_DOWNLOADS]:
        hdu = open_fits_from_urls([url], attempts=3)[0]
        if hdu is None or hdu[0].data is None:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coverage = image_coverage(hdu[0].data, WCS(hdu[0].header), ra, dec, size_arcsec)
        if coverage > best_coverage + 1e-6:
            best, best_coverage = hdu, coverage
        if coverage >= FULL_COVERAGE:
            break
    if best is None:
        return None, 0.0
    return _sha_cutout(best, ra, dec, size_arcsec), best_coverage

def get_Spitzer_images(ra: float, dec: float, size: float | u.Quantity = 3, 
                        filters: Optional[str] = None) -> list[fits.ImageHDU]:
    """Gets Spitzer fits images for the given coordinates and
    filters.

    The SEIP super mosaics (cryogenic mission) are used. If none of them covers the whole cutout
    in a filter (e.g. for positions only observed during the warm mission), the post-BCD mosaics of the
    Spitzer Heritage Archive (SHA) are used instead, if they have a larger coverage.

    Parameters
    ----------
    ra: Right Ascension in degrees.
    dec: Declination in degrees.
    size: Image size. If a float is given, the units are assumed to be arcmin.
    filters: Filters to use. If ``None``, uses ``IRAC.1, IRAC.2, IRAC.3, IRAC.4, MIPS.1``.

    Returns
    -------
    fits_files: List of fits images.
    """
    survey = "Spitzer"
    if filters is None:
        filters = get_survey_filters(survey)
    check_filters_validity(filters, survey)

    if isinstance(size, (float, int)):
        size_arcsec = (size * u.arcmin).to(u.arcsec).value
    else:
        size_arcsec = size.to(u.arcsec).value

    base_url = "https://irsa.ipac.caltech.edu/cgi-bin/Cutouts/nph-cutouts?"
    locstr = f"{str(ra)}+{str(dec)}+eq"

    mission_dict = {
        "mission": "SEIP",
        "min_size": "18",
        "max_size": "1800",
        "units": "arcsec",
        "locstr": locstr,
        "sizeX": size_arcsec,  # in arcsec (see "units")
        "ntable_cutouts": "1",
        "cutouttbl1": "science",
        "mode": "PI",  # XML output file
    }

    # check image size (in arcsec)
    if size_arcsec < int(mission_dict["min_size"]):
        mission_dict["sizeX"] = int(mission_dict["min_size"])
        print(
            f'Image cannot be smaller than {mission_dict["min_size"]} arcsec (using this value)'
        )
    elif size_arcsec > int(mission_dict["max_size"]):
        mission_dict["sizeX"] = int(mission_dict["max_size"])
        print(
            f'Image cannot be larger than {mission_dict["max_size"]} arcsec (using this value)'
        )

    mission_url = "&".join(
        [f"{key}={val}" for key, val in mission_dict.items()]
    )
    url = base_url + mission_url
    response = requests.get(url)
    list_text = response.text.split()

    # find all the mosaics of each filter (there can be several overlapping ones)
    files_dict = {filt: [] for filt in filters}
    for filt in filters:
        for line in list_text:
            if filt in line and line.endswith(".mosaic.fits") and line not in files_dict[filt]:
                files_dict[filt].append(line)
    unique_urls = list(dict.fromkeys(url for urls in files_dict.values() for url in urls))
    url_to_hdu = dict(zip(unique_urls, open_fits_from_urls(unique_urls)))

    # select the mosaic covering the whole cutout, the deepest one if there are several
    hdu_list_all = []
    for filt in filters:
        candidates = [url_to_hdu[url] for url in files_dict[filt] if url_to_hdu[url] is not None]
        images = [(hdu[0].data, WCS(hdu[0].header), hdu[0].header.get("EXPTIME")) for hdu in candidates]
        index = select_image(images, ra, dec, mission_dict["sizeX"], label=f"{survey} {filt}")
        hdu_list_all.append(None if index is None else candidates[index])

    hdu_list = []
    sha_urls = None
    for filt, hdu in zip(filters, hdu_list_all):
        coverage = 0.0
        if hdu is not None:
            hdu[0].header["MAGZP"] = hdu[0].header["ZPAB"]
            hdu[0].header["ARCHIVE"] = ("SEIP", "Spitzer Enhanced Imaging Products super mosaic")
            coverage = image_coverage(hdu[0].data, WCS(hdu[0].header), ra, dec, mission_dict["sizeX"])
        if coverage < FULL_COVERAGE:
            # fallback to the Spitzer Heritage Archive (e.g. warm-mission observations)
            try:
                if sha_urls is None:
                    sha_urls = query_sha_mosaics(ra, dec, filters)
                sha_hdu, sha_coverage = get_SHA_image(ra, dec, mission_dict["sizeX"], sha_urls[filt])
            except requests.exceptions.RequestException as exc:
                warnings.warn(f"{survey} {filt}: the Spitzer Heritage Archive could not be queried ({exc!r}).")
                sha_hdu, sha_coverage = None, 0.0
            if sha_hdu is not None and sha_coverage > coverage + 1e-6:
                warnings.warn(
                    f"{survey} {filt}: no SEIP mosaic covers the whole cutout; using a post-BCD mosaic of the Spitzer "
                    f"Heritage Archive (AOR {sha_hdu[0].header.get('AORKEY')}; {100 * sha_coverage:.0f}% coverage)."
                )
                hdu = sha_hdu
        hdu_list.append(hdu)
    return hdu_list
