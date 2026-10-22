from astropy import wcs
import astropy.units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astroquery.ukidss import Ukidss

import numpy as np

from hostphot.utils import open_fits_from_url, open_fits_from_urls, url_band, image_coverage, FULL_COVERAGE
from hostphot.surveys_utils import get_survey_filters, check_filters_validity, survey_pixel_scale

import warnings
from astropy.utils.exceptions import AstropyWarning

# maximum number of images downloaded per filter while looking for one covering the cutout
MAX_DOWNLOADS = 10


def _is_deep_stack(url: str) -> bool:
    """Whether the URL is of a deep stack (e.g. of DXS or UDS; file names ending in ``_dp``)."""
    from urllib.parse import parse_qs, urlparse

    file_name = parse_qs(urlparse(url).query).get("file", [""])[0]
    return "_dp" in file_name.rsplit("/", 1)[-1]


def _stack_exptime(hdu: fits.HDUList) -> float:
    """Total exposure time of a stack (``TEXPTIME``), or the per-exposure time times the number
    of exposures, jitter positions, micro-steps and integrations if not available."""
    for header in [hdu[1].header, hdu[0].header]:
        if "TEXPTIME" in header:
            return float(header["TEXPTIME"])

    def get(key, default=1.0):
        return float(hdu[0].header.get(key, hdu[1].header.get(key, default)))

    return get("EXP_TIME", np.nan) * get("NEXP") * get("NJITTER") * get("NUSTEP") * get("NINT")


def get_UKIDSS_images(ra: float, dec: float, size: float=3, filters: str='ZYJHK') -> list[fits.HDUList]:
    """Gets UKIDSS fits images for the given coordinates and
    filters.

    Parameters
    ----------
    ra: Right Ascension in degrees.
    dec: Declination in degrees.
    size: Image size. If a float is given, the units are assumed to be arcmin.
    filters: Filters to use. If ``None``, uses ``ZYJHK``.

    Returns
    -------
    fits_files: list
        List of fits images.
    """
    survey = "UKIDSS"
    if filters is None:
        filters = get_survey_filters(survey)
    check_filters_validity(filters, survey)

    database = 'UKIDSSDR11PLUS'
    # programme = 'LAS'  # ['UDS', 'GCS', 'GPS', 'DXS', 'LAS']

    if isinstance(size, (float, int)):
        size = (size * u.arcmin)

    u_obj = Ukidss(database=database)
    pos = SkyCoord(ra, dec, unit="deg")
    urls = u_obj.get_image_list(pos, waveband='all', frame_type="stack", image_width=size)

    # all the images of the requested filters (e.g. from different programmes or overlapping frames);
    # the band is matched exactly (e.g. H, not the H2 narrow band)
    unique_urls = [url for url in dict.fromkeys(urls) if url_band(url) in filters]

    # the images are downloaded one by one, the deep stacks (e.g. of DXS and UDS) first, until one
    # covers the whole cutout (otherwise, the one with the largest coverage is used, with a warning);
    # some positions have hundreds of frames (e.g. UDS), so they are not all downloaded
    size_arcsec = size.to(u.arcsec).value
    hdu_list_all = []
    for filt in filters:
        filt_urls = [url for url in unique_urls if url_band(url) == filt]
        filt_urls = sorted(filt_urls, key=lambda url: 0 if _is_deep_stack(url) else 1)  # (stable)
        best, best_coverage, best_depth = None, 0.0, -np.inf
        n_empty = 0
        for url in filt_urls[:MAX_DOWNLOADS]:
            hdu = open_fits_from_urls([url])[0]
            if hdu is None:
                continue
            # the archive can return frames without the image extension, which cannot be used
            if len(hdu) < 2 or hdu[1].data is None or np.ndim(hdu[1].data) != 2:
                n_empty += 1
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", AstropyWarning)
                img_wcs = wcs.WCS(hdu[1].header)
            coverage = image_coverage(hdu[1].data, img_wcs, ra, dec, size_arcsec)
            depth = _stack_exptime(hdu)
            if coverage > best_coverage + 1e-6 or (np.isclose(coverage, best_coverage) and depth > best_depth):
                best, best_coverage, best_depth = hdu, coverage, depth
            if coverage >= FULL_COVERAGE:
                break
        if n_empty > 0:
            warnings.warn(f"{survey} {filt}: {n_empty} image(s) without image data were skipped.")
        if best is not None and best_coverage < FULL_COVERAGE:
            warnings.warn(
                f"{survey} {filt}: none of the available images covers the whole cutout; using the one with the "
                f"largest coverage ({100 * best_coverage:.0f}% of the requested area)."
            )
        hdu_list_all.append(best)

    hdu_list = []
    for hdu in hdu_list_all:
        if hdu is not None:
            # update first extension with data and WCS
            data = hdu[1].data
            header_info = hdu[1].header
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", AstropyWarning)
                img_wcs = wcs.WCS(header_info)

                header = hdu[0].header
                # remove any CDELT/CD/PC keywords from PHU to avoid conflicts
                for key in list(header.keys()):
                    if key.startswith('CD') or key.startswith('PC') or key.startswith('PV'):
                        del header[key]
                header.update(img_wcs.to_header())
                # ensure CUNIT is deg
                header['CUNIT1'] = 'deg'
                header['CUNIT2'] = 'deg'

                # add some keywords to the PHU from the extension
                header['MAGZRR'] = header_info['MAGZRR']
                header['GAIN'] = header_info['GAIN']
                header['READNOIS'] = header_info['READNOIS']
                # total exposure time of the stack (e.g. deep stacks), for reference
                header['TEXPTIME'] = (_stack_exptime(hdu), "Total exposure time of the stack (s)")
                # calculate effective ZP (considering atmospheric extinction)
                # calculate extinction first
                airmass = (header['AMSTART'] + header['AMEND'])/2
                ext_coeff = header_info['EXTINCT']
                extinction = ext_coeff*(airmass - 1)
                # calculate effective ZP
                zp = header_info['MAGZPT']
                header['MAGZP'] = zp - extinction

                hdu = fits.HDUList([fits.PrimaryHDU(data=data, header=header)])
        hdu_list.append(hdu)
    return hdu_list
