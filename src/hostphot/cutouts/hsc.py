import io
import os
import tarfile
import numpy as np
import requests
from typing import Optional

from astropy.io import fits
from astropy import units as u

from hostphot.surveys_utils import get_survey_filters, check_filters_validity

# HSC SSP cutout API endpoint (PDR3)
_CUTOUT_URL = "https://hsc-release.mtk.nao.ac.jp/das_cutout/pdr3/cgi-bin/cutout"

# Mapping: HostPhot internal filter name → HSC API filter name
_FILTER_API_NAMES = {
    "g":      "HSC-G",
    "r":      "HSC-R",
    "r2":     "HSC-R2",
    "i":      "HSC-I",
    "i2":     "HSC-I2",
    "z":      "HSC-Z",
    "Y":      "HSC-Y",
    "NB387":  "NB0387",
    "NB816":  "NB0816",
    "NB921":  "NB0921",
    "NB1010": "NB1010",
}


def _get_credentials() -> tuple[str, str]:
    """Read HSC SSP credentials from environment variables.

    Returns
    -------
    username, password: Credentials for HTTP Basic Auth.

    Raises
    ------
    ValueError: If either variable is not set.
    """
    username = os.getenv("HSC_SSP_USERNAME")
    password = os.getenv("HSC_SSP_PASSWORD")
    if not username or not password:
        raise ValueError(
            "HSC SSP credentials not found. Please set the environment "
            "variables HSC_SSP_USERNAME and HSC_SSP_PASSWORD.\n"
            "Registration: https://hsc-release.mtk.nao.ac.jp/datasearch/"
        )
    return username, password


def _request_cutout(
    ra: float,
    dec: float,
    sw: float,
    sh: float,
    api_filter: str,
    rerun: str,
    image_type: str,
    auth: tuple[str, str],
) -> fits.HDUList | None:
    """Make a single HSC cutout request and return an open HDUList.

    Parameters
    ----------
    ra, dec: Position in decimal degrees.
    sw, sh: Half-width and half-height of the cutout in degrees.
    api_filter: HSC API filter name, e.g. ``HSC-G``.
    rerun: HSC rerun name, e.g. ``pdr3_wide``.
    image_type: ``coadd`` for science image or ``coadd_variance`` for
                the corresponding variance map.
    auth: (username, password) tuple for HTTP Basic Auth.

    Returns
    -------
    hdulist: Open FITS HDUList, or ``None`` if the request fails or
             no data are available at the given position.
    """
    params = {
        "ra": ra,
        "dec": dec,
        "sw": sw,
        "sh": sh,
        "filter": api_filter,
        "rerun": rerun,
        "type": image_type,
    }
    try:
        resp = requests.get(_CUTOUT_URL, params=params, auth=auth, timeout=120)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 400:
            # Typically means no coverage at this position
            return None
        print(f"Warning: HSC request failed for {api_filter} ({image_type}): {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Warning: HSC request failed for {api_filter} ({image_type}): {e}")
        return None

    raw = resp.content
    if not raw:
        return None

    # The API returns either a bare FITS or a FITS inside a tar archive
    try:
        with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
            fits_members = [m for m in tar.getmembers() if m.name.endswith(".fits")]
            if not fits_members:
                return None
            f = tar.extractfile(fits_members[0])
            return fits.open(io.BytesIO(f.read()))
    except tarfile.TarError:
        pass

    # Fallback: try to open directly as FITS
    try:
        return fits.open(io.BytesIO(raw))
    except Exception:
        return None


def get_HSC_images(
    ra: float,
    dec: float,
    size: float | u.Quantity = 3,
    filters: Optional[list] = None,
    version: str = "pdr3_wide",
) -> list[fits.HDUList] | None:
    """Get HSC SSP coadd image cutouts for the given position and filters.

    Requires the environment variables ``HSC_SSP_USERNAME`` and
    ``HSC_SSP_PASSWORD`` to be set (see :func:`_get_credentials`).

    Parameters
    ----------
    ra: Right Ascension in degrees.
    dec: Declination in degrees.
    size: Image size. If a float, the units are assumed to be arcmin.
    filters: Filters to download. If ``None``, uses all broadband filters
             ``['g', 'r', 'i', 'z', 'Y']``.
    version: HSC rerun / data layer to query. Use ``'pdr3_wide'`` (default)
             for the Wide layer or ``'pdr3_dud'`` for the Deep+UltraDeep
             layer.

    Returns
    -------
    hdu_list: List of :class:`~astropy.io.fits.HDUList` objects, one per
              filter.  Each HDUList has a primary extension (science image)
              and, when available, a first image extension (variance map).
              Entries are ``None`` for filters with no coverage.
    """
    survey = "HSC"
    if filters is None:
        filters = get_survey_filters(survey)
    check_filters_validity(filters, survey)

    # Convert requested size to cutout half-width/height in degrees
    if isinstance(size, (float, int)):
        size_deg = (size * u.arcmin).to(u.degree).value
    else:
        size_deg = size.to(u.degree).value
    # sw/sh are *half*-widths, so halve the full FOV
    sw = sh = size_deg / 2.0

    auth = _get_credentials()

    hdu_list = []
    for filt in filters:
        api_filter = _FILTER_API_NAMES[filt]

        # Science image
        sci_hdu = _request_cutout(ra, dec, sw, sh, api_filter, version, "coadd", auth)
        if sci_hdu is None:
            hdu_list.append(None)
            continue

        # Variance map (best-effort; not all reruns expose it)
        var_hdu = _request_cutout(ra, dec, sw, sh, api_filter, version, "coadd_variance", auth)

        # Build primary HDU from science data
        sci_data = sci_hdu[0].data
        sci_header = sci_hdu[0].header.copy()

        # Derive calibrated zeropoint: FLUXMAG0 is the flux in counts of a
        # zero-magnitude source, so ZP = 2.5 * log10(FLUXMAG0).  The nominal
        # value for HSC PDR3 coadds is 27.0 mag/DN.
        if "FLUXMAG0" in sci_header and sci_header["FLUXMAG0"] > 0:
            sci_header["MAGZP"] = (
                2.5 * np.log10(sci_header["FLUXMAG0"]),
                "Calibrated AB zeropoint [mag]",
            )
        else:
            sci_header["MAGZP"] = (27.0, "Nominal HSC PDR3 AB zeropoint [mag]")

        primary = fits.PrimaryHDU(data=sci_data, header=sci_header)

        if var_hdu is not None:
            var_ext = fits.ImageHDU(
                data=var_hdu[0].data,
                header=var_hdu[0].header,
                name="VARIANCE",
            )
            combined = fits.HDUList([primary, var_ext])
            var_hdu.close()
        else:
            combined = fits.HDUList([primary])

        sci_hdu.close()
        hdu_list.append(combined)

    return hdu_list
