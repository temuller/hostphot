import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from pyvo.dal import sia
import astropy.units as u
from astropy.io import fits

import hostphot
hostphot_path = Path(hostphot.__path__[0])
from hostphot.utils import open_fits_from_url, open_fits_from_urls
from hostphot.surveys_utils import get_survey_filters, check_filters_validity

def _get_SPLUS_candidates(
    ra: float, dec: float, fov: float, filters: list[str]
) -> list[list[tuple[str, Optional[str]]]] | None:
    """Obtains the URLs of the S-Plus images (and weight maps) of all the tiles covering the
    given position, for each filter, sorted by the distance of the tile centre to the position.

    Parameters
    ----------
    ra: Right ascension in degrees.
    dec: Declination in degrees.
    fov: Field of view in degrees.
    filters: Filters to use.

    Returns
    -------
    candidates: For each filter, a list of (image URL, weight-map URL) tuples (the weight-map URL
        is ``None`` if not available), or ``None`` if no images are found.
    """
    splus_url = "https://datalab.noirlab.edu/sia/splus_dr1"
    svc = sia.SIAService(splus_url)
    imgs_table = svc.search(
        (ra, dec), (fov / np.cos(dec * np.pi / 180), fov), verbosity=2
    )
    if len(imgs_table) == 0:
        print(("Warning: empty table returned for " f"ra={ra}, dec={dec}"))
        return None

    imgs_df = pd.DataFrame(imgs_table)
    imgs_df = imgs_df[imgs_df.access_format.str.endswith('fits')]
    if len(imgs_df) == 0:
        return None
    # distance of the tile centres to the position
    imgs_df = imgs_df.assign(
        distance=np.hypot((imgs_df.s_ra.astype(float) - ra) * np.cos(np.radians(dec)), imgs_df.s_dec.astype(float) - dec)
    ).sort_values("distance")
    candidates = []
    for filt in filters:
        # the broad-band filters are in upper case in the service (e.g. 'G')
        filt_df = imgs_df[imgs_df.obs_bandpass.str.upper() == filt.upper()]
        is_weight = filt_df.access_url.str.contains("weight")
        weight_urls = list(filt_df[is_weight].access_url.values)
        filt_candidates = []
        for url in filt_df[~is_weight].access_url.values:
            # weight map of the same tile (same URL, except for the file name)
            weight_url = url.replace("_swp.fz", "_swpweight.fz")
            filt_candidates.append((url, weight_url if weight_url in weight_urls else None))
        if len(filt_candidates) == 0:
            warnings.warn(f"No S-PLUS image found for filter '{filt}' at ra={ra}, dec={dec}")
        candidates.append(filt_candidates)
    return candidates


def get_SPLUS_urls(ra: float, dec: float, fov: float | u.Quantity = 3, 
                    filters: Optional[str | list] = None) -> list[str] | None:
    """Obtains the URLs of the S-Plus images (of the tile whose centre is the nearest to
    the given position).
    
    The available filters are: 'F378', 'F395', 'F410', 'F430', 
    'F515', 'F660', 'F861', 'G', 'I', 'R', 'U', 'Z'.

    Parameters
    ----------
    ra: Right ascension in degrees.
    dec: Declination in degrees.
    fov: Field of view in degrees.
    filters: Filters to use. If ``None``, uses all available filters.

    Returns
    -------
    url_list: List of URLs with S-Plus images.
    """
    candidates = _get_SPLUS_candidates(ra, dec, fov, filters)
    if candidates is None:
        return None
    return [filt_candidates[0][0] if len(filt_candidates) > 0 else None for filt_candidates in candidates]


def _covered(weight_hdu: fits.HDUList, ra: float, dec: float) -> bool:
    """Whether the given position is covered by the tile (positive weight)."""
    from astropy.wcs import WCS

    hdu = next(h for h in weight_hdu if h.data is not None)
    x, y = WCS(hdu.header, naxis=2).world_to_pixel_values(ra, dec)
    ix, iy = int(round(float(x))), int(round(float(y)))
    if not (0 <= iy < hdu.data.shape[0] and 0 <= ix < hdu.data.shape[1]):
        return False
    return bool(hdu.data[iy, ix] > 0)


def get_SPLUS_images(ra: float, dec: float, size: float | u.Quantity = 3, 
                    filters: Optional[str] = None) -> fits.HDUList | None:
    """Gets S-Plus fits images for the given coordinates and
    filters.

    S-Plus tiles overlap, and their outer parts can have no coverage (images without real data).
    For each filter, the tiles are tried in order of distance of their centres to the position, and
    the first one covering the position (positive weight) is used. Pixels without coverage (weight
    of zero) are set to NaN.

    The available filters are: 'F378', 'F395', 'F410', 'F430', 
    'F515', 'F660', 'F861', 'G', 'I', 'R', 'U', 'Z'.

    Parameters
    ----------
    ra: Right Ascension in degrees.
    dec: Declination in degrees.
    size: Image size. If a float is given, the units are assumed to be arcmin.
    filters: Filters to use. If ``None``, uses all available filters.

    Returns
    -------
    hdu_list: List of fits images.
    """
    survey = "SPLUS"
    if filters is None:
        filters = get_survey_filters(survey)
    check_filters_validity(filters, survey)
    if isinstance(size, (float, int)):
        fov = (size * u.arcmin).to(u.degree).value
    else:
        fov = size.to(u.degree).value

    candidates = _get_SPLUS_candidates(ra, dec, fov, filters)
    if candidates is None:
        return None
    # file from https://splus.cloud/documentation/dr2_3
    zps_file = hostphot_path.joinpath('filters', 'SPLUS', 'iDR3_zps.cat')
    zps_df = pd.read_csv(zps_file, sep='\\s+')

    hdu_list = []
    for filt, filt_candidates in zip(filters, candidates):
        hdu = None
        for url, weight_url in filt_candidates:
            weight = None
            if weight_url is not None:
                weight = open_fits_from_url(weight_url)
                if not _covered(weight, ra, dec):
                    continue  # the position is not covered by this tile
            hdu = open_fits_from_url(url)
            if weight is not None:
                # pixels without coverage are set to NaN
                weight_data = next(h.data for h in weight if h.data is not None)
                if weight_data.shape == hdu[0].data.shape:
                    data = hdu[0].data.astype(np.float64)
                    data[~(weight_data > 0)] = np.nan
                    hdu[0].data = data
            break
        if hdu is None:
            if len(filt_candidates) > 0:
                warnings.warn(f"No S-PLUS tile covers ra={ra}, dec={dec} in filter '{filt}' (no weight)")
            hdu_list.append(None)
            continue
        # add zeropoint
        field = hdu[0].header['OBJECT'].replace('_', '-')
        field_df = zps_df[zps_df['#field']==field]
        img_filt = hdu[0].header['FILTER']
        # the broad-band filters are in lower case in the zeropoints file (e.g. 'g')
        zp_col = [col for col in zps_df.columns if col.upper() == img_filt.upper()][0]
        zp = field_df[zp_col].values[0]
        hdu[0].header['MAGZP'] = zp
        # initial EXPTIME is normalised, so it doesn't help
        hdu[0].header['EXPTIME'] = hdu[0].header['TEXPOSED']
        hdu_list.append(hdu)
    return hdu_list
