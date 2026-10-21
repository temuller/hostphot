from astropy import wcs
import astropy.units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astroquery.ukidss import Ukidss

import numpy as np

from hostphot.utils import open_fits_from_url, open_fits_from_urls, select_image, url_band
from hostphot.surveys_utils import get_survey_filters, check_filters_validity, survey_pixel_scale

import warnings
from astropy.utils.exceptions import AstropyWarning

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
    all_hdus = open_fits_from_urls(unique_urls)
    url_to_hdu = dict(zip(unique_urls, all_hdus))

    # select the image covering the whole cutout, the deepest one if there are several
    size_arcsec = size.to(u.arcsec).value
    hdu_list_all = []
    for filt in filters:
        candidates = [url_to_hdu[url] for url in unique_urls if url_band(url) == filt]
        candidates = [hdu for hdu in candidates if hdu is not None]
        images = []
        for hdu in candidates:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", AstropyWarning)
                img_wcs = wcs.WCS(hdu[1].header)
            exptime = hdu[0].header.get("EXP_TIME", hdu[1].header.get("EXP_TIME", np.nan))
            nexp = hdu[0].header.get("NEXP", hdu[1].header.get("NEXP", 1))
            images.append((hdu[1].data, img_wcs, exptime * nexp))
        index = select_image(images, ra, dec, size_arcsec, label=f"{survey} {filt}")
        hdu_list_all.append(None if index is None else candidates[index])

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
