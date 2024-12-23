import tarfile
import requests
from pathlib import Path
from typing import Optional

import astropy.units as u
from astropy.io import fits
from astroquery.skyview import SkyView
from astropy.coordinates import SkyCoord

from hostphot.surveys_utils import get_survey_filters, check_filters_validity, survey_pixel_scale

import warnings
from astropy.utils.exceptions import AstropyWarning

def get_used_image(header: fits.header.Header) -> str:
    """Obtains the name of the image downloaded by SkyView.

    Parameters
    ----------
    header: fits header
        Header of an image.

    Returns
    -------
    used_image: str
        Name of the image.
    """
    image_line = header["HISTORY"][-3]
    used_image = image_line.split("/")[-1].split("-")[0]

    return used_image

def get_WISE_images(ra: float, dec: float, size: float | u.Quantity = 3, 
                    filters: Optional[str] = None) -> list[fits.ImageHDU]:
    """Downloads a set of WISE fits images for a given set
    of coordinates and filters.

    Parameters
    ----------
    ra: Right ascension in degrees.
    dec: Declination in degrees.
    size: Image size. If a float is given, the units are assumed to be arcmin.
    filters: Filters to use. If ``None``, uses all WISE filters.

    Return
    ------
    hdu_list: List with fits images for the given filters. ``None`` is returned if no image is found.
    """
    survey = "WISE"
    if filters is None:
        filters = get_survey_filters(survey)
    check_filters_validity(filters, survey)

    if isinstance(size, (float, int)):
        size_arcsec = (size * u.arcmin).to(u.arcsec)
    else:
        size_arcsec = size.to(u.arcsec)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        coords = SkyCoord(
            ra=ra, dec=dec, unit=(u.degree, u.degree), frame="icrs"
        )
        # just to get the original image used
        skyview_fits = SkyView.get_images(
            position=coords,
            coordinates="icrs",
            pixels="100",
            survey="WISE 3.4",
        )
        header = skyview_fits[0][0].header
        coadd_id = get_used_image(header)
        coadd_id1 = coadd_id[:4]
        coadd_id2 = coadd_id1[:2]

        # for more info: https://irsa.ipac.caltech.edu/ibe/docs/wise/allwise/p3am_cdd/#sample_code
        base_url = (
            "http://irsa.ipac.caltech.edu/ibe/data/wise/allwise/p3am_cdd/"
        )
        coadd_url = Path(coadd_id2, coadd_id1, coadd_id)
        params_url = f"center={ra},{dec}&size={size_arcsec.value}arcsec&gzip=0"  # center and size of the image
        hdu_list = []
        for filt in filters:
            i = filt[-1]
            band_url = f"{coadd_id}-w{i}-int-3.fits"
            url = Path(
                base_url, coadd_url, band_url + "?" + params_url
            )
            print(url)
            hdu = fits.open(url)
            hdu_list.append(hdu)
    return hdu_list

def get_unWISE_images(ra: float, dec: float, size: float | u.Quantity = 3, 
                    filters: Optional[str] = None, version: str = "allwise") -> list[fits.ImageHDU]:
    """Downloads a set of unWISE fits images for a given set
    of coordinates and filters.

    Parameters
    ----------
    ra: Right ascension in degrees.
    dec: Declination in degrees.
    size: Image size. If a float is given, the units are assumed to be arcmin.
    filters: Filters to use. If ``None``, uses all WISE filters.
    version:  Version of the unWISE images. Either ``allwise`` or ``neo{i}`` for {i} = 1 to 7.

    Return
    ------
    hdu_list: List with fits images for the given filters. ``None`` is returned if no image is found.
    """
    survey = "unWISE"
    if version is None:
        version = "allwise"
    else:
        # check validity of the version used
        neo_versions = [f"neo{i}" for i in range(1, 8)]
        all_versions = ["allwise"] + neo_versions
        assert (
            version in all_versions
        ), f"Not a valid version ({version}): {all_versions}"
    if filters is None:
        filters = get_survey_filters(survey)
    check_filters_validity(filters, survey)

    if "neo" in version:
        # these only have W1 and W2 data
        if "W3" in filters:
            filters.remove("W3")
        if "W4" in filters:
            filters.remove("W4")

    if isinstance(size, (float, int)):
        size_arcsec = (size * u.arcmin).to(u.arcsec)
    else:
        size_arcsec = size.to(u.arcsec)
    pixel_scale = survey_pixel_scale(survey)
    size_pixels = int(size_arcsec.value / pixel_scale)
    assert size_pixels <= 1024, "Maximum cutout size for unWISE is 1024 pixels"

    bands = "".join(filt[-1] for filt in filters)  # e.g. 1234
    # for more info: http://unwise.me/imgsearch/
    base_url = "http://unwise.me/cutout_fits?"
    params_url = (
        f"version={version}&ra={ra}&dec={dec}&size={size_pixels}&bands={bands}"
    )
    master_url = base_url + params_url

    response = requests.get(master_url, stream=True)
    target_file = Path(f"unWISE_images_{ra}_{dec}.tar.gz")  # current directory
    if response.status_code == 200:
        with open(target_file, "wb") as f:
            f.write(response.raw.read())

    hdu_list = []
    with tarfile.open(target_file) as tar_file:
        files_list = tar_file.getnames()
        for fits_file in files_list:
            for filt in filters:
                if f"{filt.lower()}-img-m.fits" in fits_file:
                    tar_file.extraction_filter = (lambda member, path: member)
                    tar_file.extract(fits_file, ".")
                    hdu = fits.open(fits_file)
                    hdu_list.append(hdu)
                    Path(fits_file).unlink()  # remove file
    # remove the tarfile
    if target_file.is_file() is True:
        target_file.unlink()
    return hdu_list