import os
import copy
import tarfile
import requests
import numpy as np
import pandas as pd
import multiprocessing as mp

from astropy.io import fits
from astropy.table import Table
from astropy import wcs, units as u
from astropy.coordinates import SkyCoord

import pyvo  # 2MASS
from pyvo.dal import sia  # DES
from astroquery.skyview import SkyView  # other surveys
from astroquery.mast import Observations  # for GALEX EXPTIME

from reproject import reproject_interp, reproject_exact
from reproject.mosaicking import find_optimal_celestial_wcs

from hostphot._constants import workdir
from hostphot.utils import (
    get_survey_filters,
    clean_dir,
    check_work_dir,
    check_survey_validity,
    check_filters_validity,
    survey_pixel_scale,
)

import warnings
from astropy.utils.exceptions import AstropyWarning


# PS1
# ----------------------------------------
def query_ps1(ra, dec, size=3, filters=None):
    """Query ps1filenames.py service to get a list of images

    Parameters
    ----------
    ra: float
        Right Ascension in degrees.
    dec: float
        Declination in degrees.
    size: float or ~astropy.units.Quantity, default ``3``
        Image size. If a float is given, the units are assumed to be arcmin.
    filters: str, default ``None``
        Filters to use. If ``None``, uses ``grizy``.

    Returns
    -------
    table: astropy Table
        Astropy table with the results.
    """
    survey = "PS1"
    check_filters_validity(filters, survey)
    if filters is None:
        filters = get_survey_filters(survey)

    pixel_scale = survey_pixel_scale(survey)
    if isinstance(size, (float, int)):
        size_arcsec = (size * u.arcmin).to(u.arcsec).value
    else:
        size_arcsec = size.to(u.arcsec).value
    size_pixels = int(size_arcsec / pixel_scale)

    service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    url = (
        f"{service}?ra={ra}&dec={dec}&size={size_pixels}&format=fits&"
        f"filters={filters}"
    )

    table = Table.read(url, format="ascii")

    return table


def get_PS1_urls(ra, dec, size=3, filters=None):
    """Get URLs for images obtained with :func:`query_ps1()`.

    Parameters
    ----------
    ra: float
        Right Ascension in degrees.
    dec: float
        Declination in degrees.
    size: float or ~astropy.units.Quantity, default ``3``
        Image size. If a float is given, the units are assumed to be arcmin.
    filters: str, default ``None``
        Filters to use. If ``None``, uses ``grizy``.

    Returns
    -------
    url_list: list
        List of URLs for the fits images.
    """
    survey = "PS1"
    check_filters_validity(filters, survey)
    if filters is None:
        filters = get_survey_filters(survey)

    pixel_scale = survey_pixel_scale(survey)
    if isinstance(size, (float, int)):
        size_arcsec = (size * u.arcmin).to(u.arcsec).value
    else:
        size_arcsec = size.to(u.arcsec).value
    size_pixels = int(size_arcsec / pixel_scale)

    table = query_ps1(ra, dec, size=size, filters=filters)
    url = (
        "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
        f"ra={ra}&dec={dec}&size={size_pixels}&format=fits"
    )

    # sort filters from blue to red
    flist = ["grizy".find(x) for x in table["filter"]]
    table = table[np.argsort(flist)]

    base_url = url + "&red="
    url_list = []
    for filename in table["filename"]:
        url_list.append(base_url + filename)

    return url_list


def get_PS1_images(ra, dec, size=3, filters=None):
    """Gets PS1 fits images for the given coordinates and
    filters.

    Parameters
    ----------
    ra: float
        Right Ascension in degrees.
    dec: float
        Declination in degrees.
    size: float or ~astropy.units.Quantity, default ``3``
        Image size. If a float is given, the units are assumed to be arcmin.
    filters: str, default ``None``
        Filters to use. If ``None``, uses ``grizy``.

    Returns
    -------
    fits_files: list
        List of fits images.
    """
    survey = "PS1"
    if filters is None:
        filters = get_survey_filters(survey)
    check_filters_validity(filters, survey)

    fits_url = get_PS1_urls(ra, dec, size, filters)

    hdu_list = []
    for url, filt in zip(fits_url, filters):
        hdu = fits.open(url)
        hdu_list.append(hdu)

    return hdu_list


# DES
# ----------------------------------------
def get_DES_urls(ra, dec, fov, filters="grizY"):
    """Obtains the URLs of the DES images+weights with the
    largest exposure times in the given filters.

    Parameters
    ----------
    ra: float
        Right ascension in degrees.
    dec: float
        Declination in degrees.
    fov: float
        Field of view in degrees.
    filters: str, default ``None``
        Filters to use. If ``None``, uses ``grizY``.

    Returns
    -------
    url_list: list
        List of URLs with DES images.
    url_w_list: list
        List of URLs with DES images weights.
    """
    if filters is None:
        filters = get_survey_filters("DES")
    check_filters_validity(filters, "DES")

    des_access_url = "https://datalab.noirlab.edu/sia/des_dr1"
    svc = sia.SIAService(des_access_url)
    imgs_table = svc.search(
        (ra, dec), (fov / np.cos(dec * np.pi / 180), fov), verbosity=2
    )
    if len(imgs_table) == 0:
        print(("Warning: empty table returned for " f"ra={ra}, dec={dec}"))
        return None, None

    imgs_df = pd.DataFrame(imgs_table)

    cond_stack = imgs_df["proctype"] == "Stack"
    cond_image = imgs_df["prodtype"] == "image"
    cond_weight = imgs_df["prodtype"] == "weight"

    url_list = []
    url_w_list = []
    for filt in filters:
        cond_filt = imgs_df["obs_bandpass"] == filt

        selected_images = imgs_df[cond_filt & (cond_stack & cond_image)]
        selected_weights = imgs_df[cond_filt & (cond_stack & cond_weight)]

        if len(selected_images) > 0:
            # pick image with longest exposure time
            max_exptime = np.argmax(selected_images["exptime"])
            row = selected_images.iloc[max_exptime]
            url = row["access_url"]  # get the download URL

            max_exptime = np.argmax(selected_weights["exptime"])
            row_w = selected_weights.iloc[max_exptime]
            url_w = row_w["access_url"]
        else:
            url = url_w = None
        url_list.append(url)
        url_w_list.append(url_w)

    return url_list, url_w_list


def get_DES_images(ra, dec, size=3, filters=None):
    """Gets DES fits images for the given coordinates and
    filters.

    Parameters
    ----------
    ra: float
        Right Ascension in degrees.
    dec: float
        Declination in degrees.
    size: float or ~astropy.units.Quantity, default ``3``
        Image size. If a float is given, the units are assumed to be arcmin.
    filters: str, default ``None``
        Filters to use. If ``None``, uses ``grizY``.

    Returns
    -------
    hdu_list: list
        List of fits images.
    """
    survey = "DES"
    if filters is None:
        filters = get_survey_filters(survey)
    check_filters_validity(filters, survey)

    if isinstance(size, (float, int)):
        fov = (size * u.arcmin).to(u.degree).value
    else:
        fov = size.to(u.degree).value

    url_list, url_w_list = get_DES_urls(ra, dec, fov, filters)

    if url_list is None:
        return None

    hdu_list = []
    for url, url_w in zip(url_list, url_w_list):
        # combine image+weights on a single fits file
        image_hdu = fits.open(url)
        weight_hdu = fits.open(url_w)
        hdu = fits.PrimaryHDU(image_hdu[0].data, header=image_hdu[0].header)
        hdu_err = fits.ImageHDU(
            weight_hdu[0].data, header=weight_hdu[0].header
        )
        hdu_sublist = fits.HDUList([hdu, hdu_err])
        hdu_list.append(hdu_sublist)

    return hdu_list


# SDSS
# ----------------------------------------
def get_SDSS_images(ra, dec, size=3, filters=None):
    """Downloads a set of SDSS fits images for a given set
    of coordinates and filters using SkyView.

    Parameters
    ----------
    ra: str or float
        Right ascension in degrees.
    dec: str or float
        Declination in degrees.
    size: float or ~astropy.units.Quantity, default ``3``
        Image size. If a float is given, the units are assumed to be arcmin.
    filters: str, default ``None``
        Filters to use. If ``None``, uses ``ugriz``.

    Return
    ------
    hdu_list: list
        List with fits images for the given filters.
        `None` is returned if no image is found.
    """
    survey = "SDSS"
    if filters is None:
        filters = get_survey_filters(survey)
    check_filters_validity(filters, survey)

    # SkyView calls the filters in a different way
    filters_dict = {
        "u": "SDSSu",
        "g": "SDSSg",
        "r": "SDSSr",
        "i": "SDSSi",
        "z": "SDSSz",
    }
    skyview_filters = [filters_dict[filt] for filt in filters]

    if isinstance(size, (float, int)):
        size_arcsec = (size * u.arcmin).to(u.arcsec)
    else:
        size_arcsec = size.to(u.arcsec)

    pixel_scale = survey_pixel_scale(survey)
    size_pixels = int(size_arcsec.value / pixel_scale)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        coords = SkyCoord(
            ra=ra, dec=dec, unit=(u.degree, u.degree), frame="icrs"
        )

        hdu_list = []
        for filt, skyview_filter in zip(filters, skyview_filters):
            hdu = SkyView.get_images(
                position=coords,
                coordinates="icrs",
                pixels=str(size_pixels),
                survey=skyview_filter,
                width=size_arcsec,
                height=size_arcsec,
            )

            hdu_list.append(hdu[0])

    return hdu_list


# GALEX
# ----------------------------------------
def get_used_image(header):
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


def get_exptime(used_image, obs_table, filt):
    """Obtains the exposure time for a GALEX image downloaded
    with SkyView.

    Parameters
    ----------
    used_image: str
        GALEX name of the image downloaded by SkyView
    obs_table: astropy.Table
        Table obtained with astroquery.Observations.
    filt: str
        GALEX filter: 'NUV' or 'FUV'.

    Returns
    -------
    texp: float
        Exposure time.
    """
    galex_df = obs_table.to_pandas()
    galex_df = galex_df[galex_df.filters == filt]

    target_df = galex_df[galex_df.target_name == used_image]
    texp = target_df.t_exptime.values[0]

    return texp


def get_GALEX_images(ra, dec, size=3, filters=None):
    """Downloads a set of GALEX fits images for a given set
    of coordinates and filters using SkyView.

    Parameters
    ----------
    ra: str or float
        Right ascension in degrees.
    dec: str or float
        Declination in degrees.
    size: float or ~astropy.units.Quantity, default ``3``
        Image size. If a float is given, the units are assumed to be arcmin.
    filters: str, default ``None``
        Filters to use. If ``None``, uses ``FUV, NUV``.

    Return
    ------
    hdu_list: list
        List with fits images for the given filters.
        `None` is returned if no image is found.
    """
    survey = "GALEX"
    if filters is None:
        filters = get_survey_filters(survey)
    check_filters_validity(filters, survey)

    # SkyView calls the filters in a different way
    filters_dict = {"NUV": "GALEX Near UV", "FUV": "GALEX Far UV"}
    skyview_filters = [filters_dict[filt] for filt in filters]

    if isinstance(size, (float, int)):
        size_arcsec = (size * u.arcmin).to(u.arcsec)
    else:
        size_arcsec = size.to(u.arcsec)

    pixel_scale = survey_pixel_scale(survey)
    size_pixels = int(size_arcsec.value / pixel_scale)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        coords = SkyCoord(
            ra=ra, dec=dec, unit=(u.degree, u.degree), frame="icrs"
        )
        obs_table = Observations.query_criteria(
            coordinates=coords, radius=size_arcsec, obs_collection=[survey]
        )

        skyview_hdu_list = SkyView.get_images(
            position=coords,
            coordinates="icrs",
            pixels=str(size_pixels),
            survey=skyview_filters,
            width=size_arcsec,
            height=size_arcsec,
        )

        hdu_list = []
        for filt, hdu in zip(filters, skyview_hdu_list):
            # add exposure time
            used_image = get_used_image(hdu[0].header)
            texp = get_exptime(used_image, obs_table, filt)
            if texp is None:
                return None
            hdu[0].header["EXPTIME"] = texp
            hdu[0].header["COMMENT"] = "EXPTIME added by HostPhot"

            hdu_list.append(hdu)

    return hdu_list


# WISE
# ----------------------------------------
def get_WISE_images(ra, dec, size=3, filters=None):
    """Downloads a set of WISE fits images for a given set
    of coordinates and filters using SkyView.

    Parameters
    ----------
    ra: str or float
        Right ascension in degrees.
    dec: str or float
        Declination in degrees.
    size: float or ~astropy.units.Quantity, default ``3``
        Image size. If a float is given, the units are assumed to be arcmin.
    filters: str, default ``None``
        Filters to use. If ``None``, uses all WISE filters.

    Return
    ------
    hdu_list: list
        List with fits images for the given filters.
        `None` is returned if no image is found.
    """
    survey = "WISE"
    if filters is None:
        filters = get_survey_filters(survey)
    check_filters_validity(filters, survey)

    if isinstance(size, (float, int)):
        size_arcsec = (size * u.arcmin).to(u.arcsec)
    else:
        size_arcsec = size.to(u.arcsec)

    pixel_scale = survey_pixel_scale(survey)
    size_pixels = int(size_arcsec.value / pixel_scale)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        coords = SkyCoord(
            ra=ra, dec=dec, unit=(u.degree, u.degree), frame="icrs"
        )

        skyview_fits = SkyView.get_images(
            position=coords,
            coordinates="icrs",
            pixels=str(size_pixels),
            survey="WISE 3.4",
            width=size_arcsec,
            height=size_arcsec,
        )
        header = skyview_fits[0][0].header

        coadd_id = get_used_image(header)
        coadd_id1 = coadd_id[:4]
        coadd_id2 = coadd_id1[:2]

        # for more info: https://irsa.ipac.caltech.edu/ibe/docs/wise/allwise/p3am_cdd/#sample_code
        base_url = (
            "http://irsa.ipac.caltech.edu/ibe/data/wise/allwise/p3am_cdd/"
        )
        coadd_url = os.path.join(coadd_id2, coadd_id1, coadd_id)
        params_url = f"center={ra},{dec}&size={size_arcsec.value}arcsec&gzip=0"  # center and size of the image

        hdu_list = []
        for filt in filters:
            i = filt[-1]
            band_url = f"{coadd_id}-w{i}-int-3.fits"
            url = os.path.join(
                base_url, coadd_url, band_url + "?" + params_url
            )
            hdu = fits.open(url)
            hdu_list.append(hdu)

    return hdu_list

def get_unWISE_images(ra, dec, size=3, filters=None, version="allwise"):
    """Downloads a set of unWISE fits images for a given set
    of coordinates and filters using SkyView.

    Parameters
    ----------
    ra: str or float
        Right ascension in degrees.
    dec: str or float
        Declination in degrees.
    size: float or ~astropy.units.Quantity, default ``3``
        Image size. If a float is given, the units are assumed to be arcmin.
    filters: str, default ``None``
        Filters to use. If ``None``, uses all WISE filters.
    version: str, default ``allwise``
        Version of the unWISE images. Either ``allwise``, ``neo1`` or ``neo2``.

    Return
    ------
    hdu_list: list
        List with fits images for the given filters.
        `None` is returned if no image is found.
    """
    survey = f"unWISE{version}"
    if filters is None:
        filters = get_survey_filters(survey)
    check_filters_validity(filters, survey)

    if version in ["neo1", "neo2"]:
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

    bands = ''.join(filt[-1] for filt in filters)  # e.g. 1234

    # for more info: http://unwise.me/imgsearch/
    base_url = "http://unwise.me/cutout_fits?"
    params_url = f"version={version}&ra={ra}&dec={dec}&size={size_pixels}&bands={bands}"
    master_url = base_url + params_url

    response = requests.get(master_url, stream=True)
    target_file = 'unWISE_images.tar.gz'  # current directory
    if response.status_code == 200:
        with open(target_file, 'wb') as f:
            f.write(response.raw.read())

    hdu_list = []
    with tarfile.open(target_file) as tar_file:
        files_list = tar_file.getnames()
        for fits_file in files_list:
            for filt in filters:
                if f'{filt.lower()}-img-m.fits' in fits_file:
                    tar_file.extract(fits_file, '.')
                    hdu = fits.open(fits_file)
                    hdu_list.append(hdu)
                    os.remove(fits_file)

    os.remove(target_file)

    return hdu_list


# 2MASS
# ----------------------------------------
def get_2MASS_images(ra, dec, size=3, filters=None):
    """Downloads a set of 2MASS fits images for a given set
    of coordinates and filters using SkyView.

    Parameters
    ----------
    ra: str or float
        Right ascension in degrees.
    dec: str or float
        Declination in degrees.
    size: float or ~astropy.units.Quantity, default ``3``
        Image size. If a float is given, the units are assumed to be arcmin.
    filters: str, default ``None``
        Filters to use. If ``None``, uses ``FUV, NUV``.

    Return
    ------
    hdu_list: list
        List with fits images for the given filters.
        `None` is returned if no image is found.
    """
    survey = "2MASS"
    if filters is None:
        filters = get_survey_filters(survey)
    check_filters_validity(filters, survey)

    if isinstance(size, (float, int)):
        size_degree = (size * u.arcmin).to(u.degree)
    else:
        size_degree = size.to(u.degree)
    size_degree = size_degree.value

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        coords = SkyCoord(
            ra=ra, dec=dec, unit=(u.degree, u.degree), frame="icrs"
        )

        twomass_services = pyvo.regsearch(
            servicetype="image", keywords=["2mass"]
        )
        table = twomass_services[0].search(pos=coords, size=size_degree)
        twomass_df = table.to_table().to_pandas()
        twomass_df = twomass_df[twomass_df.format == "image/fits"]

        # for more info: https://irsa.ipac.caltech.edu/ibe/docs/twomass/allsky/allsky/#main
        base_url = (
            "https://irsa.ipac.caltech.edu/ibe/data/twomass/allsky/allsky"
        )

        hdu_list = []
        for i, filt in enumerate(filters):
            band_df = twomass_df[twomass_df.band == filt]
            fname = band_df.download.values[0].split("=")[-1]
            hemisphere = band_df.hem.values[0]
            ordate = band_df.date.values[0]
            scanno = band_df.scan.values[0]

            tile_url = os.path.join(f"{ordate}{hemisphere}", f"s{scanno}")
            fits_url = os.path.join("image", f"{fname}.gz")
            params_url = f"center={ra},{dec}&size={size_degree}degree&gzip=0"  # center and size of the image

            url = os.path.join(base_url, tile_url, fits_url + "?" + params_url)
            hdu = fits.open(url)
            hdu_list.append(hdu)

    return hdu_list


# HST
# ----------------------------------------
def get_HST_images(ra, dec, size=3, filters=None):
    """Downloads a set of HST fits images for a given set
    of coordinates and filters using SkyView.

    Parameters
    ----------
    ra: str or float
        Right ascension in degrees.
    dec: str or float
        Declination in degrees.
    size: float or ~astropy.units.Quantity, default ``3``
        Image size. If a float is given, the units are assumed to be arcmin.
    filters: str, default ``None``
        Filters to use.

    Return
    ------
    hdu_list: list
        List with fits images for the given filters.
        `None` is returned if no image is found.
    """
    survey = "HST"
    if filters is None:
        filters = get_survey_filters(survey)
    check_filters_validity(filters, survey)

    if isinstance(size, (float, int)):
        size_degree = (size * u.arcmin).to(u.degree)
    else:
        size_degree = size.to(u.degree)
    size_degree = size_degree.value

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        coords = SkyCoord(
            ra=ra, dec=dec, unit=(u.degree, u.degree), frame="icrs"
        )

    obs_table = Observations.query_criteria(
        coordinates=coords, radius=size_degree, obs_collection=["HST"]
    )
    obs_df = obs_table.to_pandas()

    df_list = []
    for inst in obs_df.instrument_name.unique():
        inst_df = obs_df[obs_df.instrument_name == inst]

        for filt in inst_df.filters.unique():
            if not filt.startswith("F") or ";" in filt:
                continue  # skip these odd filters
            filt_df = inst_df[inst_df.filters == filt]

            dist = np.sqrt(
                (filt_df.s_dec.values - dec) ** 2
                + (filt_df.s_ra.values - ra) ** 2
            )
            filt_df = filt_df[
                dist == dist.min()
            ]  # take the most centred image
            filt_df = filt_df[:1]  # take first image
            df_list.append(filt_df)
    obs_df = pd.concat(df_list)

    # identify products to download
    data_products = Observations.get_product_list(Table.from_pandas(obs_df))
    dp_df = data_products.to_pandas()
    dp_df = dp_df[dp_df.productType == "SCIENCE"]

    df_list = []
    for filt in obs_df.filters.unique():
        if filt.lower() in dp_df.dataURI.values:
            filt_df = dp_df[dp_df.dataURI.contains(filt.lower())]
        elif filt.upper() in dp_df.dataURI.values:
            filt_df = dp_df[dp_df.dataURI.contains(filt.upper())]
        else:
            continue

        if "HLA" in filt_df.obs_collection:
            # choose images from the legacy archive
            filt_df = filt_df[filt_df.obs_collection == "HLA"]

        filt_df = filt_df[:1]  # take first image
        df_list.append(filt_df)
    dp_df = pd.concat(df_list)

    # download images
    Observations.download_products(
        Table.from_pandas(dp_df), productType="SCIENCE", extension="fits"
    )

    #return hdu_list


# Check orientation
# ------------------
def match_wcs(fits_files):
    """Matches the WCS of all the images, taking the first ones as
    a reference.

    Parameters
    ----------
    fits_files: list
        List of fits files.

    Returns
    -------
    matched_fits_files: list
        List of fits files with matched WCS.
    """
    matched_fits_files = copy.deepcopy(fits_files)
    # some hdu have data + error
    hdu0 = matched_fits_files[0][0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        wcs0 = wcs.WCS(hdu0.header)

        for hdu in matched_fits_files[1:]:
            data, footprint = reproject_interp(hdu[0], hdu0.header)
            hdu[0].data = data
            hdu[0].header.update(wcs0.to_header())

    return matched_fits_files


# Master Function
# ----------------------------------------
def download_images(
    name, ra, dec, size=3, filters=None, overwrite=False, survey="PS1"
):
    """Download images for a given object in the given filters of a
    given survey.

    Parameters
    ----------
    name: str
        Name used for tracking the object in your local
        directory.
    ra: float
        Right ascension in degrees.
    dec: float
        Declination in degrees.
    size: float or ~astropy.units.Quantity, default ``3``
        Image size. If a float is given, the units are assumed to be arcmin.
    filters: str, default ``None``
        Filters for the images.
    overwrite: bool, default ``False``
        If ``True``, the images are overwritten if they already
        exist.
    survey: str, default ``PS1``
        Survey used to download the images

    Examples
    --------
    >>> from hostphot.cutouts import download_images
    >>> name = 'SN2004eo'
    >>> host_ra, host_dec = 308.2092, 9.92755  # coords of host galaxy of SN2004eo
    >>> survey = 'PS1'
    >>> download_images(name, host_ra, host_dec, survey=survey)
    """
    check_survey_validity(survey)
    check_filters_validity(filters, survey)
    if filters is None:
        filters = get_survey_filters(survey)

    check_work_dir(workdir)
    obj_dir = os.path.join(workdir, name)
    if not os.path.isdir(obj_dir):
        os.mkdir(obj_dir)

    # download fits files
    if survey == "PS1":
        hdu_list = get_PS1_images(ra, dec, size, filters)
    elif survey == "DES":
        hdu_list = get_DES_images(ra, dec, size, filters)
    elif survey == "SDSS":
        hdu_list = get_SDSS_images(ra, dec, size, filters)
    elif survey == "GALEX":
        hdu_list = get_GALEX_images(ra, dec, size, filters)
    elif "WISE" in survey:
        if survey == "WISE":
            hdu_list = get_WISE_images(ra, dec, size, filters)
        else:
            wise_version = survey.split("WISE")[-1]
            hdu_list = get_unWISE_images(ra, dec, size,
                                           filters, wise_version)
    elif survey == "2MASS":
        hdu_list = get_2MASS_images(ra, dec, size, filters)

    if hdu_list:
        # this corrects any possible shifts between the images
        # fits_files = match_wcs(fits_files)

        # fix wcs (some have rotated wcs)
        if survey in ["XXX"]:
            for hdu in hdu_list:
                fits_file = hdu[0]
                wcs_out, shape_out = find_optimal_celestial_wcs(
                    [fits_file], auto_rotate=True
                )
                fixed_data, footprint = reproject_exact(
                    fits_file, wcs_out, shape_out=shape_out
                )
                fits_file.header.update(wcs_out.to_header())
                fits_file.data = fixed_data

        for hdu, filt in zip(hdu_list, filters):
            outfile = os.path.join(obj_dir, f"{survey}_{filt}.fits")
            if not os.path.isfile(outfile):
                hdu.writeto(outfile)
            else:
                if overwrite:
                    hdu.writeto(outfile, overwrite=overwrite)
                else:
                    continue

    # remove directory if it ends up empty
    clean_dir(obj_dir)


def pool_download(
    df=None,
    name=None,
    ra=None,
    dec=None,
    size=600,
    filters=None,
    overwrite=False,
    survey="PS1",
    processes=8,
):
    """Downloads images for multiple objects using parallelisation.

    Parameters
    ----------
    df: DataFrame, default ``None``
        DataFrame with the values of the argmuents. If this is given,
        ``name``, ``ra`` and ``dec`` should be the names of the columns in
        ``df``.
    name: str or list-like, default ``None``
        Name used for tracking the object in your local
        directory.
    ra: float or list-like, default ``None``
        Right ascension in degrees of the center of the image.
    dec: float or list-like, default ``None``
        Declination in degrees of the center of the image.
    size: int, default ``600``
        Image size in pixels.
    filters: str, default ``None``
        Filters for the images. If ``None``, use all the available
        filters.
    overwrite: bool, default ``False``
        If ``True``, the images are overwritten if they already
        exist.
    survey: str, default ``PS1``
        Survey used to download the images
    processes: floar, default ``8``
        Number of processes to use for the parallelisation.
    """
    local_dict = locals()  # get dictionary of input arguments
    variable_args = ["name", "ra", "dec"]
    ignore_args = ["df", "processes"]
    args_dict = {
        key: local_dict[key]
        for key in local_dict.keys()
        if key not in ignore_args
    }

    # adapt the input to feed it to the parallelisation function
    input_dict = args_dict.copy()
    if df is None:
        length = len(name)
    else:
        length = len(df)
        for key in variable_args:
            var_name = args_dict[key]
            input_dict[key] = df[var_name].values

    for key in args_dict.keys():
        if key not in variable_args:
            input_dict[key] = [args_dict[key]] * length

    # transpose list
    input_args = list(map(list, zip(*input_dict.values())))

    pool = mp.Pool(processes)
    pool.starmap(download_images, (args for args in input_args))
