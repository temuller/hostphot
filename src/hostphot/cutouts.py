import os
import copy
import gzip
import shutil
import tarfile
import requests  # for several surveys
import numpy as np
import pandas as pd

# for VISTA
import re
import urllib

from astropy.io import fits
from astropy.table import Table
from astropy import wcs, units as u
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D

import pyvo  # 2MASS
from pyvo.dal import sia  # DES, DELVE
from astroquery.sdss import SDSS
from astroquery.skyview import SkyView  # other surveys
from astroquery.mast import Observations  # for GALEX

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

    if isinstance(size, (float, int)):
        size_arcsec = (size * u.arcmin).to(u.arcsec)
    else:
        size_arcsec = size.to(u.arcsec)

    pixel_scale = survey_pixel_scale(survey)
    size_pixels = int(size_arcsec.value / pixel_scale)

    pos = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)
    for radius in np.arange(1, 30):
        ids = SDSS.query_region(pos, radius=radius * u.arcsec)
        if ids is not None:
            break

    if ids is None:
        return None

    # get images
    hdu_id = len(ids) // 2
    if hdu_id == 0:
        hdu_list = SDSS.get_images(matches=ids, band=filters)
    else:
        sub_id = ids[hdu_id - 1 : hdu_id]  # get the one in the middle
        hdu_list = SDSS.get_images(matches=sub_id, band=filters)

    # SDSS images are large so need to be trimmed
    for hdu in hdu_list:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AstropyWarning)
            img_wcs = wcs.WCS(hdu[0].header)
        pos = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)

        trimmed_data = Cutout2D(hdu[0].data, pos, size_pixels, img_wcs)
        hdu[0].data = trimmed_data.data
        hdu[0].header.update(trimmed_data.wcs.to_header())

    return hdu_list


# GALEX
# ----------------------------------------
def get_img_dist(host_ra, host_dec, header):
    """Obtains the distance from the image center
    to the galaxy position.

    Parameters
    ----------
    host_ra: str or float
        Right ascension of the galaxy in degrees.
    host_dec: str or float
        Declination of the galaxy in degrees.
    header: fits header
        Header of an image.

    Returns
    -------
    dist_from_center: float
        Distance of from the image center to the galaxy
        position in pixel units.
    """
    # coordinates of the image center
    img_wcs = wcs.WCS(header)
    img_ra, img_dec = header['RA_CENT'], header['DEC_CENT']
    img_coord = SkyCoord(
        ra=img_ra, dec=img_dec, unit=(u.degree, u.degree), frame="icrs"
    )
    img_x, img_y = img_wcs.world_to_pixel(img_coord)

    # coordinates of the galaxy
    host_coord = SkyCoord(
        ra=host_ra, dec=host_dec, unit=(u.degree, u.degree), frame="icrs"
    )
    host_x, host_y = img_wcs.world_to_pixel(host_coord)

    dist_from_center = np.sqrt((img_x - host_x) ** 2 + (img_y - host_y) ** 2)

    return dist_from_center

def get_GALEX_images(ra, dec, size=3, filters=None, version=None):
    """Downloads a set of GALEX fits images for a given set
    of coordinates and filters.

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
    version: str, default ``None``
        Version of GALEX images. Either Deep (``DIS``), Medium (``MIS``) or
        All-Sky Imaging Survey (``AIS``), or Nearby Galaxy Survey (``NGS``) or
        Guest Investigator Survey (``GII``). By default, ``AIS`` is used.

    Return
    ------
    hdu_list: list
        List with fits images for the given filters.
        ``None`` is returned if no image is found.
    """
    survey = "GALEX"
    if version is None:
        version = 'AIS'
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

    obs_table = Observations.query_criteria(
        coordinates=coords, radius=size_arcsec, obs_collection=["GALEX"]
    )
    obs_df = obs_table.to_pandas()
    print(obs_df.jpegURL.values)
    print(obs_df.dataURL.values)
    print(obs_df.project.values)
    obs_df = obs_df[obs_df.project == version]  # only all sky survey

    # get unique image sectors
    nuv_files, fuv_files = [], []
    for file in obs_df.dataURL.values:
        nuv_file = '-'.join(string for string in file.split('-')[:-2]) + '-nd-int.fits.gz'
        if nuv_file not in nuv_files:
            nuv_files.append(nuv_file)
            fuv_file = nuv_file.replace('-nd-', '-fd-')
            fuv_files.append(fuv_file)

    # download the FITS images
    nuv_hdu_list, fuv_hdu_list = [], []
    for nuv_file, fuv_file in zip(nuv_files, fuv_files):
        try:
            nuv_hdu = fits.open(nuv_file)
            nuv_hdu_list.append(nuv_hdu)
        except:
            pass
        try:
            fuv_hdu = fits.open(fuv_file)
            fuv_hdu_list.append(fuv_hdu)
        except:
            pass

    # calculate the distance of the galaxy to the image center
    nuv_distances = []
    for hdu in nuv_hdu_list:
        header = hdu[0].header
        dist_from_center = get_img_dist(ra, dec, header)
        nuv_distances.append(dist_from_center)

    fuv_distances = []
    for hdu in fuv_hdu_list:
        header = hdu[0].header
        dist_from_center = get_img_dist(ra, dec, header)
        fuv_distances.append(dist_from_center)

    # get the image with the galaxy closest to the center
    if len(nuv_distances)!=0:
        id_nuv_file = np.argmin(np.array(nuv_distances))
        nuv_hdu = nuv_hdu_list[id_nuv_file]
    else:
        nuv_hdu = None

    if len(fuv_distances)!=0:
        id_fuv_file = np.argmin(np.array(fuv_distances))
        fuv_hdu = fuv_hdu_list[id_fuv_file]
    else:
        fuv_hdu = None

    hdu_dict = {'NUV': nuv_hdu, 'FUV': fuv_hdu}
    hdu_list = []
    for filt in filters:
        hdu = hdu_dict[filt]
        if hdu is None:
            continue  # no image in this filter
        # trim data to requested size
        img_wcs = wcs.WCS(hdu[0].header)
        pos = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)

        trimmed_data = Cutout2D(hdu[0].data, pos, size_pixels, img_wcs)
        hdu[0].data = trimmed_data.data
        hdu[0].header.update(trimmed_data.wcs.to_header())
        hdu_list.append(hdu)

    return hdu_list


# WISE
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


def get_WISE_images(ra, dec, size=3, filters=None):
    """Downloads a set of WISE fits images for a given set
    of coordinates and filters.

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
            #width=size_arcsec,
            #height=size_arcsec,
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
    of coordinates and filters.

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
    survey = "unWISE"
    if version is None:
        version = 'allwise'
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

    bands = "".join(filt[-1] for filt in filters)  # e.g. 1234

    # for more info: http://unwise.me/imgsearch/
    base_url = "http://unwise.me/cutout_fits?"
    params_url = (
        f"version={version}&ra={ra}&dec={dec}&size={size_pixels}&bands={bands}"
    )
    master_url = base_url + params_url

    response = requests.get(master_url, stream=True)
    target_file = "unWISE_images.tar.gz"  # current directory
    if response.status_code == 200:
        with open(target_file, "wb") as f:
            f.write(response.raw.read())

    hdu_list = []
    with tarfile.open(target_file) as tar_file:
        files_list = tar_file.getnames()
        for fits_file in files_list:
            for filt in filters:
                if f"{filt.lower()}-img-m.fits" in fits_file:
                    tar_file.extract(fits_file, ".")
                    hdu = fits.open(fits_file)
                    hdu_list.append(hdu)
                    os.remove(fits_file)

    os.remove(target_file)

    return hdu_list


# 2MASS
# ----------------------------------------
def get_2MASS_images(ra, dec, size=3, filters=None):
    """Downloads a set of 2MASS fits images for a given set
    of coordinates and filters.

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
            if filt=='Ks':
                filt = 'K'
            band_df = twomass_df[twomass_df.band == filt]
            if len(band_df) == 0:
                # no data for this band:
                hdu_list.append(None)
                continue

            fname = band_df.download.values[0].split("=")[-1]
            hemisphere = band_df.hem.values[0]
            ordate = band_df.date.values[0]
            scanno = band_df.scan.values[0]
            # add leading zeros for scanno bellow 100
            n_zeros = 3 - len(str(scanno))
            scanno = n_zeros * '0' + str(scanno)

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

    # return hdu_list


# Legacy Survey
def get_LegacySurvey_images(ra, dec, size=3, filters=None, version='dr10'):
    """Gets Legacy Survey fits images for the given coordinates and
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
        Filters to use. If ``None``, uses ``grz``.
    version: str, default ``dr10``
        Data release version. E.g. ``dr10``, ``dr9``, ``dr8``, etc..

    Returns
    -------
    fits_files: list
        List of fits images.
    """
    survey = "LegacySurvey"
    if filters is None:
        filters = get_survey_filters(survey)
        check_filters_validity(filters, survey)

    pixel_scale = survey_pixel_scale(survey)
    if isinstance(size, (float, int)):
        size_arcsec = (size * u.arcmin).to(u.arcsec).value
    else:
        size_arcsec = size.to(u.arcsec).value
    size_pixels = int(size_arcsec / pixel_scale)

    if version is None:
        version = 'dr10'  # latest data release

    base_url = "https://www.legacysurvey.org/viewer/fits-cutout?"
    params = f"ra={ra}&dec={dec}&layer=ls-{version}&pixscale={pixel_scale}&bands={filters}&size={size_pixels}"
    url = base_url + params
    master_hdu = fits.open(url)
    header = master_hdu[0].header

    hdu_list = []
    # assuming order grz of the filters
    for i, filt in enumerate(filters):
        data = master_hdu[0].data[i]
        hdu = fits.PrimaryHDU(data=data, header=header)
        hdu_list.append(hdu)

    return hdu_list


# Spitzer
def get_Spitzer_images(ra, dec, size=3, filters=None):
    """Gets Spitzer fits images for the given coordinates and
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
        Filters to use. If ``None``, uses ``IRAC.1, IRAC.2, IRAC.3, IRAC.4, MIPS.1``.

    Returns
    -------
    fits_files: list
        List of fits images.
    """
    survey = "Spitzer"
    if filters is None:
        filters = get_survey_filters(survey)
    check_filters_validity(filters, survey)

    pixel_scale = survey_pixel_scale(survey, "IRAC.1")  # IRAC resolution
    if isinstance(size, (float, int)):
        size_arcsec = (size * u.arcmin).to(u.arcsec).value
    else:
        size_arcsec = size.to(u.arcsec).value
    size_pixels = int(size_arcsec / pixel_scale)

    base_url = "https://irsa.ipac.caltech.edu/cgi-bin/Cutouts/nph-cutouts?"
    locstr = f"{str(ra)}+{str(dec)}+eq"

    mission_dict = {
        "mission": "SEIP",
        "min_size": "18",
        "max_size": "1800",
        "units": "arcsec",
        "locstr": locstr,
        "sizeX": size_pixels,
        "ntable_cutouts": "1",
        "cutouttbl1": "science",
        "mode": "PI",  # XML output file
    }

    # check image size
    if size_pixels < int(mission_dict["min_size"]):
        mission_dict["sizeX"] = int(mission_dict["min_size"])
        print(
            f'Image cannot be smaller than {mission_dict["min_size"]} arcsec (using this value)'
        )
    elif size_pixels > int(mission_dict["max_size"]):
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

    # find the fits file for each fitler (None if not found)
    files_dict = {filt: None for filt in filters}
    for filt in filters:
        for line in list_text:
            if filt in line and line.endswith(".mosaic.fits"):
                files_dict[filt] = line
                # pick the first image and move to the next filter
                break

    hdu_list = []
    for filt, file in files_dict.items():
        if file is not None:
            hdu = fits.open(line)
            hdu[0].header["MAGZP"] = hdu[0].header["ZPAB"]
            hdu_list.append(hdu)
        else:
            hdu_list.append(None)

    return hdu_list


# VISTA
def get_VISTA_images(ra, dec, size=3, filters=None, version="VHS"):
    """Gets VISTA fits images for the given coordinates and
    filters.

    Note: the different surveys included in VISTA cover different
    parts of the sky and do not necessarily contain the same filters.

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
    version: str, default ``VHS``
        Survey to use: ``VHS``, ``VIDEO`` or ``VIKING``.

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

    if version is None:
        version = 'VHS'
    # These are final data releases except for VHS(?):
    # VHSDR5: https://www.eso.org/sci/publications/announcements/sciann17290.html
    # VHSDR6: https://b2find.eudat.eu/dataset/0b10d3a0-1cfe-5e67-8a5c-0949db9d19cb
    # VIDEODR5: https://www.eso.org/sci/publications/announcements/sciann17491.html
    # VIKINGDR4: https://www.eso.org/sci/publications/announcements/sciann17289.html
    database_dict = {
        "VHS": "VHSDR6",
        "VIDEO": "VIDEODR5",
        "VIKING": "VIKINGDR4",
    }
    valid_surveys = list(database_dict.keys())
    assert (
        version in valid_surveys
    ), f"Not a valid VISTA survey: choose from {valid_surveys}"
    database = database_dict[version]

    base_url = "http://horus.roe.ac.uk:8080/vdfs/GetImage?archive=VSA&"
    survey_dict = {
        "database": database,
        "ra": ra,
        "dec": dec,
        "sys": "J",
        "filterID": "all",
        "size": size,  # in arcmin
        "obsType": "object",
        "frameType": "tilestack",
    }
    survey_url = "&".join([f"{key}={val}" for key, val in survey_dict.items()])

    url = base_url + survey_url
    results = urllib.request.urlopen(url).read()
    links = re.findall('href="(http://.*?)"', results.decode("utf-8"))

    # find url for each filter (None if not found)
    urls_dict = {filt: None for filt in filters}
    for filt in filters:
        for link in links:
            url = link.replace("getImage", "getFImage", 1)
            if f"band={filt}" in url:
                urls_dict[filt] = url
                break

    hdu_list = []
    for filt, url in urls_dict.items():
        if url is not None:
            hdu = fits.open(url)
            # to be consistent with the naming of other surveys
            hdu[1].header["MAGZP"] = hdu[1].header["MAGZPT"]
            hdu_list.append(hdu[1])  # extension 1
        else:
            hdu_list.append(None)

    return hdu_list


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
    name,
    ra,
    dec,
    size=3,
    filters=None,
    overwrite=False,
    survey="PS1",
    version=None,
):
    """Download images for a given object in the given filters of a
    given survey.

    The surveys that use the ``version`` parameter are: GALEX (``AIS``, ``MIS``,
    ``DIS``, ``NGS`` and ``GII``),  unWISE (``allwise``, ``neo1`` and ``neo2``)
    and VISTA (``VHS``, ``VIDEO`` and ``VIKING``).

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
        Survey used to download the images.
    version: str, default ``None``
        Version used by some surveys including multiple surveys. E.g. ``VHS`` for VISTA.

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
        hdu_list = get_GALEX_images(ra, dec, size, filters, version)
    elif survey == "WISE":
        hdu_list = get_WISE_images(ra, dec, size, filters)
    elif survey == "unWISE":
        hdu_list = get_unWISE_images(ra, dec, size, filters, version)
    elif survey == "2MASS":
        hdu_list = get_2MASS_images(ra, dec, size, filters)
    elif survey == "LegacySurvey":
        hdu_list = get_LegacySurvey_images(ra, dec, size, filters, version)
    elif survey == "Spitzer":
        hdu_list = get_Spitzer_images(ra, dec, size, filters)
    elif survey == "VISTA":
        hdu_list = get_VISTA_images(ra, dec, size, filters, version)

    if hdu_list:
        # NOT WORKING: this corrects any possible shifts between the images
        # fits_files = match_wcs(fits_files)
        # fix wcs (some have rotated wcs)
        """
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
        """

        for hdu, filt in zip(hdu_list, filters):
            if hdu is None:
                continue  # skip missing filter/image
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