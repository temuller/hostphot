import os
import copy
import numpy as np
import pandas as pd
import multiprocessing as mp

from astropy.io import fits
from astropy.table import Table
from astropy import wcs, units as u
from astropy.coordinates import SkyCoord

import pyvo  # 2MASS
from pyvo.dal import sia  # DES
from astroquery.sdss import SDSS
from astroquery.skyview import SkyView  # other surveys
from astroquery.mast import Observations  # for GALEX EXPTIME

from reproject import reproject_interp, reproject_exact
from reproject.mosaicking import find_optimal_celestial_wcs

from hostphot._constants import __workdir__
from hostphot.utils import (
    get_survey_filters,
    clean_dir,
    check_work_dir,
    check_survey_validity,
    check_filters_validity,
    survey_pixel_scale,
)
from hostphot.image_cleaning import trim_images

import warnings
from astropy.utils.exceptions import AstropyWarning

# ----------------------------------------
def _choose_workdir(workdir):
    """Updates the work directory.

    Parameters
    ----------
    workdir: str
        Path to the work directory.
    """
    global __workdir__
    __workdir__ = workdir


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
    survey = 'PS1'
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
    survey = 'PS1'
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
    survey = 'PS1'
    if filters is None:
        filters = get_survey_filters(survey)
    check_filters_validity(filters, survey)

    fits_url = get_PS1_urls(ra, dec, size, filters)

    fits_files = []
    for url, filt in zip(fits_url, filters):
        fits_file = fits.open(url)
        fits_files.append(fits_file)

    return fits_files


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
    fits_files: list
        List of fits images.
    """
    survey = 'DES'
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

    fits_files = []
    for url, url_w in zip(url_list, url_w_list):
        # combine image+weights on a single fits file
        image_fits = fits.open(url)
        weight_fits = fits.open(url_w)
        hdu = fits.PrimaryHDU(image_fits[0].data, header=image_fits[0].header)
        hdu_err = fits.ImageHDU(
            weight_fits[0].data, header=weight_fits[0].header
        )
        hdu_list = fits.HDUList([hdu, hdu_err])
        fits_files.append(hdu_list)

    return fits_files


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
    size: int, default ``3``
        Image size in arcmin.
    filters: str, default ``None``
        Filters to use. If ``None``, uses ``ugriz``.

    Return
    ------
    fits_files: list
        List with fits images for the given filters.
        `None` is returned if no image is found.
    """
    survey = "SDSS"
    if filters is None:
        filters = get_survey_filters(survey)
    check_filters_validity(filters, survey)

    # SkyView calls the filters in a different way
    filters_dict = {'u': 'SDSSu',
                    'g': 'SDSSg',
                    'r': 'SDSSr',
                    'i': 'SDSSi',
                    'z': 'SDSSz',
                    }
    skyview_filters = [filters_dict[filt] for filt in filters]

    if isinstance(size, (float, int)):
        size_arcsec = (size * u.arcmin).to(u.arcsec)
    else:
        size_arcsec = size.to(u.arcsec)

    pixel_scale = survey_pixel_scale(survey)
    size_pixels = int(size_arcsec.value / pixel_scale)

    fits_files = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', AstropyWarning)
        coords = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='icrs')

        for filt, skyview_filter in zip(filters, skyview_filters):
            fits_file = SkyView.get_images(position=coords,
                                           coordinates='icrs', pixels=str(size_pixels),
                                           survey=skyview_filter, width=size_arcsec,
                                           height=size_arcsec)

            fits_files.append(fits_file[0])

    return fits_files

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
    image_line = header['HISTORY'][-3]
    used_image = image_line.split('/')[-1].split('-')[0]

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

    target_df = galex_df[galex_df.target_name==used_image]
    texp = target_df.t_exptime.values[0]

    if texp is None:
        raise valueError('Galex image not found.')

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
    size: int, default ``3``
        Image size in arcmin.
    filters: str, default ``None``
        Filters to use. If ``None``, uses ``FUV, NUV``.

    Return
    ------
    fits_files: list
        List with fits images for the given filters.
        `None` is returned if no image is found.
    """
    survey = "GALEX"
    if filters is None:
        filters = get_survey_filters(survey)
    check_filters_validity(filters, survey)

    # SkyView calls the filters in a different way
    filters_dict = {'NUV': 'GALEX Near UV',
                    'FUV': 'GALEX Far UV'}
    skyview_filters = [filters_dict[filt] for filt in filters]

    if isinstance(size, (float, int)):
        size_arcsec = (size * u.arcmin).to(u.arcsec)
    else:
        size_arcsec = size.to(u.arcsec)

    pixel_scale = survey_pixel_scale(survey)
    size_pixels = int(size_arcsec.value / pixel_scale)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', AstropyWarning)
        coords = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='icrs')
        obs_table = Observations.query_criteria(coordinates=coords, radius=size_arcsec,
                                                obs_collection=[survey])

        skyview_fits = SkyView.get_images(position=coords,
                                           coordinates='icrs', pixels=str(size_pixels),
                                           survey=skyview_filters, width=size_arcsec,
                                           height=size_arcsec)

        fits_files = []
        for filt, fits_file in zip(filters, skyview_fits):
            # add exposure time
            used_image = get_used_image(fits_file[0].header)
            texp = get_exptime(used_image, obs_table, filt)
            fits_file[0].header['EXPTIME'] = texp
            fits_file[0].header['COMMENT'] = "EXPTIME added by HostPhot"

            fits_files.append(fits_file)

    return fits_files

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
    size: int, default ``3``
        Image size in arcmin.
    filters: str, default ``None``
        Filters to use. If ``None``, uses ``FUV, NUV``.

    Return
    ------
    fits_files: list
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
        warnings.simplefilter('ignore', AstropyWarning)
        coords = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='icrs')

        skyview_fits = SkyView.get_images(position=coords,
                                       coordinates='icrs', pixels=str(size_pixels),
                                       survey="WISE 3.4", width=size_arcsec,
                                       height=size_arcsec)
        header = skyview_fits[0][0].header

        coadd_id = get_used_image(header)
        coadd_id1 = coadd_id[:4]
        coadd_id2 = coadd_id1[:2]

        # for more info: https://irsa.ipac.caltech.edu/ibe/docs/wise/allwise/p3am_cdd/#sample_code
        base_url = "http://irsa.ipac.caltech.edu/ibe/data/wise/allwise/p3am_cdd/"
        coadd_url = os.path.join(coadd_id2, coadd_id1, coadd_id)
        params_url = f"center={ra},{dec}&size={size_arcsec.value}arcsec&gzip=0"  # center and size of the image

        fits_files = []
        for filt in filters:
            i = filt[-1]
            band_url = f"{coadd_id}-w{i}-int-3.fits"
            url = os.path.join(base_url, coadd_url, band_url + "?" + params_url)
            fits_file = fits.open(url)
            fits_files.append(fits_file)

    return fits_files

# WISE
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
    size: int, default ``3``
        Image size in arcmin.
    filters: str, default ``None``
        Filters to use. If ``None``, uses ``FUV, NUV``.

    Return
    ------
    fits_files: list
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
        warnings.simplefilter('ignore', AstropyWarning)
        coords = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='icrs')

        twomass_services = pyvo.regsearch(servicetype='image', keywords=['2mass'])
        table = twomass_services[0].search(pos=coords, size=size_degree)
        twomass_df = table.to_table().to_pandas()
        twomass_df = twomass_df[twomass_df.format=='image/fits']

        # for more info: https://irsa.ipac.caltech.edu/ibe/docs/twomass/allsky/allsky/#main
        base_url = "https://irsa.ipac.caltech.edu/ibe/data/twomass/allsky/allsky"

        fits_files = []
        for i, filt in enumerate(filters):
            band_df = twomass_df[twomass_df.band==filt]
            fname = band_df.download.values[0].split('=')[-1]
            hemisphere = band_df.hem.values[0]
            ordate = band_df.date.values[0]
            scanno = band_df.scan.values[0]

            tile_url = os.path.join(f"{ordate}{hemisphere}", f"s{scanno}")
            fits_url = os.path.join("image", f"{fname}.gz")
            params_url = f"center={ra},{dec}&size={size_degree}degree&gzip=0"  # center and size of the image

            url = os.path.join(base_url, tile_url, fits_url + "?" + params_url)
            fits_file = fits.open(url)
            fits_files.append(fits_file)

    return fits_files

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

    global __workdir__
    check_work_dir(__workdir__)
    obj_dir = os.path.join(__workdir__, name)
    if not os.path.isdir(obj_dir):
        os.mkdir(obj_dir)

    # download fits files
    if survey == "PS1":
        fits_files = get_PS1_images(ra, dec, size, filters)
    elif survey == "DES":
        fits_files = get_DES_images(ra, dec, size, filters)
    elif survey == "SDSS":
        fits_files = get_SDSS_images(ra, dec, size, filters)
    elif survey == "GALEX":
        fits_files = get_GALEX_images(ra, dec, size, filters)
    elif survey == "WISE":
        fits_files = get_WISE_images(ra, dec, size, filters)
    elif survey == "2MASS":
        fits_files = get_2MASS_images(ra, dec, size, filters)

    if fits_files:
        # this corrects any possible shifts between the images
        # fits_files = match_wcs(fits_files)

        # fix wcs (some have rotated wcs)
        if survey in ['XXX']:
            for fits_file in fits_files:
                fits_file = fits_file[0]
                wcs_out, shape_out = find_optimal_celestial_wcs([fits_file], auto_rotate=True)
                fixed_data, footprint = reproject_exact(fits_file, wcs_out, shape_out=shape_out)
                fits_file.header.update(wcs_out.to_header())
                fits_file.data = fixed_data


        for fits_file, filt in zip(fits_files, filters):
            outfile = os.path.join(obj_dir, f"{survey}_{filt}.fits")
            if not os.path.isfile(outfile):
                fits_file.writeto(outfile)
            else:
                if overwrite:
                    fits_file.writeto(outfile, overwrite=overwrite)
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
