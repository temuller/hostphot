import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy import wcs
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.utils import calc_total_error

import warnings
from astropy.utils.exceptions import AstropyWarning

import hostphot
hostphot_path = hostphot.__path__[0]
config_file = os.path.join(hostphot_path, 'filters', 'config.txt')
config_df = pd.read_csv(config_file, delim_whitespace=True)


def calc_sky_unc(image, exptime):
    """Calculates the uncertainty of the image from the
    sky standard deviation, sigma-clipped STD.

    Parameters
    ----------
    image: ndarray
        Image in a 2D numpy array.
    exptime: float
        Exposure time of the image.

    Returns
    -------
    error: float
        Estimated error of the image.
    """
    mask = image != 0
    avg, sky, sky_std = sigma_clipped_stats(image[mask], sigma=3.0)
    error = calc_total_error(image, sky_std, exptime)

    return error


def pixel2pixel(x1, y1, img_wcs1, img_wcs2):
    """Convert the pixel coordinates from one image to another."""
    coord1 = img_wcs1.pixel_to_world(x1, y1)
    x2, y2 = img_wcs2.world_to_pixel(coord1)

    return x2, y2


def check_survey_validity(survey):
    """Check whether the given survey is whithin the valid
    options.

    Parameters
    ----------
    survey: str
        Survey name: ``PS1``, ``DES``, ``SDSS``, ``GALEX``, ``WISE``, ``2MASS``.
    """
    global config_df
    surveys = list(config_df.survey)
    assert survey in surveys, (
        f"survey '{survey}' not" f" in {surveys}"
    )


def get_survey_filters(survey):
    """Gets all the valid filters for the given survey.

    Parameters
    ----------
    survey: str
        Survey name: ``PS1``, ``DES``, ``SDSS``, ``GALEX``, ``WISE``, ``2MASS``.

    Returns
    -------
    filters: str
        Filters for the given survey.
    """
    check_survey_validity(survey)

    global config_df
    survey_df = config_df[config_df.survey == survey]
    filters = survey_df.filters.values[0]

    if ',' in filters:
        filters = filters.split(',')

    return filters


def survey_zp(survey):
    """Returns the zero-point for a given survey.

    **Note:** for ``PS1``, an extra :math:`+2.5*np.log10(exptime)`
    needs to be added afterwards.

    Parameters
    ----------
    survey: str
        Survey name: ``PS1``, ``DES``, ``SDSS``, ``GALEX``, ``WISE``, ``2MASS``.

    Returns
    -------
    zp_dict: dict or str
        Zero-points for all the filters in the given survey. If the survey zero-point
        is different for each image, the string ``header`` is returned.
    """
    check_survey_validity(survey)
    filters = get_survey_filters(survey)

    global config_df
    survey_df = config_df[config_df.survey == survey]
    zps = survey_df.zp.values[0]

    if zps=='header':
        return zps

    if ',' in zps:
        zps = zps.split(',')
        zp_dict = {filt: float(zp) for filt, zp in zip(filters, zps)}
    else:
        zp_dict = {filt: float(zps) for filt in filters}

    return zp_dict


def get_image_gain(header, survey):
    """Returns the gain from an image's header.

    **Note:** for ``SDSS`` this is assumed to be one
    as it should already be included.

    Parameters
    ----------
    header: fits header
        Header of an image.
    survey: str
        Survey name: ``PS1``, ``DES``, ``SDSS``, ``GALEX``, ``WISE``, ``2MASS``.

    Returns
    -------
    gain: float
        Gain value.
    """
    check_survey_validity(survey)
    if survey == "PS1":
        gain = header["HIERARCH CELL.GAIN"]
    elif survey == "DES":
        gain = header["GAIN"]
    elif survey == "SDSS":
        gain = 1.0
    else:
        gain = 1.0

    return gain


def get_image_readnoise(header, survey):
    """Returns the read noise from an image's header.
    All values are per-pixel values.

    **Note:** for ``SDSS`` this is assumed to be zero
    as it should already be included.

    Parameters
    ----------
    header: fits header
        Header of an image.
    survey: str
        Survey name: ``PS1``, ``DES``, ``SDSS``, ``GALEX``, ``WISE``, ``2MASS``.

    Returns
    -------
    readnoise: float
        Read noise value.
    """
    check_survey_validity(survey)
    if survey == "PS1":
        readnoise = header["HIERARCH CELL.READNOISE"]
    elif survey == "DES":
        # see https://arxiv.org/pdf/0810.3600.pdf
        readnoise = 7.0  # electrons per pixel
    elif survey == "SDSS":
        readnoise = 0.0
    else:
        readnoise = 0.0

    return readnoise

def get_image_exptime(header, survey):
    """Returns the exposure time from an image's header.

    Parameters
    ----------
    header: fits header
        Header of an image.
    survey: str
        Survey name: ``PS1``, ``DES``, ``SDSS``, ``GALEX``, ``WISE``, ``2MASS``.

    Returns
    -------
    exptime: float
        Exposure time in seconds.
    """
    check_survey_validity(survey)
    if survey in ["PS1", "DES", "GALEX"]:
        exptime = float(header["EXPTIME"])
    else:
        exptime = 0.0

    return exptime

def uncertainty_calc(flux, survey, filt=None, ap_area=0.0, readnoise=0.0, gain=1.0, exptime=0.0):
    """Calculates the uncertainty propagation.

    Parameters
    ----------
    flux: float
        Aperture flux.
    survey: str
        Survey name: ``PS1``, ``DES``, ``SDSS``, ``GALEX``, ``WISE``, ``2MASS``.
    filt: str, default ``None``
        Survey-specific filter.
    ap_area: float, default ``0.0``
        Aperture area.
    readnoise: float, default ``0.0``
        Image readnoise.
    gain: str, default ``1.0``
        Image gain.
    exptime: str, default ``0.0``
        Image exposure time.

    Returns
    -------
    mag_err: float
        Extra uncertainty in magnitudes.
    """
    mag_err = 0.0
    if survey in ["PS1", "DES", "SDSS"]:
        # 1.0857 = 2.5/ln(10)
        extra_err = (
                1.0857
                * np.sqrt(ap_area * (readnoise ** 2) + flux / gain)
                / flux
        )
        mag_err = np.sqrt(mag_err ** 2 + extra_err ** 2)

    elif survey=="GALEX":
        CPS = flux.copy()
        if filt == 'FUV':
            uv_err = -2.5 * (
                        np.log10(CPS) - np.log10(CPS + (CPS * exptime + (0.050 * CPS * exptime) ** 2) ** 0.5 / exptime))
        elif filt == 'NUV':
            uv_err = -2.5 * (
                        np.log10(CPS) - np.log10(CPS + (CPS * exptime + (0.027 * CPS * exptime) ** 2) ** 0.5 / exptime))
        mag_err = np.sqrt(mag_err ** 2 + uv_err ** 2)

    return mag_err


def survey_pixel_scale(survey):
    """Returns the pixel scale for a given survey.

    Parameters
    ----------
    survey: str
        Survey name: ``PS1``, ``DES``, ``SDSS``, ``GALEX``, ``WISE``, ``2MASS``.

    Returns
    -------
    pixel_scale: float
        Pixel scale in units of arcsec/pixel.
    """
    check_survey_validity(survey)

    global config_df
    survey_df = config_df[config_df.survey == survey]
    pixel_scale = survey_df.pixel_scale.values[0]

    return pixel_scale


def check_filters_validity(filters, survey):
    """Check whether the given filters are whithin the valid
    options for the given survey.

    Parameters
    ----------
    filters: str
        Filters to use, e,g, ``griz``.
    survey: str
        Survey name: ``PS1``, ``DES``, ``SDSS``, ``GALEX``, ``WISE``, ``2MASS``.
    """
    if filters is not None:
        valid_filters = get_survey_filters(survey)

        for filt in filters:
            message = (
                f"filter '{filt}' is not a valid option for "
                f"'{survey}' survey ({valid_filters})"
            )
            assert filt in valid_filters, message


def extract_filters(filters, survey):
    """Extracts transmission functions.

    Parameters
    ----------
    filters: str
        Filters to extract.
    survey: str
        Survey of the filters.

    Returns
    -------
    filters_dict: dict
        Dictionary with transmission functions
        and their respective wavelengths.
    """
    check_survey_validity(survey)
    check_filters_validity(filters, survey)

    filters_dict = {filt: None for filt in filters}
    filters_path = os.path.join(hostphot.__path__[0], "filters", survey)

    for filt in filters:
        filt_file = os.path.join(filters_path, f"{survey.upper()}_{filt}.dat")
        wave, transmission = np.loadtxt(filt_file).T

        filters_dict[filt] = {"wave": wave, "transmission": transmission}

    return filters_dict


def integrate_filter(
    spectrum_wave,
    spectrum_flux,
    filter_wave,
    filter_response,
    response_type="photon",
):
    """Calcultes the flux density of an SED given a filter response.

    Parameters
    ----------
    spectrum_wave : array
        Spectrum's wavelength range.
    spectrum_flux : array
        Spectrum's flux density distribution.
    filter_wave : array
        Filter's wavelength range.
    filter_response : array
        Filter's response function.
    response_type : str, default ``photon``
        Filter's response type. Either ``photon`` or ``energy``.

    Returns
    -------
    flux_filter : float
        Flux density.
    """
    if response_type == "energy":
        filter_response = filter_response.copy() / filter_wave

    interp_response = np.interp(
        spectrum_wave, filter_wave, filter_response, left=0.0, right=0.0
    )
    int1 = np.trapz(
        spectrum_flux * interp_response * spectrum_wave, spectrum_wave
    )
    int2 = np.trapz(filter_response * filter_wave, filter_wave)
    flux_filter = int1 / int2

    return flux_filter


def check_work_dir(wokrdir):
    """Checks if the working directory exists. If it
    does not, one is created.

    Parameters
    ----------
    wokrdir: str
        Working directory path.
    """
    if not os.path.isdir(wokrdir):
        os.mkdir(wokrdir)


def clean_dir(directory):
    """Removes the directory if it is empty.

    Parameters
    ----------
    directory: str
        Directory path.
    """
    try:
        os.rmdir(directory)
    except:
        pass

def plot_fits(fits_file):
    """Plots a fits file.

    Parameters
    ----------
    fits_file: str
        Path to fits file.
    """
    img = fits.open(fits_file)
    header = img[0].header
    data = img[0].data

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        img_wcs = wcs.WCS(header, naxis=2)

    m, s = np.nanmean(data), np.nanstd(data)

    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(projection=img_wcs)
    update_axislabels(ax)

    im = ax.imshow(
        data,
        interpolation="nearest",
        cmap="gray",
        vmin=m - s,
        vmax=m + s,
        origin="lower",
    )

    plt.show()


def update_axislabels(ax):
    """Updates the labels and ticks of a plot.

    Parameters
    ----------
    ax: `.axes.SubplotBase`.
        The axis of a subplot.
    """
    for i in range(2):
        ax.coords[i].set_ticklabel(size=16)
        if i == 0:
            ax.coords[i].set_axislabel("RA (J2000)", fontsize=20)
        else:
            ax.coords[i].set_axislabel("Dec (J2000)", fontsize=20)
            # ax.coords[i].set_ticklabel(rotation=65)
