import os
import sys
import tarfile
import numpy as np

import sfdmap
import extinction
import hostphot

from astropy import wcs
from astropy.stats import sigma_clipped_stats
from photutils.utils import calc_total_error

def calc_sky_unc(image, exptime):
    """Calculates the uncertainty of the image from the
    sky standard deviation, sigma-clipped STD.

    Parameters
    ----------
    image: 2D array
        Image in a 2D numpy array.
    exptime: float
        Exposure time of the image.

    Returns
    -------
    error: float
        Estimated error of the image.
    """

    avg, sky, sky_std = sigma_clipped_stats(image[(image!= 0)],
                                                        sigma=3.0)
    error = calc_total_error(image, sky_std, exptime)

    return error

def pixel2pixel(x1, y1, img_wcs1, img_wcs2):
    """Convert the pixel coordinates from one image to another.
    """
    coord1 = img_wcs1.pixel_to_world(x1, y1)
    x2, y2 = img_wcs2.world_to_pixel(coord1)

    return x2, y2

def check_survey_validity(survey):
    """Check whether the given survey is whithin the valid
    options.

    Parameters
    ----------
    survey: str
        Survey name: ``PS1``, ``DES`` or ``SDSS``.
    """
    valid_surveys = ['PS1', 'DES', 'SDSS']
    assert survey in valid_surveys, (f"survey '{survey}' not"
                                     f" in {valid_surveys}")

def get_survey_filters(survey):
    """Gets all the valid filters for the given survey.

    Parameters
    ----------
    survey: str
        Survey name: ``PS1``, ``DES`` or ``SDSS``.

    Returns
    -------
    filters: str
        Filters for the given survey.
    """
    check_survey_validity(survey)
    filters_dict = {'PS1':'grizy',
                    'DES':'grizY',
                    'SDSS':'ugriz'}
    filters = filters_dict[survey]

    return filters

def survey_zp(survey):
    """Returns the zero-point for a given survey.

    **Note:** for ``PS1``, an extra :math:`+2.5*np.log10(exptime)`
    needs to be added afterwards.

    Parameters
    ----------
    survey: str
        Survey name: ``PS1``, ``DES`` or ``SDSS``.

    Returns
    -------
    zp: float
        Zero-point.
    """
    check_survey_validity(survey)
    zp_dict = {'PS1':25 ,  # + 2.5*np.log10(exptime)
               'DES':30,
               'SDSS':22.5}
    zp = zp_dict[survey]

    return zp

def get_image_gain(header, survey):
    """Returns the gain from an image's header.

    **Note:** for ``SDSS`` this is assumed to be zero
    as it should already be included.

    Parameters
    ----------
    header: fits header
        Header of an image.
    survey: str
        Survey name: ``PS1``, ``DES`` or ``SDSS``.

    Returns
    -------
    gain: float
        Gain value.
    """
    check_survey_validity(survey)
    if survey=='PS1':
        gain = header['HIERARCH CELL.GAIN']
    elif survey=='DES':
        gain = header['GAIN']
    elif survey=='SDSS':
        gain = 0.0

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
        Survey name: ``PS1``, ``DES`` or ``SDSS``.

    Returns
    -------
    readnoise: float
        Read noise value.
    """
    check_survey_validity(survey)
    if survey=='PS1':
        readnoise = header['HIERARCH CELL.READNOISE']
    elif survey=='DES':
        # see https://arxiv.org/pdf/0810.3600.pdf
        readnoise = 7.0  # electrons per pixel
    elif survey=='SDSS':
        readnoise = 0.0

    return readnoise

def survey_pixel_scale(survey):
        """Returns the pixel scale for a given survey.

        Parameters
        ----------
        survey: str
            Survey name: ``PS1``, ``DES`` or ``SDSS``.

        Returns
        -------
        pixel_scale: float
            Pixel scale.
        """
        check_survey_validity(survey)
        # units of arcsec/pix
        pixel_scale_dict = {'PS1':0.25,
                            'DES':0.263,
                            'SDSS':0.396}
        pixel_scale = pixel_scale_dict[survey]

        return pixel_scale

def check_filters_validity(filters, survey):
    """Check whether the given filters are whithin the valid
    options for the given survey.

    Parameters
    ----------
    filters: str
        Filters to use, e,g, ``griz``.
    survey: str
        Survey name: ``PS1``, ``DES`` or ``SDSS``.
    """
    if filters is not None:
        valid_filters = get_survey_filters(survey)

        for filt in filters:
            message = (f"filter '{filt}' is not a valid option for "
                       f"'{survey}' survey ({valid_filters})")
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

    filters_dict = {filt:None for filt in filters}
    filters_path = os.path.join(hostphot.__path__[0],
                                'filters', survey)

    for filt in filters:
        filt_file = os.path.join(filters_path,
                                 f'{survey.lower()}_{filt}.dat')
        wave, transmission = np.loadtxt(filt_file).T

        filters_dict[filt] = {'wave':wave,
                              'transmission':transmission}

    return filters_dict

def integrate_filter(spectrum_wave, spectrum_flux, filter_wave,
                                    filter_response, response_type='photon'):
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
    if response_type == 'energy':
        filter_response = filter_response.copy()/filter_wave

    interp_response = np.interp(spectrum_wave, filter_wave, filter_response,
                                                            left=0.0, right=0.0)
    I1 = np.trapz(spectrum_flux*interp_response*spectrum_wave, spectrum_wave)
    I2 = np.trapz(filter_response*filter_wave, filter_wave)
    flux_filter = I1/I2

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

def clean_dir(dir):
    """Removes the directory if it is empty.

    Parameters
    ----------
    dir: str
        Directory path.
    """
    try:
        os.rmdir(dir)
    except:
        pass
