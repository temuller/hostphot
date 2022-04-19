import os
import sys
import wget
import tarfile

import numpy as np

import sfdmap
import hostphot
import extinction

from astropy import wcs
from astropy.nddata.utils import Cutout2D
from astropy.stats import sigma_clipped_stats
from photutils.utils import calc_total_error

def trim_images(fits_files, pos, size):
    """Trims the size of the given fits images.

    Parameters
    ----------
    fits_files: list
        List of fits images.
    pos: `~astropy.coordinates.SkyCoord`
        The position of the cutout array's center.
    size: int
        Image size in pixels.

    Return
    ------
    trimmed_fits_files: list
        List of the trimmed fits images.
    """

    trimmed_fits_files = []
    for fits_file in fits_files:
        data = fits_file[0].data.copy()
        header = fits_file[0].header
        img_wcs = wcs.WCS(header, naxis=2)

        trimmed_data = Cutout2D(data, pos, size, img_wcs)
        fits_file[0].data = trimmed_data.data
        trimmed_fits_files.append(fits_file)

    return trimmed_fits_files

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

def get_survey_filters(survey):
    """Gets all the valid filters for the given survey.

    Parameters
    ----------
    survey: str
        Survey name: `PS1`, `DES` or `SDSS`.

    Returns
    -------
    filters: str
        Filters for the given survey.
    """
    filters_dict = {'PS1':'grizy',
                    'DES':'grizY',
                    'SDSS':'ugriz'}
    filters = filters_dict[survey]

    return filters

def check_survey_validity(survey):
    """Check whether the given survey is whithin the valid
    options.

    Parameters
    ----------
    survey: str
        Survey name: `PS1`, `DES` or `SDSS`.
    """
    valid_surveys = ['PS1', 'DES', 'SDSS']
    assert survey in valid_surveys, (f"survey '{survey}' not"
                                     f" in {valid_surveys}")

def check_filters_validity(filters, survey):
    """Check whether the given filters are whithin the valid
    options for the given survey.

    Parameters
    ----------
    filters: str
        Filters to use, e,g, `griz`.
    survey: str
        Survey name: `PS1`, `DES` or `SDSS`.
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

def deredden(wave, flux, ra, dec, scaling=0.86, reddening_law='fitzpatrick99',
                                                    dustmaps_dir=None, r_v=3.1, ebv=None):
    """Dereddens the given spectrum, given a right ascension and declination or `E(B-V)`.

    Parameters
    ----------
    wave : array
        Wavelength values.
    flux : array
        Flux density values.
    ra : float
        Right ascension in degrees.
    dec : float
        Declination in degrees.
    scaling: float, default ``0.86``
        Calibration of the Milky Way dust maps. Either ``0.86``
        for the Schlafly & Finkbeiner (2011) recalibration or ``1.0`` for the original
        dust map of Schlegel, Fikbeiner & Davis (1998).
    reddening_law: str, default ``fitzpatrick99``
        Reddening law. The options are: ``ccm89`` (Cardelli, Clayton & Mathis 1989),
        ``odonnell94`` (O’Donnell 1994), ``fitzpatrick99`` (Fitzpatrick 1999), ``calzetti00``
        (Calzetti 2000) and ``fm07`` (Fitzpatrick & Massa 2007 with `R_V` = 3.1.)
    dustmaps_dir : str, default ``None``
        Directory where the dust maps of Schlegel, Fikbeiner & Davis (1998) are found.
    r_v : float, default ``3.1``
        Total-to-selective extinction ratio (:math:`R_V`)
    ebv : float, default ``None``
        Colour excess (:math:`E(B-V)`). If given, this is used instead of the dust map value.

    Returns
    -------
    deredden_flux : array
        Deredden flux values.
    """
    hostphot_path = hostphot.__path__[0]
    if dustmaps_dir is None:
        dustmaps_dir = os.path.join(hostphot_path, 'sfddata-master')

    if ebv is None:
        m = sfdmap.SFDMap(mapdir=dustmaps_dir, scaling=scaling)
        ebv = m.ebv(ra, dec) # RA and DEC in degrees

    a_v = r_v*ebv

    rl_list = ['ccm89', 'odonnell94', 'fitzpatrick99', 'calzetti00', 'fm07']
    assert reddening_law in rl_list, f'Choose one of the available reddening laws: {rl_list}'

    if reddening_law=='ccm89':
        ext = extinction.ccm89(wave, a_v, r_v)
    elif reddening_law=='odonnell94':
        ext = extinction.odonnell94(wave, a_v, r_v)
    elif reddening_law=='fitzpatrick99':
        ext = extinction.fitzpatrick99(wave, a_v, r_v)
    elif reddening_law=='calzetti00':
        ext = extinction.calzetti00(wave, a_v, r_v)
    elif reddening_law=='fm07':
        ext = extinction.fm07(wave, a_v)

    deredden_flux = extinction.remove(ext, flux)

    return deredden_flux

def calc_ext(filter_wave, filter_response, ra, dec, scaling=0.86,
                        reddening_law='fitzpatrick99', dustmaps_dir=None, r_v=3.1, ebv=None):
    """Calculates the extinction for a given filter, right ascension and declination
    or `E(B-V)`.

    Parameters
    ----------
    filter_wave : array
        Filter's wavelength range.
    filter_response : array
        Filter's response function.
    ra : float
        Right ascension.
    dec : float
        Declinationin degrees.
    scaling: float, default ``0.86``
        Calibration of the Milky Way dust maps. Either ``0.86``
        for the Schlafly & Finkbeiner (2011) recalibration or ``1.0`` for the original
        dust map of Schlegel, Fikbeiner & Davis (1998).
    reddening_law: str, default ``fitzpatrick99``
        Reddening law. The options are: ``ccm89`` (Cardelli, Clayton & Mathis 1989),
        ``odonnell94`` (O’Donnell 1994), ``fitzpatrick99`` (Fitzpatrick 1999), ``calzetti00``
        (Calzetti 2000) and ``fm07`` (Fitzpatrick & Massa 2007 with `R_V` = 3.1.)
    dustmaps_dir : str, default ``None``
        Directory where the dust maps of Schlegel, Fikbeiner & Davis (1998) are found.
    r_v : float, default ``3.1``
        Total-to-selective extinction ratio (:math:`R_V`)
    ebv : float, default ``None``
        Colour excess (:math:`E(B-V)`). If given, this is used instead of the dust map value.

    Returns
    -------
    A_ext : float
        Extinction value in magnitudes.
    """
    flux = 100
    deredden_flux = deredden(filter_wave, flux, ra, dec, scaling,
                            reddening_law, dustmaps_dir, r_v, ebv)

    f1 = integrate_filter(filter_wave, flux, filter_wave, filter_response)
    f2 = integrate_filter(filter_wave, deredden_flux, filter_wave, filter_response)
    A_ext = -2.5*np.log10(f1/f2)

    return A_ext

def download_dustmaps():
    """ Downloads dust maps of Schlegel, Fikbeiner & Davis (1998).
    """

    sfdmaps_url = 'https://github.com/kbarbary/sfddata/archive/master.tar.gz'
    master_tar = wget.download(sfdmaps_url)

    # extract tar file under mapsdir directory
    mapsdir = hostphot.__path__[0]
    tar = tarfile.open(master_tar)
    tar.extractall(mapsdir)
    tar.close()

    os.remove(master_tar)

def clean_sn_dir(sn_dir):
    """Removes the SN directory if it is empty.

    Parameters
    ----------
    sn_dir: str
        SN directory.
    """
    try:
        os.rmdir(sn_dir)
    except:
        pass
