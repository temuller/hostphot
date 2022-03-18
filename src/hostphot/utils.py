import os
import piscola
import numpy as np
from astropy import wcs
from astropy.nddata.utils import Cutout2D
from .validate import (check_survey_validity, 
                        check_filters_validity)

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
                    'DES':'griz',
                    'SDSS':'griz'}
    filters = filters_dict[survey]

    return filters

def extract_filters(filters, survey):
    """Extracts transmission functions from PISCOLA.

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
    if survey=='PS1':
        filters_path = os.path.join(piscola.__path__[0],
                                    'filters/Pan-Starrs')
    else:
        filters_path = os.path.join(piscola.__path__[0],
                                    f'filters/{survey}')

    for filt in filters:
        filt_file = os.path.join(filters_path,
                                 f'{survey.lower()}_{filt}.dat')
        wave, transmission = np.loadtxt(filt_file).T

        filters_dict[filt] = {'wave':wave,
                              'transmission':transmission}

    return filters_dict


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
