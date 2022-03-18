import os
from astropy import wcs
from astropy.nddata.utils import Cutout2D
from .utils import get_survey_filters

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
