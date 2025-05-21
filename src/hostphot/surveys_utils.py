import yaml
import numpy as np
from pathlib import Path
from typing import Optional
import warnings

import hostphot

hostphot_path = Path(hostphot.__path__[0])
filters_file = hostphot_path.joinpath("filters", "filters.yml")

# surveys that need background subtraction
bkg_surveys = [
    "2MASS",
    "WISE",
    "VISTA",
    "SkyMapper",
    "UKIDSS",
]
# surveys which are flipped respect to most others
flipped_surveys = [
    "DES",
    "VISTA",
    "UKIDSS",
]
# add warnings for the users
survey_warnings = {
    "SkyMapper": "WARNING: The photometric calibration of SkyMapper is not trustworthy at the moment (DR4)!",
    "VISTA": "WARNING: The photometric calibration of VISTA is currently not correct (might be a HostPhot problem)!",
    "UKIDSS": "WARNING: The photometric calibration of UKIDSS is currently under revision",
}


def load_yml(file: str) -> dict:
    """Simply loads a YAML file."""
    with open(file) as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def check_survey_validity(survey: str) -> None:
    """Check whether the given survey is whithin the valid
    options.

    Parameters
    ----------
    survey: Survey name: e.g. ``PanSTARRS``, ``GALEX``.
    """
    filters_config = load_yml(filters_file)
    surveys = list(filters_config.keys())
    assert survey in surveys, f"Survey '{survey}' not in {surveys}"
    if survey in survey_warnings.keys():
        warnings.warn(survey_warnings[survey])


def get_survey_filters(survey: str) -> str | list:
    """Gets all the valid filters for the given survey.

    Parameters
    ----------
    survey: Survey name: e.g. ``PanSTARRS``, ``GALEX``.

    Returns
    -------
    filters: Filters for the given survey.
    """
    check_survey_validity(survey)
    if survey in ["HST", "JWST"]:
        # For HST, the filter needs to be specified
        return None
    filters_config = load_yml(filters_file)
    filters = list(filters_config[survey].keys())
    return filters


def survey_zp(survey: str, filt: str) -> str | dict:
    """Returns the zero-point for a given survey.

    Parameters
    ----------
    survey: Survey name: e.g. ``PanSTARRS``, ``GALEX``.

    Returns
    -------
    zp_dict: Zero-points for all the filters in the given survey. If the survey zero-point
             is different for each image, the string ``header`` is returned.
    """
    check_survey_validity(survey)
    check_filters_validity([filt], survey)
    filters_config = load_yml(filters_file)
    zp = filters_config[survey][filt]["zeropoint"]
    return zp


def survey_pixel_scale(survey: str, filt: Optional[str] = None) -> float:
    """Returns the pixel scale for a given survey.

    Parameters
    ----------
    survey: Survey name: e.g. ``PanSTARRS``, ``GALEX``.
    filt: Filter to use by surveys that have different pixel scales
          for different filters (e.g. Spitzer's IRAC and MIPS instruments).

    Returns
    -------
    pixel_scale: Pixel scale in units of arcsec/pixel.
    """
    check_survey_validity(survey)
    filters_config = load_yml(filters_file)
    pixel_scale = float(filters_config[survey][filt]["pixel_scale"])
    return pixel_scale


def survey_pixel_units(survey: str, filt: str) -> str:
    """Returns the pixel units for a given survey.

    Parameters
    ----------
    survey: Survey name: e.g. ``PanSTARRS``, ``GALEX``.

    Returns
    -------
    pixel_units: Pixel units (e.g. ``counts`` or ``counts/second``).
    """
    check_survey_validity(survey)
    check_filters_validity([filt], survey)
    filters_config = load_yml(filters_file)
    pixel_units = filters_config[survey][filt]["pixel_units"]
    return pixel_units


def check_filters_validity(filters: str | list, survey: str) -> None:
    """Check whether the given filters are whithin the valid
    options for the given survey.

    Parameters
    ----------
    filters: Filters to use, e,g, ``griz``.
    survey: Survey name: e.g. ``PanSTARRS``, ``GALEX``.
    """
    if survey == "HST":
        if isinstance(filters, list):
            for filt in filters:
                check_HST_filters(filt)
        else:
            check_HST_filters(filters)
    elif survey == "JWST":
        if isinstance(filters, list):
            for filt in filters:
                check_JWST_filters(filt)
        else:
            check_JWST_filters(filters)
    else:
        valid_filters = get_survey_filters(survey)
        for filt in filters:
            message = (
                f"filter '{filt}' is not a valid option for "
                f"'{survey}' survey ({valid_filters})"
            )
            assert filt in valid_filters, message


def check_HST_filters(filt: str) -> None:
    """Check whether the given filter is whithin the valid
    options for HST.

    Parameters
    ----------
    filt: Filter to use, e,g, ``WFC3_UVIS_F225W``.
    """
    if filt is None:
        raise ValueError(f"'{filt}' is not a valid HST filter.")
    # For UVIS, only the filters of UVIS1 are used as the
    # detector 2 is scaled to match detector 1
    detector = filt.split("_")[1]
    if detector == "UVIS":
        filt_ = filt.replace("UVIS", "UVIS1")
    else:
        filt_ = filt
    filters_dir = hostphot_path / "filters/HST"
    hst_filters = [file.name.split(".")[0] for file in filters_dir.rglob("*.dat")]
    assert filt_ in hst_filters, f"Not a valid HST filter ({filt_}): {hst_filters}"


def check_JWST_filters(filt: str) -> None:
    """Check whether the given filter is within the valid
    options for JWST.

    Parameters
    ----------
    filt: str
        Filter to use, e,g, ``NIRCam_F150W``.
    """
    if filt is None:
        raise ValueError(f"'{filt}' is not a valid JWST filter.")
    filters_dir = hostphot_path / "filters/JWST"
    jwst_filters = [file.name.split(".")[0] for file in filters_dir.rglob("*.dat")]
    assert filt in jwst_filters, f"Not a valid JWST filter ({filt}): {jwst_filters}"


def extract_filter(
    filt: str, survey: str, version: Optional[str] = None
) -> tuple[np.array, np.array]:
    """Extracts the transmission function for the filter.

    Parameters
    ----------
    filt: Filter to extract.
    survey: Survey of the filters.
    version: Version of the filters to use. E.g. for the
             Legacy Survey as it uses DECam for the south
             and BASS+MzLS for the north.

    Returns
    -------
    wave: Wavelength range of the filter.
    transmission: Transmission function.
    """
    check_survey_validity(survey)
    if survey in ["HST", "JWST"]:
        check_filters_validity(filt, survey)
    else:
        check_filters_validity([filt], survey)
    if "WISE" in survey:
        survey = "WISE"  # for unWISE to use the same filters as WISE
    filters_path = Path(hostphot_path, "filters", survey)
    # Assume DECaLS filters below 32 degrees and BASS+MzLS above
    # https://www.legacysurvey.org/status/
    if survey == "LegacySurvey":
        if version == "BASS+MzLS":
            if filt == "z":
                filt_file = filters_path / "MzLS_z.dat"
            else:
                filt_file = filters_path / f"BASS_{filt}.dat"
        elif version == "DECam":
            filt_file = filters_path / f"DECAM_{filt}.dat"
    elif survey == "HST":
        if "UVIS" in filt:
            # Usually UVIS2 is used, but there is no large difference
            filt_ = filt.replace("UVIS", "UVIS2")
        else:
            filt_ = filt
        filt_file = [
            file for file in Path(filters_path, "HST").rglob("*.dat") if filt_ in file
        ][0]
    elif survey == "JWST":
        detector = filt.split("_")[0]
        filt_file = filters_path / f"{detector}/{filt}.dat"
    else:
        filt_file = filters_path / f"{survey}_{filt}.dat"
    wave, transmission = np.loadtxt(filt_file).T
    return wave, transmission
