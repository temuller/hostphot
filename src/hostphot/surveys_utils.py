import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

import hostphot

hostphot_path = Path(hostphot.__path__[0])
filters_file = hostphot_path.joinpath("filters", "filters.yml")

#config_file = hostphot_path.joinpath("filters", "config.txt")
#config_df = pd.read_csv(config_file, sep="\\s+")

# surveys that need background subtraction
bkg_surveys = ["2MASS", "WISE", "VISTA", "SkyMapper", "UKIDSS"]
# surveys which are flipped respect to most others
flipped_surveys = ["DES", "VISTA", "UKIDSS"]

def load_yml(file: str) -> dict:
    """Simply loads a YAML file.
    """
    with open(file) as stream:
        try:
            return (yaml.safe_load(stream))
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
    #if "," in filters:
    #    filters = filters.split(",")
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

    #if zp == "header":
    #    return zp
    #if "," in zps:
    #    zps = zps.split(",")
    #    zp_dict = {filt: float(zp) for filt, zp in zip(filters, zps)}
    #else:
    #    zp_dict = {filt: float(zps) for filt in filters}
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
    pixel_scale = filters_config[survey][filt]["pixel_scale"]
    """
    # some surveys have multiple scales (separated by ',') depending on the instrument
    if len(pixel_scale.split(",")) > 1:
        if survey in ["HST", "JWST"]:
            filters = filt  # needs to be explicitly specified
        else:
            filters = get_survey_filters(survey)
        pixel_scale_dict = {
            f: float(ps) for f, ps in zip(filters, pixel_scale.split(","))
        }
        if filt in pixel_scale_dict.keys():
            pixel_scale = pixel_scale_dict[filt]
        elif survey == "HST":
            if "UVIS" in filt:
                pixel_scale = list(pixel_scale_dict.values())[0]
            elif "IR" in filt:
                pixel_scale = list(pixel_scale_dict.values())[1]
            else:
                raise ValueError(f"Not a valid HST filter: {filt}")
        else:
            print(f"No pixel scale found for filter {filt}.")
            filt_used = list(pixel_scale_dict.keys())[0]
            print(f"Using the pixel scale of filter {filt_used} ({survey}).")
            pixel_scale = list(pixel_scale_dict.values())[0]
        return pixel_scale
    """
    return float(pixel_scale)


def check_filters_validity(filters: str | list, survey: str) -> None:
    """Check whether the given filters are whithin the valid
    options for the given survey.

    Parameters
    ----------
    filters: Filters to use, e,g, ``griz``.
    survey: Survey name: e.g. ``PanSTARRS``, ``GALEX``.
    """
    if survey == "HST":
        check_HST_filters(filters)
    elif survey == "JWST":
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
    if "UVIS" in filt:
        filt = filt.replace("UVIS", "UVIS1")
    global hostphot_path
    filters_dir = hostphot_path / "filters/HST"
    hst_filters = [file.name.split(".")[0] for file in filters_dir.rglob("*.dat")]
    assert filt in hst_filters, f"Not a valid HST filter ({filt}): {hst_filters}"


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
    global hostphot_path
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
    global hostphot_path
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
            filt = filt.replace("UVIS", "UVIS2")
        filt_file = [
            file for file in Path(filters_path, "HST").rglob("*.dat") if filt in file
        ][0]
    elif survey == "JWST":
        filt_file = filters_path / f"{filt}.dat"
    else:
        filt_file = filters_path / f"{survey}_{filt}.dat"
    wave, transmission = np.loadtxt(filt_file).T
    return wave, transmission
