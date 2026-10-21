import copy
import yaml
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
    "Spitzer",  # SEIP mosaics only have a model of the zodiacal light subtracted
    "JWST",  # the sky is not subtracted from the i2d images (BKGSUB = False)
    "Herschel",  # SPIRE extended-emission maps have an absolute (Planck-based) sky level, and PACS maps small offsets
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
    "Herschel": "Herschel support is currently in beta testing. Results may not be fully validated.",
    "SPLUS": (
        "S-PLUS support is in beta: the images (DR1) can have no coverage in parts of the tiles, "
        "which are set to NaN using the weight maps, and the photometry agrees with other surveys within ~0.1 mag."
    ),
}


# parsed YAML files, as parsing them is slow and they are read many times
_yml_cache = {}


def load_yml(file: str) -> dict:
    """Loads a YAML file (cached until the file is modified)."""
    try:
        return copy.deepcopy(_load_yml_cached(file))
    except yaml.YAMLError as exc:
        print(exc)
        return None


def _load_yml_cached(file: str) -> dict:
    """Loads a YAML file (cached until the file is modified), returning the cached
    object itself, so it must not be modified (use :func:`load_yml` otherwise)."""
    file = Path(file)
    mtime = file.stat().st_mtime
    if file not in _yml_cache or _yml_cache[file][0] != mtime:
        loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)  # C loader if available (faster)
        with open(file) as stream:
            _yml_cache[file] = (mtime, yaml.load(stream, Loader=loader))
    return _yml_cache[file][1]


def check_survey_validity(survey: str) -> None:
    """Check whether the given survey is whithin the valid
    options.

    Parameters
    ----------
    survey: Survey name: e.g. ``PanSTARRS``, ``GALEX``.
    """
    filters_config = _load_yml_cached(filters_file)  # read only
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
    filters_config = _load_yml_cached(filters_file)  # read only
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
    filters_config = _load_yml_cached(filters_file)  # read only
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
    filters_config = _load_yml_cached(filters_file)  # read only
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
    filters_config = _load_yml_cached(filters_file)  # read only
    pixel_units = filters_config[survey][filt]["pixel_units"]
    return pixel_units


def normalize_filters(filters: Optional[str | list], survey: str) -> Optional[list]:
    """Returns the filters as a list.

    A string is either a single filter (e.g. ``FUV``, ``Ks``, ``W1``) or, for surveys with
    single-character filter names, a concatenation of filters (e.g. ``grizy``).

    Parameters
    ----------
    filters: Filters, e.g. ``griz``, ``FUV`` or ``['g', 'r']``.
    survey: Survey name: e.g. ``PanSTARRS``, ``GALEX``.

    Returns
    -------
    filters: List of filters (``None`` if ``filters`` is ``None``).
    """
    if filters is None:
        return None
    if isinstance(filters, str):
        if survey in ["HST", "JWST"] or filters in get_survey_filters(survey):
            return [filters]
        return list(filters)
    return list(filters)


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
    # the supported filters (WFC3/UVIS, WFC3/IR, ACS/WFC and NICMOS) are those in the configuration
    hst_filters = list(load_yml(filters_file)["HST"])
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
    filters_dir = hostphot_path / "filters/JWST"
    jwst_filters = [file.name.split(".")[0] for file in filters_dir.rglob("*.dat")]
    assert filt in jwst_filters, f"Not a valid JWST filter ({filt}): {jwst_filters}"
