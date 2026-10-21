import warnings
import importlib
import pandas as pd
from pathlib import Path
import astropy.units as u
from typing import Optional

from hostphot._constants import workdir
from hostphot.utils import check_work_dir, store_input, image_has_data
from hostphot.surveys_utils import (
    get_survey_filters,
    get_default_filters,
    check_survey_validity,
    normalize_filters,
)


def download_images(
    name: str,
    ra: float,
    dec: float,
    survey: str,
    size: float | u.Quantity = 3,
    filters: Optional[str | list] = None,
    overwrite: bool = True,
    version: str = None,
    save_input: bool = True,
) -> None:
    """Download images for a given object in the given filters of a
    given survey.

    The surveys that use the ``version`` parameter are: GALEX (``AIS``, ``MIS``,
    ``DIS``, ``NGS`` and ``GII``),  unWISE (``allwise`` and ``neo{i}`` for {i}=1-7),
    VISTA (``VHS``, ``VIDEO`` and ``VIKING``), SDSS (``dr{i}`` for {i}=12-17),
    LegacySurvey (``dr{i}`` for {i}=?-10), and HSC (``pdr3_wide`` or ``pdr3_dud``).

    Parameters
    ----------
    name: Name used for tracking the object in your localdirectory.
    ra: Right ascension in degrees.
    dec: Declination in degrees.
    survey: Survey used to download the images.
    size: Image size. If a float is given, the units are assumed to be arcmin.
    filters: Filters for the images.
    overwrite: If ``True``, the images are overwritten if they alreadyexist.
    version: Version used by some surveys including multiple surveys. E.g. ``VHS`` for VISTA.
    save_input: Whether to save the input parameters.

    Examples
    --------
    >>> from hostphot.cutouts import download_images
    >>> name = 'SN2004eo'
    >>> host_ra, host_dec = 308.2092, 9.92755  # coords of host galaxy of SN2004eo
    >>> survey = 'PanSTARRS'
    >>> download_images(name, host_ra, host_dec, survey=survey)
    """
    input_params = locals()  # dictionary
    # Normalise the right ascension into the valid [0, 360) degree range so
    # that negative or wrapped-around values (e.g. ra=-15 -> 345) are accepted
    # by the survey services. IRSA's SIA endpoint (WISE) rejects negative RA,
    # which otherwise surfaces as a confusing downstream error (issue #17).
    # ``input_params`` above intentionally keeps the caller's original value.
    ra = ra % 360
    # initial checks
    check_survey_validity(survey)
    check_work_dir(workdir)
    # check filters
    if filters is None:
        if survey in ["HST", "JWST"]:
            raise ValueError(f"For {survey}, the filters need to be specified!")
        filters = get_default_filters(survey)
    else:
        filters = normalize_filters(filters, survey)
    # check output directory
    survey_dir = Path(workdir, name, survey)
    if survey_dir.is_dir() is False:
        survey_dir.mkdir(parents=True)

    # save input parameters
    if save_input is True:
        inputs_file = survey_dir / "input_cutouts.csv"
        store_input(input_params, inputs_file)

    # check existing images
    if overwrite is False:
        filters_without_image = []
        for filt in filters:
            # same file name as used when saving the images (below)
            filt_image = survey_dir / f"{survey}_{filt}.fits"
            if filt_image.is_file() is False:
                filters_without_image.append(filt)
        # only download images not found locally
        filters = filters_without_image
        if len(filters) == 0:
            return None

    # extract download function for the given survey
    if survey == "2MASS":
        survey_module = importlib.import_module("hostphot.cutouts.twomass")
    elif survey == "unWISE":
        survey_module = importlib.import_module("hostphot.cutouts.wise")
    else:
        survey_module = importlib.import_module(f"hostphot.cutouts.{survey.lower()}")

    # download the images
    get_images = getattr(survey_module, f"get_{survey}_images")  # this is a function
    if survey in ["SDSS", "GALEX", "unWISE", "LegacySurvey", "HSC", "VISTA"]:
        hdu_list = get_images(ra, dec, size, filters, version)
    elif survey == "Herschel":
        hdu_list = get_images(ra, dec, size, filters, output_dir=survey_dir, overwrite=overwrite)
    else:
        hdu_list = get_images(ra, dec, size, filters)
    if hdu_list is None:
        return None
    # save the images
    for hdu, filt in zip(hdu_list, filters):
        if hdu is None:
            # no coverage or failed download (see previous warnings, if any)
            warnings.warn(f"No {survey} image was downloaded for filter {filt} of {name}.")
            continue
        if not image_has_data(hdu[0].data):
            warnings.warn(
                f"The {survey} image for filter {filt} of {name} contains no data (zeros, constant or "
                "compression noise only), so it was not saved."
            )
            hdu.close()
            continue
        # get output file name
        if survey == "HST":
            #inst = version.replace("/", "_")
            #outfile = Path(survey_dir, f"{survey}_{inst}_{filters}.fits")
            outfile = Path(survey_dir, f"{survey}_{filt}.fits")
        else:
            outfile = Path(survey_dir, f"{survey}_{filt}.fits")
        if overwrite is True or outfile.is_file() is False:
            hdu.writeto(outfile, overwrite=overwrite)
            hdu.close()
        else:
            hdu.close()

    # remove directory if it remains empty
    try:
        survey_dir.rmdir()
    except:
        pass
