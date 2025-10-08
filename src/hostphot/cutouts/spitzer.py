import requests
from typing import Optional

import astropy.units as u
from astropy.io import fits

from hostphot.utils import open_fits_from_url
from hostphot.surveys_utils import get_survey_filters, check_filters_validity, survey_pixel_scale

def get_Spitzer_images(ra: float, dec: float, size: float | u.Quantity = 3, 
                        filters: Optional[str] = None) -> list[fits.ImageHDU]:
    """Gets Spitzer fits images for the given coordinates and
    filters.

    Parameters
    ----------
    ra: Right Ascension in degrees.
    dec: Declination in degrees.
    size: Image size. If a float is given, the units are assumed to be arcmin.
    filters: Filters to use. If ``None``, uses ``IRAC.1, IRAC.2, IRAC.3, IRAC.4, MIPS.1``.

    Returns
    -------
    fits_files: List of fits images.
    """
    survey = "Spitzer"
    if filters is None:
        filters = get_survey_filters(survey)
    check_filters_validity(filters, survey)

    pixel_scale = survey_pixel_scale(survey, "IRAC.1")  # IRAC resolution
    if isinstance(size, (float, int)):
        size_arcsec = (size * u.arcmin).to(u.arcsec).value
    else:
        size_arcsec = size.to(u.arcsec).value
    size_pixels = int(size_arcsec / pixel_scale)

    base_url = "https://irsa.ipac.caltech.edu/cgi-bin/Cutouts/nph-cutouts?"
    locstr = f"{str(ra)}+{str(dec)}+eq"

    mission_dict = {
        "mission": "SEIP",
        "min_size": "18",
        "max_size": "1800",
        "units": "arcsec",
        "locstr": locstr,
        "sizeX": size_pixels,
        "ntable_cutouts": "1",
        "cutouttbl1": "science",
        "mode": "PI",  # XML output file
    }

    # check image size
    if size_pixels < int(mission_dict["min_size"]):
        mission_dict["sizeX"] = int(mission_dict["min_size"])
        print(
            f'Image cannot be smaller than {mission_dict["min_size"]} arcsec (using this value)'
        )
    elif size_pixels > int(mission_dict["max_size"]):
        mission_dict["sizeX"] = int(mission_dict["max_size"])
        print(
            f'Image cannot be larger than {mission_dict["max_size"]} arcsec (using this value)'
        )

    mission_url = "&".join(
        [f"{key}={val}" for key, val in mission_dict.items()]
    )
    url = base_url + mission_url
    response = requests.get(url)
    list_text = response.text.split()

    # find the fits file for each fitler (None if not found)
    files_dict = {filt: None for filt in filters}
    for filt in filters:
        for line in list_text:
            if filt in line and line.endswith(".mosaic.fits"):
                files_dict[filt] = line
                # pick the first image and move to the next filter
                break

    hdu_list = []
    for filt, file in files_dict.items():
        if file is not None:
            hdu = open_fits_from_url(file)
            hdu[0].header["MAGZP"] = hdu[0].header["ZPAB"]
            hdu_list.append(hdu)
        else:
            hdu_list.append(None)
    return hdu_list