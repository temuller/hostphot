import requests
from typing import Optional

import astropy.units as u
from astropy.io import fits

from astropy.wcs import WCS

from hostphot.utils import open_fits_from_url, open_fits_from_urls, select_image
from hostphot.surveys_utils import get_survey_filters, check_filters_validity

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

    if isinstance(size, (float, int)):
        size_arcsec = (size * u.arcmin).to(u.arcsec).value
    else:
        size_arcsec = size.to(u.arcsec).value

    base_url = "https://irsa.ipac.caltech.edu/cgi-bin/Cutouts/nph-cutouts?"
    locstr = f"{str(ra)}+{str(dec)}+eq"

    mission_dict = {
        "mission": "SEIP",
        "min_size": "18",
        "max_size": "1800",
        "units": "arcsec",
        "locstr": locstr,
        "sizeX": size_arcsec,  # in arcsec (see "units")
        "ntable_cutouts": "1",
        "cutouttbl1": "science",
        "mode": "PI",  # XML output file
    }

    # check image size (in arcsec)
    if size_arcsec < int(mission_dict["min_size"]):
        mission_dict["sizeX"] = int(mission_dict["min_size"])
        print(
            f'Image cannot be smaller than {mission_dict["min_size"]} arcsec (using this value)'
        )
    elif size_arcsec > int(mission_dict["max_size"]):
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

    # find all the mosaics of each filter (there can be several overlapping ones)
    files_dict = {filt: [] for filt in filters}
    for filt in filters:
        for line in list_text:
            if filt in line and line.endswith(".mosaic.fits") and line not in files_dict[filt]:
                files_dict[filt].append(line)
    unique_urls = list(dict.fromkeys(url for urls in files_dict.values() for url in urls))
    url_to_hdu = dict(zip(unique_urls, open_fits_from_urls(unique_urls)))

    # select the mosaic covering the whole cutout, the deepest one if there are several
    hdu_list_all = []
    for filt in filters:
        candidates = [url_to_hdu[url] for url in files_dict[filt] if url_to_hdu[url] is not None]
        images = [(hdu[0].data, WCS(hdu[0].header), hdu[0].header.get("EXPTIME")) for hdu in candidates]
        index = select_image(images, ra, dec, mission_dict["sizeX"], label=f"{survey} {filt}")
        hdu_list_all.append(None if index is None else candidates[index])

    hdu_list = []
    for hdu in hdu_list_all:
        if hdu is not None:
            hdu[0].header["MAGZP"] = hdu[0].header["ZPAB"]
        hdu_list.append(hdu)
    return hdu_list
