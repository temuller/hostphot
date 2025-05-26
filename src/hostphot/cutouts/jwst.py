import numpy as np
from pathlib import Path

import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord

from astroquery.esa.jwst import Jwst

from hostphot._constants import workdir
from hostphot.utils import check_work_dir, suppress_stdout
from hostphot.surveys_utils import check_JWST_filters, survey_pixel_scale

import warnings
from astropy.utils.exceptions import AstropyWarning


def update_JWST_header(hdu: fits.hdu.ImageHDU) -> None:
    """Updates the JWST image header with the necessary keywords.

    Parameters
    ----------
    hdu : JWST FITS image.
    """
    # get WCS - SCI is hdu[1] - hostphot always assumes hdu[0] is used, so need to move them over
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        img_wcs = WCS(hdu[1].header)
    hdu[0].header.update(img_wcs.to_header())
    hdu[0].header["PIXAR_SR"] = hdu[1].header["PIXAR_SR"]
    hdu[0].data = hdu[1].data
    # add zeropoints
    # https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-performance/nircam-absolute-flux-calibration-and-zeropoints
    pixar_sr = hdu[0].header["PIXAR_SR"]
    hdu[0].header["MAGZP"] = (
       -6.10 - (2.5 * np.log10(pixar_sr))
    )

def set_JWST_image(file: str, filt: str, name: str) -> None:
    """Moves a previously downloaded JWST image into the work directory.

    The image's header is updated with the necessary keywords to obtain
    photometry and is also moved under the objects directory inside the
    work directory.

    JWST images take very long to download, so the user might prefer to
    download the images manually and then use this function to include
    the image into the workflow.

    Parameters
    ----------
    file: JWST image to use.
    filt: JWST filter, e.g. ``NIRCam_F150W``.
    name: Object's name.
    """
    # check output directory
    check_work_dir(workdir)
    check_JWST_filters(filt)
    obj_dir = Path(workdir, name, "JWST")
    if obj_dir.is_dir() is False:
        obj_dir.mkdir(parents=True, exist_ok=True)
    # update header and save file
    hdu = fits.open(file)
    update_JWST_header(hdu)
    outfile = obj_dir / f"JWST_{filt}.fits"
    hdu.writeto(outfile, overwrite=True)
    
def get_JWST_images(ra: float, dec: float, size: float | u.Quantity = 3, 
                        filters: list = ["WFC3_UVIS_F225W"]) -> list[fits.ImageHDU]:
    """Downloads a set of JWST fits images for a given set
    of coordinates and filters using astroquery.

    Parameters
    ----------
    ra: Right ascension in degrees.
    dec: Declination in degrees.
    size: Image size. If a float is given, the units are assumed to be arcmin.
    filters: Filters to use, e.g. ``WFC3_UVIS_F225W``.

    Return
    ------
    hdu_list: List with fits image for the given filter. ``None`` is returned if no image is found.
    """
    Jwst.get_status_messages()
    
    if isinstance(size, (float, int)):
        size_arcsec = (size * u.arcmin).to(u.arcsec)
    else:
        size_arcsec = size.to(u.arcsec)
    size_arcsec = size_arcsec.value
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        coords = SkyCoord(
            ra=ra, dec=dec, unit=(u.degree, u.degree), frame="icrs"
        )

    # query observations at the given coordinates
    with suppress_stdout():
        result = Jwst.cone_search(
            radius=3*u.arcsec,
            coordinate=coords,
            cal_level="Top",
            prod_type="image",
            #instrument_name=instrument,
            #filter_name=filt,
            only_public=True,
            async_job=True,
        ).get_data()
    results_df = result.to_pandas()
            
    hdu_list = []
    for filt in filters:
        check_JWST_filters(filt)
        # separate the instrument name from the actual filter
        split_filt = filt.split("_")
        filt = split_filt[-1]
        instrument = split_filt[0].upper() + "/IMAGE"
        # filter by filter and instrument
        obs_df = results_df[results_df["instrument_name"] == instrument]
        obs_df = obs_df[obs_df["energy_bandpassname"] == filt]
        obs_df = obs_df[obs_df["calibrationlevel"] == 3]

        # start download
        temp_file = None
        for obs_id in obs_df.observationid:
            try:
                product_list = Jwst.get_product_list(observation_id=obs_id, product_type='science', cal_level=3).to_pandas()
                # choose first file as there es no exptime info and the images take too lonk to download multiple ones
                file_name = [file for file in product_list.filename if file.endswith(".fits")][0] 
                temp_file = Jwst.get_product(file_name=file_name)
                break
            except:
                pass
        if temp_file is None:
            hdu_list.append(None)
            continue
        hdu = fits.open(temp_file)
        # remove the temporary files
        Path(temp_file).unlink()
        # add necessary information to the header
        update_JWST_header(hdu)
        hdu_list.append(hdu)
    # JWST images are large so need to be trimmed
    for hdu, filt in zip(hdu_list, filters):
        if hdu is None:
            continue
        pixel_scale = survey_pixel_scale("JWST", filt)  # same pixel scale for all filters
        size_pixels = int(size_arcsec / pixel_scale)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AstropyWarning)
            img_wcs = WCS(hdu[0].header)
        trimmed_data = Cutout2D(hdu[0].data, coords, size_pixels, img_wcs)
        hdu[0].data = trimmed_data.data
        hdu[0].header.update(trimmed_data.wcs.to_header())
    return hdu_list