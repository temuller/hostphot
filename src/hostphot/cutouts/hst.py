import numpy as np
from typing import Optional
from pathlib import Path

import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord

from astroquery.esa.hubble import ESAHubble  # HST

from hostphot._constants import workdir
from hostphot.utils import check_work_dir, suppress_stdout
from hostphot.surveys_utils import check_HST_filters, survey_pixel_scale

import warnings
from astropy.utils.exceptions import AstropyWarning


def update_HST_header(hdu: fits.hdu.ImageHDU) -> None:
    """Updates the HST image header with the necessary keywords.

    Parameters
    ----------
    hdu : Header Data Unit.
        HST FITS image.
    """
    # Drizzlepac images have the data on hdu[0]
    # MAST-archive images have the data (SCI) on hdu[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        if hdu[0].data is None and len(hdu) > 1:
            # MAST image: move things to hdu[0] to homogenise
            img_wcs = WCS(hdu[1].header)
            hdu[0].header.update(img_wcs.to_header())
            hdu[0].data = hdu[1].data
        for key in ["PHOTFLAM", "PHOTPLAM"]:
            if key not in hdu[0].header and len(hdu) > 1 and key in hdu[1].header:
                hdu[0].header[key] = hdu[1].header[key]
    # add zeropoints
    # https://www.stsci.edu/hst/instrumentation/acs/data-analysis/zeropoints
    photflam = hdu[0].header["PHOTFLAM"]
    photplam = hdu[0].header["PHOTPLAM"]
    hdu[0].header["MAGZP"] = (
        -2.5 * np.log10(photflam) - 5 * np.log10(photplam) - 2.408
    )

def set_HST_image(file: str, filt: str, name: str) -> None:
    """Moves a previously downloaded HST image into the work directory.

    The image's header is updated with the necessary keywords to obtain
    photometry and is also moved under the objects directory inside the
    work directory.

    HST images take very long to download, so the user might prefer to
    download the images manually and then use this function to include
    the image into the workflow.

    Parameters
    ----------
    file: HST image to use.
    filt: HST filter, e.g. ``WFC3_UVIS_F275W``.
    name: Object's name under which the image is saved.
    """
    # check output directory
    check_work_dir(workdir)
    check_HST_filters(filt)
    obj_dir = Path(workdir, name, "HST")
    if obj_dir.is_dir() is False:
        obj_dir.mkdir(parents=True, exist_ok=True)
    # update file and save file
    hdu = fits.open(file)
    update_HST_header(hdu)
    outfile = obj_dir / f"HST_{filt}.fits"
    hdu.writeto(outfile, overwrite=True)

def get_HST_images(ra: float, dec: float, size: float | u.Quantity = 3, 
                        filters: Optional[list] = None) -> list[fits.ImageHDU]:
    """Downloads a set of HST fits images for a given set
    of coordinates and filters using astroquery.

    Parameters
    ----------
    ra: Right ascension in degrees.
    dec: Declination in degrees.
    size: Image size. If a float is given, the units are assumed to be arcmin.
    filters: Filters to use, e.g. ``WFC3_UVIS_F225W`` (they need to be specified).

    Return
    ------
    hdu_list: List with fits image for the given filter. ``None`` is returned if no image is found.
    """
    if filters is None:
        raise ValueError("For HST, the filters need to be specified!")
    if isinstance(filters, str):
        filters = [filters]
    esahubble = ESAHubble()
    esahubble.get_status_messages()
    
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
        result = esahubble.cone_search_criteria(
            radius=3,
            coordinates=coords,
            calibration_level="PRODUCT",
            data_product_type="image",
            #instrument_name=instrument,
            #filters=filt,
            async_job=True,
        )
    results_df = result.to_pandas()
    
    from concurrent.futures import ThreadPoolExecutor

    def download_hst_filter(filt):
        check_HST_filters(filt)
        # separate the instrument name from the actual filter
        split_filt = filt.split("_")
        filt_name = split_filt[-1]
        if split_filt[0].startswith("NICMOS"):
            # e.g. NICMOS2_F110W -> NICMOS/NIC2
            instrument = f"NICMOS/NIC{split_filt[0][-1]}"
        elif len(split_filt) == 3:
            # e.g. WFC3_UVIS_F275W -> WFC3/UVIS, ACS_WFC_F814W -> ACS/WFC
            instrument = f"{split_filt[0]}/{split_filt[1]}"
        else:
            raise ValueError(f"Incorrect filter name: {filt}")
        # filter by filter and instrument (the ACS filters can be given together with the
        # clear filter of the other filter wheel, e.g. CLEAR1L;F814W)
        filter_match = results_df["filter"].astype(str).apply(
            lambda value: filt_name in [part.strip() for part in value.split(";")]
        )
        obs_df = results_df[filter_match]
        obs_df = obs_df[obs_df.instrument_name == instrument]
        # get only exposures shorter than one hour
        obs_df = obs_df[obs_df.exposure_duration < 3600]
        obs_df.sort_values(
            by=["exposure_duration"], ascending=False, inplace=True
        )

        # start download 
        filename = f"HST_tmp_{ra}_{dec}_{filt}"  # the extension is added below
        download_success = False
        for obs_id in obs_df.observation_id:
            try:
                esahubble.download_product(
                    observation_id=obs_id,
                    product_type="SCIENCE",
                    calibration_level="PRODUCT",
                    filename=filename,
                )
                download_success = True
                break
            except:
                pass
        temp_file = Path(f"{filename}.fits.gz")
        if download_success and temp_file.is_file():
            hdu = fits.open(temp_file)
            # copy the data to avoid closing issue with temp_file deletion (the primary HDU has no
            # data and there are table extensions, e.g. HDRTAB, so each HDU is copied as it is)
            hdu_copy = fits.HDUList([extension.copy() for extension in hdu])
            hdu.close()
            # remove the temporary files
            temp_file.unlink()
            # add necessary information to the header
            update_HST_header(hdu_copy)
            # only the science image (moved into the primary HDU) is kept, as the other extensions are
            # not used and are as large as the (full) image
            if hdu_copy[0].data is not None:
                hdu_copy = fits.HDUList([hdu_copy[0]])
            return hdu_copy
        else:
            return None

    with ThreadPoolExecutor() as executor:
        hdu_list = list(executor.map(download_hst_filter, filters))
    # HST images are large so need to be trimmed
    for hdu, filt in zip(hdu_list, filters):
        if hdu is None:
            continue
        pixel_scale = survey_pixel_scale("HST", filt)  # same pixel scale for all filters
        size_pixels = int(size_arcsec / pixel_scale)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AstropyWarning)
            img_wcs = WCS(hdu[0].header)
        trimmed_data = Cutout2D(hdu[0].data, coords, size_pixels, img_wcs)
        hdu[0].data = trimmed_data.data
        hdu[0].header.update(trimmed_data.wcs.to_header())
    return hdu_list