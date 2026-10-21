import tempfile
import zipfile
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
from hostphot.utils import check_work_dir, suppress_stdout, image_coverage, FULL_COVERAGE
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
        candidates = _candidate_observations(results_df, filt, coords)
        if len(candidates) == 0:
            warnings.warn(f"HST {filt}: no images covering the position were found.")
            return None
        # the images are downloaded from the deepest to the shallowest, until one covers the
        # whole cutout (otherwise, the one with the largest coverage is used, with a warning)
        best, best_coverage = None, 0.0
        n_downloads = 0
        with tempfile.TemporaryDirectory(prefix="hostphot_hst_") as tmp_dir:
            for _, row in candidates.iterrows():
                if n_downloads >= MAX_DOWNLOADS:
                    break
                n_downloads += 1
                hdu = _download_observation(esahubble, row["observation_id"], Path(tmp_dir))
                if hdu is None:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", AstropyWarning)
                    img_wcs = WCS(hdu[0].header, naxis=2)
                coverage = image_coverage(hdu[0].data, img_wcs, ra, dec, size_arcsec)
                if coverage > best_coverage:
                    best, best_coverage = hdu, coverage
                if coverage >= FULL_COVERAGE:
                    break
        if best is not None and best_coverage < FULL_COVERAGE:
            warnings.warn(
                f"HST {filt}: none of the downloaded images covers the whole cutout; using the one with the "
                f"largest coverage ({100 * best_coverage:.0f}% of the requested area)."
            )
        return best

    with ThreadPoolExecutor() as executor:
        hdu_list = list(executor.map(download_hst_filter, filters))
    # HST images are large so need to be trimmed
    for hdu, filt in zip(hdu_list, filters):
        if hdu is None:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AstropyWarning)
            img_wcs = WCS(hdu[0].header, naxis=2)
        pixel_scale = float(np.sqrt(np.abs(np.linalg.det(img_wcs.pixel_scale_matrix))) * 3600)
        size_pixels = int(size_arcsec / pixel_scale)
        trimmed_data = Cutout2D(hdu[0].data, coords, size_pixels, img_wcs)
        hdu[0].data = trimmed_data.data
        hdu[0].header.update(trimmed_data.wcs.to_header())
    return hdu_list


# maximum number of images downloaded per filter while looking for one covering the cutout
MAX_DOWNLOADS = 5


def _instrument_names(filt: str) -> tuple[str, str]:
    """Returns the archive name of the instrument/detector of a filter (e.g. ``ACS/WFC``) and the
    token identifying the detector in the names of the combined products (e.g. ``_acs_wfc_``)."""
    split_filt = filt.split("_")
    if split_filt[0].startswith("NICMOS"):
        camera = split_filt[0][-1]  # e.g. NICMOS2_F110W -> NICMOS/NIC2
        return f"NICMOS/NIC{camera}", f"_nic_nic{camera}_"
    if len(split_filt) == 3:
        # e.g. WFC3_UVIS_F275W -> WFC3/UVIS, ACS_WFC_F814W -> ACS/WFC
        return f"{split_filt[0]}/{split_filt[1]}", f"_{split_filt[0].lower()}_{split_filt[1].lower()}_"
    raise ValueError(f"Incorrect filter name: {filt}")


def _candidate_observations(results_df, filt: str, coords: SkyCoord):
    """Selects the observations of a filter that may cover the position, from the deepest."""
    check_HST_filters(filt)
    filt_name = filt.split("_")[-1]
    detector, token = _instrument_names(filt)
    instrument = detector.split("/")[0]
    # the ACS filters can be given together with the clear filter of the other filter wheel
    # (e.g. CLEAR1L;F814W); multi-filter products (e.g. F814W/F555W) are not used
    filter_match = results_df["filter"].astype(str).apply(
        lambda value: filt_name in [part.strip() for part in value.split(";")]
    )
    obs_ids = results_df["observation_id"].astype(str).str.lower()
    # individual products are listed with the detector (e.g. ACS/WFC) and the combined ones
    # (mosaics) with the instrument only (e.g. ACS), with the detector in their names
    instrument_match = (results_df["instrument_name"] == detector) | (
        (results_df["instrument_name"] == instrument) & obs_ids.str.contains(token, regex=False)
    )
    # the "total" products are detection images (combining all the filters), not calibrated images
    not_total = ~obs_ids.str.contains("_total", regex=False)
    obs_df = results_df[filter_match & instrument_match & not_total].copy()
    if len(obs_df) == 0:
        return obs_df
    # images whose centre is too far for them to cover the position are skipped
    if {"ra", "dec", "fov_size"}.issubset(obs_df.columns):
        centres = SkyCoord(obs_df["ra"].values, obs_df["dec"].values, unit="deg")
        separation = centres.separation(coords).deg
        fov = obs_df["fov_size"].astype(float).values
        obs_df = obs_df[~(np.isfinite(fov) & (separation > fov))]
    return obs_df.sort_values(by=["exposure_duration"], ascending=False)


def _download_observation(esahubble, obs_id: str, tmp_dir: Path) -> Optional[fits.HDUList]:
    """Downloads the science product of an observation and returns its science image (in the
    primary HDU, see :func:`update_HST_header`), or ``None`` if it fails."""
    try:
        with suppress_stdout():
            downloaded = esahubble.download_product(
                observation_id=obs_id,
                product_type="SCIENCE",
                calibration_level="PRODUCT",
                filename=obs_id,
                folder=str(tmp_dir),
            )
    except Exception as exc:
        warnings.warn(f"HST: the observation {obs_id} could not be downloaded: {exc}")
        return None
    path = Path(downloaded)
    if not path.is_absolute() and not path.exists():
        path = tmp_dir / path.name
    if path.suffix == ".zip" or zipfile.is_zipfile(path):
        # some products are zip files with the drizzled images
        with zipfile.ZipFile(path) as zip_file:
            names = [name for name in zip_file.namelist() if name.endswith((".fits", ".fits.gz"))]
            names = sorted(names, key=lambda name: (0 if "_drc" in name else 1 if "_drz" in name else 2, name))
            if len(names) == 0:
                warnings.warn(f"HST: no image found in the product of {obs_id}.")
                return None
            path = Path(zip_file.extract(names[0], tmp_dir))
    try:
        with fits.open(path) as hdu:
            # copy the data to avoid closing issue with the temporary files (the primary HDU has no
            # data and there are table extensions, e.g. HDRTAB, so each HDU is copied as it is)
            hdu_copy = fits.HDUList([extension.copy() for extension in hdu])
    except Exception as exc:
        warnings.warn(f"HST: the image of {obs_id} could not be read: {exc}")
        return None
    # add necessary information to the header
    try:
        update_HST_header(hdu_copy)
    except KeyError as exc:
        warnings.warn(f"HST: the image of {obs_id} has no photometric calibration keywords ({exc}).")
        return None
    # only the science image (moved into the primary HDU) is kept, as the other extensions are
    # not used and are as large as the (full) image
    if hdu_copy[0].data is None:
        return None
    return fits.HDUList([hdu_copy[0]])