# Check the following urls for more info about Pan-STARRS:
#
#     https://outerspace.stsci.edu/display/PANSTARRS/PS1+Image+Cutout+Service#PS1ImageCutoutService-ImportantFITSimageformat,WCS,andflux-scalingnotes
#     https://outerspace.stsci.edu/display/PANSTARRS/PS1+Stack+images#PS1Stackimages-Photometriccalibration
#
# For DES:
#
#     https://des.ncsa.illinois.edu/releases/dr1/dr1-docs/processing
#
# For SDSS:
#
#     https://www.sdss.org/dr12/algorithms/fluxcal/#SDSStoAB
#     https://data.sdss.org/datamodel/files/BOSS_PHOTOOBJ/frames/RERUN/RUN/CAMCOL/frame.html
#
# Some parts of this notebook are based on https://github.com/djones1040/PS1_surface_brightness/blob/master/Surface%20Brightness%20Tutorial.ipynb and codes from Lluís Galbany

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

import sep
from astropy.wcs import WCS
from astropy.io import fits

from photutils.utils import calc_total_error
from photutils.aperture import aperture_photometry, EllipticalAperture

from hostphot._constants import workdir
from hostphot.processing.objects_detection import extract_objects, plot_detected_objects
from hostphot.processing.cleaning import remove_nan
from hostphot.photometry.dust import calc_extinction
from hostphot.utils import check_work_dir, store_input
from hostphot.photometry.photometry_utils import magnitude_calculation
from hostphot.photometry.image_utils import adapt_aperture, get_image_exptime
from hostphot.surveys_utils import (
    get_survey_filters,
    survey_pixel_units,
    check_filters_validity,
    check_survey_validity,
    flipped_surveys,
    bkg_surveys,
)

import warnings
from astropy.utils.exceptions import AstropyWarning

sep.set_sub_object_limit(1e4)

def kron_flux(
    data: np.ndarray,
    bkg: float,
    exptime: float,
    objects: np.ndarray,
    kronrad: float,
    scale: float,
) -> tuple[float, float]:
    """Calculates the Kron flux.

    Parameters
    ----------
    data: Data of an image.
    bkg: Background error of the images.
    exptime: Exposure time.
    objects: Objects detected with `sep.extract()`.
    kronrad: Kron radius.
    scale: Scale of the Kron radius.

    Returns
    -------
    flux: Kron flux.
    flux_err: Kron flux error.
    """
    # theta must be in the range [-pi/2, pi/2] for sep.sum_ellipse()
    if objects["theta"] > np.pi / 2:
        objects["theta"] -= np.pi
    elif objects["theta"] < -np.pi / 2:
        objects["theta"] += np.pi
        
    error = calc_total_error(data, bkg, exptime)
    positions = np.array([objects["x"], objects["y"]]).T
    aperture = EllipticalAperture(positions, 
                                    objects["a"][0] * scale * kronrad, 
                                    objects["b"][0] * scale * kronrad,
                                    objects["theta"][0]
                                    )
    phot_table = aperture_photometry(data, aperture, error=error)
    flux = phot_table["aperture_sum"].value[0]
    flux_err = phot_table["aperture_sum_err"].value[0]

    return flux, flux_err


def optimize_aperture(
    data: np.ndarray, bkg: float, exptime: float, objects: np.ndarray, eps: float = 0.001
) -> tuple[float, float, float, float]:
    """Optimizes the Kron flux by iteration over different scales.

    The stop condition is met when the fractional change in flux between iterations
    is less that ``eps``.

    Parameters
    ----------
    data: Data of an image.
    bkg: Background error of the images.
    exptime: Exposure time.
    objects: Objects detected with :func:`sep.extract()`.
    eps: Minimum percent change in flux allowed between iterations.

    Returns
    -------
    opt_flux: Optimized Kron flux.
    opt_flux_err: Optimized Kron flux error.
    kronrad: Kron radius.
    opt_scale: Optimized scale for the Kron radius.
    """
    kronrad, _ = sep.kron_radius(
        data,
        objects["x"],
        objects["y"],
        objects["a"],
        objects["b"],
        objects["theta"],
        r=6.0,  # following sep docs
    )
    kronrad = kronrad[0]

    opt_flux = 0.0
    # iterate over scales
    scales = np.arange(1, 10, 0.01)
    for scale in scales:
        flux, flux_err = kron_flux(data, bkg, exptime, objects, kronrad, scale)
        calc_eps = np.abs(opt_flux - flux) / flux
        opt_flux, opt_flux_err = flux, flux_err
        opt_scale = scale
        if calc_eps < eps:
            opt_scale = scale
            break
        elif np.isnan(calc_eps):
            opt_scale = scales[-2]
            warnings.warn("Warning: the aperture might not fit in the image!")
            break

    return opt_flux, opt_flux_err, kronrad, opt_scale


def extract_aperture(
    name: str,
    host_ra: float,
    host_dec: float,
    filt: str,
    survey: str,
    ra: float = None,
    dec: float = None,
    bkg_sub: bool = False,
    threshold: float = 10,
    use_mask: bool = True,
    optimize_kronrad: bool = True,
    eps: float = 0.0001,
    gal_dist_thresh: float = -1,
    deblend_cont: float = 0.005,
    save_plots: bool = True,
    save_aperture_params: bool = True,
) -> tuple[np.ndarray, WCS, float, float, bool]:
    """Calculates the aperture for a given galaxy.

    Parameters
    ----------
    name: Name of the object to find the path of the fits file.
    host_ra: Host-galaxy right ascension of the galaxy in degrees.
    host_dec: Host-galaxy declination of the galaxy in degrees.
    filt: Filter to use to load the fits file. List is commonly used for coadds.
    survey: Survey to use for the zero-points and pixel scale.
    ra: Right ascension of an object, in degrees. Used for plotting the position of the object.
    dec: Declination of an object, in degrees. Used for plotting the position of the object.
    bkg_sub: If `True`, the image gets background subtracted. By default, only
        the images that need it get background subtracted (WISE, 2MASS and
        VISTA).
    threshold: Threshold used by `sep.extract()` to extract objects.
    use_mask: If `True`, the masked fits files are used. These must have
        been created beforehand.
    optimize_kronrad: If `True`, the Kron-radius scale is optimized, increasing the
        aperture size until the change in flux is less than ``eps``.
    eps: The Kron radius is increased until the change in flux is lower than ``eps``.
        A value of 0.0001 means 0.01% change in flux.
    gal_dist_thresh: Distance in arcsec to crossmatch the galaxy coordinates with a detected object,
        where the object nearest to the galaxy position is considered as the galaxy (within
        the given threshold). If no objects are found within the given distance threshold,
        the galaxy is considered as not found and a warning is printed. If a non-positive value
        is given, the threshold is considered as infinite, i.e. the closest detected object is
        considered as the galaxy (default option).
    deblend_cont : Minimum contrast ratio used for object deblending. Default is 0.005.
        To entirely disable deblending, set to 1.0.
    save_plots: If `True`, the mask and galaxy aperture figures are saved.
    save_aperture_params: If `True`, the extracted mask parameters are saved into a pickle file.

    Returns
    -------
    gal_obj: Galaxy object.
    wcs: Image's WCS.
    kronrad: Kron radius.
    scale: Scale for the Kron radius.
    flip: Whether to flip the orientation of the aperture.
    """
    # initial checks
    check_survey_validity(survey)
    check_work_dir(workdir)
    obj_dir = Path(workdir, name)
    if use_mask:
        suffix = "_masked"
    else:
        suffix = ""
    if isinstance(filt, list):
        filt = "".join(f for f in filt)
    fits_file = obj_dir / survey / f"{survey}_{filt}{suffix}.fits"

    # image information
    hdu = fits.open(fits_file)
    hdu = remove_nan(hdu)
    header = hdu[0].header
    data = hdu[0].data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        wcs = WCS(header, naxis=2)

    data = data.astype(np.float64)
    bkg = sep.Background(data)
    # background subtraction, if needed
    if (bkg_sub is None and survey in bkg_surveys) or bkg_sub is True:
        data_sub = np.copy(data - bkg.back())
    else:
        data_sub = np.copy(data)

    # extract galaxy
    gal_obj, _ = extract_objects(
        data_sub,
        bkg.globalrms,
        host_ra,
        host_dec,
        threshold,
        wcs,
        gal_dist_thresh,
        deblend_cont,
    )
    if optimize_kronrad:
        exptime = get_image_exptime(header, survey)
        try:
            pixel_units = survey_pixel_units(survey, filt)
        except:
            # probably a coadd, so only need one filter
            pixel_units = survey_pixel_units(survey, filt[0])
        if pixel_units == "counts":
            _exptime = 1
        else:
            _exptime = exptime
        opt_res = optimize_aperture(data_sub, bkg.rms(), _exptime, gal_obj, eps)
        _, _, kronrad, scale = opt_res
    else:
        scale = 2.5
        kronrad, _ = sep.kron_radius(
            data_sub,
            gal_obj["x"],
            gal_obj["y"],
            gal_obj["a"],
            gal_obj["b"],
            gal_obj["theta"],
            6.0,
        )

    if save_plots is True:
        outfile = obj_dir / survey / f"global_{survey}_{filt}.jpg"
        title = rf"{name}: {survey}-${filt}$"
        plot_detected_objects(
            hdu,
            gal_obj,
            scale * kronrad,
            ra,
            dec,
            host_ra,
            host_dec,
            title,
            outfile,
        )

    if survey in flipped_surveys:
        flip = True
    else:
        flip = False

    if save_aperture_params is True:
        # save galaxy object and aperture parameters
        gal_df = pd.DataFrame(gal_obj)
        gal_df["kronrad"] = kronrad
        gal_df["scale"] = scale
        gal_df["flip"] = flip
        gal_df["filt"] = filt
        gal_df["survey"] = survey
        outfile = obj_dir / survey / f"aperture_parameters_{filt}.csv"
        gal_df.to_csv(outfile, index=False)
    hdu.close()

    return gal_obj, wcs, kronrad, scale, flip


def load_aperture_params(
    name: str, filt: str | list, survey: str
) -> tuple[np.ndarray, WCS, float, float, bool]:
    """Loads previously saved aperture parameters.

    Parameters
    ----------
    name: Name of the object to find the path of the aperture-parameters file.
    filt: Name of the filter used for the aperture parameters. Coadds are
        also valid.
    survey: Survey name to be used.

    Returns
    -------
    gal_obj, nongal_objs, wcs, sigma, r, flip: Mask parameters.
    """
    if isinstance(filt, list):
        filt = "".join(f for f in filt)
    obj_dir = Path(workdir, name)
    aper_params_file = obj_dir / survey / f"aperture_parameters_{filt}.csv"
    gal_df = pd.read_csv(aper_params_file)

    # split parameters
    kronrad = gal_df.pop("kronrad").values[0]
    scale = gal_df.pop("scale").values[0]  # remove host-galaxy row
    flip = gal_df.pop("flip").values[0]
    # remove unused columns
    _ = gal_df.pop("filt")
    _ = gal_df.pop("survey")
    # DataFrame to structured/record array
    gal_obj = gal_df.to_records()

    # load image WCS
    fits_file = obj_dir / survey / f"{survey}_{filt}.fits"
    hdu = fits.open(fits_file)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        wcs = WCS(hdu[0].header, naxis=2)

    return gal_obj, wcs, kronrad, scale, flip


def photometry(
    name: str,
    host_ra: float,
    host_dec: float,
    filt: str,
    survey: str,
    ra: Optional[float] = None,
    dec: Optional[float] = None,
    bkg_sub: Optional[bool] = None,
    threshold: float = 10,
    use_mask: bool = True,
    correct_extinction: float = True,
    optimize_kronrad: bool = True,
    eps: float = 0.0001,
    gal_dist_thresh: float = -1,
    deblend_cont: float = 0.005,
    common_aperture: bool = False,
    ref_filt: Optional[str] = None,
    ref_survey: Optional[str] = None,
    save_plots: bool = True,
    save_aperture_params: bool = False,
) -> tuple[float, float, float, float, float]:
    """Calculates the global aperture photometry of a galaxy using
    the Kron flux.

    **Note:** the galaxy must be ideally centred in the image.

    Parameters
    ----------
    name: Name of the object to find the path of the fits file.
    host_ra: Host-galaxy right ascension of the galaxy in degrees.
    host_dec: Host-galaxy declination of the galaxy in degrees.
    filt: Filter to use to load the fits file.
    survey: Survey to use for the zero-points and pixel scale.
    ra: Right ascension of an object, in degrees. Used for plotting the position of the object.
    dec: Declination of an object, in degrees. Used for plotting the position of the object.
    bkg_sub: If ``True``, the image gets background subtracted. By default, only
        the images that need it get background subtracted (WISE, 2MASS and
        VISTA).
    threshold: Threshold used by `sep.extract()` to extract objects.
    use_mask: If `True`, the masked fits files are used. These must have
        been created beforehand.
    correct_extinction: If `True`, corrects for Milky-Way extinction using the recalibrated dust maps
        by Schlafly & Finkbeiner (2011) and the extinction law from Fitzpatrick (1999).
    optimize_kronrad: If `True`, the Kron-radius scale is optimized, increasing the
        aperture size until the change in flux is less than ``eps``.
    eps: The Kron radius is increased until the change in flux is lower than ``eps``.
        A value of 0.0001 means 0.01% change in flux.
    gal_dist_thresh: Distance in arcsec to crossmatch the galaxy coordinates with a detected object,
        where the object nearest to the galaxy position is considered as the galaxy (within
        the given threshold). If no objects are found within the given distance threshold,
        the galaxy is considered as not found and a warning is printed. If a non-positive value
        is given, the threshold is considered as infinite, i.e. the closest detected object is
        considered as the galaxy (default option).
    deblend_cont : Minimum contrast ratio used for object deblending. Default is 0.005.
        To entirely disable deblending, set to 1.0.
    common_aperture: Whether to use common aperture photometry using the reference filter(s) and survey.
    ref_filt: Reference filter (or coadd filters) from which to use the aperture parameters. Note that the parameters
        need to have been previously saved with ``save_aperture_params=True``.
    ref_survey: Reference survey from which to use the aperture parameters. Note that the parameters
        need to have been previously saved with ``save_aperture_params=True``.
    save_plots: If `True`, the mask and galaxy aperture figures are saved.
    save_aperture_params: Whether to save the aperture parameters.

    Returns
    -------
    mag: Aperture magnitude.
    mag_err: Error on the aperture magnitude.
    flux: Aperture flux.
    flux_err: Total flux error on the aperture flux.
    zp: Zeropoint.
    """
    # initial checks
    check_survey_validity(survey)
    check_work_dir(workdir)
    obj_dir = Path(workdir, name)
    if use_mask:
        suffix = "_masked"
    else:
        suffix = ""
    fits_file = obj_dir / survey / f"{survey}_{filt}{suffix}.fits"

    # image information
    hdu = fits.open(fits_file)
    hdu = remove_nan(hdu)
    header = hdu[0].header
    data = hdu[0].data
    exptime = get_image_exptime(header, survey)
    pixel_units = survey_pixel_units(survey, filt)
    if pixel_units == "counts":
        _exptime = 1
    else:
        _exptime = exptime
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        wcs = WCS(header, naxis=2)

    data = data.astype(np.float64)
    bkg = sep.Background(data)
    # background subtraction, if needed
    if (bkg_sub is None and survey in bkg_surveys) or bkg_sub is True:
        data_sub = np.copy(data - bkg.back())
    else:
        data_sub = np.copy(data)

    if (ref_filt is None) | (ref_survey is None) | (common_aperture is False):
        # independent photometry for each filter
        gal_obj, wcs, kronrad, scale, _ = extract_aperture(
            name,
            host_ra,
            host_dec,
            filt,
            survey,
            ra,
            dec,
            bkg_sub,
            threshold,
            use_mask,
            optimize_kronrad,
            eps,
            gal_dist_thresh,
            deblend_cont,
            save_plots,
            save_aperture_params,
        )
    elif ((common_aperture is True) & (ref_filt is None)) | (
        (common_aperture is True) & (ref_survey is None)
    ):
        # error
        raise ValueError(
            "Reference filter(s) 'ref_filt' and survey 'ref_survey' must be given for common aperture photometry"
        )
    else:
        # common aperture photometry
        gal_obj, master_wcs, kronrad, scale, flip2 = load_aperture_params(
            name, ref_filt, ref_survey
        )
        if survey != ref_survey:
            # adapt different surveys
            if survey in flipped_surveys:
                flip = True
            else:
                flip = False
            if flip == flip2:
                flip_ = False
            else:
                flip_ = True
            # factor used for scaling the Kron radius between different pixel-scales / surveys
            gal_obj, conv_factor = adapt_aperture(
                gal_obj, master_wcs, wcs, flip_
            )
    # get Kron flux
    flux, flux_err = kron_flux(
        data_sub, bkg.rms(), _exptime, gal_obj, kronrad, scale
    )
    
    # aperture area for an ellipse (or circle)
    ap_area = np.pi * gal_obj["a"][0] * gal_obj["b"][0]
    mag, mag_err, flux, flux_err, zp = magnitude_calculation(
        flux,
        flux_err,
        survey,
        filt,
        ap_area,
        header,
        bkg.globalrms,
    )

    if correct_extinction is True:
        A_ext = calc_extinction(filt, survey, host_ra, host_dec)
        mag -= A_ext
        flux *= 10 ** (0.4 * A_ext)

    if save_plots is True:
        outfile = obj_dir / survey / f"global_{survey}_{filt}.jpg"
        filt_ = filt.replace("_", "-")
        title = f"{name}: {survey}-${filt_}$"
        plot_detected_objects(
            hdu,
            gal_obj,
            scale * kronrad,
            ra,
            dec,
            host_ra,
            host_dec,
            title,
            outfile,
        )
    hdu.close()

    return mag, mag_err, flux, flux_err, zp


def multi_band_phot(
    name: str,
    host_ra: float,
    host_dec: float,
    filters: Optional[str | list] = None,
    survey: str = "PanSTARRS",
    ra: Optional[float] = None,
    dec: Optional[float] = None,
    bkg_sub: Optional[bool] = None,
    threshold: float = 10,
    use_mask: bool = True,
    correct_extinction: bool = True,
    common_aperture: bool = False,
    ref_filt: Optional[str | list] = None,
    ref_survey: Optional[str | list] = None,
    optimize_kronrad: bool = True,
    eps: float = 0.0001,
    gal_dist_thresh: float = -1,
    deblend_cont: float = 0.005,
    save_plots: bool = True,
    save_aperture_params: bool = True,
    save_results: bool = True,
    raise_exception: bool = True,
    save_input: bool = True,
):
    """Calculates multi-band aperture photometry of the host galaxy
    for an object.

    Parameters
    ----------
    name: Name of the object to find the path of the fits file.
    host_ra: Host-galaxy right ascension of the galaxy in degrees.
    host_dec: Host-galaxy declination of the galaxy in degrees.
    filters: Filters to use to load the fits files. If `None` use all
        the filters of the given survey.
    survey: Survey to use for the zero-points and pixel scale.
    ra: Right ascension of an object, in degrees. Used for plotting the position of the object.
    dec: Declination of an object, in degrees. Used for plotting the position of the object.
    bkg_sub: If ``True``, the image gets background subtracted. By default, only
        the images that need it get background subtracted (WISE, 2MASS and
        VISTA).
    threshold: Threshold used by :func:`sep.extract()` to extract objects.
    use_mask: If ``True``, the masked fits files are used. These must have
        been created beforehand.
    correct_extinction: If ``True``, the magnitudes are corrected for extinction.
    common_aperture: Whether to use common aperture photometry using the reference filter(s) and survey.
    ref_filt: Reference filter(s) used for common aperture photometry.
    ref_survey: Reference survey used for common aperture photometry. By default uses the same
        as the input 'survey'.
    optimize_kronrad: If `True`, the Kron-radius scale is optimized, increasing the
        aperture size until the change in flux is less than ``eps``.
    eps: The Kron radius is increased until the change in flux is lower than ``eps``.
        A value of 0.0001 means 0.01% change in flux.
        when optimizing the Kron radius.
    gal_dist_thresh: Distance in arcsec to crossmatch the galaxy coordinates with a detected object,
        where the object nearest to the galaxy position is considered as the galaxy (within
        the given threshold). If no objects are found within the given distance threshold,
        the galaxy is considered as not found and a warning is printed. If a non-positive value
        is given, the threshold is considered as infinite, i.e. the closest detected object is
        considered as the galaxy (default option).
    deblend_cont : Minimum contrast ratio used for object deblending. Default is 0.005.
        To entirely disable deblending, set to 1.0.
    save_plots: If ``True``, the mask and galaxy aperture figures are saved.
    save_aperture_params: Whether to save the aperture parameters.
    save_results: If ``True``, the magnitudes are saved into a csv file.
    raise_exception: If ``True``, an exception is raised if the photometry fails for any filter.
    save_input: Whether to save the input parameters.

    Returns
    -------
    phot_df: DataFrame with the object's photometry and other info.

    Examples
    --------
    >>> from hostphot.photometry import global_photometry as gp
    >>> name = 'SN2004eo'
    >>> host_ra, host_dec = 308.2092, 9.92755  # coords of host galaxy of SN2004eo
    >>> ra, dec =  308.22579, 9.92853 # coords of SN2004eo
    >>> results = gp.multi_band_phot(name, 
                                     host_ra, 
                                     host_dec, 
                                     survey=survey, 
                                     ra=ra, 
                                     dec=dec, 
                                     use_mask=True,
                                     common_aperture=True, 
                                     ref_filt='riz',
                                     ref_survey='PanSTARRS', 
                                     save_plots=True)
    """
    input_params = locals()  # dictionary
    check_survey_validity(survey)
    if filters is None:
        if survey in ["HST", "JWST"]:
            raise ValueError(f"For {survey}, the filter needs to be specified!")
        filters = get_survey_filters(survey)
    else:
        check_filters_validity(filters, survey)
    if survey in ["HST", "JWST"]:
        filters = [filters]

    # save input parameters
    if save_input is True:
        inputs_file = Path(workdir, name, survey, "input_global_photometry.csv")
        store_input(input_params, inputs_file)

    results_dict = {
        "name": name,
        "host_ra": host_ra,
        "host_dec": host_dec,
        "survey": survey,
    }

    if ref_survey is None:
        ref_survey = survey
    """
    if (common_aperture is True) & (ref_filt is not None):
        # use common aperture
        _ = extract_aperture(
            name,
            host_ra,
            host_dec,
            ref_filt,
            ref_survey,
            ra=ra,
            dec=dec,
            bkg_sub=bkg_sub,
            threshold=threshold,
            use_mask=use_mask,
            optimize_kronrad=optimize_kronrad,
            eps=eps,
            gal_dist_thresh=gal_dist_thresh,
            deblend_cont=deblend_cont,
            save_plots=save_plots,
            save_aperture_params=True,
        )
    elif (common_aperture is True) & (ref_filt is None):
        raise ValueError(
            "Reference filter(s) 'ref_filt' must be given for common aperture"
        )
    """

    for filt in filters:
        try:
            mag, mag_err, flux, flux_err, zp = photometry(
                name,
                host_ra,
                host_dec,
                filt,
                survey,
                ra=ra,
                dec=dec,
                bkg_sub=bkg_sub,
                threshold=threshold,
                use_mask=use_mask,
                correct_extinction=correct_extinction,
                optimize_kronrad=optimize_kronrad,
                eps=eps,
                gal_dist_thresh=gal_dist_thresh,
                deblend_cont=deblend_cont,
                common_aperture=common_aperture,
                ref_filt=ref_filt,
                ref_survey=ref_survey,
                save_plots=save_plots,
                save_aperture_params=save_aperture_params,
            )
            results_dict[filt] = mag
            results_dict[f"{filt}_err"] = mag_err
            results_dict[f"{filt}_flux"] = flux
            results_dict[f"{filt}_flux_err"] = flux_err
            results_dict[f"{filt}_zeropoint"] = zp
        except Exception as exc:
            if raise_exception is True:
                raise Exception(f"{filt}-band: {exc}")
            else:
                results_dict[filt] = np.nan
                results_dict[f"{filt}_err"] = np.nan
                results_dict[f"{filt}_flux"] = np.nan
                results_dict[f"{filt}_flux_err"] = np.nan
                results_dict[f"{filt}_zeropoint"] = np.nan

    phot_df = pd.DataFrame({key: [val] for key, val in results_dict.items()})
    if save_results is True:
        outfile = Path(workdir, name, survey, "global_photometry.csv")
        phot_df.to_csv(outfile, index=False)

    return phot_df
