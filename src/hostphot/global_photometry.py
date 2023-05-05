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

import os
import pickle
import numpy as np
import pandas as pd

import sep
from astropy import wcs
from astropy.io import fits

from hostphot._constants import workdir
from hostphot.utils import (
    get_survey_filters,
    check_survey_validity,
    check_filters_validity,
    get_image_gain,
    check_work_dir,
    magnitude_calculation,
    adapt_aperture,
    bkg_surveys,
)
from hostphot.objects_detect import extract_objects, plot_detected_objects
from hostphot.image_cleaning import remove_nan
from hostphot.dust import calc_extinction

import warnings
from astropy.utils.exceptions import AstropyWarning

sep.set_sub_object_limit(1e4)


# -------------------------------
def kron_flux(data, err, gain, objects, kronrad, scale):
    """Calculates the Kron flux.

    Parameters
    ----------
    data: ndarray
        Data of an image.
    err: float or 2D array
        Background error of the images.
    gain: float
        Gain value.
    objects: array
        Objects detected with `sep.extract()`.
    kronrad: float
        Kron radius.
    scale: float
        Scale of the Kron radius.

    Returns
    -------
    flux: array
        Kron flux.
    flux_err: array
        Kron flux error.
    """
    r_min = 1.75  # minimum diameter = 3.5

    if kronrad * np.sqrt(objects["a"] * objects["b"]) < r_min:
        print(f"Warning: using circular photometry")
        flux, flux_err, _ = sep.sum_circle(
            data,
            objects["x"],
            objects["y"],
            r_min,
            err=err,
            subpix=5,
            gain=1.0,
        )
    else:
        # theta must be in the range [-pi/2, pi/2] for sep.sum_ellipse()
        if objects["theta"] > np.pi / 2:
            objects["theta"] -= np.pi
        elif objects["theta"] < -np.pi / 2:
            objects["theta"] += np.pi

        flux, flux_err, _ = sep.sum_ellipse(
            data,
            objects["x"],
            objects["y"],
            objects["a"],
            objects["b"],
            objects["theta"],
            scale * kronrad,
            err=err,
            subpix=5,
            gain=gain,
        )

    return flux, flux_err


def optimize_kron_flux(data, err, gain, objects, eps=0.0001):
    """Optimizes the Kron flux by iteration over different values.
    The stop condition is met when the change in flux is less that ``eps``.

    Parameters
    ----------
    data: ndarray
        Data of an image.
    err: float or ndarray
        Background error of the images.
    gain: float
        Gain value.
    objects: array
        Objects detected with :func:`sep.extract()`.
    eps: float, default ``0.0001`` (0.1%)
        Minimum percent change in flux allowed between iterations.

    Returns
    -------
    opt_flux: float
        Optimized Kron flux.
    opt_flux_err: float
        Optimized Kron flux error.
    opt_kronrad: float
        Optimized Kron radius.
    opt_scale: float
        Optimized scale of the Kron radius.
    """
    # iterate over kron radii
    for r in np.arange(1, 6.05, 0.05)[::-1]:
        kronrad, _ = sep.kron_radius(
            data,
            objects["x"],
            objects["y"],
            objects["a"],
            objects["b"],
            objects["theta"],
            r,
        )
        opt_kronrad = kronrad
        if ~np.isnan(opt_kronrad):
            break

    if np.isnan(opt_kronrad):
        raise ValueError(
            "The Kron radius cannot be calculated. The image might have NaNs or the aperture is too close to the edge."
        )

    opt_flux = 0.0
    # iterate over scale
    scales = np.arange(1, 10, 0.01)
    for scale in scales:
        flux, flux_err = kron_flux(
            data, err, gain, objects, opt_kronrad, scale
        )
        flux, flux_err = flux[0], flux_err[0]

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

    return opt_flux, opt_flux_err, opt_kronrad[0], opt_scale


def extract_kronparams(
    name,
    host_ra,
    host_dec,
    filt,
    survey,
    ra=None,
    dec=None,
    bkg_sub=False,
    threshold=10,
    use_mask=True,
    optimize_kronrad=True,
    eps=0.0001,
    gal_dist_thresh=-1,
    save_plots=True,
    save_aperture_params=True,
):
    """Calculates the aperture parameters for common aperture.

    Parameters
    ----------
    name: str
        Name of the object to find the path of the fits file.
    host_ra: float
        Host-galaxy right ascension of the galaxy in degrees.
    host_dec: float
        Host-galaxy declination of the galaxy in degrees.
    filt: str or list
        Filter to use to load the fits file. List is commonly used for coadds.
    survey: str
        Survey to use for the zero-points and pixel scale.
    ra: float, default ``None``
       Right ascension of an object, in degrees. Used for plotting the position of the object.
    dec: float, default ``None``
       Declination of an object, in degrees. Used for plotting the position of the object.
    bkg_sub: bool, default `None`
        If `True`, the image gets background subtracted. By default, only
        the images that need it get background subtracted (WISE, 2MASS and
        VISTA).
    threshold: float, default `10`
        Threshold used by `sep.extract()` to extract objects.
    use_mask: bool, default `True`
        If `True`, the masked fits files are used. These must have
        been created beforehand.
    optimize_kronrad: bool, default `True`
        If `True`, the Kron radius is optimized, increasing the
        aperture size until the flux does not increase.
    eps: float, default ``0.0001``
        The Kron radius is increased until the change in flux is lower than ``eps``.
        A value of 0.0001 means 0.01% change in flux.
    gal_dist_thresh: float, default ``-1``.
        Distance in arcsec to crossmatch the galaxy coordinates with a detected object,
        where the object nearest to the galaxy position is considered as the galaxy (within
        the given threshold). If no objects are found within the given distance threshold,
        the galaxy is considered as not found and a warning is printed. If a non-positive value
        is given, the threshold is considered as infinite, i.e. the closest detected object is
        considered as the galaxy (default option).
    save_plots: bool, default `True`
        If `True`, the mask and galaxy aperture figures are saved.
    save_aperture_params: bool, default `True`
        If `True`, the extracted mask parameters are saved into a pickle file.

    Returns
    -------
    gal_obj: array
        Galaxy object.
    img_wcs: WCS
        Image's WCS.
    kronrad: float
        Kron radius.
    scale: float
        Scale for the Kron radius.
    flip: bool
        Whether to flip the orientation of the
        aperture. Only used for DES images.
    """
    check_survey_validity(survey)
    check_work_dir(workdir)
    obj_dir = os.path.join(workdir, name)
    if use_mask:
        suffix = "masked_"
    else:
        suffix = ""
    if isinstance(filt, list):
        filt = "".join(f for f in filt)
    fits_file = os.path.join(obj_dir, f"{suffix}{survey}_{filt}.fits")

    img = fits.open(fits_file)
    img = remove_nan(img)

    header = img[0].header
    data = img[0].data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        img_wcs = wcs.WCS(header, naxis=2)

    data = data.astype(np.float64)
    bkg = sep.Background(data)
    bkg_rms = bkg.globalrms
    if (bkg_sub is None and survey in bkg_surveys) or bkg_sub is True:
        data_sub = np.copy(data - bkg)
    else:
        data_sub = np.copy(data)

    # extract galaxy
    gal_obj, _ = extract_objects(
        data_sub,
        bkg_rms,
        host_ra,
        host_dec,
        threshold,
        img_wcs,
        gal_dist_thresh,
    )
    if optimize_kronrad:
        gain = 1  # doesn't matter here
        opt_res = optimize_kron_flux(data_sub, bkg_rms, gain, gal_obj, eps)
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
        outfile = os.path.join(obj_dir, f"global_{survey}_{filt}.jpg")
        plot_detected_objects(
            data_sub, gal_obj, scale * kronrad, img_wcs, ra, dec, outfile
        )

    if survey == "DES":
        flip = True
    else:
        flip = False

    if save_aperture_params is True:
        outfile = os.path.join(
            obj_dir, f"{survey}_{filt}_aperture_parameters.pickle"
        )
        with open(outfile, "wb") as fp:
            aperture_parameters = gal_obj, img_wcs, kronrad, scale, flip
            pickle.dump(aperture_parameters, fp, protocol=4)

    return gal_obj, img_wcs, kronrad, scale, flip


def photometry(
    name,
    host_ra,
    host_dec,
    filt,
    survey,
    ra=None,
    dec=None,
    bkg_sub=None,
    threshold=10,
    use_mask=True,
    correct_extinction=True,
    aperture_params=None,
    optimize_kronrad=True,
    eps=0.0001,
    gal_dist_thresh=-1,
    save_plots=True,
):
    """Calculates the global aperture photometry of a galaxy using
    the Kron flux.

    **Note:** the galaxy must be ideally centred in the image.

    Parameters
    ----------
    name: str
        Name of the object to find the path of the fits file.
    host_ra: float
        Host-galaxy right ascension of the galaxy in degrees.
    host_dec: float
        Host-galaxy declination of the galaxy in degrees.
    filt: str
        Filter to use to load the fits file.
    survey: str
        Survey to use for the zero-points and pixel scale.
    ra: float, default ``None``
       Right ascension of an object, in degrees. Used for plotting the position of the object.
    dec: float, default ``None``
       Declination of an object, in degrees. Used for plotting the position of the object.
    bkg_sub: bool, default ``None``
        If ``True``, the image gets background subtracted. By default, only
        the images that need it get background subtracted (WISE, 2MASS and
        VISTA).
    threshold: float, default `10`
        Threshold used by `sep.extract()` to extract objects.
    use_mask: bool, default `True`
        If `True`, the masked fits files are used. These must have
        been created beforehand.
    correct_extinction: bool, default `True`
        If `True`, corrects for Milky-Way extinction using the recalibrated dust maps
        by Schlafly & Finkbeiner (2011) and the extinction law from Fitzpatrick (1999).
    aperture_params: tuple, default `None`
        Tuple with objects info and Kron parameters. Used for
        common aperture. If given, the Kron parameters are not
        re-calculated
    optimize_kronrad: bool, default `True`
        If `True`, the Kron radius is optimized, increasing the
        aperture size until the flux does not increase.
    eps: float, default ``0.0001``
        The Kron radius is increased until the change in flux is lower than ``eps``.
        A value of 0.0001 means 0.01% change in flux.
    gal_dist_thresh: float, default ``-1``.
        Distance in arcsec to crossmatch the galaxy coordinates with a detected object,
        where the object nearest to the galaxy position is considered as the galaxy (within
        the given threshold). If no objects are found within the given distance threshold,
        the galaxy is considered as not found and a warning is printed. If a non-positive value
        is given, the threshold is considered as infinite, i.e. the closest detected object is
        considered as the galaxy (default option).
    save_plots: bool, default `True`
        If `True`, the mask and galaxy aperture figures are saved.

    Returns
    -------
    mag: float
        Aperture magnitude.
    mag_err: float
        Error on the aperture magnitude.
    flux: float
        Aperture flux.
    total_flux_err: float
        Total flux error on the aperture flux.
    """
    check_survey_validity(survey)
    check_work_dir(workdir)
    obj_dir = os.path.join(workdir, name)
    if use_mask:
        suffix = "masked_"
    else:
        suffix = ""
    fits_file = os.path.join(obj_dir, f"{suffix}{survey}_{filt}.fits")

    img = fits.open(fits_file)
    img = remove_nan(img)

    header = img[0].header
    data = img[0].data
    gain = get_image_gain(header, survey)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        img_wcs = wcs.WCS(header, naxis=2)

    data = data.astype(np.float64)
    bkg = sep.Background(data)
    bkg_rms = bkg.globalrms
    if (bkg_sub is None and survey in bkg_surveys) or bkg_sub is True:
        data_sub = np.copy(data - bkg)
    else:
        data_sub = np.copy(data)

    if aperture_params is not None:
        gal_obj, master_img_wcs, kronrad, scale, flip2 = aperture_params

        if survey == "DES":
            flip = True
        else:
            flip = False
        if flip == flip2:
            flip_ = False
        else:
            flip_ = True

        a0 = gal_obj["a"]
        gal_obj = adapt_aperture(gal_obj, master_img_wcs, img_wcs, flip_)
        # factor used for scaling the Kron radius between different pixel scales
        conv_factor = gal_obj["a"] / a0

        flux, flux_err = kron_flux(
            data_sub, bkg_rms, gain, gal_obj, kronrad * conv_factor, scale
        )
        flux, flux_err = flux[0], flux_err[0]
    else:
        # extract galaxy
        gal_obj, _ = extract_objects(
            data_sub,
            bkg_rms,
            host_ra,
            host_dec,
            threshold,
            img_wcs,
            gal_dist_thresh,
        )

        # aperture photometry
        # This uses what would be the default SExtractor parameters.
        # See https://sep.readthedocs.io/en/v1.1.x/apertures.html
        if optimize_kronrad:
            opt_res = optimize_kron_flux(data_sub, bkg_rms, gain, gal_obj, eps)
            flux, flux_err, kronrad, scale = opt_res
        else:
            kronrad, _ = sep.kron_radius(
                data_sub,
                gal_obj["x"],
                gal_obj["y"],
                gal_obj["a"],
                gal_obj["b"],
                gal_obj["theta"],
                6.0,
            )
            scale = 2.5
            flux, flux_err = kron_flux(
                data_sub, bkg_rms, gain, gal_obj, kronrad, scale
            )
            flux, flux_err = flux[0], flux_err[0]

    ap_area = np.pi * gal_obj["a"][0] * gal_obj["b"][0]

    mag, mag_err, flux, flux_err = magnitude_calculation(
        flux,
        flux_err,
        survey,
        filt,
        ap_area,
        header,
        bkg_rms,
    )

    if correct_extinction is True:
        A_ext = calc_extinction(filt, survey, host_ra, host_dec)
        mag -= A_ext

    if save_plots is True:
        outfile = os.path.join(obj_dir, f"global_{survey}_{filt}.jpg")
        plot_detected_objects(
            data_sub, gal_obj, scale * kronrad, img_wcs, ra, dec, outfile
        )

    return mag, mag_err, flux, flux_err


def multi_band_phot(
    name,
    host_ra,
    host_dec,
    filters=None,
    survey="PS1",
    ra=None,
    dec=None,
    bkg_sub=None,
    threshold=10,
    use_mask=True,
    correct_extinction=True,
    aperture_params=None,
    common_aperture=True,
    coadd_filters="riz",
    optimize_kronrad=True,
    eps=0.0001,
    gal_dist_thresh=-1,
    save_plots=True,
    save_results=True,
    raise_exception=False,
):
    """Calculates multi-band aperture photometry of the host galaxy
    for an object.

    Parameters
    ----------
    name: str
        Name of the object to find the path of the fits file.
    host_ra: float
        Host-galaxy right ascension of the galaxy in degrees.
    host_dec: float
        Host-galaxy declination of the galaxy in degrees.
    filters: str, default, ``None``
        Filters to use to load the fits files. If `None` use all
        the filters of the given survey.
    survey: str, default ``PS1``
        Survey to use for the zero-points and pixel scale.
    ra: float, default ``None``
       Right ascension of an object, in degrees. Used for plotting the position of the object.
    dec: float, default ``None``
       Declination of an object, in degrees. Used for plotting the position of the object.
    bkg_sub: bool, default ``None``
        If ``True``, the image gets background subtracted. By default, only
        the images that need it get background subtracted (WISE, 2MASS and
        VISTA).
    threshold: float, default ``10``
        Threshold used by :func:`sep.extract()` to extract objects.
    use_mask: bool, default ``True``
        If ``True``, the masked fits files are used. These must have
        been created beforehand.
    correct_extinction: bool, default ``True``
        If ``True``, the magnitudes are corrected for extinction.
    aperture_params: tuple, default `None`
        Tuple with objects info and Kron parameters. Used for common aperture. If given,
        the Kron parameters are not re-calculated. If given, this supersedes the use of
        coadds for common aperture (``common_aperture`` parameter).
    common_aperture: bool, default ``True``
        If ``True``, use a coadd image for common aperture photometry. This is not used
        if ``aperture_params`` is given.
    coadd_filters: str, default ``riz``
        Filters of the coadd image. Used for common aperture photometry.
    optimize_kronrad: bool, default ``True``
        If ``True``, the Kron radius is optimized, increasing the
        aperture size until the flux does not increase.
    eps: float, default ``0.0001``
        The Kron radius is increased until the change in flux is lower than ``eps``.
        A value of 0.0001 means 0.01% change in flux.
        when optimizing the Kron radius.
    gal_dist_thresh: float, default ``-1``.
        Distance in arcsec to crossmatch the galaxy coordinates with a detected object,
        where the object nearest to the galaxy position is considered as the galaxy (within
        the given threshold). If no objects are found within the given distance threshold,
        the galaxy is considered as not found and a warning is printed. If a non-positive value
        is given, the threshold is considered as infinite, i.e. the closest detected object is
        considered as the galaxy (default option).
    save_plots: bool, default ``True``
        If ``True``, the mask and galaxy aperture figures are saved.
    save_results: bool, default ``True``
        If ``True``, the magnitudes are saved into a csv file.
    raise_exception: bool, default ``False``
        If ``True``, an exception is raised if the photometry fails for any filter.

    Returns
    -------
    results_dict: dict
        Dictionary with the object's photometry and other info.

    Examples
    --------
    >>> import hostphot.global_photometry as gp
    >>> name = 'SN2004eo'
    >>> host_ra, host_dec = 308.2092, 9.92755  # coords of host galaxy of SN2004eo
    >>> ra, dec =  308.22579, 9.92853 # coords of SN2004eo
    >>> results = gp.multi_band_phot(name, host_ra, host_dec,
                            survey=survey, ra=ra, dec=dec,
                            use_mask=True, common_aperture=True,
                            coadd_filters='riz', save_plots=True)
    """
    check_survey_validity(survey)
    if filters is None:
        if survey=='HST':
            raise ValueError("For HST, the filter needs to be specified!")
        filters = get_survey_filters(survey)
    else:
        check_filters_validity(filters, survey)
    if survey=='HST':
        filters = [filters]

    results_dict = {
        "name": name,
        "host_ra": host_ra,
        "host_dec": host_dec,
        "survey": survey,
    }

    if common_aperture is True and aperture_params is None:
        aperture_params = extract_kronparams(
            name,
            host_ra,
            host_dec,
            coadd_filters,
            survey,
            ra=ra,
            dec=dec,
            bkg_sub=bkg_sub,
            threshold=threshold,
            use_mask=use_mask,
            optimize_kronrad=optimize_kronrad,
            eps=eps,
            gal_dist_thresh=gal_dist_thresh,
            save_plots=save_plots,
        )

    for filt in filters:
        try:
            mag, mag_err, flux, flux_err = photometry(
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
                aperture_params=aperture_params,
                optimize_kronrad=optimize_kronrad,
                eps=eps,
                gal_dist_thresh=gal_dist_thresh,
                save_plots=save_plots,
            )
            results_dict[filt] = mag
            results_dict[f"{filt}_err"] = mag_err
            results_dict[f"{filt}_flux"] = flux
            results_dict[f"{filt}_flux_err"] = flux_err
        except Exception as exc:
            if raise_exception is True:
                raise Exception(f"{filt}-band: {exc}")
            else:
                results_dict[filt] = np.nan
                results_dict[f"{filt}_err"] = np.nan

    if save_results is True:
        outfile = os.path.join(workdir, name, f"{survey}_global.csv")
        phot_df = pd.DataFrame(
            {key: [val] for key, val in results_dict.items()}
        )
        phot_df.to_csv(outfile, index=False)

    return results_dict
