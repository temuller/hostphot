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
import inspect
import numpy as np
import pandas as pd
import multiprocessing as mp

import sep
from astropy.io import fits
from astropy import wcs

from hostphot._constants import __workdir__
from hostphot.utils import (get_survey_filters, check_survey_validity,
                            check_filters_validity, survey_zp, get_image_gain,
                            get_image_readnoise, pixel2pixel, check_work_dir)
from hostphot.objects_detect import extract_objects,  plot_detected_objects
from hostphot.image_cleaning import remove_nan
from hostphot.dust import calc_extinction

import warnings

sep.set_sub_object_limit(1e4)

#----------------------------------------
def _choose_workdir(workdir):
    """Updates the work directory.

    Parameters
    ----------
    workdir: str
        Path to the work directory.
    """
    global __workdir__
    __workdir__ = workdir

#-------------------------------
def kron_flux(data, err, gain, objects, kronrad, scale):
    """Calculates the Kron flux.

    Parameters
    ----------
    data: 2D array
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

    if kronrad*np.sqrt(objects['a']*objects['b']) < r_min:
        print(f'Warning: using circular photometry')
        flux, flux_err, flag = sep.sum_circle(data, objects['x'], objects['y'],
                                              r_min, err=err, subpix=5, gain=1.0)
    else:
        flux, flux_err, flag = sep.sum_ellipse(data, objects['x'], objects['y'],
                                          objects['a'], objects['b'],
                                          objects['theta'], scale*kronrad,
                                          err=err, subpix=5, gain=gain)

    return flux, flux_err

def optimize_kron_flux(data, err, gain, objects, eps=0.001):
    """Optimizes the Kron flux by iteration over different values.
    The stop condition is met when the change in flux is less that ``eps``.

    Parameters
    ----------
    data: 2D array
        Data of an image.
    err: float or 2D array
        Background error of the images.
    gain: float
        Gain value.
    objects: array
        Objects detected with :func:`sep.extract()`.
    eps: float, default ``0.001``
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
        kronrad, krflag = sep.kron_radius(data, objects['x'], objects['y'],
                                      objects['a'], objects['b'],
                                      objects['theta'], r)
        if ~np.isnan(kronrad):
            opt_kronrad = kronrad
            break
    if np.isnan(kronrad):
        print(f'kronrad = {kronrad}')
        raise ValueError('The Kron radius cannot be calculated '
                         '(something went wrong!)')

    opt_flux = 0.0
    # iterate over scale
    scales = np.arange(1, 10, 0.01)
    for scale in scales:
        flux, flux_err = kron_flux(data, err, gain, objects,
                                    opt_kronrad, scale)
        flux, flux_err = flux[0], flux_err[0]

        calc_eps = np.abs(opt_flux - flux)/flux
        if calc_eps<eps:
            opt_scale = scale
            opt_flux = flux
            opt_flux_err = flux_err
            break
        elif np.isnan(calc_eps):
            opt_scale = scales[-2]
            warnings.warn("Warning: the aperture might not fit in the image!")
            break
        else:
            opt_flux = flux
            opt_flux_err = flux_err

    return opt_flux, opt_flux_err, opt_kronrad, opt_scale

def extract_kronparams(name, host_ra, host_dec, filt, survey, bkg_sub=False,
                       threshold=10, use_mask=True, optimze_kronrad=True,
                       eps=0.001, save_plots=True):
    """Calculates the aperture parameters for common aperture.

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
    bkg_sub: bool, default `False`
        If `True`, the image gets background subtracted.
    threshold: float, default `10`
        Threshold used by `sep.extract()` to extract objects.
    use_mask: bool, default `True`
        If `True`, the masked fits files are used. These must have
        been created beforehand.
    optimze_kronrad: bool, default `True`
        If `True`, the Kron radius is optimized, increasing the
        aperture size until the flux does not increase.
    eps: float, default `0.001`
        Minimum percent change in flux allowed between iterations
        when optimizing the Kron radius.
    save_plots: bool, default `False`
        If `True`, the mask and galaxy aperture figures are saved.

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
    """
    check_survey_validity(survey)
    check_work_dir(__workdir__)
    obj_dir = os.path.join(__workdir__, name)
    if use_mask:
        suffix = 'masked_'
    else:
        suffix = ''
    fits_file = os.path.join(obj_dir, f'{suffix}{survey}_{filt}.fits')

    img = fits.open(fits_file)
    img = remove_nan(img)

    header = img[0].header
    data = img[0].data
    img_wcs = wcs.WCS(header, naxis=2)

    data = data.astype(np.float64)
    bkg = sep.Background(data)
    bkg_rms = bkg.globalrms
    if bkg_sub:
        data_sub = np.copy(data - bkg)
    else:
        data_sub = np.copy(data)

    # extract objects
    gal_obj, nogal_objs = extract_objects(data_sub, bkg_rms,
                                          host_ra, host_dec,
                                          threshold, img_wcs)
    if optimze_kronrad:
        gain = 1  # doesn't matter here
        opt_res = optimize_kron_flux(data_sub, bkg_rms,
                                     gain, gal_obj, eps)
        flux, flux_err, kronrad, scale = opt_res
    else:
        scale = 2.5
        kronrad, krflag = sep.kron_radius(data_sub,
                                          gal_obj['x'], gal_obj['y'],
                                          gal_obj['a'], gal_obj['b'],
                                          gal_obj['theta'], 6.0)

    if save_plots:
        outfile = os.path.join(obj_dir, f'global_{survey}_{filt}.jpg')
        plot_detected_objects(data_sub, gal_obj,
                                scale*kronrad, outfile)

    return gal_obj, img_wcs, kronrad, scale


def photometry(name, host_ra, host_dec, filt, survey, bkg_sub=False,
               threshold=10, use_mask=True, aperture_params=None,
               optimze_kronrad=True, eps=0.001, save_plots=True):
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
    bkg_sub: bool, default `False`
        If `True`, the image gets background subtracted.
    threshold: float, default `10`
        Threshold used by `sep.extract()` to extract objects.
    use_mask: bool, default `True`
        If `True`, the masked fits files are used. These must have
        been created beforehand.
    aperture_params: tuple, default `None`
        Tuple with objects info and Kron parameters. Used for
        common aperture. If given, the Kron parameters are not
        re-calculated
    optimze_kronrad: bool, default `True`
        If `True`, the Kron radius is optimized, increasing the
        aperture size until the flux does not increase.
    eps: float, default `0.001`
        Minimum percent change in flux allowed between iterations
        when optimizing the Kron radius.
    save_plots: bool, default `True`
        If `True`, the mask and galaxy aperture figures are saved.

    Returns
    -------
    mag: float
        Aperture magnitude.
    mag_err: float
        Error on the aperture magnitude.
    """
    check_survey_validity(survey)
    check_work_dir(__workdir__)
    obj_dir = os.path.join(__workdir__, name)
    if use_mask:
        suffix = 'masked_'
    else:
        suffix = ''
    fits_file = os.path.join(obj_dir, f'{suffix}{survey}_{filt}.fits')

    img = fits.open(fits_file)
    img = remove_nan(img)

    header = img[0].header
    data = img[0].data
    exptime = float(header['EXPTIME'])
    gain = get_image_gain(header, survey)
    readnoise = get_image_readnoise(header, survey)
    img_wcs = wcs.WCS(header, naxis=2)

    data = data.astype(np.float64)
    bkg = sep.Background(data)
    bkg_rms = bkg.globalrms
    if bkg_sub:
        data_sub = np.copy(data - bkg)
    else:
        data_sub = np.copy(data)

    if aperture_params is not None:
        gal_obj, img_wcs0, kronrad, scale = aperture_params

        gal_obj['x'], gal_obj['y'] = pixel2pixel(gal_obj['x'],
                                                gal_obj['y'],
                                                img_wcs0, img_wcs)

        flux, flux_err =  kron_flux(data_sub, bkg_rms, gain,
                                    gal_obj, kronrad, scale)
        flux, flux_err = flux[0], flux_err[0]
    else:
        # extract objects
        gal_obj, nogal_objs = extract_objects(data_sub, bkg_rms,
                                              host_ra, host_dec,
                                              threshold, img_wcs)

        # aperture photometry
        # This uses what would be the default SExtractor parameters.
        # See https://sep.readthedocs.io/en/v1.1.x/apertures.html
        if optimze_kronrad:
            opt_res = optimize_kron_flux(data_sub, bkg_rms,
                                         gain, gal_obj, eps)
            flux, flux_err, kronrad, scale = opt_res
        else:
            kronrad, krflag = sep.kron_radius(data_sub,
                                              gal_obj['x'], gal_obj['y'],
                                              gal_obj['a'], gal_obj['b'],
                                              gal_obj['theta'], 6.0)
            scale = 2.5
            flux, flux_err =  kron_flux(data_sub, bkg_rms, gain,
                                        gal_obj, kronrad, scale)
            flux, flux_err = flux[0], flux_err[0]

    zp = survey_zp(survey)
    if survey=='PS1':
        zp += 2.5*np.log10(exptime)

    mag = -2.5*np.log10(flux) + zp
    mag_err = 2.5/np.log(10)*flux_err/flux

    # correct extinction
    A_ext = calc_extinction(filt, survey, host_ra, host_dec)
    mag -= A_ext

    # error budget
    # 1.0857 = 2.5/ln(10)
    if survey!='SDSS':
        # ellipse area = pi*a*b
        ap_area = np.pi*gal_obj['a'][0]*gal_obj['b'][0]
        extra_err = 1.0857*np.sqrt(ap_area*(readnoise**2)+flux/gain)/flux
        mag_err = np.sqrt(mag_err**2 + extra_err**2)

    if save_plots:
        outfile = os.path.join(obj_dir, f'global_{survey}_{filt}.jpg')
        plot_detected_objects(data_sub, gal_obj,
                                scale*kronrad, outfile)

    return mag, mag_err

def multi_band_phot(name, host_ra, host_dec, filters=None, survey='PS1',
                    bkg_sub=False, threshold=10, use_mask=True,
                    common_aperture=True, coadd_filters='riz',
                    optimze_kronrad=True, eps=0.001, save_plots=True):
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
    bkg_sub: bool, default ``False``
        If `True`, the image gets background subtracted.
    threshold: float, default ``10``
        Threshold used by :func:`sep.extract()` to extract objects.
    use_mask: bool, default ``True``
        If ``True``, the masked fits files are used. These must have
        been created beforehand.
    common_aperture: bool, default ``True``
        If ``True``, use a coadd image for common aperture photometry.
    coadd_filters: str, default ``riz``
        Filters of the coadd image. Used for common aperture photometry.
    optimze_kronrad: bool, default ``True``
        If ``True``, the Kron radius is optimized, increasing the
        aperture size until the flux does not increase.
    eps: float, default ``0.001``
        Minimum percent change in flux allowed between iterations
        when optimizing the Kron radius.
    save_plots: bool, default ``True``
        If ``True``, the mask and galaxy aperture figures are saved.

    Returns
    -------
    results_dict: dict
        Dictionary with the object's photometry and other info.
    """
    check_survey_validity(survey)
    if filters is None:
        filters = get_survey_filters(survey)
    else:
        check_filters_validity(filters, survey)

    results_dict = {'name':name,
                    'host_ra':host_ra, 'host_dec':host_dec,
                    'survey':survey}

    if common_aperture:
        aperture_params = extract_kronparams(name, host_ra, host_dec,
                                            coadd_filters, survey, bkg_sub,
                                            threshold, use_mask,
                                            optimze_kronrad, eps, save_plots)
    else:
        aperture_params = None

    for filt in filters:
        mag, mag_err = photometry(name, host_ra, host_dec, filt, survey, bkg_sub,
                                  threshold, use_mask, aperture_params,
                                  optimze_kronrad, eps, save_plots)
        results_dict[filt] = mag
        results_dict[f'{filt}_err'] = mag_err

    return results_dict
