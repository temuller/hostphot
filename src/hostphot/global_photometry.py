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
import glob
import subprocess

import numpy as np
import pandas as pd

import sep
from astropy.io import fits
from astropy.modeling import models, fitting
from astropy import coordinates as coords, units as u, wcs

from photutils import (CircularAnnulus,
                       CircularAperture,
                       aperture_photometry)

import reproject
from reproject.mosaicking import reproject_and_coadd

from .utils import (get_survey_filters, extract_filters,
                    check_survey_validity, check_filters_validity,
                    calc_sky_unc, survey_zp)
from .objects_detect import (extract_objects, find_gaia_objects,
                             cross_match, plot_detected_objects)
from .image_masking import mask_image, plot_masked_image
from .image_cleaning import remove_nan
from .coadd import coadd_images
from .dust import calc_extinction

sep.set_sub_object_limit(1e4)

# Photometry
#-------------------------------
def kron_flux(data, err, objects, kronrad, scale):
    """Calculates the Kron flux.

    Parameters
    ----------
    data: 2D array
        Data of an image.
    err: float or 2D array
        Background error of the images.
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
                                          err=err, subpix=5, gain=1.0)

    return flux, flux_err

def optimize_kron_flux(data, err, objects, eps=0.001):
    """Optimizes the Kron flux by iteration over different values.
    The stop condition is met when the change in flux is less that `eps`.

    Parameters
    ----------
    data: 2D array
        Data of an image.
    err: float or 2D array
        Background error of the images.
    objects: array
        Objects detected with `sep.extract()`.
    eps: float, default `0.001`
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
    for r in np.arange(1, 6.1, 0.1)[::-1]:
        kronrad, krflag = sep.kron_radius(data, objects['x'], objects['y'],
                                      objects['a'], objects['b'],
                                      objects['theta'], r)
        if ~np.isnan(kronrad):
            opt_kronrad = kronrad
            break
        else:
            print(f'kronrad = {kronrad}')
            raise ValueError('The Kron radius cannot be calculated '
                             '(something went wrong!)')

    opt_flux = 0.0
    # iterate over scale
    scales = np.arange(1, 10, 0.1)
    for scale in scales:
        flux, flux_err = kron_flux(data, err, objects, opt_kronrad, scale)
        flux, flux_err = flux[0], flux_err[0]

        calc_eps = np.abs(opt_flux - flux)/flux
        if calc_eps<eps:
            opt_scale = scale
            opt_flux = flux
            opt_flux_err = flux_err
            break
        elif np.isnan(calc_eps):
            opt_scale = scale_list[-2]
            warnings.warn("Warning: the aperture might not fit in the image!")
            break
        else:
            opt_flux = flux
            opt_flux_err = flux_err

    return opt_flux, opt_flux_err, opt_kronrad, opt_scale

def quick_load(fits_file, bkg_sub):
    """Extracts the data and backrgound rms of an image.

    Parameters
    ----------
    fits_file: str
        Path to the fits file.
    bkg_sub: bool
        If `True`, the image gets background subtracted.

    Returns
    -------
    data_sub: 2D array
        Image data.
    bkg_rms: float
        RMS of the image's background.
    """
    img = fits.open(fits_file)
    img = remove_nan(img)

    data = img[0].data
    data = data.astype(np.float64)
    bkg = sep.Background(data)
    bkg_rms = bkg.globalrms
    if bkg_sub:
        data_sub = np.copy(data - bkg)
    else:
        data_sub = np.copy(data)

    return data_sub, bkg_rms

def photometry(fits_file, host_ra, host_dec, bkg_sub=False, threshold=7,
               mask_stars=True, coadd_file=None, optimze_kronrad=True,
               filt=None, survey='PS1', correct_extinction=False,
               save_plots=False, plots_path=''):
    """Calculates the global aperture photometry of a galaxy using Kron flux.

    **Note:** the galaxy must be ideally centred in the image.

    Parameters
    ----------
    fits_file: str
        Path to the fits file.
    host_ra: float
        Host-galaxy Right ascension of the galaxy in degrees.
    host_dec: float
        Host-galaxy Declination of the galaxy in degrees.
    bkg_sub: bool, default `False`
        If `True`, the image gets background subtracted.
    threshold: float, default `7`
        Threshold used by `sep.extract()` to extract objects.
    mask_stars: bool, default `True`
        If `True`, the stars identified are masked by using
        a convolution with a 2D Gaussian kernel.
    coadd_file: 'str', default `None`
        Path to the coadd image for common aperture photometry.
        If none is given, it is not used.
    optimze_kronrad: bool, default `True`
        If `True`, the Kron radius is optimized, increasing the
        aperture size until the flux does not increase.
    filt: str, default `None`
        Filter to use for extinction correction and saving outputs.
    survey: str, default `PS1`
        Survey to use for the zero-points and correct filter path.
    correct_extinction: bool, default `False`
        If `True`, the magnitude is corrected for Milky-Way extinction.
    save_plots: bool, default `False`
        If `True`, the mask and galaxy aperture figures are saved.
    plots_path: str, default `''`
        Path where to save the output plots. By default uses the current
        directory.

    Returns
    -------
    mag: float
        Aperture magnitude.
    mag_err: float
        Error on the aperture magnitude.
    """
    check_survey_validity(survey)

    img = fits.open(fits_file)
    img = remove_nan(img)

    header = img[0].header
    data = img[0].data
    exptime = float(header['EXPTIME'])
    img_wcs = wcs.WCS(header, naxis=2)

    data = data.astype(np.float64)
    bkg = sep.Background(data)
    bkg_rms = bkg.globalrms
    if bkg_sub:
        data_sub = np.copy(data - bkg)
    else:
        data_sub = np.copy(data)

    # extract objects and cross-match with gaia
    gal_obj, nogal_objs = extract_objects(data_sub, bkg_rms,
                                          host_ra, host_dec,
                                          threshold, img_wcs)
    gaia_coord, pix_coords = find_gaia_objects(host_ra, host_dec,
                                                img_wcs)
    nogal_objs = cross_match(nogal_objs, img_wcs, gaia_coord)

    # preprocessing
    if mask_stars:
        masked_data = mask_image(data_sub, nogal_objs)
        if save_plots:
            outfile = os.path.join(plots_path, f'mask_{filt}.jpg')
            plot_masked_image(data_sub, masked_data,
                                nogal_objs, outfile)
    else:
        masked_data = data_sub.copy()

    if coadd_file:
        # use the coadd image for the galaxy parameters
        coadd_data, coadd_err = quick_load(coadd_file, bkg_sub)
        gal_obj, _ = extract_objects(coadd_data, coadd_err,
                                      host_ra, host_dec,
                                      threshold, img_wcs)

    # aperture photometry
    # This uses what would be the default SExtractor parameters.
    # See https://sep.readthedocs.io/en/v1.1.x/apertures.html
    if optimze_kronrad:
        opt_res = optimize_kron_flux(masked_data, bkg_rms, gal_obj)
        flux, flux_err, kronrad, scale = opt_res
    else:
        kronrad, krflag = sep.kron_radius(masked_data,
                                          gal_obj['x'], gal_obj['y'],
                                          gal_obj['a'], gal_obj['b'],
                                          gal_obj['theta'], 6.0)
        scale = 2.5
        flux, flux_err =  kron_flux(masked_data, bkg_rms,
                                    gal_obj, kronrad, scale)
        flux, flux_err = flux[0], flux_err[0]

    zp = survey_zp(survey)
    if survey=='PS1':
        zp += 2.5*np.log10(exptime)

    mag = -2.5*np.log10(flux) + zp
    mag_err = 2.5/np.log(10)*flux_err/flux

    if save_plots:
        outfile = os.path.join(plots_path, f'gal_{filt}.jpg')
        plot_detected_objects(masked_data, gal_obj,
                                scale*kronrad, outfile)

    if correct_extinction:
        if filt is None:
            raise ValueError('A filter must be given to calculate '
                             'the extinction.')
        A_ext = calc_extinction(filt, survey, host_ra, host_dec)
        mag -= A_ext

    return mag, mag_err

def multi_band_phot(name, host_ra, host_dec, bkg_sub=False, threshold=7,
                   mask_stars=True, coadd_filters=None, optimze_kronrad=True,
                   filters=None, survey='PS1', correct_extinction=False,
                   work_dir='', save_plots=False):
    """Calculates multi-band photometry of the host galaxy for an object.

    Parameters
    ----------
    name: str
        Obejct's name. This is used for the directory path.
    host_ra: float
        Host-galaxy Right ascension of the galaxy in degrees.
    host_dec: float
        Host-galaxy Declination of the galaxy in degrees.
    bkg_sub: bool, default `False`
        If `True`, the image gets background subtracted.
    threshold: float, default `7`
        Threshold used by `sep.extract()` to extract objects.
    mask_stars: bool, default `True`
        If `True`, the stars identified are masked by using
        a convolution with a 2D Gaussian kernel.
    coadd_filters: 'str', default `None`
        Filters to use for the coadd image. E.g. `riz`.
    optimze_kronrad: bool, default `True`
        If `True`, the Kron radius is optimized, increasing the
        aperture size until the flux does not increase.
    filters: str, default `None`
        Filter to use for for the photometry.
    survey: str, default `PS1`
        Survey to use for the zero-points and correct filter path.
    correct_extinction: bool, default `False`
        If `True`, the magnitude is corrected for Milky-Way extinction.
    work_dir: str, default `''`
        Working directory where to find the objects.
    save_plots: bool, default `False`
        If `True`, the mask and galaxy aperture figures are saved.

    Returns
    -------
    results_dict: dict
        Dictionary with the results: name, host_ra, host_dec,
        filter magnitudes and errors
    """
    check_survey_validity(survey)
    if filters is None:
        filters = get_survey_filters(survey)
    else:
        check_filters_validity(filters, survey)

    # dictionary to save the results (mag + err)
    results_dict = {'name':name, 'host_ra':host_ra, 'host_dec':host_dec}

    if coadd_filters:
        coadd_file = coadd_images(name, coadd_filters, work_dir, survey)
    else:
        coadd_file = None

    obj_dir = os.path.join(work_dir, name)
    fits_files = [os.path.join(obj_dir, f'{survey}_{filt}.fits')
                                                for filt in filters]

    for fits_file, filt in zip(fits_files, filters):
        mag, mag_err = photometry(fits_file, host_ra, host_dec, bkg_sub, threshold,
                                   mask_stars, coadd_file, optimze_kronrad, filt,
                                   survey, correct_extinction, save_plots, obj_dir)
        results_dict.update({filt:mag, f'{filt}_err':mag_err})

    return results_dict

def multi_global_photometry(name_list, host_ra_list, host_dec_list, work_dir='',
                            filters=None, coadd=True, coadd_filters='riz',
                            mask_stars=True, threshold=3, bkg_sub=True, survey="PS1",
                            correct_extinction=True, plot_output=False):
    """Extract global photometry for multiple SNe.

    Parameters
    ==========
    name_list: list-like
        List of SN names.
    host_ra_list: list-like
        List of host-galaxy right ascensions in degrees.
    host_dec_list: list-like
        List of host-galaxy declinations in degrees.
    work_dir: str, default ''
        Working directory where to find the objects'
        directories with the images. Default, current directory.
    filters: str, defaul `None`
        Filters used to extract photometry. If `None`, use all
        the available filters for the given survey.
    coadd: bool, default `True`
        If `True`, a coadd image is created for common aperture.
    coadd_filters: str, default `riz`
        Filters to use for the coadd image.
    mask_stars: bool, default `True`
        If `True`, the stars identified inside the common aperture
        are masked with the mean value of the background around them.
    threshold: float, default `3`
        Threshold used by `sep.extract()` to extract objects.
    bkg_sub: bool, default `True`
        If `True`, the image gets background subtracted.
    survey: str, default `PS1`
        Survey to use for the zero-points.
    correct_extinction: bool, default `True`
        If `True`, the magnitudes are corrected for extinction.
    plot_output: bool, default `False`
        If `True`, saves the output plots.

    Returns
    =======
    global_phot_df: DataFrame
        Dataframe with the photometry, errors and SN name.
    """
    check_survey_validity(survey)
    check_filters_validity(filters, survey)
    if filters is None:
        filters = get_survey_filters(survey)

    # dictionary to save results
    mag_dict = {filt:[] for filt in filters}
    mag_err_dict = {filt+'_err':[] for filt in filters}
    mag_dict.update(mag_err_dict)

    results_dict = {'name':[], 'host_ra':[], 'host_dec':[]}
    results_dict.update(mag_dict)

    # filter functions for extinction correction
    filters_dict = extract_filters(filters, survey)

    for name, host_ra, host_dec in zip(name_list, host_ra_list,
                                                      host_dec_list):

        sn_dir = os.path.join(work_dir, name)
        image_files = [os.path.join(sn_dir, f'{survey}_{filt}.fits')
                                                    for filt in filters]

        if coadd:
            coadd_images(name, coadd_filters, work_dir, survey)
            coadd_file = os.path.join(sn_dir, f'{survey}_{coadd_filters}.fits')
            gal_object, _ = extract_aperture_params(coadd_file,
                                                    host_ra, host_dec,
                                                    threshold, bkg_sub)
        else:
            gal_object = None

        for image_file, filt in zip(image_files, filters):
            try:
                if plot_output:
                    plot_output = os.path.join(sn_dir, f'global_{filt}.jpg')
                else:
                    plot_output = None
                mag, mag_err = extract_global_photometry(image_file,
                                                         host_ra,
                                                         host_dec,
                                                         gal_object,
                                                         mask_stars,
                                                         threshold,
                                                         bkg_sub,
                                                         survey,
                                                         plot_output)
                if correct_extinction:
                    wave = filters_dict[filt]['wave']
                    transmission = filters_dict[filt]['transmission']
                    A_ext = calc_ext(wave, transmission,
                                        host_ra, host_dec)
                    mag -= A_ext
                results_dict[filt].append(mag)
                results_dict[filt+'_err'].append(mag_err)
            except Exception as message:
                results_dict[filt].append(np.nan)
                results_dict[filt+'_err'].append(np.nan)
                print(f'{name} failed with {filt} band: {message}')
        results_dict['name'].append(name)
        results_dict['host_ra'].append(host_ra)
        results_dict['host_dec'].append(host_dec)

    global_phot_df = pd.DataFrame(results_dict)

    return global_phot_df
