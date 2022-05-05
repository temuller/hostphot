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
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import sep
from astropy.io import fits
from astropy.modeling import models, fitting
from astropy import coordinates as coords, units as u, wcs

from photutils import (CircularAnnulus,
                       CircularAperture,
                       aperture_photometry)
from photutils.detection import DAOStarFinder

import reproject
from reproject.mosaicking import reproject_and_coadd

from .utils import (get_survey_filters, extract_filters,
                    check_survey_validity, check_filters_validity,
                    calc_ext, calc_sky_unc)
from .objects_detect import (extract_objects, find_gaia_objects,
                             plot_detected_objects)
from .image_masking import mask_image, plot_masked_image

sep.set_sub_object_limit(1e4)


# Coadding images
#-------------------------------
def coadd_images(sn_name, filters='riz', work_dir='', survey='PS1'):
    """Reprojects and coadds images for the choosen filters for
    common-aperture photometry.

    Parameters
    ----------
    sn_name: str
        SN name to be used for finding the images locally.
    filters: str, default `riz`
        Filters to use for the coadd image.
    work_dir: str, default ''
        Working directory where to find the objects'
        directories with the images. Default, current directory.
    survey: str, default `PS1`
        Survey to use as prefix for the images.

    Returns
    -------
    A fits file with the coadded images is created with the filters
    used as the name of the file at the SN directory.
    """

    init_dir = os.path.abspath('.')
    sn_dir = os.path.join(work_dir, sn_name)
    fits_files = [os.path.join(sn_dir,
                               f'{survey}_{filt}.fits') for filt in filters]

    hdu_list = []
    for fits_file in fits_files:
        fits_image = fits.open(fits_file)
        hdu_list.append(fits_image[0])

    hdu_list = fits.HDUList(hdu_list)
    # use the last image as reference
    coadd = reproject_and_coadd(hdu_list, fits_image[0].header,
                                reproject_function=reproject.reproject_interp)
    fits_image[0].data = coadd[0]
    outfile = os.path.join(sn_dir, f'{survey}_{filters}.fits')
    fits_image.writeto(outfile, overwrite=True)

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
    for r in np.arange(2, 6.1, 0.1)[::-1]:
        kronrad, krflag = sep.kron_radius(data, objects['x'], objects['y'],
                                      objects['a'], objects['b'],
                                      objects['theta'], r)
        if ~np.isnan(kronrad):
            opt_kronrad = kronrad
            break

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

def calculate_global_photometry(fits_file, host_ra, host_dec, gal_object=None,
                              mask_stars=True, threshold=3, bkg_sub=True,
                              survey='PS1', plot_output=None):
    """Calculates the global photometry of a galaxy. The use
    of `gal_object` is intended for common-aperture photometry.

    **Note:** the galaxy must be ideally centred in the image.

    Parameters
    ==========
    fits_file: str
        Path to the fits file.
    host_ra: float
        Host-galaxy Right ascension of the galaxy in degrees.
    host_dec: float
        Host-galaxy Declination of the galaxy in degrees.
    gal_object: numpy array, default `None`
        Galaxy object extracted with `extract_aperture_params()`.
        Use this for common-aperture photometry only.
    mask_stars: bool, default `True`
        If `True`, the stars identified inside the common aperture
        are masked with the mean value of the background around them.
    threshold: float, default `3`
        Threshold used by `sep.extract()` to extract objects.
    bkg_sub: bool, default `True`
        If `True`, the image gets background subtracted.
    survey: str, default `PS1`
        Survey to use for the zero-points.
    plot_output: str, default `None`
        If not `None`, saves the output plots with the given name.

    Returns
    =======
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

    data = data.astype(np.float64)
    bkg = sep.Background(data)
    bkg_rms = bkg.globalrms
    if bkg_sub:
        data_sub = np.copy(data - bkg)
    else:
        data_sub = np.copy(data)

    if gal_object is None:
        # no common aperture
        gal_object, objects = extract_aperture_params(fits_file,
                                                      host_ra,
                                                      host_dec,
                                                      threshold,
                                                      bkg_sub)
    else:
        gal_object2, objects = extract_aperture_params(fits_file,
                                             host_ra,
                                             host_dec,
                                             threshold,
                                             bkg_sub)
        # sometimes, one of the filter images can be flipped, so the position of the
        # aperture from the coadd image might not match that of the filter image. This
        # is a workaround as we only need the semi-major and -minor axes from the coadd.
        gal_object['x'] = gal_object2['x']
        gal_object['y'] = gal_object2['y']
        gal_object['theta'] = gal_object2['theta']  # this might be slightly different

    if mask_stars:
        masked_data = mask_image(data_sub, objects)
        if plot_output:
            plot_masked_image(data_sub, masked_data, objects)
    else:
        masked_data = data_sub.copy()

    # aperture photometry
    # This uses what would be the default SExtractor parameters.
    # See https://sep.readthedocs.io/en/v1.1.x/apertures.html
    if optimze_kronrad:
        opt_res = optimize_kron_flux(masked_data, bkg_rms, gal_object)
        flux, flux_err, kronrad, scale = opt_res
    else:
        kronrad, krflag = sep.kron_radius(masked_data,
                                          gal_object['x'], gal_object['y'],
                                          gal_object['a'], gal_object['b'],
                                          gal_object['theta'], 6.0)
        scale = 2.5
        flux, flux_err =  kron_flux(masked_data, bkg_rms, gal_object, kronrad, scale)
        flux, flux_err = flux[0], flux_err[0]

    zp = survey_zp(survey)
    if survey=='PS1':
        zp += 2.5*np.log10(exptime)

    mag = -2.5*np.log10(flux) + zp
    mag_err = 2.5/np.log(10)*flux_err/flux

    if plot_output:
        plot_detected_objects(data, objects, scale=6)

    return mag, mag_err

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

    # filter funcstions for extinction correction
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
