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
import matplotlib.pyplot as plt
import multiprocessing as mp

import sep
from photutils import CircularAperture
from photutils import aperture_photometry

from astropy.io import fits
from astropy.table import Table
from astropy import coordinates, units as u, wcs
from astropy.cosmology import FlatLambdaCDM
from astropy.stats import sigma_clipped_stats

from .utils import (get_survey_filters, extract_filters,
                    check_survey_validity, check_filters_validity,
                     calc_sky_unc, survey_pixel_scale, survey_zp,
                     get_image_gain, get_image_readnoise, pixel2pixel)
from .objects_detect import (extract_objects, find_gaia_objects,
                             cross_match, plot_detected_objects)
from .image_masking import mask_image, plot_masked_image
from .image_cleaning import remove_nan
from .coadd import coadd_images
from .dust import calc_extinction

H0 = 70
Om0 = 0.3
__cosmo__ = FlatLambdaCDM(H0, Om0)
sep.set_sub_object_limit(1e4)
#-------------------------------
def choose_cosmology(cosmo):
    """Updates the cosmology used to calculate the aperture size.

    Parameters
    ----------
    cosmo: `astropy.cosmology` object
        Cosmological model. E.g. `FlatLambdaCDM(70, 0.3)`.
    """
    global __cosmo__
    __cosmo__ = cosmo

#-------------------------------
def calc_aperture_size(z, ap_radius):
    """Calculates the size of the aperture in arsec,
    for aperture photometry, given a physical size.

    Parameters
    ----------
    z: float
        Redshift.
    ap_radius: float
        Physical aperture size in kpc.

    Returns
    -------
    radius_arcsec: float
        Aperture size in arcsec.
    """
    ap_radius = ap_radius*u.kpc

    # transverse separations
    transv_sep_per_arcmin = __cosmo__.kpc_proper_per_arcmin(z)
    transv_sep_per_arcsec = transv_sep_per_arcmin.to(u.kpc/u.arcsec)

    radius_arcsec = ap_radius/transv_sep_per_arcsec

    return radius_arcsec.value

def extract_aperture_flux(data, error, px, py, radius):
    """Extracts aperture photometry of a single image.

    Parameters
    ----------
    data: array
        Image data in a 2D numpy array.
    error: array
        Errors of `data`.
    px: float
        x-axis pixel coordinate of the aperture center.
    py: float
        y-axis pixel coordinate of the aperture center.
    radius: float
        Aperture radius in pixels.

    Returns
    -------
    raw_flux: float
        Aperture photometry ("raw" flux).
    raw_flux_err: float
        Uncertainty on the aperture photometry.
    """
    aperture = CircularAperture((px, py), r=radius)
    ap_results = aperture_photometry(data, aperture,
                                         error=error)
    raw_flux = ap_results['aperture_sum'][0]
    raw_flux_err = ap_results['aperture_sum_err'][0]

    return raw_flux, raw_flux_err

def plot_aperture(data, px, py, radius_pix, outfile=None):
    """Plots the aperture for the given parameters.

    Parameters
    ----------
    data: 2D array
        Data of an image.
    px: float
        X-axis center of the aperture in pixels.
    py: float
        Y-axis center of the aperture in pixels.
    radius_pix: float
        Aperture radius in pixels.
    outfile: str, default `None`
        If given, path where to save the output figure.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    m, s = np.nanmean(data), np.nanstd(data)
    im = ax.imshow(data, interpolation='nearest',
                   cmap='gray',
                   vmin=m-s, vmax=m+s,
                   origin='lower')

    circle = plt.Circle((px, py), radius_pix, color='r', fill=False)
    ax.add_patch(circle)

    if outfile:
        plt.tight_layout()
        plt.savefig(outfile)
        plt.close(fig)
    else:
        plt.show()

def common_aperture(fits_file, host_ra, host_dec, bkg_sub=False, threshold=7,
                   mask_stars=True,filt=None, survey=None, save_plots=False,
                   plots_path=''):
    """Calculates the aperture parameters for common aperture.

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
    filt: str, default `None`
        Filter to use for extinction correction and saving outputs.
    survey: str, default `None`
        Survey to use for the zero-points and correct filter path.
    save_plots: bool, default `False`
        If `True`, the mask and galaxy aperture figures are saved.
    plots_path: str, default `''`
        Path where to save the output plots. By default uses the current
        directory.

    Returns
    -------
    gal_obj: array
        Galaxy object.
    gal_obj: array
        Non-galaxy objects.
    img_wcs: WCS
        Image's WCS.
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

    # extract objects
    gal_obj, nogal_objs = extract_objects(data_sub, bkg_rms,
                                          host_ra, host_dec,
                                          threshold, img_wcs)

    # preprocessing
    # cross-match extracted objects with gaia
    gaia_coord = find_gaia_objects(host_ra, host_dec, img_wcs)
    nogal_objs = cross_match(nogal_objs, img_wcs, gaia_coord)
    masked_data = mask_image(data_sub, nogal_objs)

    if save_plots:
        outfile = os.path.join(plots_path,
                                f'local_mask_{survey}_{filt}.jpg')
        plot_masked_image(data_sub, masked_data,
                            nogal_objs, outfile)

        outfile = os.path.join(plots_path, f'local_{survey}_{filt}.jpg')
        plot_detected_objects(masked_data, gal_obj,
                                6, outfile)

    return gal_obj, nogal_objs, img_wcs

def photometry(fits_file, ra, dec, z, ap_radius, host_ra=None, host_dec=None,
                threshold=7, bkg_sub=False, mask_stars=True, aperture_params=None,
                filt=None, survey=None, correct_extinction=True, save_plots=False,
                plots_path=''):
    """Calculates the local aperture photometry in a given radius.

    Parameters
    ----------
    fits_file: str
        Path to the fits file.
    ra: float
        Right Ascensions in degrees.
    dec: float
        Declinations in degrees.
    z: float
        Redshift of the SN.
    ap_radius: float
        Physical size of the aperture in kpc. This is used
        for aperture photometry.
    host_ra: float, default `None`
        Host-galaxy Right ascension of the galaxy in degrees.
        Used for masking objects in the image.
    host_dec: float, default `None`
        Host-galaxy Declination of the galaxy in degrees.
        Used for masking objects in the image.
    threshold: float, default `7`
        Threshold used by `sep.extract()` to extract objects.
    bkg_sub: bool, default `False`
        If `True`, the image gets background subtracted.
    mask_stars: bool, default `True`
        If `True`, the stars identified are masked by using
        a convolution with a 2D Gaussian kernel.
    aperture_params: tuple, default `None`
        Tuple with objects info and Kron parameters. Used for
        common aperture.
    filt: str, default `None`
        Filter to use for extinction correction and saving outputs.
    survey: str, default `None`
        Survey to use for the zero-points and pixel scale.
    correct_extinction: bool, default `True`
        If `True`, the magnitude is corrected for Milky-Way extinction.
    save_plots: bool, default `False`
        If `True`, the a figure with the aperture is saved.
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

    # preprocessing
    if mask_stars:
        if aperture_params is None:
            # extract objects and cross-match with gaia
            gal_obj, nogal_objs = extract_objects(data_sub, bkg_rms,
                                                  host_ra, host_dec,
                                                  threshold, img_wcs)
            gaia_coord = find_gaia_objects(host_ra, host_dec, img_wcs)
            nogal_objs = cross_match(nogal_objs, img_wcs, gaia_coord)
        else:
            gal_obj, nogal_objs, img_wcs0 = aperture_params
            gal_obj['x'], gal_obj['y'] = pixel2pixel(gal_obj['x'],
                                                    gal_obj['y'],
                                                    img_wcs0, img_wcs)
            nogal_objs['x'], nogal_objs['y'] = pixel2pixel(nogal_objs['x'],
                                                    nogal_objs['y'],
                                                    img_wcs0, img_wcs)
        masked_data = mask_image(data_sub, nogal_objs)
        if save_plots:
            outfile = os.path.join(plots_path,
                                    f'local_mask_{survey}_{filt}.jpg')
            plot_masked_image(data_sub, masked_data,
                                nogal_objs, outfile)
    else:
        masked_data = data_sub.copy()

    # aperture photometry
    radius_arcsec = calc_aperture_size(z, ap_radius)
    pixel_scale = survey_pixel_scale(survey)
    radius_pix  = radius_arcsec/pixel_scale

    px, py = img_wcs.wcs_world2pix(ra, dec, 1)
    error = calc_sky_unc(masked_data, exptime)

    flux, flux_err = extract_aperture_flux(masked_data, error,
                                            px, py, radius_pix)

    zp = survey_zp(survey)
    if survey=='PS1':
        zp += 2.5*np.log10(exptime)

    mag = -2.5*np.log10(flux) + zp
    mag_err = 2.5/np.log(10)*flux_err/flux

    if save_plots:
        outfile = os.path.join(plots_path, f'local_{survey}_{filt}.jpg')
        plot_aperture(masked_data, px, py, radius_pix, outfile)

    if correct_extinction:
        if filt is None:
            raise ValueError('A filter must be given to calculate '
                             'the extinction.')
        A_ext = calc_extinction(filt, survey, ra, dec)
        mag -= A_ext

    # error budget
    # 1.0857 = 2.5/ln(10)
    if survey!='SDSS':
        ap_area = 2*np.pi*(radius_pix**2)
        extra_err = 1.0857*np.sqrt(ap_area*(readnoise**2) + flux/gain)/flux
        mag_err = np.sqrt(mag_err**2 + extra_err**2)

    return mag, mag_err

def multi_band_phot(name, ra, dec, z, ap_radius, host_ra=None, host_dec=None,
                    threshold=7, bkg_sub=False, mask_stars=True, filters=None,
                    survey=None, correct_extinction=True, work_dir='',
                    save_plots=False):
    """Calculates the local aperture photometry in a given radius.

    Parameters
    ----------
    name: str
        Obejct's name. This is used for the directory path.
    ra: float
        Right Ascensions in degrees.
    dec: float
        Declinations in degrees.
    z: float
        Redshift of the SN.
    ap_radius: float
        Physical size of the aperture in kpc.
    host_ra: float, default `None`
        Host-galaxy Right ascension of the galaxy in degrees.
        Used for masking objects in the image.
    host_dec: float, default `None`
        Host-galaxy Declination of the galaxy in degrees.
        Used for masking objects in the image.
    threshold: float, default `7`
        Threshold used by `sep.extract()` to extract objects.
    bkg_sub: bool, default `False`
        If `True`, the image gets background subtracted.
    mask_stars: bool, default `True`
        If `True`, the stars identified are masked by using
        a convolution with a 2D Gaussian kernel.
    filters: str, default `None`
        Filter to use for for the photometry.
    survey: str, default `None`
        Survey to use for the zero-points and correct filter path.
    correct_extinction: bool, default `True`
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

    obj_dir = os.path.join(work_dir, name)
    # get parameters in common between `multi_band_phot()` and `photometry()`
    kwargs = locals()
    cap_args = inspect.getargspec(common_aperture).args
    cap_kwargs = {key:val for key, val in kwargs.items() if key in cap_args}
    phot_args = inspect.getargspec(photometry).args
    phot_kwargs = {key:val for key, val in kwargs.items() if key in phot_args}

    # dictionary to save the results (mag + err)
    results_dict = {'name':name, 'ra':ra, 'dec':dec, 'ap_radius':ap_radius,
                    'host_ra':host_ra, 'host_dec':host_dec}

    if mask_stars:
        coadd_file = coadd_images(name, 'riz', work_dir, survey)
        aperture_params = common_aperture(coadd_file, filt='riz',
                                      plots_path=obj_dir, **cap_kwargs)
    else:
        aperture_params = None

    fits_files = [os.path.join(obj_dir, f'{survey}_{filt}.fits')
                                                for filt in filters]
    for fits_file, filt in zip(fits_files, filters):
        mag, mag_err = photometry(fits_file, filt=filt, plots_path=obj_dir,
                                aperture_params=aperture_params, **phot_kwargs)
        results_dict.update({filt:mag, f'{filt}_err':mag_err})

    return results_dict

def pool_phot(input_args, n_cores):
    """Parallelises `multi_band_phot()` for multiple objects.

    Parameters
    ----------
    input_args: dict or DataFrame
        Dictionary or DataFrame with keys/columns names as the
        parameters of `multi_band_phot()`.
    n_cores: int
        Number of CPU cores to use.

    Returns
    -------
    results_df: DataFrame
        Results with the same outputs of `multi_band_phot()` as columns
    """
    if isinstance(input_args, pd.DataFrame):
        input_args = input_args.to_dict('list')

    _results = []
    def _collect_result(result):
        nonlocal _results
        _results.append(result)

    pool = mp.Pool(n_cores)
    for i, name in enumerate(input_args['name']):
        kwds = {key:input_args[key][i] for key in input_args.keys()}
        pool.apply_async(multi_band_phot, kwds=kwds,
                         callback=_collect_result)
    # the callback allows the colection of the outputs
    pool.close()
    pool.join()

    results_df = pd.DataFrame(_results)

    return results_df
