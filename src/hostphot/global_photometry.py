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
import inspect
import numpy as np
import pandas as pd
import multiprocessing as mp

import sep
from astropy.io import fits
from astropy import coordinates as coords, units as u, wcs

from photutils import (CircularAnnulus,
                       CircularAperture,
                       aperture_photometry)

import reproject
from reproject.mosaicking import reproject_and_coadd

from .utils import (get_survey_filters, extract_filters,
                    check_survey_validity, check_filters_validity,
                    calc_sky_unc, survey_zp, get_image_gain,
                    get_image_readnoise, pixel2pixel)
from .objects_detect import (extract_objects, find_gaia_objects,
                             cross_match, plot_detected_objects)
from .image_masking import mask_image, plot_masked_image
from .image_cleaning import remove_nan
from .coadd import coadd_images
from .dust import calc_extinction

sep.set_sub_object_limit(1e4)

# Photometry
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
    The stop condition is met when the change in flux is less that `eps`.

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
    if np.isnan(kronrad):
        print(f'kronrad = {kronrad}')
        raise ValueError('The Kron radius cannot be calculated '
                         '(something went wrong!)')

    opt_flux = 0.0
    # iterate over scale
    scales = np.arange(1, 10, 0.1)
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

def common_aperture(fits_file, host_ra, host_dec, bkg_sub=False, threshold=7,
                   mask_stars=True, optimze_kronrad=True, eps=0.001, filt=None,
                   survey=None, save_plots=False, plots_path=''):
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
    optimze_kronrad: bool, default `True`
        If `True`, the Kron radius is optimized, increasing the
        aperture size until the flux does not increase.
    eps: float, default `0.001`
        Minimum percent change in flux allowed between iterations
        when optimizing the Kron radius.
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
    kronrad: float
        Kron radius.
    scale: float
        Scale for the kron radius.
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
    if mask_stars:
        # cross-match extracted objects with gaia
        gaia_coord = find_gaia_objects(host_ra, host_dec, img_wcs)
        nogal_objs = cross_match(nogal_objs, img_wcs, gaia_coord)

        masked_data = mask_image(data_sub, nogal_objs)
        if save_plots:
            outfile = os.path.join(plots_path,
                                    f'global_mask_{survey}_{filt}.jpg')
            plot_masked_image(data_sub, masked_data,
                                nogal_objs, outfile)
    else:
        masked_data = data_sub.copy()

    if optimze_kronrad:
        gain = 1  # doesn't matter here
        opt_res = optimize_kron_flux(masked_data, bkg_rms,
                                     gain, gal_obj, eps)
        flux, flux_err, kronrad, scale = opt_res
    else:
        scale = 2.5
        kronrad, krflag = sep.kron_radius(masked_data,
                                          gal_obj['x'], gal_obj['y'],
                                          gal_obj['a'], gal_obj['b'],
                                          gal_obj['theta'], 6.0)

    if save_plots:
        outfile = os.path.join(plots_path, f'global_{survey}_{filt}.jpg')
        plot_detected_objects(masked_data, gal_obj,
                                scale*kronrad, outfile)

    return gal_obj, nogal_objs, img_wcs, kronrad, scale


def photometry(fits_file, host_ra, host_dec, bkg_sub=False, threshold=7,
               mask_stars=True, aperture_params=None, optimze_kronrad=True,
               eps=0.001, filt=None, survey=None, correct_extinction=True,
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
    aperture_params: tuple, default `None`
        Tuple with objects info and Kron parameters. Used for
        common aperture.
    optimze_kronrad: bool, default `True`
        If `True`, the Kron radius is optimized, increasing the
        aperture size until the flux does not increase.
    eps: float, default `0.001`
        Minimum percent change in flux allowed between iterations
        when optimizing the Kron radius.
    filt: str, default `None`
        Filter to use for extinction correction and saving outputs.
    survey: str, default `None`
        Survey to use for the zero-points and correct filter path.
    correct_extinction: bool, default `True`
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

    if get_kronrad_scale`==True`
        gal_obj: array
            Galaxy obejct.
        kronrad: float
            Kron radius.
        scale: float
            Scale for the kron radius.
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

    if aperture_params is not None:
        gal_obj, nogal_objs, img_wcs0, _, _ = aperture_params
        gal_obj['x'], gal_obj['y'] = pixel2pixel(gal_obj['x'],
                                                gal_obj['y'],
                                                img_wcs0, img_wcs)
        nogal_objs['x'], nogal_objs['y'] = pixel2pixel(nogal_objs['x'],
                                                nogal_objs['y'],
                                                img_wcs0, img_wcs)
    else:
        # extract objects
        gal_obj, nogal_objs = extract_objects(data_sub, bkg_rms,
                                              host_ra, host_dec,
                                              threshold, img_wcs)

    # preprocessing
    if mask_stars:
        if aperture_params is None:
            # cross-match extracted objects with gaia
            gaia_coord = find_gaia_objects(host_ra, host_dec, img_wcs)
            nogal_objs = cross_match(nogal_objs, img_wcs, gaia_coord)
            masked_data = mask_image(data_sub, nogal_objs)
        else:
            masked_data = mask_image(data_sub, nogal_objs)

        if save_plots:
            outfile = os.path.join(plots_path, f'global_mask_{filt}.jpg')
            plot_masked_image(data_sub, masked_data,
                                nogal_objs, outfile)
    else:
        masked_data = data_sub.copy()

    if aperture_params is None:
        # aperture photometry
        # This uses what would be the default SExtractor parameters.
        # See https://sep.readthedocs.io/en/v1.1.x/apertures.html
        if optimze_kronrad:
            opt_res = optimize_kron_flux(masked_data, bkg_rms,
                                         gain, gal_obj, eps)
            flux, flux_err, kronrad, scale = opt_res
        else:
            kronrad, krflag = sep.kron_radius(masked_data,
                                              gal_obj['x'], gal_obj['y'],
                                              gal_obj['a'], gal_obj['b'],
                                              gal_obj['theta'], 6.0)
            scale = 2.5
            flux, flux_err =  kron_flux(masked_data, bkg_rms, gain,
                                        gal_obj, kronrad, scale)
            flux, flux_err = flux[0], flux_err[0]

        if save_plots:
            outfile = os.path.join(plots_path,
                                    f'global_{survey}_{filt}.jpg')
            plot_detected_objects(masked_data, gal_obj,
                                    scale*kronrad, outfile)
    else:
        _, _, _, kronrad, scale = aperture_params
        flux, flux_err =  kron_flux(masked_data, bkg_rms, gain,
                                    gal_obj, kronrad, scale)
        flux, flux_err = flux[0], flux_err[0]

        if save_plots:
            outfile = os.path.join(plots_path,
                                    f'global_{survey}_{filt}.jpg')
            plot_detected_objects(masked_data, gal_obj,
                                    scale*kronrad, outfile)

    zp = survey_zp(survey)
    if survey=='PS1':
        zp += 2.5*np.log10(exptime)

    mag = -2.5*np.log10(flux) + zp
    mag_err = 2.5/np.log(10)*flux_err/flux

    if correct_extinction:
        if filt is None:
            raise ValueError('A filter must be given to calculate '
                             'the extinction.')
        A_ext = calc_extinction(filt, survey, host_ra, host_dec)
        mag -= A_ext

    # error budget
    # 1.0857 = 2.5/ln(10)
    if survey!='SDSS':
        ap_area = np.pi*gal_obj['a'][0]*gal_obj['b'][0]  # ellipse area = pi*a*b
        extra_err = 1.0857*np.sqrt(ap_area*(readnoise**2) + flux/gain)/flux
        mag_err = np.sqrt(mag_err**2 + extra_err**2)

    return mag, mag_err

def multi_band_phot(name, host_ra, host_dec, bkg_sub=False, threshold=7,
                   mask_stars=True, coadd_filters='riz', optimze_kronrad=True,
                   eps=0.001, filters=None, survey=None, correct_extinction=True,
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
    coadd_filters: 'str', default `riz`
        Filters to use for the coadd image.
    optimze_kronrad: bool, default `True`
        If `True`, the Kron radius is optimized, increasing the
        aperture size until the flux does not increase.
    eps: float, default `0.001`
        Minimum percent change in flux allowed between iterations
        when optimizing the Kron radius.
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
    # get parameters in common between the functions
    kwargs = locals()
    cap_args = inspect.getargspec(common_aperture).args
    cap_kwargs = {key:val for key, val in kwargs.items() if key in cap_args}
    phot_args = inspect.getargspec(photometry).args
    phot_kwargs = {key:val for key, val in kwargs.items() if key in phot_args}

    # dictionary to save the results (mag + err)
    results_dict = {'name':name, 'host_ra':host_ra, 'host_dec':host_dec}

    if coadd_filters:
        coadd_file = coadd_images(name, coadd_filters, work_dir, survey)
        aperture_params = common_aperture(coadd_file, filt=coadd_filters,
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
