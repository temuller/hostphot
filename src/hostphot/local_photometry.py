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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sep
from photutils import CircularAperture
from photutils import aperture_photometry

from astropy.io import fits
from astropy import units as u, wcs
from astropy.cosmology import FlatLambdaCDM

from hostphot._constants import __workdir__
from hostphot.utils import (get_survey_filters, check_survey_validity,
                            check_filters_validity, calc_sky_unc,
                            survey_pixel_scale, survey_zp, get_image_gain,
                            get_image_readnoise, check_work_dir)
from hostphot.image_cleaning import remove_nan
from hostphot.dust import calc_extinction

H0 = 70
Om0 = 0.3
__cosmo__ = FlatLambdaCDM(H0, Om0)
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

def choose_cosmology(cosmo):
    """Updates the cosmology used to calculate the aperture size.

    Parameters
    ----------
    cosmo: `astropy.cosmology` object
        Cosmological model. E.g. :func:`FlatLambdaCDM(70, 0.3)`.
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
        Errors of ``data``.
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
    outfile: str, default ``None``
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

def photometry(name, ra, dec, z, filt, survey, ap_radii=1, bkg_sub=False,
                use_mask=True, save_plots=True):
    """Calculates the local aperture photometry in a given radius.

    Parameters
    ----------
    name: str
        Name of the object to find the path of the fits file.
    ra: float
        Right Ascensions in degrees to center the aperture.
    dec: float
        Declinations in degrees to center the aperture.
    z: float
        Redshift of the object to estimate the physical calculate
        of the aperture.
    filt: str
        Filter to use to load the fits file.
    survey: str
        Survey to use for the zero-points and pixel scale.
    ap_radii: float or list-like, default ``1``
        Physical size of the aperture in kpc.
    bkg_sub: bool, default ``False``
        If ``True``, the image gets background subtracted.
    use_mask: bool, default ``True``
        If ``True``, the masked fits files are used. These must have
        been created beforehand.
    save_plots: bool, default ``True``
        If ``True``, the a figure with the aperture is saved.

    Returns
    -------
    mags: list
        List of aperture magnitudes for the given aperture radii.
    mags_err: list
        List of aperture magnitude errors for the given aperture radii.
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

    # turn float into a list
    if isinstance(ap_radii, float):
        ap_radii = [ap_radii]

    mags, mags_err = [], []
    for ap_radius in ap_radii:
        # aperture photometry
        radius_arcsec = calc_aperture_size(z, ap_radius)
        pixel_scale = survey_pixel_scale(survey)
        radius_pix  = radius_arcsec/pixel_scale

        px, py = img_wcs.wcs_world2pix(ra, dec, 1)
        error = calc_sky_unc(data_sub, exptime)

        flux, flux_err = extract_aperture_flux(data_sub, error,
                                                px, py, radius_pix)

        zp = survey_zp(survey)
        if survey=='PS1':
            zp += 2.5*np.log10(exptime)

        mag = -2.5*np.log10(flux) + zp
        mag_err = 2.5/np.log(10)*flux_err/flux

        # correct extinction
        A_ext = calc_extinction(filt, survey, ra, dec)
        mag -= A_ext

        # error budget
        # 1.0857 = 2.5/ln(10)
        if survey!='SDSS':
            ap_area = 2*np.pi*(radius_pix**2)
            extra_err = 1.0857*np.sqrt(ap_area*(readnoise**2) + flux/gain)/flux
            mag_err = np.sqrt(mag_err**2 + extra_err**2)

        mags.append(mag)
        mags_err.append(mag_err)

        if save_plots:
            outfile = os.path.join(obj_dir,
                                    f'local_{survey}_{filt}_{ap_radius}kpc.jpg')
            plot_aperture(data_sub, px, py, radius_pix, outfile)

    return mags, mags_err

def multi_band_phot(name, ra, dec, z, filters=None, survey='PS1', ap_radii=1,
                    bkg_sub=False, use_mask=True, save_plots=True):
    """Calculates the local aperture photometry for multiple filters.

    Parameters
    ----------
    name: str
        Name of the object to find the path of the fits file.
    ra: float
        Right Ascensions in degrees to center the aperture.
    dec: float
        Declinations in degrees to center the aperture.
    z: float
        Redshift of the object to estimate the physical calculate
        of the aperture.
    filters: str, default, ``None``
        Filters to use to load the fits files. If ``None`` use all
        the filters of the given survey.
    survey: str, default ``PS1``
        Survey to use for the zero-points and pixel scale.
    ap_radii: float or list-like, default ``1``
        Physical size of the aperture in kpc.
    bkg_sub: bool, default ``False``
        If ``True``, the image gets background subtracted.
    use_mask: bool, default ``True``
        If ``True``, the masked fits files are used. These must have
        been created beforehand.
    save_plots: bool, default ``True``
        If ``True``, the a figure with the aperture is saved.

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

    # turn float into a list
    if isinstance(ap_radii, float):
        ap_radii = [ap_radii]

    results_dict = {'name':name, 'ra':ra, 'dec':dec,
                    'zspec':z, 'survey':survey}

    for filt in filters:
        mags, mags_err = photometry(name, ra, dec, z, filt, survey,
                                    ap_radii, bkg_sub, use_mask,
                                    save_plots)
        for i, ap in enumerate(ap_radii):
            results_dict[f'{filt}{ra}'] = mags[i]
            results_dict[f'{filt}{ra}_err'] = mags_err[i]

    return results_dict
