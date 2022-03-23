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

from photutils import CircularAperture
from photutils import aperture_photometry

from astropy.io import fits
from astropy.table import Table
from astropy import coordinates, units as u, wcs
from astropy.cosmology import FlatLambdaCDM
from astropy.stats import sigma_clipped_stats

import piscola
from piscola.extinction_correction import extinction_filter

from .phot_utils import calc_sky_unc
from .utils import (get_survey_filters, extract_filters,
                check_survey_validity, check_filters_validity)


H0 = 70
Om0 = 0.3
cosmo = FlatLambdaCDM(H0, Om0)

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
    transv_sep_per_arcmin = cosmo.kpc_proper_per_arcmin(z)
    transv_sep_per_arcsec = transv_sep_per_arcmin.to(u.kpc/u.arcsec)

    radius_arcsec = ap_radius/transv_sep_per_arcsec

    return radius_arcsec.value


def extract_aperture(data, error, px, py, radius):
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


def extract_local_photometry(fits_file, ra, dec, z,
                                     ap_radius=4, survey="PS1"):
    """Extracts local photometry of a given fits file.

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
    ap_radius: float, default `4`
        Physical size of the aperture in kpc. This is used
        for aperture photometry.
    survey: str, default `PS1`
        Survey to use for the zero-points and pixel scale.

    Returns
    -------
    mag: float
        Magnitude.
    mag_err: float
        Error on the magnitude.
    """
    check_survey_validity(survey)

    img = fits.open(fits_file)

    header = img[0].header
    data = img[0].data
    img_wcs = wcs.WCS(header, naxis=2)

    exptime = float(header['EXPTIME'])
    radius_arcsec = calc_aperture_size(z, ap_radius)

    # arcsec to number of pixels (0.XXX arcsec/pix)
    pixel_scale_dict = {'PS1':0.25, 'DES':0.263, 'SDSS':0.396}
    pixel_scale = pixel_scale_dict[survey]
    radius_pix  = radius_arcsec/pixel_scale

    px, py = img_wcs.wcs_world2pix(ra, dec, 1)
    error = calc_sky_unc(data, exptime)

    raw_flux, raw_flux_err = extract_aperture(data, error,
                                              px, py, radius_pix)

    zp_dict = {'PS1':25 + 2.5*np.log10(exptime),
               'DES':30,
               'SDSS':22.5}
    zp = zp_dict[survey]

    mag = -2.5*np.log10(raw_flux) + zp
    mag_err = 2.5/np.log(10)*raw_flux_err/raw_flux

    return mag, mag_err

def multi_local_photometry(name_list, ra_list, dec_list, z_list,
                             ap_radius, work_dir='', filters=None,
                               survey="PS1", correct_extinction=True):
    """Extract local photometry for multiple SNe.

    Parameters
    ----------
    name_list: list-like
        List of SN names.
    ra_list: list-like
        List of right ascensions in degrees.
    dec_list: list-like
        List of declinations in degrees.
    z_list: list-like
        List of redshifts.
    ap_radius: float
        Physical size of the aperture in kpc. This is used
        for aperture photometry.
    work_dir: str, default ''
        Working directory where to find the objects'
        directories with the images. Default, current directory.
    filters: str, defaul `None`
        Filters used to extract photometry. If `None`, use all
        the available filters for the given survey.
    survey: str, default `PS1`
        Survey to use for the zero-points and pixel scale.
    correct_extinction: bool, default `True`
        If `True`, the magnitudes are corrected for extinction.

    Returns
    -------
    local_phot_df: DataFrame
        Dataframe with the photometry, errors and SN info.
    """
    check_survey_validity(survey)
    check_filters_validity(filters, survey)
    if filters is None:
        filters = get_survey_filters(survey)

    # dictionary to save results
    mag_dict = {filt:[] for filt in filters}
    mag_err_dict = {filt+'_err':[] for filt in filters}
    mag_dict.update(mag_err_dict)

    results_dict = {'name':[], 'ra':[], 'dec':[], 'zspec':[]}
    results_dict.update(mag_dict)

    # filter funcstions for extinction correction
    filters_dict = extract_filters(filters, survey)

    for name, ra, dec, z in zip(name_list, ra_list,
                                    dec_list, z_list):

        sn_dir = os.path.join(work_dir, name)
        image_files = [os.path.join(sn_dir, f'{survey}_{filt}.fits')
                                                    for filt in filters]

        for image_file, filt in zip(image_files, filters):
            try:
                mag, mag_err = extract_local_photometry(image_file,
                                                  ra, dec, z,
                                                  ap_radius=ap_radius,
                                                  survey=survey)
                if correct_extinction:
                    wave = filters_dict[filt]['wave']
                    transmission = filters_dict[filt]['transmission']
                    A_ext = extinction_filter(wave, transmission, ra, dec)
                    mag -= A_ext
                results_dict[filt].append(mag)
                results_dict[filt+'_err'].append(mag_err)
            except Exception as message:
                results_dict[filt].append(np.nan)
                results_dict[filt+'_err'].append(np.nan)
                print(f'{name} failed with {filt} band: {message}')
        results_dict['name'].append(name)
        results_dict['ra'].append(ra)
        results_dict['dec'].append(dec)
        results_dict['zspec'].append(z)

    local_phot_df = pd.DataFrame(results_dict)

    return local_phot_df
