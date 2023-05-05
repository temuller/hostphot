import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy import wcs
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.utils import calc_total_error
from photutils.aperture import EllipticalAperture

import warnings
from astropy.utils.exceptions import AstropyWarning

import hostphot

hostphot_path = hostphot.__path__[0]
config_file = os.path.join(hostphot_path, "filters", "config.txt")
config_df = pd.read_csv(config_file, delim_whitespace=True)

# surveys that need background subtraction
bkg_surveys = ['2MASS', 'WISE', 'VISTA']

def calc_sky_unc(image, exptime):
    """Calculates the uncertainty of the image from the
    sky standard deviation, sigma-clipped STD.

    Parameters
    ----------
    image: ndarray
        Image in a 2D numpy array.
    exptime: float
        Exposure time of the image.

    Returns
    -------
    error: float
        Estimated error of the image.
    """
    mask = image != 0
    avg, sky, sky_std = sigma_clipped_stats(image[mask], sigma=3.0)
    error = calc_total_error(image, sky_std, exptime)

    return error


def pixel2pixel(x1, y1, img_wcs1, img_wcs2):
    """Convert the pixel coordinates from one image to another."""
    coord1 = img_wcs1.pixel_to_world(x1, y1)
    x2, y2 = img_wcs2.world_to_pixel(coord1)

    return x2, y2


def check_survey_validity(survey):
    """Check whether the given survey is whithin the valid
    options.

    Parameters
    ----------
    survey: str
        Survey name: e.g. ``PS1``, ``GALEX``.
    """
    global config_df
    surveys = list(config_df.survey)
    assert survey in surveys, f"Survey '{survey}' not in {surveys}"


def get_survey_filters(survey):
    """Gets all the valid filters for the given survey.

    Parameters
    ----------
    survey: str
        Survey name: e.g. ``PS1``, ``GALEX``.

    Returns
    -------
    filters: str
        Filters for the given survey.
    """
    check_survey_validity(survey)

    if survey=='HST':
        # For HST, the filter needs to be specified
        return None

    global config_df
    survey_df = config_df[config_df.survey == survey]
    filters = survey_df.filters.values[0]

    if "," in filters:
        filters = filters.split(",")

    return filters


def survey_zp(survey):
    """Returns the zero-point for a given survey.

    **Note:** for ``PS1``, an extra :math:`+2.5*np.log10(exptime)`
    needs to be added afterwards.

    Parameters
    ----------
    survey: str
        Survey name: e.g. ``PS1``, ``GALEX``.

    Returns
    -------
    zp_dict: dict or str
        Zero-points for all the filters in the given survey. If the survey zero-point
        is different for each image, the string ``header`` is returned.
    """
    check_survey_validity(survey)
    filters = get_survey_filters(survey)

    global config_df
    survey_df = config_df[config_df.survey == survey]
    zps = survey_df.zp.values[0]

    if zps == "header":
        return zps

    if "," in zps:
        zps = zps.split(",")
        zp_dict = {filt: float(zp) for filt, zp in zip(filters, zps)}
    elif survey=='SDSS':
        # SDSS zero-points are not exactly in AB:
        # https://www.sdss4.org/dr12/algorithms/fluxcal/#SDSStoAB
        zp_dict = {filt: float(zps) for filt in filters}
        zp_dict['u'] -= 0.04
        zp_dict['z'] += 0.02
    else:
        zp_dict = {filt: float(zps) for filt in filters}

    return zp_dict


def get_image_gain(header, survey):
    """Returns the gain from an image's header.

    **Note:** for ``SDSS`` this is assumed to be one
    as it should already be included.

    Parameters
    ----------
    header: fits header
        Header of an image.
    survey: str
        Survey name: e.g. ``PS1``, ``GALEX``.

    Returns
    -------
    gain: float
        Gain value.
    """
    check_survey_validity(survey)
    if survey == "PS1":
        gain = header["HIERARCH CELL.GAIN"]
    elif survey == "DES":
        gain = header["GAIN"]
    elif survey == "SDSS":
        gain = 1.0
    elif survey == "2MASS":
        # the value comes from https://wise2.ipac.caltech.edu/staff/jarrett/2mass/3chan/noise/
        # Note that it is different from the value of
        # https://iopscience.iop.org/article/10.1086/498708/pdf (8 e ADU^-1)
        gain = 10.0
    elif survey == "LegacySurvey":
        gain = 30  # assumed similar to DES
    elif survey == "Spitzer":
        # also in the "EXPGAIN" keyword
        if header["INSTRUME"] == "IRAC":
            # Table 2.4 of IRAC Instrument Handbook
            gain = 3.8  # all filters have similar gain
            gain *= header["EFCONV"]  # convert DN/s to MJy/sr
        elif header["INSTRUME"] == "MIPS":
            # Table 2.4 of MIPS Instrument Handbook
            gain = 5.0
    elif survey == "VISTA":
        # use median from http://casu.ast.cam.ac.uk/surveys-projects/vista/technical/vista-gain
        gain = 4.19
    elif survey=='HST':
        gain = header['CCDGAIN']
    else:
        gain = 1.0

    return gain


def get_image_readnoise(header, survey):
    """Returns the read noise from an image's header.
    All values are per-pixel values.

    **Note:** for ``SDSS`` this is assumed to be zero
    as it should already be included.

    Parameters
    ----------
    header: fits header
        Header of an image.
    survey: str
        Survey name: e.g. ``PS1``, ``GALEX``.

    Returns
    -------
    readnoise: float
        Read noise value.
    """
    check_survey_validity(survey)
    if survey == "PS1":
        readnoise = header["HIERARCH CELL.READNOISE"]
    elif survey == "DES":
        # see https://arxiv.org/pdf/0810.3600.pdf
        readnoise = 7.0  # electrons per pixel
    elif survey == "SDSS":
        readnoise = 0.0
    elif survey == "2MASS":
        # https://iopscience.iop.org/article/10.1086/498708/pdf
        # 6 combined images
        readnoise = 4.5 * np.sqrt(6)  # not used
    elif survey == "LegacySurvey":
        readnoise = 7.0  # assumed similar to DES
    elif survey == "Spitzer":
        if header["INSTRUME"] == "IRAC":
            # Table 2.3 of IRAC Instrument Handbook
            # very rough average
            readnoise_dict = {1: 16.0, 2: 12.0, 3: 10.0, 4: 8.0}
            channel = header["CHNLNUM"]
            readnoise = readnoise_dict[channel]
        elif header["INSTRUME"] == "MIPS":
            # Table 2.4 of MIPS Instrument Handbook
            readnoise = 40.0
    elif survey == "VISTA":
        # very rough average for all filters in
        # http://casu.ast.cam.ac.uk/surveys-projects/vista/technical/vista-gain
        readnoise = 24.0
    elif survey=='HST':
        # tipically 0.0
        readnoise = header['PCTERNOI']
    else:
        readnoise = 0.0

    return readnoise


def get_image_exptime(header, survey):
    """Returns the exposure time from an image's header.

    Parameters
    ----------
    header: fits header
        Header of an image.
    survey: str
        Survey name: e.g. ``PS1``, ``GALEX``.

    Returns
    -------
    exptime: float
        Exposure time in seconds.
    """
    check_survey_validity(survey)
    if survey in ["PS1", "DES", "GALEX", "VISTA", "Spitzer", "HST"]:
        exptime = float(header["EXPTIME"])
    elif survey == "WISE":
        # see: https://wise2.ipac.caltech.edu/docs/release/allsky/
        # and https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec1_1.html
        if header["BAND"] in [1, 2]:
            exptime = 7.7
        elif header["BAND"] in [3, 4]:
            exptime = 8.8
    elif survey == "2MASS":
        # https://iopscience.iop.org/article/10.1086/498708/pdf
        exptime = 7.8
    elif survey == "LegacySurvey":
        exptime = 900.0  # assumed similar to DES
    else:
        exptime = 1.0

    return exptime


def correct_HST_aperture(filt, ap_area, header):
    """Get the aperture correction for the given configuration.

    Parameters
    ----------
    filt : str
        HST filter, e.g. ``WFC3_UVIS_F225W``.
    ap_area: float
        Aperture area.
    header: fits header
        Header of an image.

    Returns
    -------
    correction: float
        Aperture correction (encircled energy fraction).
    """
    # split instrument and filter
    filt_split = filt.split('_')
    filt = filt_split[-1]
    instrument = filt_split[-2]

    if instrument=='UVIS':
        instrument = header['APERTURE']

    # assuming circular aperture
    # for an ellipse, this would take the average of the axes
    ap_radius = np.sqrt(ap_area/np.pi)

    # get correction curve
    ac_files = glob.glob(os.path.join(hostphot_path, 'filters/HST/*'))
    ac_file = [file for file in ac_files 
               if f'{instrument.lower()}_aper' in file][0]
    ac_df = pd.read_csv(ac_file)

    # linear interpolation of the aperture correction
    apertures = np.array([float(col.replace('APER#', '')) for col in ac_df.columns 
                          if col.startswith('AP')])
    ap_corr = ac_df[ac_df.FILTER==filt].values[0][2:].astype(float)

    cont_apertures = np.arange(0, 9, 0.01)
    cont_ap_corr = np.interp(cont_apertures, apertures, ap_corr)

    # get the closest value
    ind = np.argmin(np.abs(cont_apertures-ap_radius))
    correction = cont_ap_corr[ind]

    return correction

def magnitude_calculation(
    flux,
    flux_err,
    survey,
    filt=None,
    ap_area=0.0,
    header=None,
    bkg_rms=0.0,
):
    """Calculates the calibrated magnitudes and errors.

    Parameters
    ----------
    flux: float
        Aperture flux.
    survey: str
        Survey name: e.g. ``PS1``, ``GALEX``.
    filt: str, default ``None``
        Survey-specific filter.
    ap_area: float, default ``0.0``
        Aperture area.
    header: fits header, default ``None``
        Header of an image.
    bkg_rms: float, default ``0.0``
        Background noise.

    Returns
    -------
    mag: float
        Apparent magnitude.
    mag_err: float
        Apparent magnitude uncertainty.
    flux: float
        Total flux.
    flux_error: float
        Total uncertainty in flux.
    """
    # get zeropoint
    zp_dict = survey_zp(survey)
    if zp_dict == "header":
        zp = header["MAGZP"]
    else:
        zp = zp_dict[filt]
    if survey == "PS1":
        exptime = get_image_exptime(header, survey)
        zp += 2.5 * np.log10(exptime)

    # error propagation
    mag_err, flux_err = uncertainty_calculation(
        flux,
        flux_err,
        survey,
        filt,
        ap_area,
        header,
        bkg_rms,
    )

    if survey == 'HST':
        # HST needs and aperture correction for the flux
        # see, e.g. https://www.stsci.edu/hst/instrumentation/acs/data-analysis/aperture-corrections
        ap_corr = correct_HST_aperture(filt, ap_area, header)
        flux = flux*ap_corr

    mag = -2.5 * np.log10(flux) + zp

    return mag, mag_err, flux, flux_err


def get_HST_err(filt, header):
    """Obtaines the error propagation from the zeropoint.

    Parameters
    ----------
    filt : str
        HST filter, e.g. ``WFC3_UVIS_F225W``.
    header: fits header
        Header of an image.

    Returns
    -------
    flux_err: float
        Error on PHOTFLAM.
    mag_err: float
        Magnitude error on PHOTFLAM.
    """
    # split instrument and filter
    filt_split = filt.split('_')
    filt = filt_split[-1]
    instrument = filt_split[-2]

    if instrument=='UVIS':
        # APERTURE usually point to UVIS2
        instrument = header['APERTURE']

    # get uncertainty file
    err_file = os.path.join(hostphot_path, 
                            'filters/HST/', 
                            f'{instrument}_err.txt')
    err_df = pd.read_csv(err_file, delim_whitespace=True)
    filt_err_df = err_df[err_df.Filter==filt]

    flux = filt_err_df.PHOTFLAM.values[0]
    flux_err = filt_err_df.ERR_PHOTFLAM.values[0]
    mag_err = np.abs(2.5 * flux_err / (flux * np.log(10)))
    
    return flux_err, mag_err

def uncertainty_calculation(
    flux, flux_err, survey, filt=None, ap_area=0.0, header=None, bkg_rms=0.0
):
    """Calculates the uncertainty propagation.

    Parameters
    ----------
    flux: float
        Aperture flux.
    flux_err: float
        Aperture flux error.
    survey: str
        Survey name: e.g. ``PS1``, ``GALEX``.
    filt: str, default ``None``
        Survey-specific filter.
    ap_area: float, default ``0.0``
        Aperture area.
    header: fits header, default ``None``
        Header of an image.
    bkg_rms: float, default ``0.0``
        Background noise.

    Returns
    -------
    mag_err: float
        Extra uncertainty in magnitudes.
    flux_err: float
        Total uncertainty in flux units.
    """
    exptime = get_image_exptime(header, survey)
    gain = get_image_gain(header, survey)
    readnoise = get_image_readnoise(header, survey)

    mag_err = 2.5 / np.log(10) * flux_err / flux
    
    if survey in ["PS1", "DES", "LegacySurvey", "Spitzer", "VISTA"]:
        if survey == "Spitzer":
            flux /= header["EFCONV"]  # conv. factor (MJy/sr)/(DN/s)
        # 1.0857 = 2.5/ln(10)
        extra_err = (
            1.0857 * np.sqrt(ap_area * (readnoise**2) + flux / gain) / flux
        )
        mag_err = np.sqrt(mag_err**2 + extra_err**2)

        extra_flux_err = np.sqrt(ap_area * (readnoise**2) + flux / gain)
        flux_err = np.sqrt(flux_err**2 + extra_flux_err**2)
    
    if survey == "DES":
        # see the photometry section in https://des.ncsa.illinois.edu/releases/dr1/dr1-docs/quality
        # statistical uncertainties on the AB magnitud system zeropoints
        unc_dict = {
            "g": 2.6e-3,
            "r": 2.9e-3,
            "i": 3.4e-3,
            "z": 2.5e-3,
            "Y": 4.5e-3,
        }
        extra_err = unc_dict[filt]
        mag_err = np.sqrt(mag_err**2 + extra_err**2)

        # median coadd zeropoint statistical uncertainty
        unc_dict = {
            "g": 5e-3,
            "r": 4e-3,
            "i": 5e-3,
            "z": 6e-3,
            "Y": 5e-3,
        }
        extra_err = unc_dict[filt]
        mag_err = np.sqrt(mag_err**2 + extra_err**2)

        extra_flux_err = np.abs(flux*0.4*np.log(10)*extra_err)
        flux_err = np.sqrt(flux_err ** 2 + extra_flux_err ** 2)

    elif survey == "PS1":
        # add floor systematic error from:
        # https://iopscience.iop.org/article/10.3847/1538-4365/abb82a/pdf
        unc_dict = {
            "g": 14e-3,
            "r": 14e-3,
            "i": 15e-3,
            "z": 15e-3,
            "y": 18e-3,
        }
        floor_err = unc_dict[filt]
        mag_err = np.sqrt(mag_err ** 2 + floor_err ** 2)

        extra_flux_err = np.abs(flux * 0.4 * np.log(10) * floor_err)
        flux_err = np.sqrt(flux_err ** 2 + extra_flux_err ** 2)

    elif survey == "SDSS":
        # https://data.sdss.org/datamodel/files/BOSS_PHOTOOBJ/frames/RERUN/RUN/CAMCOL/frame.html
        camcol = header["CAMCOL"]
        run = header["RUN"]

        gain_dict = {
            "u": {1: 1.62, 2: 1.595, 3: 1.59, 4: 1.6, 5: 1.47, 6: 2.17},
            "g": {1: 3.32, 2: 3.855, 3: 3.845, 4: 3.995, 5: 4.05, 6: 4.035},
            "r": {1: 4.71, 2: 4.6, 3: 4.72, 4: 4.76, 5: 4.725, 6: 4.895},
            "i": {1: 5.165, 2: 6.565, 3: 4.86, 4: 4.885, 5: 4.64, 6: 4.76},
            "z": {1: 4.745, 2: 5.155, 3: 4.885, 4: 4.775, 5: 3.48, 6: 4.69},
        }
        # dark variance
        dv_dict = {
            "u": {
                1: 9.61,
                2: 12.6025,
                3: 8.7025,
                4: 12.6025,
                5: 9.3025,
                6: 7.0225,
            },
            "g": {
                1: 15.6025,
                2: 1.44,
                3: 1.3225,
                4: 1.96,
                5: 1.1025,
                6: 1.8225,
            },
            "r": {
                1: 1.8225,
                2: 1.00,
                3: 1.3225,
                4: 1.3225,
                5: 0.81,
                6: 0.9025,
            },
            "i": {1: 7.84, 2: 5.76, 3: 4.6225, 4: 6.25, 5: 7.84, 6: 5.0625},
            "z": {1: 0.81, 2: 1.0, 3: 1.0, 4: 9.61, 5: 1.8225, 6: 1.21},
        }
        gain = gain_dict[filt][camcol]
        dark_variance = dv_dict[filt][camcol]

        if filt == "u" and camcol == 2 and run > 1100:
            gain = 1.825
        if filt == "i" and run > 1500:
            if camcol == 2:
                dark_variance = 6.25
            if camcol == 4:
                dark_variance = 7.5625
        if filt == "z" and run > 1500:
            if camcol == 4:
                dark_variance = 12.6025
            if camcol == 5:
                dark_variance = 2.1025

        extra_err = 1.0857 * np.sqrt(dark_variance + flux / gain) / flux
        mag_err = np.sqrt(mag_err**2 + extra_err**2)

        extra_flux_err = np.sqrt(dark_variance + flux / gain)
        flux_err = np.sqrt(flux_err ** 2 + extra_flux_err ** 2)

    elif survey == "GALEX":
        # https://asd.gsfc.nasa.gov/archive/galex/FAQ/counts_background.html
        CPS = flux
        if filt == "FUV":
            uv_err = -2.5 * (
                np.log10(CPS)
                - np.log10(
                    CPS
                    + (CPS * exptime + (0.050 * CPS * exptime) ** 2) ** 0.5
                    / exptime
                )
            )
            flux_uv_err = np.sqrt(CPS*exptime + (0.050 * CPS * exptime)**2)/exptime
        elif filt == "NUV":
            uv_err = -2.5 * (
                np.log10(CPS)
                - np.log10(
                    CPS
                    + (CPS * exptime + (0.027 * CPS * exptime) ** 2) ** 0.5
                    / exptime
                )
            )
            flux_uv_err = np.sqrt(CPS * exptime + (0.027 * CPS * exptime) ** 2) / exptime

        mag_err = np.sqrt(mag_err**2 + uv_err**2)

        flux_err = np.sqrt(flux_err ** 2 + flux_uv_err ** 2)

    elif survey == "2MASS":
        # see: https://wise2.ipac.caltech.edu/staff/jarrett/2mass/3chan/noise/
        S = flux
        N_c = 6  # number of coadd pixels
        k_z = 1.7  # kernel smoothing factor
        n_f = ap_area  # number of frame pixels in the aperture; aprox. as aperture area
        n_c = 4 * n_f  # number of coadd pixels in the aperture
        sigma_c = bkg_rms  # coadd noise; assumed to be ~background noise

        SNR = S / np.sqrt(
            S / (gain * N_c)
            + n_c * (2 * k_z * sigma_c) ** 2
            + (n_c * 0.024 * sigma_c) ** 2
        )

        mag_err = 1.0857 / SNR

        extra_flux_err = flux/SNR
        flux_err = np.sqrt(flux_err ** 2 + extra_flux_err ** 2)

    elif "WISE" in survey:
        # see: https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec2_3f.html
        # see Table 5 of
        # https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4c.html#wpro
        # correction assumed to be 0 mags as PSF fitting is not used.
        m_apcor = 0.0
        f_apcor = 10 ** (-0.4 * m_apcor)
        F_src = f_apcor * flux

        N_p = (
            ap_area  # effective number of noise-pixels characterizing the PRF
        )
        # ratio of input (detector) pixel scale to output (Atlas Image) pixel scale
        pixel_scale_ratios = {"W1": 2, "W2": 2, "W3": 2, "W4": 4}
        Sin_Sout = pixel_scale_ratios[filt]
        F_corr = N_p * Sin_Sout**2

        k = 1
        N_A = N_B = ap_area
        sigma_conf = flux_err  # assumed to be the ~error in the aperture sum
        sigma_src = np.sqrt(
            f_apcor**2
            * F_corr
            * (flux_err**2 + k * (N_A**2) / N_B * bkg_rms**2)
            + sigma_conf**2
        )

        mag_err = np.sqrt(1.179 * sigma_src**2 / F_src**2)

        flux_err = f_apcor ** 2

        # add uncertainty from the ZP
        if survey=="unWISE":
            # These values are the same for all Atlas Images of a given band...
            # see: https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec2_3f.html
            unc_dict = {'W1':0.006, 'W2':0.007, 'W3':0.012, 'W4':0.012}
            zp_unc = unc_dict[filt]
        elif survey=="WISE":
            zp_unc = header["MAGZPUNC"]

        mag_err = np.sqrt(mag_err**2 + zp_unc**2)

        extra_flux_err = np.abs(flux * 0.4 * np.log(10) * zp_unc)
        flux_err = np.sqrt(flux_err ** 2 + extra_flux_err ** 2)

    elif survey == "LegacySurvey":
        # already added at the beginning
        pass
    elif survey == "Spitzer":
        # already added at the beginning
        pass
    elif survey == "VISTA":
        # add uncertainty from the ZP
        zp_unc = header["MAGZRR"]
        mag_err = np.sqrt(mag_err**2 + zp_unc**2)

        extra_flux_err = np.abs(flux * 0.4 * np.log(10) * zp_unc)
        flux_err = np.sqrt(flux_err ** 2 + extra_flux_err ** 2)
    elif survey == "HST":
        flux_zp_unc, zp_unc = get_HST_err(filt, header)

        mag_err = np.sqrt(mag_err**2 + zp_unc**2)
        flux_err = np.sqrt(flux_err ** 2 + flux_zp_unc ** 2)
    else:
        raise Exception(
            f"Survey {survey} has not been added for error propagation."
        )

    return mag_err, flux_err


def survey_pixel_scale(survey, filt=None):
    """Returns the pixel scale for a given survey.

    Parameters
    ----------
    survey: str
        Survey name: e.g. ``PS1``, ``GALEX``.
    filt: str, default ``None``
        Filter to use by surveys that have different pixel scales
        for different filters (e.g. Spitzer's IRAC and MIPS instruments).

    Returns
    -------
    pixel_scale: float
        Pixel scale in units of arcsec/pixel.
    """
    check_survey_validity(survey)

    global config_df
    survey_df = config_df[config_df.survey == survey]
    pixel_scale = survey_df.pixel_scale.values[0]

    if len(pixel_scale.split(",")) > 1:
        filters = get_survey_filters(survey)
        pixel_scale_dict = {
            f: float(ps) for f, ps in zip(filters, pixel_scale.split(","))
        }

        if filt in pixel_scale_dict.keys():
            pixel_scale = pixel_scale_dict[filt]
        else:
            print(f"No pixel scale found for filter {filt}.")
            filt_used = list(pixel_scale_dict.keys())[0]
            print(f"Using the pixel scale of filter {filt_used} ({survey}).")
            pixel_scale = list(pixel_scale_dict.values())[0]

        return pixel_scale

    return float(pixel_scale)


def check_filters_validity(filters, survey):
    """Check whether the given filters are whithin the valid
    options for the given survey.

    Parameters
    ----------
    filters: str or list
        Filters to use, e,g, ``griz``.
    survey: str
        Survey name: e.g. ``PS1``, ``GALEX``.
    """
    if survey=='HST':
        check_HST_filters(filters)

    else:
        valid_filters = get_survey_filters(survey)

        for filt in filters:
            message = (
                f"filter '{filt}' is not a valid option for "
                f"'{survey}' survey ({valid_filters})"
            )
            assert filt in valid_filters, message

def check_HST_filters(filt):
    """Check whether the given filter is whithin the valid
    options for HST.

    Parameters
    ----------
    filt: str 
        Filter to use, e,g, ``WFC3_UVIS_F225W``.
    """        
    if filt is None:
        raise ValueError(f"'{filt}' is not a valid HST filter.")
    
    # For UVIS, only the filters of UVIS1 are used as the
    # detector 2 is scaled to match detector 1
    if 'UVIS' in filt:
        filt = filt.replace('UVIS', 'UVIS1')

    hostphot_path = hostphot.__path__[0]
    hst_file = glob.glob(os.path.join(hostphot_path, "filters/HST/*/*"))
    hst_filters = [os.path.basename(file).split('.')[0] for file in hst_file]

    assert filt in hst_filters, f"Not a valid HST filter ({filt}): {hst_filters}"


def extract_filter(filt, survey, version=None):
    """Extracts the transmission function for the filter.

    Parameters
    ----------
    filt: str
        Filter to extract.
    survey: str
        Survey of the filters.
    version: str, default ``None``
        Version of the filters to use. E.g. for the
        Legacy Survey as it uses DECam for the south
        and BASS+MzLS for the north.

    Returns
    -------
    wave: array
        Wavelength range of the filter.
    transmission: array
        Transmission function.
    """
    check_survey_validity(survey)
    if survey=='HST':
        check_filters_validity(filt, survey)
    else:
        check_filters_validity([filt], survey)

    if "WISE" in survey:
        survey = "WISE"  # for unWISE to use the same filters as WISE

    filters_path = os.path.join(hostphot.__path__[0], "filters", survey)

    # Assume DECaLS filters below 32 degrees and BASS+MzLS above
    # https://www.legacysurvey.org/status/
    if survey == "LegacySurvey":
        if version == "BASS+MzLS":
            if filt == "z":
                filt_file = os.path.join(filters_path, f"MzLS_z.dat")
            else:
                filt_file = os.path.join(filters_path, f"BASS_{filt}.dat")       
        elif version == "DECam":
            filt_file = os.path.join(filters_path, f"DECAM_{filt}.dat")

    elif survey == "HST":
        if 'UVIS' in filt:
            # Usually UVIS2 is used, but there is no large difference
            filt = filt.replace('UVIS', 'UVIS2')
        hst_files = glob.glob(os.path.join(filters_path, '*/*'))
        filt_file = [file for file in hst_files if filt in file][0]
    else:
        filt_file = os.path.join(filters_path, f"{survey}_{filt}.dat")
    
    wave, transmission = np.loadtxt(filt_file).T

    return wave, transmission


def integrate_filter(
    spectrum_wave,
    spectrum_flux,
    filter_wave,
    filter_response,
    response_type="photon",
):
    """Calcultes the flux density of an SED given a filter response.

    Parameters
    ----------
    spectrum_wave : array
        Spectrum's wavelength range.
    spectrum_flux : array
        Spectrum's flux density distribution.
    filter_wave : array
        Filter's wavelength range.
    filter_response : array
        Filter's response function.
    response_type : str, default ``photon``
        Filter's response type. Either ``photon`` or ``energy``.

    Returns
    -------
    flux_filter : float
        Flux density.
    """
    if response_type == "energy":
        filter_response = filter_response.copy() / filter_wave

    interp_response = np.interp(
        spectrum_wave, filter_wave, filter_response, left=0.0, right=0.0
    )
    int1 = np.trapz(
        spectrum_flux * interp_response * spectrum_wave, spectrum_wave
    )
    int2 = np.trapz(filter_response * filter_wave, filter_wave)
    flux_filter = int1 / int2

    return flux_filter

def adapt_aperture(objects, img_wcs, img_wcs2, flip=False):
    """Changes the aperture parameters to consider differences
    in WCS between surveys.

    The values of ``center``, ``a`` and ``b`` should in pixel
    units. DES images are flipped, so these need to be corrected
    with ``theta -> -theta``, i.e. using ``flip=True``.

    Parameters
    ----------
    objects: ndarray
        Objects with apertures.
    img_wcs: ~wcs.WCS
        WCS of the image from where the objects were extracted.
    img_wcs2: ~wcs.WCS
        WCS used to adapt the apertures.
    flip: bool, default ``False``
        Whether to flip the orientation of the aperture. Only 
        used for DES images.

    Returns
    -------
    objects_: ndarray
        Objects with adapted apertures.
    """
    objects_ = objects.copy()  # avoid modifying the intial objects
    for obj in objects_:
        center = (obj['x'], obj['y'])
        apertures = EllipticalAperture(center, obj['a'],
                                       obj['b'], obj['theta'])
        sky_apertures = apertures.to_sky(img_wcs)

        new_apertures = sky_apertures.to_pixel(img_wcs2)
        new_center = new_apertures.positions
        obj['x'], obj['y'] = new_center
        obj['a'] = new_apertures.a
        obj['b'] = new_apertures.b
        obj['theta'] = new_apertures.theta

        if flip is True:
            # flip aperture orientation
            obj['theta'] *= -1

    return objects_

def check_work_dir(wokrdir):
    """Checks if the working directory exists. If it
    does not, one is created.

    Parameters
    ----------
    wokrdir: str
        Working directory path.
    """
    if not os.path.isdir(wokrdir):
        os.mkdir(wokrdir)


def clean_dir(directory):
    """Removes the directory if it is empty.

    Parameters
    ----------
    directory: str
        Directory path.
    """
    try:
        os.rmdir(directory)
    except:
        pass


def plot_fits(fits_file, ext=0):
    """Plots a fits file.

    Parameters
    ----------
    fits_file: str or HDU.
        Path to fits file.
    ext: int
        Extension index.
    """
    if isinstance(fits_file, str):
        hdu = fits.open(fits_file)
        title = os.path.splitext(os.path.basename(fits_file))[0]
    else:
        hdu = fits_file
        title = None
    header = hdu[ext].header
    data = hdu[ext].data

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        img_wcs = wcs.WCS(header, naxis=2)

    m, s = np.nanmean(data), np.nanstd(data)

    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(projection=img_wcs)
    ax.set_title(title, fontsize=18)
    update_axislabels(ax)

    im = ax.imshow(
        data,
        interpolation="nearest",
        cmap="gray",
        vmin=m - s,
        vmax=m + s,
        origin="lower",
    )
    
    plt.show()


def update_axislabels(ax):
    """Updates the labels and ticks of a plot.

    Parameters
    ----------
    ax: `.axes.SubplotBase`.
        The axis of a subplot.
    """
    for i in range(2):
        ax.coords[i].set_ticklabel(size=16)
        if i == 0:
            ax.coords[i].set_axislabel("RA (J2000)", fontsize=20)
        else:
            ax.coords[i].set_axislabel("Dec (J2000)", fontsize=20)
            # ax.coords[i].set_ticklabel(rotation=65)
