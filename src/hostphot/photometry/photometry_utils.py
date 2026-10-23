import sep
import warnings
import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes
from pathlib import Path
from typing import Optional

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area
from astropy.stats import sigma_clipped_stats
from photutils.utils import calc_total_error
from photutils.aperture import CircularAperture, EllipticalAperture, ApertureMask

import hostphot
from hostphot.surveys_utils import (
    check_survey_validity,
    check_filters_validity,
    survey_pixel_units,
    survey_pixel_scale,
    survey_zp,
)
from hostphot.photometry.image_utils import (
    get_image_exptime,
    get_image_gain,
)

hostphot_path = Path(hostphot.__path__[0])

# surveys without (inverse-)variance maps for which a blank-aperture noise floor is applied
# to the aperture uncertainties (accounts for correlated noise in the coadds, e.g. drizzled HST and
# JWST images: https://hst-docs.stsci.edu/drizzpac/chapter-3-description-of-the-drizzle-algorithm/3-3-weight-maps-and-correlated-noise)
blank_floor_surveys = [
    "PanSTARRS", "DES", "SDSS", "2MASS", "SkyMapper", "SPLUS", "VISTA", "UKIDSS", "Spitzer", "HST", "JWST"
]


def calc_sky_unc(image: np.ndarray, exptime: float) -> float:
    """Calculates the uncertainty of the image from the
    sky's standard deviation, sigma-clipped STD.

    Parameters
    ----------
    image: Image in a 2D numpy array.
    exptime: Exposure time of the image.

    Returns
    -------
    error: Estimated error of the image.
    """
    mask = image != 0
    avg, sky, sky_std = sigma_clipped_stats(image[mask], sigma=3.0)
    error = calc_total_error(image, sky_std, exptime)

    return error


def get_HST_err(filt: str, header: fits.Header) -> float:
    """Obtaines the error propagation from the zeropoint.

    Parameters
    ----------
    filt: HST filter, e.g. ``WFC3_UVIS_F225W``.
    header: Header of an image.

    Returns
    -------
    mag_err: Magnitude error on PHOTFLAM.
    """
    # split instrument and filter
    filt_split = filt.split("_")
    filt = filt_split[-1]
    instrument = filt_split[-2]

    if filt_split[0] == "ACS":
        # absolute flux calibration of ACS/WFC accurate to ~1% (Bohlin 2016, AJ, 152, 60)
        return 2.5 / np.log(10) * 0.01
    if filt_split[0].startswith("NICMOS"):
        # NICMOS photometric calibration accurate to ~5% (NICMOS Data Handbook; e.g. it improved
        # from 10-15% to ~5% after the first calibrations)
        return 2.5 / np.log(10) * 0.05

    if instrument == "UVIS":
        # chip from the aperture (e.g. UVIS1, UVIS1-FIX, UVIS2-C512C-SUB); the full-detector apertures
        # (e.g. UVIS, UVIS-CENTER, UVIS-FIX, UVIS-QUAD) span both chips, see
        # https://hst-docs.stsci.edu/wfc3ihb/chapter-6-uvis-imaging-with-wfc3/6-4-uvis-field-geometry
        # (the zeropoint uncertainties of both chips are the same)
        aperture = str(header.get("APERTURE", "")).strip().upper()
        instrument = "UVIS1" if aperture.startswith("UVIS1") else "UVIS2"

    # get uncertainty file
    err_file = hostphot_path.joinpath("filters", "HST", f"{instrument}_err.txt")
    err_df = pd.read_csv(err_file, sep="\\s+")
    filt_err_df = err_df[err_df.Filter == filt]
    if len(filt_err_df) == 0:
        # e.g. the WFC3/UVIS quad filters (FQ*) are not in the tables
        warnings.warn(
            f"HST {instrument} {filt}: no zeropoint uncertainty available, so 2% is assumed."
        )
        return 2.5 / np.log(10) * 0.02
    # error propagation
    flux = filt_err_df.PHOTFLAM.values[0]
    flux_err = filt_err_df.ERR_PHOTFLAM.values[0]
    mag_err = np.abs(2.5 * flux_err / (flux * np.log(10)))

    return mag_err


def uncertainty_calculation(
    flux: float,
    flux_err: float,
    survey: str,
    filt: Optional[str] = None,
    ap_area: float = 0.0,
    header: Optional[fits.Header] = None,
    bkg_rms: float = 0.0,
) -> float:
    """Calculates the uncertainty propagation.

    Parameters
    ----------
    flux: Aperture flux.
    flux_err: Aperture flux error.
    survey: Survey name: e.g. ``PanSTARRS``, ``GALEX``.
    filt: Survey-specific filter.
    ap_area: Aperture area.
    header: Header of an image.
    bkg_rms: Background noise.

    Returns
    -------
    flux_err: Total uncertainty in flux units.
    """
    exptime = get_image_exptime(header, survey)
    gain = get_image_gain(header, survey)

    # 1.0857 = 2.5/ln(10)
    mag_err = 2.5 / np.log(10) * flux_err / flux

    # PanSTARRS, DES, SkyMapper, SPLUS, VISTA, UKIDSS and Spitzer: the aperture error already includes the
    # background (with read noise) and source Poisson noise, with a blank-aperture floor for correlated noise
    # (calculated together with the flux, outside this function), so no extra term is added here

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

    elif survey == "PanSTARRS":
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
        mag_err = np.sqrt(mag_err**2 + floor_err**2)


    elif survey == "SDSS":
        # the Poisson noise of the source is included in the aperture error, using the effective
        # gain (see get_effective_gain()), while the dark current and read noise are included in
        # the background noise (and the blank-aperture floor)
        # calibration: ~1% in gri and ~2% in u and z (Padmanabhan et al. 2008;
        # https://www.sdss4.org/dr17/algorithms/fluxcal/), where the AB offsets of u and z
        # are also uncertain at the >0.01 mag level
        calib_err = 0.02 if filt in ["u", "z"] else 0.01
        mag_err = np.sqrt(mag_err**2 + calib_err**2)

    elif survey == "GALEX":
        # the Poisson noise of the source and the background noise are included in the aperture
        # error, so only the calibration uncertainty is added: 5% (FUV) and 2.7% (NUV), see
        # https://asd.gsfc.nasa.gov/archive/galex/FAQ/counts_background.html
        calib_frac = {"FUV": 0.050, "NUV": 0.027}[filt]
        calib_err = 2.5 / np.log(10) * calib_frac
        mag_err = np.sqrt(mag_err**2 + calib_err**2)

    elif survey == "2MASS":
        # see: https://wise2.ipac.caltech.edu/staff/jarrett/2mass/3chan/noise/
        S = flux
        N_c = 6  # number of coadd pixels
        k_z = 1.7  # kernel smoothing factor
        gain = 10  # typical gain in e-/dn
        # number of coadd (1 arcsec) pixels in the aperture, i.e. the aperture area
        # (the number of frame pixels, of 2 arcsec, is a quarter of it)
        n_c = ap_area
        sigma_c = bkg_rms  # coadd noise; assumed to be ~background noise

        # (absolute values, so that negative fluxes do not reduce the noise or give a smaller error)
        SNR = S / np.sqrt(
            np.abs(S) / (gain * N_c)
            + 4 * n_c * k_z ** 2 * sigma_c ** 2
        )
        # the aperture error includes the blank-aperture floor, which is kept if larger
        mag_err = max(np.abs(1.0857 / SNR), np.abs(mag_err))
        # global photometric calibration accuracy of ~1-2% (2MASS Explanatory Supplement)
        calib_err = 0.01
        mag_err = np.sqrt(mag_err**2 + calib_err**2)


    elif survey == "WISE":
        # Eq. 2 of https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec2_3f.html:
        # sigma_src^2 = f_apcor^2 * F_corr * (sum_i^N_A sigma_i^2 + k * N_A^2 / N_B * sigma_B^2) + sigma_conf^2
        # correction assumed to be 0 mags as PSF fitting is not used.
        m_apcor = 0.0
        f_apcor = 10 ** (-0.4 * m_apcor)
        F_src = f_apcor * flux

        # correlated-noise correction factor: F_corr ~ N_p * (s_in / s_out)^2 (Section II.3.f.i).
        # Atlas Images are interpolated with the PRF, with N_p the effective number of
        # noise-pixels of the PRF in native detector pixels (Table 1 of Section IV.6.c.i:
        # https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_6ci.html)
        N_p = {"W1": 13.772, "W2": 17.636, "W3": 35.476, "W4": 24.462}[filt]
        # ratio of input (detector) pixel scale to output (Atlas Image, 1.375") pixel scale
        Sin_Sout = {"W1": 2, "W2": 2, "W3": 2, "W4": 4}[filt]
        F_corr = N_p * Sin_Sout**2
        N_A = ap_area  # number of pixels in the aperture
        # the variance of the sum can not exceed that of fully correlated pixels (small apertures)
        F_corr = min(F_corr, max(N_A, 1.0))

        k = np.pi / 2  # sep uses a median-related background estimate
        # the background is estimated from the whole image
        N_B = header["NAXIS1"] * header["NAXIS2"] if "NAXIS1" in header else 10 * N_A
        sigma_i = bkg_rms  # per-pixel noise; assumed background dominated
        sigma_src = np.sqrt(
            f_apcor**2 * F_corr * (N_A * sigma_i**2 + k * N_A**2 / N_B * bkg_rms**2)
        )  # confusion noise (sigma_conf) not included

        mag_err = np.sqrt(1.179 * sigma_src**2 / F_src**2)

        # add uncertainty from the ZP
        zp_unc = header["MAGZPUNC"]
        mag_err = np.sqrt(mag_err**2 + zp_unc**2)

    elif survey == "unWISE":
        # flux uncertainties from the inverse-variance maps (Lang 2014, AJ, 147, 108), with a
        # blank-aperture floor, are calculated together with the flux (outside this function)
        # ZP uncertainties are the same as for the WISE Atlas Images (Table 1 of
        # https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec2_3f.html)
        unc_dict = {"W1": 0.006, "W2": 0.007, "W3": 0.015, "W4": 0.012}
        zp_unc = unc_dict[filt]
        mag_err = np.sqrt(mag_err**2 + zp_unc**2)

    elif survey == "LegacySurvey":
        # photometry uncertainties for DR10 from https://ui.adsabs.harvard.edu/abs/2023RNAAS...7..105Z/abstract
        # LS also includes uncertainties from inverse-variance maps, calculated together with the flux
        # (outside this function), as mentioned by Dustin Lang
        unc_dict = {
            "g": 5.0e-3,
            "r": 3.9e-3,
            "i": 4.3e-3,
            "z": 5.5e-3,
        }
        extra_err = unc_dict[filt]
        mag_err = np.sqrt(mag_err**2 + extra_err**2)

    elif survey == "Spitzer":
        # absolute calibration uncertainty (in flux): 3% for IRAC (Section 4.3 of the IRAC Instrument Handbook:
        # https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/iracinstrumenthandbook/14/) and 4% for MIPS 24 micron
        # (https://irsa.ipac.caltech.edu/data/SPITZER/docs/mips/mipsinstrumenthandbook/42/)
        flux_unc = 0.04 if "MIPS" in filt else 0.03
        zp_unc = 2.5 / np.log(10) * flux_unc
        mag_err = np.sqrt(mag_err**2 + zp_unc**2)
    elif survey in ["VISTA", "UKIDSS"]:
        # uncertainty of the ZP of the image (MAGZRR) and systematic uncertainty of the calibration
        # (from 2MASS): 1.5% in JHK and 2% in ZY for UKIDSS (Hodgkin et al. 2009, MNRAS, 394, 675) and
        # 2% in YJHKs and 3% in Z for VISTA (Gonzalez-Fernandez et al. 2018, MNRAS, 474, 5459)
        if survey == "UKIDSS":
            calib_frac = 0.02 if filt in ["Z", "Y"] else 0.015
        else:
            calib_frac = 0.03 if filt == "Z" else 0.02
        calib_err = 2.5 / np.log(10) * calib_frac
        zp_unc = header["MAGZRR"]
        mag_err = np.sqrt(mag_err**2 + zp_unc**2 + calib_err**2)

    elif survey == "HST":
        zp_unc = get_HST_err(filt, header)
        mag_err = np.sqrt(mag_err**2 + zp_unc**2)

    elif survey == "SkyMapper":
        zp_unc = header["ZPTERR"]
        mag_err = np.sqrt(mag_err**2 + zp_unc**2)

    elif survey == "SPLUS":
        # zero-point uncertainties from Section 4.4 of Almeida-Fernandes et al. (2022):
        # 25 mmag (u and J0395), 15 mmag (J0378) and 10 mmag (all other filters)
        zp_uncs = {
            "u": 25e-3,
            "F395": 25e-3,
            "F378": 15e-3,
        }
        zp_unc = zp_uncs.get(filt, 10e-3)
        mag_err = np.sqrt(mag_err**2 + zp_unc**2)

    elif survey == "JWST":
        # see https://jwst-docs.stsci.edu/jwst-calibration-status/nircam-calibration-status/nircam-imaging-calibration-status#NIRCamImagingCalibrationStatus-Photometriccalibration
        # absolute flux calibration uncertainties < 1%, but set to 1% error (in flux) for now
        zp_unc = 2.5 / np.log(10) * 0.01  # ~0.011 mag
        mag_err = np.sqrt(mag_err**2 + zp_unc**2)

    elif survey == "HSC":
        # flux uncertainties from the variance maps are calculated together with the flux
        # (outside this function); ~1% photometric calibration uncertainty assumed
        # (see Aihara et al. 2022, PASJ, 74, 247)
        zp_unc = 0.01
        mag_err = np.sqrt(mag_err**2 + zp_unc**2)

    else:
        raise Exception(f"Survey {survey} has not been added for error propagation.")

    flux_err = np.abs(flux * 0.4 * np.log(10) * mag_err)

    return flux_err


def irac_extended_source_correction(filt: str, radius: float) -> tuple[float, float]:
    """Extended-source aperture correction for IRAC photometry.

    The IRAC calibration is tied to point sources, so the photometry of extended sources
    (e.g. galaxies) needs to be corrected for the light scattered across the IRAC focal plane
    (Section 8.2.2 of the IRAC Instrument Handbook:
    https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/iracinstrumenthandbook/46/):
    ``true flux = measured flux * (A * exp(-radius**B) + C)``. The correction should not be used
    for small radii (< 8 arcsec), so it is not applied in that case. The corrections are good
    to 10%, which is propagated as an uncertainty of 10% of the size of the correction.

    Parameters
    ----------
    filt: IRAC filter, e.g. ``IRAC.1``.
    radius: (Effective) aperture radius in arcsec.

    Returns
    -------
    correction: Correction factor (multiplicative, in flux).
    correction_err: Uncertainty of the correction factor.
    """
    # Table 8.1 of the IRAC Instrument Handbook
    coefficients = {
        "IRAC.1": (0.82, 0.370, 0.910),
        "IRAC.2": (1.16, 0.433, 0.94),
        "IRAC.3": (1.49, 0.207, 0.66),
        "IRAC.4": (1.37, 0.330, 0.740),
    }
    if filt not in coefficients or radius < 8.0:
        return 1.0, 0.0
    A, B, C = coefficients[filt]
    correction = A * np.exp(-(radius**B)) + C
    correction_err = 0.1 * np.abs(1 - correction)
    return correction, correction_err


def sdss_gain(filt: str, camcol: int, run: int) -> float:
    """Returns the gain (e-/DN) of an SDSS frame.

    See: https://data.sdss.org/datamodel/files/BOSS_PHOTOOBJ/frames/RERUN/RUN/CAMCOL/frame.html
    """
    gain_dict = {
        "u": {1: 1.62, 2: 1.595, 3: 1.59, 4: 1.6, 5: 1.47, 6: 2.17},
        "g": {1: 3.32, 2: 3.855, 3: 3.845, 4: 3.995, 5: 4.05, 6: 4.035},
        "r": {1: 4.71, 2: 4.6, 3: 4.72, 4: 4.76, 5: 4.725, 6: 4.895},
        "i": {1: 5.165, 2: 6.565, 3: 4.86, 4: 4.885, 5: 4.64, 6: 4.76},
        "z": {1: 4.745, 2: 5.155, 3: 4.885, 4: 4.775, 5: 3.48, 6: 4.69},
    }
    gain = gain_dict[filt][camcol]
    if filt == "u" and camcol == 2 and run > 1100:
        gain = 1.825
    return gain


def get_effective_gain(header: fits.Header, survey: str, filt: str) -> float:
    """Returns the effective gain used for the Poisson noise of the sources
    (i.e. the factor converting the image units into electrons).

    Parameters
    ----------
    header: Header of the image.
    survey: Survey name.
    filt: Filter name.

    Returns
    -------
    effective_gain: Effective gain.
    """
    if survey == "SDSS" and len(filt) == 1 and "NMGY" in header:
        # images in nanomaggies: gain [e-/DN] / NMGY [nanomaggies/DN]
        return sdss_gain(filt, int(header["CAMCOL"]), int(header["RUN"])) / float(header["NMGY"])
    try:
        pixel_units = survey_pixel_units(survey, filt)
    except Exception:
        # probably a coadd, so only need one filter
        pixel_units = survey_pixel_units(survey, filt[0])
    if pixel_units == "counts":
        # note: for images in counts (ADU), e.g. 2MASS, VISTA and UKIDSS, the detector gain and the
        # number of combined exposures are not included, so the Poisson noise of the sources is
        # overestimated (conservative; e.g. <= 0.003 mag for VISTA stars, below the calibration terms)
        return 1
    if survey == "Spitzer" and float(header.get("GAIN", 0)) > 0:
        # SEIP mosaics (MJy/sr): mean gain in e-/(MJy/sr), from the exposure time and coverage
        # (GAIN = EXPGAIN * EXPTIME / EFCONV; see the SEIP Explanatory Supplement)
        return float(header["GAIN"])
    if survey == "JWST" and float(header.get("EFFGAIN", 0)) > 0:
        # images in MJy/sr: effective gain in e-/(MJy/sr), from the Poisson variance (see update_JWST_header())
        return float(header["EFFGAIN"])
    if survey == "HST" and "ELECTRON" not in str(header.get("BUNIT", "ELECTRONS/S")).upper():
        # images in counts (DN) per second, e.g. NICMOS: gain (e-/DN) times the exposure time
        return get_image_exptime(header, survey) * float(header.get("ADCGAIN", header.get("CCDGAIN", 1.0)))
    return get_image_exptime(header, survey)


def magnitude_calculation(
    flux: float,
    flux_err: float,
    survey: str,
    filt: str = None,
    ap_area: float = 0.0,
    header: Optional[fits.Header] = None,
    bkg_rms: float = 0.0,
) -> tuple[float, float, float, float, float]:
    """Calculates the calibrated magnitudes and errors.

    Error propagation is included here, both for magnitudes
    and fluxes.

    Parameters
    ----------
    flux: Aperture flux.
    survey: Survey name: e.g. ``PanSTARRS``, ``GALEX``.
    filt: Survey-specific filter.
    ap_area: Aperture area.
    header: Header of an image.
    bkg_rms: Background noise.

    Returns
    -------
    mag: Apparent magnitude.
    mag_err: Apparent magnitude uncertainty.
    flux: Total flux.
    flux_err: Total uncertainty in flux.
    zp: Zeropoint.
    """
    # get zeropoint
    zp = survey_zp(survey, filt)
    if zp == "header":
        zp = header["MAGZP"]

    if survey == "SDSS":
        # SDSS zero-points are not exactly in AB:
        # https://www.sdss4.org/dr12/algorithms/fluxcal/#SDSStoAB
        if filt == "u":
            offset = -0.04  # mag
        elif filt == "z":
            offset = 0.02  # mag
        else:
            offset = 0
        flux *= 10 ** (-0.4 * offset)
        flux_err *= 10 ** (-0.4 * offset)

    if survey == "unWISE":
        # To improve the agreement between unWISE and AllWISE fluxes,
        # these offsets need to be applied: https://catalog.unwise.me/catalogs.html#absolute
        if filt == "W1":
            offset = -4e-3  # 4 mmag
        elif filt == "W2":
            offset = -32e-3  # 32 mmag
        else:
            offset = 0
        flux *= 10 ** (-0.4 * offset)
        flux_err *= 10 ** (-0.4 * offset)

    # extended-source aperture correction for IRAC, using the effective circular radius of the aperture
    ap_corr, ap_corr_err = 1.0, 0.0
    if survey == "Spitzer" and ap_area > 0:
        radius_arcsec = np.sqrt(ap_area / np.pi) * survey_pixel_scale(survey, filt)
        ap_corr, ap_corr_err = irac_extended_source_correction(filt, radius_arcsec)
        flux *= ap_corr
        flux_err *= ap_corr

    # note: no (point-source) aperture correction is applied to HST images, as HostPhot targets
    # extended sources (galaxies), for which the light scattered by the PSF out of the aperture
    # is largely compensated by the light scattered into it, and the HST PSF is much smaller
    # than the usual apertures

    # error propagation
    flux_err = uncertainty_calculation(
        flux,
        flux_err,
        survey,
        filt,
        ap_area,
        header,
        bkg_rms,
    )
    if ap_corr_err > 0:
        # uncertainty of the IRAC extended-source correction
        flux_err = np.sqrt(flux_err**2 + (flux * ap_corr_err / ap_corr) ** 2)

    pixel_units = survey_pixel_units(survey, filt)
    if pixel_units == "counts" and survey not in ["2MASS", "VISTA", "UKIDSS"]:
        # flux needs to be in units of counts per second
        # but only after the error propagation
        exptime = get_image_exptime(header, survey)
        flux /= exptime
        flux_err /= exptime
    elif survey in ["VISTA", "UKIDSS"]:
        # VISTA and UKIDSS images are always in counts (not counts/second)
        # regardless of the pixel_units config setting
        exptime = get_image_exptime(header, survey)
        flux /= exptime
        flux_err /= exptime

    mag = -2.5 * np.log10(flux) + zp
    mag_err = np.abs(2.5 * flux_err / (flux * np.log(10)))

    return mag, mag_err, flux, flux_err, zp


def extract_filter(
    filt: str, survey: str, version: Optional[str] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Extracts the transmission function for the filter.

    Parameters
    ----------
    filt: Filter to extract.
    survey: Survey of the filters.
    version: Version of the filters to use. E.g. for the
        Legacy Survey as it uses DECam for the south
        and BASS+MzLS for the north.

    Returns
    -------
    wave: Wavelength range of the filter.
    transmission: Transmission function.
    """
    check_survey_validity(survey)
    if survey in ["HST", "JWST"]:
        check_filters_validity(filt, survey)
    else:
        check_filters_validity([filt], survey)

    if "WISE" in survey:
        survey = "WISE"  # for unWISE to use the same filters as WISE

    filters_path = hostphot_path.joinpath("filters", survey)

    # Assume DECaLS filters below 32 degrees and BASS+MzLS above
    # https://www.legacysurvey.org/status/
    if survey == "LegacySurvey":
        if version is None:
            version = "DECam"  # covers most of the footprint (BASS+MzLS only in the north)
        if version == "BASS+MzLS":
            if filt == "z":
                filt_file = filters_path / "MzLS_z.dat"
            else:
                filt_file = filters_path / f"BASS_{filt}.dat"
        elif version == "DECam":
            filt_file = filters_path / f"DECAM_{filt}.dat"
    elif survey == "HST":
        detector = filt.split("_")[1]
        if detector == "UVIS":
            # Usually UVIS2 is used, but there is no large difference
            filt_ = filt.replace("UVIS", "UVIS2")
        else:
            filt_ = filt
        filt_file = [file for file in filters_path.rglob(f"{filt_}.dat")][0]  # exact name
    elif survey == "JWST":
        # the transmission functions are in a directory per instrument (e.g. NIRCam/NIRCam_F150W.dat)
        filt_file = filters_path / filt.split("_")[0] / f"{filt}.dat"
    else:
        filt_file = filters_path / f"{survey}_{filt}.dat"
    wave, transmission = np.loadtxt(filt_file).T

    return wave, transmission


def integrate_filter(
    spectrum_wave: np.ndarray,
    spectrum_flux: np.ndarray,
    filter_wave: np.ndarray,
    filter_response: np.ndarray,
    response_type: str = "photon",
) -> float:
    """Calcultes the flux density of an SED given a filter response.

    Parameters
    ----------
    spectrum_wave: Spectrum's wavelength range.
    spectrum_flux: Spectrum's flux density distribution.
    filter_wave: Filter's wavelength range.
    filter_response: Filter's response function.
    response_type: Filter's response type. Either ``photon`` or ``energy``.

    Returns
    -------
    flux_filter: Flux density.
    """
    assert response_type in ["energy", "photon"], "Not a valid response type!"
    if response_type == "energy":
        filter_response = filter_response.copy() / filter_wave

    interp_response = np.interp(
        spectrum_wave, filter_wave, filter_response, left=0.0, right=0.0
    )
    int1 = np.trapezoid(spectrum_flux * interp_response * spectrum_wave, spectrum_wave)
    int2 = np.trapezoid(filter_response * filter_wave, filter_wave)
    flux_filter = int1 / int2

    return flux_filter


# Aperture photometry with inverse-variance maps and a blank-aperture noise floor
# (used for surveys providing (inverse-)variance maps, e.g. Legacy Survey, HSC and unWISE)
def get_invvar_map(hdu: fits.HDUList, survey: str) -> np.ndarray | None:
    """Returns the inverse-variance map of an image, if available.

    Parameters
    ----------
    hdu: Image HDU list. The (inverse-)variance map is the first extension.
    survey: Survey name: e.g. ``LegacySurvey``, ``HSC``.

    Returns
    -------
    invvar: Inverse-variance map or ``None`` if not available for this survey/image.
    """
    if len(hdu) < 2 or hdu[1].data is None:
        return None
    if survey in ["LegacySurvey", "unWISE"]:
        return hdu[1].data
    elif survey == "HSC":
        # HSC provides a variance map (sigma^2); invert to get inverse-variance
        if "VARIANCE" not in hdu:
            return None
        var = hdu["VARIANCE"].data
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(var > 0, 1.0 / var, 0.0)
    return None

def _normalized_theta(theta: float) -> float:
    """Wrap ellipse position angle into the range expected by photutils/sep."""
    return (theta + np.pi / 2) % np.pi - np.pi / 2


def _aperture_weighted_sum(
    data: np.ndarray, px: float, py: float, radius: float
) -> tuple[float, np.ndarray, ApertureMask]:
    """Return the exact-weight circular aperture sum."""
    aperture = CircularAperture((px, py), r=radius)
    mask = aperture.to_mask(method="exact")
    cutout = mask.cutout(data, fill_value=np.nan)
    if cutout is None:
        raise RuntimeError("aperture cutout failed")

    weights = mask.data
    valid = np.isfinite(cutout) & np.isfinite(weights)
    if not np.any(valid):
        raise RuntimeError("no valid pixels in aperture")

    flux = float(np.nansum(cutout[valid] * weights[valid]))
    return flux, weights, mask


def _elliptical_weighted_sum(
    data: np.ndarray, x: float, y: float, a: float, b: float, theta: float
) -> tuple[float, np.ndarray, ApertureMask]:
    """Return the exact-weight elliptical (e.g. Kron) aperture sum."""
    aperture = EllipticalAperture((x, y), a, b, theta)
    mask = aperture.to_mask(method="exact")
    cutout = mask.cutout(data, fill_value=np.nan)
    if cutout is None:
        raise RuntimeError("aperture cutout failed")

    weights = mask.data
    valid = np.isfinite(cutout) & np.isfinite(weights)
    if not np.any(valid):
        raise RuntimeError("no valid pixels in aperture")

    flux = float(np.nansum(cutout[valid] * weights[valid]))
    return flux, weights, mask


def _aperture_invvar_variance(
    invvar: np.ndarray, weights: np.ndarray, mask: ApertureMask
) -> float:
    """Propagate inverse-variance maps into aperture variance."""
    # Use mask.cutout to ensure shapes match even if aperture is partially off-image
    cutout = mask.cutout(invvar, fill_value=np.nan)
    valid = np.isfinite(cutout) & np.isfinite(weights) & (cutout > 0)
    if not np.any(valid):
        raise RuntimeError("no valid invvar pixels in aperture")
    return float(np.nansum((weights[valid] ** 2) / cutout[valid]))


def _source_mask(
    data: np.ndarray, pixel_scale: float = 1.0, host_xy: Optional[tuple[float, float]] = None
) -> np.ndarray:
    """Return a boolean mask of the pixels covered by detected sources (incl. the host).

    The same mask as for the background estimation is used (see
    :func:`hostphot.processing.background.source_mask`): sources detected at 3 sigma (a lower
    threshold also masks positive noise fluctuations, biasing the blank-aperture noise low),
    dilated by the FWHM, and an enlarged ellipse around the host galaxy (if its position is given).
    """
    from hostphot.processing.background import source_mask

    valid = np.isfinite(data)
    data = np.asarray(data, dtype=np.float64)
    try:
        # if the (enlarged) host ellipse covers (almost) the whole image, smaller ellipses are used,
        # so that blank apertures can still be placed
        for host_factor in [2.0, 1.5, 1.0, None]:
            if host_factor is None or host_xy is None:
                mask, info = source_mask(data, pixel_scale)
                break
            mask, info = source_mask(data, pixel_scale, host_xy=host_xy, host_factor=host_factor)
            if info["masked_frac"] <= 0.9:
                break
    except Exception:
        mask = np.zeros(data.shape, dtype=bool)
    # fill holes inside sources (e.g. faint regions between spiral arms), which are not blank sky
    mask = binary_fill_holes(mask)
    return mask | ~valid


def get_source_mask(
    data: np.ndarray, pixel_scale: float = 1.0, host_xy: Optional[tuple[float, float]] = None
) -> np.ndarray:
    """Returns the mask of the detected sources used for the blank-aperture noise floor.

    It can be calculated once per image and given to :func:`blank_aperture_floor` and
    :func:`extract_invvar_aperture_flux` for several apertures.

    Parameters
    ----------
    data: Image data.
    pixel_scale: Pixel scale in arcsec (used for the default FWHM and the host-galaxy detection).
    host_xy: Pixel coordinates of the host galaxy, which is masked with an enlarged ellipse.
    """
    return _source_mask(data, pixel_scale, host_xy)


def _background_residuals(
    data: np.ndarray, source_mask: np.ndarray, mesh_size: int = 64
) -> np.ndarray:
    """Return the data minus a smooth background estimated with the sources masked.

    The background mesh size should be larger than the aperture, so that only structure on
    larger scales (e.g. faint wings of the galaxy) is removed, and not the noise itself.
    """
    filled = np.ascontiguousarray(np.where(np.isfinite(data), data, 0.0), dtype=np.float64)
    try:
        bkg = sep.Background(filled, mask=source_mask, bw=mesh_size, bh=mesh_size)
        return np.where(np.isfinite(data), filled - bkg.back(), np.nan)
    except Exception:
        return data


def _blank_rms(
    data: np.ndarray,
    x0: float,
    y0: float,
    make_aperture,
    half_size: float,
    rmin: float,
    rmax: float,
    n_apertures: int = 80,
    source_mask: Optional[np.ndarray] = None,
) -> float | None:
    """Estimate an empirical noise floor from the scatter of blank (source-free) apertures.

    Apertures overlapping detected sources (including the host galaxy) are rejected, and the
    fluxes are measured on the residuals from a smooth background (estimated with the sources
    masked), so that faint, undetected wings of the galaxy do not contribute to the scatter.
    The apertures are first drawn from an annulus around the source; if not enough fit there,
    they are drawn from the whole image.

    Parameters
    ----------
    data: Image data.
    x0, y0: Centre of the source aperture.
    make_aperture: Function of (x, y) returning the (photutils) aperture at that position.
    half_size: Half-size of the aperture (for the image-edge check).
    rmin, rmax: Inner and outer radii of the annulus where apertures are drawn first.
    n_apertures: Number of blank apertures.
    source_mask: Mask of the detected sources (see :func:`get_source_mask`). It is calculated if not given.

    Returns
    -------
    rms: Robust RMS of the blank-aperture fluxes, or ``None`` if fewer than 10 apertures fit.
    """
    rng = np.random.default_rng(12345)
    ny, nx = data.shape
    if source_mask is None:
        source_mask = _source_mask(data, host_xy=(x0, y0))
    # background mesh of at least 4 times the aperture size
    mesh_size = int(max(16, 4 * half_size))
    residuals = _background_residuals(data, source_mask, mesh_size)
    max_trials = max(400, n_apertures * 20)
    blank_fluxes = []

    # the aperture weights are computed once (centred on a pixel) and shifted by whole pixels
    weights = make_aperture(0.0, 0.0).to_mask(method="exact")
    template, bbox = weights.data, weights.bbox
    footprint = template > 0
    for region in ["annulus", "image"]:
        for _ in range(max_trials):
            if len(blank_fluxes) >= n_apertures:
                break
            if region == "annulus":
                phi = rng.uniform(0, 2 * np.pi)
                rho = rng.uniform(rmin, rmax)
                x, y = x0 + rho * np.cos(phi), y0 + rho * np.sin(phi)
            else:
                x, y = rng.uniform(0, nx), rng.uniform(0, ny)
            ix, iy = int(round(x)), int(round(y))
            xmin, ymin = bbox.ixmin + ix, bbox.iymin + iy
            xmax, ymax = bbox.ixmax + ix, bbox.iymax + iy
            if xmin < 0 or ymin < 0 or xmax > nx or ymax > ny:
                continue  # aperture not fully inside the image
            if np.any(source_mask[ymin:ymax, xmin:xmax][footprint]):
                continue  # aperture overlaps a source
            cutout = residuals[ymin:ymax, xmin:xmax]
            if not np.all(np.isfinite(cutout[footprint])):
                continue
            blank_fluxes.append(float(np.sum(cutout[footprint] * template[footprint])))
        if len(blank_fluxes) >= 10:
            break

    if len(blank_fluxes) < 10:
        return None

    arr = np.asarray(blank_fluxes, dtype=float)
    med = np.nanmedian(arr)
    return float(1.4826 * np.nanmedian(np.abs(arr - med)))


def _blank_aperture_rms(
    data: np.ndarray,
    px: float,
    py: float,
    radius: float,
    n_apertures: int = 80,
    rmin_factor: float = 2.5,
    rmax_factor: float = 6.0,
    source_mask: Optional[np.ndarray] = None,
) -> float | None:
    """Estimate an empirical noise floor from blank circular apertures (see :func:`_blank_rms`)."""
    return _blank_rms(
        data,
        px,
        py,
        lambda x, y: CircularAperture((x, y), r=radius),
        radius,
        rmin_factor * radius,
        rmax_factor * radius,
        n_apertures,
        source_mask,
    )


def _blank_elliptical_rms(
    data: np.ndarray,
    x0: float,
    y0: float,
    a: float,
    b: float,
    theta: float,
    n_apertures: int = 80,
    rmin_factor: float = 2.5,
    rmax_factor: float = 6.0,
    source_mask: Optional[np.ndarray] = None,
) -> float | None:
    """Estimate an empirical noise floor from blank elliptical apertures (see :func:`_blank_rms`)."""
    step_scale = max(a, b)
    return _blank_rms(
        data,
        x0,
        y0,
        lambda x, y: EllipticalAperture((x, y), a, b, theta=theta),
        step_scale,
        rmin_factor * step_scale,
        rmax_factor * step_scale,
        n_apertures,
        source_mask,
    )


def _extrapolated_blank_rms(
    data: np.ndarray,
    x0: float,
    y0: float,
    area: float,
    source_mask: np.ndarray,
    fractions: tuple = (1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2),
) -> float | None:
    """Estimate the blank-aperture noise of apertures too large to fit in the image.

    The noise of smaller blank circular apertures (with ``fractions`` of the given area) is
    fitted as :math:`\\sigma \\propto A^\\beta` (in log space), with :math:`\\beta` between 0.5
    (uncorrelated noise) and 1 (fully correlated), and extrapolated to the given area.

    Parameters
    ----------
    data: Image data.
    x0, y0: Centre of the source aperture.
    area: Area of the source aperture in pixels.
    source_mask: Mask of the detected sources (see :func:`get_source_mask`).
    fractions: Areas of the smaller apertures, as fractions of ``area``.

    Returns
    -------
    rms: Extrapolated noise, or ``None`` if fewer than three smaller apertures could be measured.
    """
    target_radius = np.sqrt(area / np.pi)
    points = []
    for fraction in fractions:
        radius = target_radius * np.sqrt(fraction)
        if radius < 1.5:
            continue
        rms = _blank_aperture_rms(data, x0, y0, radius, source_mask=source_mask)
        if rms is not None and rms > 0:
            points.append((np.pi * radius**2, rms))
    if len(points) < 3:
        return None
    log_area, log_rms = np.log(np.array(points)).T
    beta = float(np.clip(np.polyfit(log_area, log_rms, 1)[0], 0.5, 1.0))
    intercept = np.median(log_rms - beta * log_area)
    return float(np.exp(intercept + beta * np.log(area)))


def aperture_missing_fraction(data: np.ndarray, aperture) -> float:
    """Returns the fraction of an aperture with no data (non-finite pixels or outside the image).

    Parameters
    ----------
    data: Image data.
    aperture: Photutils aperture (e.g. circular or elliptical).

    Returns
    -------
    missing_fraction: Fraction of the aperture area (weighted by the exact pixel overlap) without data.
    """
    mask = aperture.to_mask(method="exact")
    cutout = mask.cutout(data, fill_value=np.nan)
    if cutout is None:
        return 1.0  # aperture completely outside the image
    weights = mask.data
    total = np.sum(weights)
    if total <= 0:
        return 1.0
    return float(np.sum(weights[~np.isfinite(cutout)]) / total)


def elliptical_missing_flux_fraction(data: np.ndarray, aperture, nbins: int = 10) -> float:
    """Estimates the fraction of the flux of an elliptical aperture (centred on a galaxy) in pixels without data.

    The flux of each pixel without data is estimated from the mean surface brightness of the valid pixels
    at the same elliptical radius (in ``nbins`` rings), assuming that the galaxy is roughly symmetric. If a
    ring has no valid pixels, the next ring inwards with data is used (conservative).

    Parameters
    ----------
    data: Image data.
    aperture: Photutils elliptical aperture.
    nbins: Number of elliptical rings.

    Returns
    -------
    missing_fraction: Estimated fraction of the flux in pixels without data.
    """
    mask = aperture.to_mask(method="exact")
    cutout = mask.cutout(data, fill_value=np.nan)
    if cutout is None:
        return 1.0
    weights = mask.data
    yy, xx = np.indices(weights.shape)
    xc, yc = aperture.positions
    dx, dy = xx + mask.bbox.ixmin - xc, yy + mask.bbox.iymin - yc
    theta = aperture.theta.to_value("rad") if hasattr(aperture.theta, "unit") else float(aperture.theta)
    u = dx * np.cos(theta) + dy * np.sin(theta)
    v = -dx * np.sin(theta) + dy * np.cos(theta)
    rings = np.minimum((np.hypot(u / aperture.a, v / aperture.b) * nbins).astype(int), nbins - 1)
    inside = weights > 0
    missing = inside & ~np.isfinite(cutout)
    valid = inside & np.isfinite(cutout)
    measured = np.sum((cutout * weights)[valid])
    estimated = 0.0
    for ring in range(nbins):
        ring_missing = missing & (rings == ring)
        if not ring_missing.any():
            continue
        brightness = np.nan
        for inner in range(ring, -1, -1):
            ring_valid = valid & (rings == inner)
            if ring_valid.sum() > 10:
                brightness = np.mean(cutout[ring_valid])
                break
        if not np.isfinite(brightness):
            return 1.0
        estimated += brightness * np.sum(weights[ring_missing])
    total = measured + estimated
    if total <= 0:
        # no positive flux to compare with: fall back to the missing area
        return aperture_missing_fraction(data, aperture)
    return float(np.clip(estimated / total, 0, 1))


def check_saturation(flag_mask: Optional[np.ndarray], aperture, label: str) -> bool:
    """Checks whether an aperture contains saturated pixels (without measured data, see
    :func:`hostphot.processing.cleaning.saturated_mask`), with a warning if it does. As these are
    the brightest pixels of the source, the flux cannot be measured (it would be underestimated).

    Parameters
    ----------
    flag_mask: Mask of the saturated pixels (``None`` if not available).
    aperture: Photutils aperture (e.g. circular or elliptical).
    label: Description of the measurement, used in the warnings (e.g. survey, filter and aperture).

    Returns
    -------
    usable: Whether the flux can be measured (no saturated pixels within the aperture).
    """
    if flag_mask is None:
        return True
    mask = aperture.to_mask(method="center")
    cutout = mask.cutout(flag_mask.astype(float), fill_value=0.0)
    if cutout is None:
        return True
    n_saturated = int(np.sum(cutout[mask.data > 0] > 0))
    if n_saturated == 0:
        return True
    warnings.warn(
        f"{label}: the aperture contains {n_saturated} pixels without measured data (e.g. saturated in "
        "all the exposures), so the flux would be underestimated; returning NaN."
    )
    return False


def check_aperture_coverage(
    data: np.ndarray, aperture, label: str, tolerance: float = 0.05, flux_weighted: bool = False
) -> bool:
    """Checks whether an aperture has enough data to measure its flux.

    If part of the aperture has no data (e.g. image edge or chip gap), a warning is raised. The flux
    can still be measured from the remaining pixels if the missing fraction is within the tolerance.

    Parameters
    ----------
    data: Image data.
    aperture: Photutils aperture (e.g. circular or elliptical).
    label: Description of the measurement, used in the warnings (e.g. survey, filter and aperture).
    tolerance: Maximum fraction of the aperture without data.
    flux_weighted: If ``True``, the tolerance applies to the estimated fraction of the flux without data
        (see :func:`elliptical_missing_flux_fraction`; for elliptical apertures centred on a galaxy, e.g.
        Kron apertures, whose outer parts contain little flux) instead of the fraction of the area.

    Returns
    -------
    usable: Whether the flux can be measured.
    """
    missing_area = aperture_missing_fraction(data, aperture)
    if missing_area == 0:
        return True
    area_msg = f"{100 * missing_area:.1f}% of the aperture has no data (NaN pixels or outside the image)"
    if flux_weighted:
        missing = elliptical_missing_flux_fraction(data, aperture)
        msg = f"{area_msg}, containing an estimated {100 * missing:.1f}% of the flux"
    else:
        missing = missing_area
        msg = area_msg
    if missing <= tolerance:
        warnings.warn(
            f"{label}: {msg}; the flux is measured from the remaining pixels and may be underestimated "
            f"by up to ~{100 * missing:.1f}%."
        )
        return True
    warnings.warn(f"{label}: {msg}, more than the {100 * tolerance:.0f}% tolerance; returning NaN.")
    return False


def _kron_ellipse(objects: np.ndarray, kronrad: float, scale: float) -> tuple[float, float, float, float, float]:
    """Return the (x, y, a, b, theta) parameters of the (scaled) Kron ellipse."""
    # theta must be in the range [-pi/2, pi/2] for sep.sum_ellipse()
    theta = _normalized_theta(float(np.ravel(objects["theta"])[0]))
    x = float(np.ravel(objects["x"])[0])
    y = float(np.ravel(objects["y"])[0])
    a = float(np.ravel(objects["a"])[0]) * scale * kronrad
    b = float(np.ravel(objects["b"])[0]) * scale * kronrad
    return x, y, a, b, theta


def _apply_floor(flux_err: float, blank_rms: float | None) -> float:
    """Return the largest between the flux error and the blank-aperture floor (if available)."""
    if blank_rms is None:
        return flux_err
    return max(flux_err, blank_rms)


def blank_aperture_floor(
    data: np.ndarray,
    flux_err: float,
    px: float,
    py: float,
    radius: float,
    source_mask: Optional[np.ndarray] = None,
) -> float:
    """Applies a blank-aperture noise floor to the error of a circular-aperture flux.

    The floor (the scatter of the fluxes in blank, source-free apertures) accounts for
    correlated noise, which is not captured by pixel-wise uncertainties.

    Parameters
    ----------
    data: Image data.
    flux_err: Flux uncertainty.
    px, py: Aperture centre in pixels.
    radius: Aperture radius in pixels.
    source_mask: Mask of the detected sources (see :func:`get_source_mask`). Giving it avoids
        recalculating it for every aperture on the same image.

    Returns
    -------
    flux_err: Flux uncertainty including the floor.
    """
    return _apply_floor(flux_err, _blank_aperture_rms(data, px, py, radius, source_mask=source_mask))


def blank_kron_floor(
    data: np.ndarray,
    flux_err: float,
    objects: np.ndarray,
    kronrad: float,
    scale: float,
    pixel_scale: float = 1.0,
    label: str = "Kron aperture",
) -> float:
    """Applies a blank-aperture noise floor to the error of a Kron flux.

    See :func:`blank_aperture_floor`. If blank apertures of the size of the Kron aperture do not fit
    in the image (e.g. a large galaxy in a small image), the floor is extrapolated from smaller
    apertures (see :func:`_extrapolated_blank_rms`), with a warning.

    Parameters
    ----------
    data: Image data.
    flux_err: Flux uncertainty.
    objects: Objects detected with `sep.extract()`.
    kronrad: Kron radius.
    scale: Scale of the Kron radius.
    pixel_scale: Pixel scale in arcsec (for the source mask).
    label: Description of the measurement, used in the warnings (e.g. survey and filter).

    Returns
    -------
    flux_err: Flux uncertainty including the floor.
    """
    x, y, a, b, theta = _kron_ellipse(objects, kronrad, scale)
    source_mask = _source_mask(data, pixel_scale, host_xy=(x, y))
    rms = _blank_elliptical_rms(data, x, y, a, b, theta, source_mask=source_mask)
    if rms is None:
        rms = _extrapolated_blank_rms(data, x, y, np.pi * a * b, source_mask)
        if rms is None:
            warnings.warn(
                f"{label}: the blank-aperture noise floor could not be estimated (not enough blank sky "
                "in the image), so the uncertainty might be underestimated. Consider downloading a larger image."
            )
        else:
            warnings.warn(
                f"{label}: blank apertures of the size of the Kron aperture do not fit in the image, so the "
                "noise floor was extrapolated from smaller apertures (less accurate). Consider downloading a larger image."
            )
    return _apply_floor(flux_err, rms)


def extract_invvar_aperture_flux(
    data: np.ndarray,
    invvar: np.ndarray,
    px: float,
    py: float,
    radius: float,
    source_mask: Optional[np.ndarray] = None,
) -> tuple[float, float]:
    """Aperture flux with inverse-variance propagation and a blank-aperture floor."""
    flux, weights, mask = _aperture_weighted_sum(data, px, py, radius)
    invvar_var = _aperture_invvar_variance(invvar, weights, mask)
    invvar_err = np.sqrt(invvar_var) if invvar_var > 0 else np.nan
    return flux, blank_aperture_floor(data, invvar_err, px, py, radius, source_mask=source_mask)


def extract_invvar_kron_flux(
    data: np.ndarray,
    invvar: np.ndarray,
    objects: np.ndarray,
    kronrad: float,
    scale: float,
    pixel_scale: float = 1.0,
    label: str = "Kron aperture",
) -> tuple[float, float]:
    """Kron flux with inverse-variance propagation and a blank-aperture floor."""
    x, y, a, b, theta = _kron_ellipse(objects, kronrad, scale)
    flux, weights, mask = _elliptical_weighted_sum(data, x, y, a, b, theta)
    invvar_var = _aperture_invvar_variance(invvar, weights, mask)
    invvar_err = np.sqrt(invvar_var) if invvar_var > 0 else np.nan
    return flux, blank_kron_floor(data, invvar_err, objects, kronrad, scale, pixel_scale, label)

# Herschel-specific functions (similar to the inverse-variance ones above)
# absolute calibration uncertainties (fraction of the flux)
HERSCHEL_CALIBRATION_FRACTION = {
    "PACS100": 0.05,
    "PACS160": 0.05,
    "SPIRE250": 0.06,
    "SPIRE350": 0.06,
    "SPIRE500": 0.06,
}


def _herschel_pixel_area_sr(wcs: WCS) -> float:
    """Return projected pixel area in steradians for a Herschel map."""
    pixel_area_deg2 = abs(float(proj_plane_pixel_area(wcs)))
    if not np.isfinite(pixel_area_deg2) or pixel_area_deg2 <= 0:
        raise RuntimeError("could not determine Herschel pixel area from WCS")
    return pixel_area_deg2 * (np.pi / 180.0) ** 2


def _herschel_data_to_jy_per_pixel(
    data: np.ndarray, header: fits.Header, wcs: WCS
) -> tuple[np.ndarray, str, float]:
    """Convert Herschel image data into Jy/pixel for aperture summation."""
    bunit = str(header.get("BUNIT", "")).strip().lower().replace(" ", "")
    pixel_area_sr = _herschel_pixel_area_sr(wcs)

    if bunit in {"jy/pixel", "jy/pix", "jy"}:
        return data, "Jy", 1.0

    if bunit == "mjy/sr":
        factor = 1.0e6 * pixel_area_sr
        return data * factor, "Jy", factor

    if bunit == "jy/beam":
        bmaj = header.get("BMAJ")
        bmin = header.get("BMIN")
        if bmaj is None or bmin is None:
            raise RuntimeError("Herschel Jy/beam map has no BMAJ/BMIN beam keywords")
        beam_area_sr = 1.1331 * float(bmaj) * float(bmin) * (np.pi / 180.0) ** 2
        factor = pixel_area_sr / beam_area_sr
        return data * factor, "Jy", factor

    raise RuntimeError(f"unsupported Herschel BUNIT={header.get('BUNIT')!r}")


def _herschel_ellipse_params(
    objects: np.ndarray, kronrad: float, scale: float
) -> tuple[float, float, float, float, float]:
    """Return the Herschel Kron-ellipse parameters for a single object."""
    x = float(np.ravel(objects["x"])[0])
    y = float(np.ravel(objects["y"])[0])
    a = float(np.ravel(objects["a"])[0]) * scale * kronrad
    b = float(np.ravel(objects["b"])[0]) * scale * kronrad
    theta = _normalized_theta(float(np.ravel(objects["theta"])[0]))
    return x, y, a, b, theta

def _herschel_blank_elliptical_rms(
    data: np.ndarray,
    x0: float,
    y0: float,
    a: float,
    b: float,
    theta: float,
    n_apertures: int = 80,
    rmin_factor: float = 2.5,
    rmax_factor: float = 6.0,
) -> float | None:
    """Estimate a local empirical floor from blank elliptical apertures around the host."""
    rng = np.random.default_rng(12345)
    ny, nx = data.shape
    blank_fluxes = []
    max_trials = max(400, n_apertures * 20)
    step_scale = max(a, b)
    rmin = rmin_factor * step_scale
    rmax = rmax_factor * step_scale

    for _ in range(max_trials):
        if len(blank_fluxes) >= n_apertures:
            break
        phi = rng.uniform(0, 2 * np.pi)
        rho = rng.uniform(rmin, rmax)
        x = x0 + rho * np.cos(phi)
        y = y0 + rho * np.sin(phi)
        if x - step_scale < 0 or x + step_scale >= nx or y - step_scale < 0 or y + step_scale >= ny:
            continue
        try:
            flux, _, _ = _elliptical_weighted_sum(data, x, y, a, b, theta)
        except Exception:
            continue
        if np.isfinite(flux):
            blank_fluxes.append(flux)

    if len(blank_fluxes) < 10:
        return None

    arr = np.asarray(blank_fluxes, dtype=float)
    med = np.nanmedian(arr)
    return float(1.4826 * np.nanmedian(np.abs(arr - med)))
