import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.utils import calc_total_error

import hostphot
from hostphot.surveys_utils import (
    check_survey_validity,
    check_filters_validity,
    survey_pixel_units,
    survey_zp,
)
from hostphot.photometry.image_utils import (
    get_image_exptime,
    get_image_gain,
    get_image_readnoise,
    correct_HST_aperture,
)

hostphot_path = Path(hostphot.__path__[0])


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

    if instrument == "UVIS":
        # APERTURE usually points to UVIS2
        instrument = header["APERTURE"]
        # some images have a different APERTURE value
        # see: https://hst-docs.stsci.edu/wfc3ihb/chapter-6-uvis-imaging-with-wfc3/6-4-uvis-field-geometry
        # not sure if this is the correct solution
        if instrument == "UVIS-CENTER":
            instrument = "UVIS2"
        if instrument == "UVIS":
            instrument = "UVIS1"

    # get uncertainty file
    err_file = hostphot_path.joinpath("filters", "HST", f"{instrument}_err.txt")
    err_df = pd.read_csv(err_file, sep="\\s+")
    filt_err_df = err_df[err_df.Filter == filt]
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
    readnoise = get_image_readnoise(header, survey)

    # 1.0857 = 2.5/ln(10)
    mag_err = 2.5 / np.log(10) * flux_err / flux

    if survey in [
        "PanSTARRS",
        "DES",
        "LegacySurvey",
        "Spitzer",
        "VISTA",
        "SkyMapper",
        "SPLUS",
        "UKIDSS",
    ]:
        if survey == "Spitzer":
            flux /= header["EFCONV"]  # conv. factor (MJy/sr)/(DN/s)

        extra_err = (
            2.5 / np.log(10) * np.sqrt(ap_area * (readnoise**2) + flux / gain) / flux
        )
        mag_err = np.sqrt(mag_err**2 + extra_err**2)

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

    elif survey == "GALEX":
        # https://asd.gsfc.nasa.gov/archive/galex/FAQ/counts_background.html
        CPS = flux
        if filt == "FUV":
            uv_err = -2.5 * (
                np.log10(CPS)
                - np.log10(
                    CPS
                    + (CPS * exptime + (0.050 * CPS * exptime) ** 2) ** 0.5 / exptime
                )
            )
            # flux_uv_err = (
            #    np.sqrt(CPS * exptime + (0.050 * CPS * exptime) ** 2) / exptime
            # )
        elif filt == "NUV":
            uv_err = -2.5 * (
                np.log10(CPS)
                - np.log10(
                    CPS
                    + (CPS * exptime + (0.027 * CPS * exptime) ** 2) ** 0.5 / exptime
                )
            )
            # flux_uv_err = (
            #    np.sqrt(CPS * exptime + (0.027 * CPS * exptime) ** 2) / exptime
            # )

        mag_err = np.sqrt(mag_err**2 + uv_err**2)

    elif survey == "2MASS":
        # see: https://wise2.ipac.caltech.edu/staff/jarrett/2mass/3chan/noise/
        S = flux
        N_c = 6  # number of coadd pixels
        k_z = 1.7  # kernel smoothing factor
        gain = 10  # typical gain in e-/dn
        n_f = ap_area  # number of frame pixels in the aperture; aprox. as aperture area
        n_c = 4 * n_f  # number of coadd pixels in the aperture
        sigma_c = bkg_rms  # coadd noise; assumed to be ~background noise

        SNR = S / np.sqrt(
            S / (gain * N_c)
            + 4 * n_c * k_z ** 2 * sigma_c ** 2
        )
        mag_err = 1.0857 / SNR


    elif "WISE" in survey:
        # see: https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec2_3f.html
        # see Table 5 of
        # https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4c.html#wpro
        # correction assumed to be 0 mags as PSF fitting is not used.
        m_apcor = 0.0
        f_apcor = 10 ** (-0.4 * m_apcor)
        F_src = f_apcor * flux

        N_p = ap_area  # effective number of noise-pixels characterizing the PRF
        # ratio of input (detector) pixel scale to output (Atlas Image) pixel scale
        pixel_scale_ratios = {"W1": 2, "W2": 2, "W3": 2, "W4": 4}
        Sin_Sout = pixel_scale_ratios[filt]
        F_corr = N_p * Sin_Sout**2
        
        k = np.pi / 2
        N_A = ap_area
        N_B = N_A / 10
        sigma_conf = bkg_rms  # assumed to be ~bkg
        sigma_src = np.sqrt(
            f_apcor ** 2 * F_corr * (bkg_rms ** 2 + k * (N_A ** 2) / N_B * bkg_rms ** 2)
            + sigma_conf ** 2
        )
        
        mag_err = np.sqrt(1.179 * sigma_src**2 / F_src**2)

        # add uncertainty from the ZP
        if survey == "unWISE":
            # These values are the same for all Atlas Images of a given band...
            # see: https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec2_3f.html
            unc_dict = {"W1": 0.006, "W2": 0.007, "W3": 0.012, "W4": 0.012}
            zp_unc = unc_dict[filt]
        elif survey == "WISE":
            zp_unc = header["MAGZPUNC"]

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
        # already added at the beginning
        pass
    elif survey in ["VISTA", "UKIDSS"]:
        # add uncertainty from the ZP
        zp_unc = header["MAGZRR"]
        mag_err = np.sqrt(mag_err**2 + zp_unc**2)

    elif survey == "HST":
        zp_unc = get_HST_err(filt, header)
        mag_err = np.sqrt(mag_err**2 + zp_unc**2)

    elif survey == "SkyMapper":
        zp_unc = header["ZPTERR"]
        mag_err = np.sqrt(mag_err**2 + zp_unc**2)

    elif survey == "SPLUS":
        # following Section 4.4 of Almeida-Fernandes et al. (2022)
        zp_uncs = {
            "U": 25e-3,
            "F395": 25e-3,
            "F378": 15e-3,
        }
        if filt in zp_uncs.keys():
            zp_unc = zp_uncs[filt]
        else:
            zp_unc = 1e-3
        mag_err = np.sqrt(mag_err**2 + zp_unc**2)

    elif survey == "JWST":
        # see https://jwst-docs.stsci.edu/jwst-calibration-status/nircam-calibration-status/nircam-imaging-calibration-status#NIRCamImagingCalibrationStatus-Photometriccalibration
        # absolute flux calibration uncertainties < 1%, but set to 1% error for now
        zp_unc = 0.01 * (header["MAGZP"])
        mag_err = np.sqrt(mag_err**2 + zp_unc**2)

    else:
        raise Exception(f"Survey {survey} has not been added for error propagation.")

    flux_err = np.abs(flux * 0.4 * np.log(10) * mag_err)

    return flux_err


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

    if survey == "HST":
        # HST needs and aperture correction for the flux
        # see, e.g. https://www.stsci.edu/hst/instrumentation/acs/data-analysis/aperture-corrections
        ap_corr = correct_HST_aperture(filt, ap_area, header)
        flux = flux * ap_corr

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

    pixel_units = survey_pixel_units(survey, filt)
    if pixel_units == "counts" and survey!="2MASS":
        # flux needs to be in units of counts per second
        # but only after the error propagation
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
        filt_file = [file for file in filters_path.rglob(f"*{filt_}*")][0]
    elif survey == "JWST":
        filt_file = filters_path / f"{filt}.dat"
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
