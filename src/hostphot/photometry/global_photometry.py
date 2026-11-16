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

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

import sep
from scipy.optimize import curve_fit
from scipy.special import gamma, gammainc
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area, proj_plane_pixel_scales
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

from photutils.utils import calc_total_error
from photutils.aperture import EllipticalAperture

from hostphot._constants import workdir
from hostphot.processing.objects_detection import extract_objects, plot_detected_objects
from hostphot.processing.cleaning import remove_nan, saturated_mask
from hostphot.photometry.dust import calc_extinction
from hostphot.utils import check_work_dir, store_input, image_provenance
from hostphot._version import __version__
from hostphot.flags import GLOBAL_FLAGS, ApertureWarning, collect_flags, header_flags
from hostphot.photometry.photometry_utils import (
    magnitude_calculation,
    extract_invvar_kron_flux,
    get_invvar_map,
    blank_kron_floor,
    floor_residual_background,
    blank_floor_surveys,
    check_global_coverage,
    correct_missing_flux,
    check_saturation,
    _kron_ellipse,
    _elliptical_weighted_sum,
    _herschel_ellipse_params,
    HERSCHEL_CALIBRATION_FRACTION,
    _herschel_pixel_area_sr,
    _herschel_data_to_jy_per_pixel,
    get_effective_gain,
    image_psf_fwhm,
    warn_non_detection,
    ab_photometry,
)
from hostphot.photometry.image_utils import (
    add_sky_ellipses,
    adapt_aperture,
    has_sky_ellipses,
    objects_from_sky,
    wrap_theta,
)
from hostphot.processing.background import background_flag, masked_background, sep_background
from hostphot.processing.host_model import _bn, elliptical_radius, estimate_fwhm_pix
from hostphot.surveys_utils import (
    get_survey_filters,
    get_default_filters,
    check_filters_validity,
    check_survey_validity,
    flipped_surveys,
    bkg_surveys,
    normalize_filters,
)

import functools
import inspect
import warnings
from astropy.utils.exceptions import AstropyWarning

sep.set_sub_object_limit(1e4)

def herschel_kron_flux(
    data: np.ndarray,
    header: fits.Header,
    wcs: WCS,
    objects: np.ndarray,
    kronrad: float,
    scale: float,
    filt: str,
) -> tuple[float, float, dict]:
    """Measure Herschel Kron-aperture flux directly in Jy.

    PACS selected maps are already in Jy/pixel.  SPIRE extended-source maps are
    in MJy/sr and are converted to Jy/pixel using the WCS pixel area.  The random
    uncertainty is estimated from nearby blank apertures and a configurable
    absolute-calibration term is added in quadrature.
    """
    jy_data, flux_unit, conversion_factor = _herschel_data_to_jy_per_pixel(data, header, wcs)
    x, y, a, b, theta = _herschel_ellipse_params(objects, kronrad, scale)
    flux, weights, _ = _elliptical_weighted_sum(jy_data, x, y, a, b, theta)

    # blank-aperture noise, as for the other surveys (with the sources masked; extrapolated from smaller
    # apertures if needed, or NaN with a warning if it cannot be estimated)
    pixel_scale = float(np.sqrt(np.abs(np.linalg.det(wcs.pixel_scale_matrix))) * 3600)
    blank_rms = blank_kron_floor(
        jy_data, 0.0, objects, kronrad, scale, pixel_scale=pixel_scale, label=f"Herschel {filt} (global aperture)"
    )
    if not np.isfinite(blank_rms) or blank_rms <= 0:
        blank_rms = np.nan

    cal_fraction = HERSCHEL_CALIBRATION_FRACTION.get(filt, 0.0)
    cal_err = abs(flux) * cal_fraction
    if np.isfinite(blank_rms):
        flux_err = float(np.sqrt(blank_rms**2 + cal_err**2))
    else:
        flux_err = float(cal_err)

    aperture_area_pix = float(np.nansum(weights[np.isfinite(weights)]))
    pixel_area_sr = _herschel_pixel_area_sr(wcs)
    aperture_area_arcsec2 = aperture_area_pix * pixel_area_sr * 206265.0**2
    details = {
        "flux_unit": flux_unit,
        "input_bunit": header.get("BUNIT"),
        "conversion_factor_to_jy_per_pixel": conversion_factor,
        "blank_rms_jy": blank_rms,
        "calibration_fraction": cal_fraction,
        "calibration_err_jy": cal_err,
        "aperture_area_pix": aperture_area_pix,
        "aperture_area_arcsec2": aperture_area_arcsec2,
        "snr": flux / flux_err if flux_err > 0 else np.nan,
    }
    return float(flux), flux_err, details


def kron_aperture_area(objects: np.ndarray, kronrad: float, scale: float) -> float:
    """Calculates the area of the Kron aperture used by :func:`kron_flux`.

    Parameters
    ----------
    objects: Objects detected with `sep.extract()`.
    kronrad: Kron radius.
    scale: Scale of the Kron radius.

    Returns
    -------
    ap_area: Aperture area in pixels.
    """
    return np.pi * objects["a"][0] * objects["b"][0] * (scale * kronrad) ** 2


def _background(data: np.ndarray) -> sep.Background:
    """Estimates the background of an image, masking the pixels without data (NaN)."""
    nan_mask = ~np.isfinite(data)
    filled = np.ascontiguousarray(np.where(nan_mask, 0.0, data), dtype=np.float64)
    return sep_background(filled, mask=nan_mask)


def _kron_radius(data: np.ndarray, objects: np.ndarray) -> float:
    """Calculates the Kron radius of an object, masking the pixels without data (NaN)."""
    nan_mask = ~np.isfinite(data)
    filled = np.ascontiguousarray(np.where(nan_mask, 0.0, data), dtype=np.float64)
    kronrad, _ = sep.kron_radius(
        filled,
        objects["x"],
        objects["y"],
        objects["a"],
        objects["b"],
        objects["theta"],
        r=6.0,  # following sep docs
        mask=nan_mask,
    )
    kronrad = float(kronrad[0])
    if not np.isfinite(kronrad) or kronrad <= 0:
        raise ValueError(f"Invalid Kron radius ({kronrad}) for the galaxy.")
    return kronrad


def kron_flux(
    data: np.ndarray,
    bkg: float,
    exptime: float,
    objects: np.ndarray,
    kronrad: float,
    scale: float,
) -> tuple[float, float]:
    """Calculates the Kron flux.

    Parameters
    ----------
    data: Data of an image.
    bkg: Background error of the images.
    exptime: Exposure time.
    objects: Objects detected with `sep.extract()`.
    kronrad: Kron radius.
    scale: Scale of the Kron radius.

    Returns
    -------
    flux: Kron flux.
    flux_err: Kron flux error.
    """
    # theta must be in the range [-pi/2, pi/2] for sep.sum_ellipse()
    objects["theta"] = wrap_theta(objects["theta"])

    data_valid, error_valid = _kron_arrays(data, bkg, exptime)
    return _sum_kron_ellipse(data_valid, error_valid, objects, kronrad, scale)


def _kron_arrays(data: np.ndarray, bkg: float, exptime: float) -> tuple[np.ndarray, np.ndarray]:
    """Returns the data and total error with the pixels without data set to zero (i.e. excluded).

    These only depend on the image, so they can be calculated once for several apertures.
    """
    error = calc_total_error(data, bkg, exptime)
    # pixels without data are excluded (see check_global_coverage())
    nan_mask = ~np.isfinite(data) | ~np.isfinite(error)
    data_valid = np.ascontiguousarray(np.where(nan_mask, 0.0, data), dtype=np.float64)
    error_valid = np.ascontiguousarray(np.where(nan_mask, 0.0, error), dtype=np.float64)
    return data_valid, error_valid


def _sum_kron_ellipse(
    data_valid: np.ndarray,
    error_valid: np.ndarray,
    objects: np.ndarray,
    kronrad: float,
    scale: float,
) -> tuple[float, float]:
    """Sums the flux and error in the Kron ellipse with exact pixel overlap.

    Equivalent to ``photutils.aperture_photometry`` with an ``EllipticalAperture`` (same
    exact-overlap geometry), but much faster, which matters when optimising the aperture.
    """
    flux, flux_err, _ = sep.sum_ellipse(
        data_valid,
        [objects["x"][0]],
        [objects["y"][0]],
        [objects["a"][0] * scale * kronrad],
        [objects["b"][0] * scale * kronrad],
        [objects["theta"][0]],
        r=1.0,
        err=error_valid,
        subpix=0,
    )
    return flux[0], flux_err[0]


# maximum scale of the Kron radius (and step) when optimising the aperture
MAX_KRON_SCALE = 6.0
KRON_SCALE_STEP = 0.1


def optimize_aperture(
    data: np.ndarray, bkg: float, exptime: float, objects: np.ndarray, eps: float = 0.001
) -> tuple[float, float, float, float]:
    """Optimizes the Kron flux by iteration over different scales.

    The scale of the Kron radius is increased (in steps of ``KRON_SCALE_STEP``) until the flux
    gained between steps is consistent with the noise of the added ring (or smaller than a
    fraction ``eps`` of the flux), up to ``MAX_KRON_SCALE`` (with a warning). Without the noise
    criterion, the noise and residuals of the sky prevent the convergence, and the apertures
    grow too large (with fluxes biased by the sky).

    Parameters
    ----------
    data: Data of an image.
    bkg: Background error of the images.
    exptime: Exposure time.
    objects: Objects detected with :func:`sep.extract()`.
    eps: Minimum fractional change in flux used to stop the iterations.

    Returns
    -------
    opt_flux: Optimized Kron flux.
    opt_flux_err: Optimized Kron flux error.
    kronrad: Kron radius.
    opt_scale: Optimized scale for the Kron radius.
    """
    kronrad = _kron_radius(data, objects)

    # theta must be in the range [-pi/2, pi/2] for sep.sum_ellipse()
    objects["theta"] = wrap_theta(objects["theta"])
    data_valid, error_valid = _kron_arrays(data, bkg, exptime)  # same for all the scales

    scales = np.arange(1.0, MAX_KRON_SCALE + KRON_SCALE_STEP / 2, KRON_SCALE_STEP)
    prev = None  # flux, error and scale of the previous step
    for scale in scales:
        flux, flux_err = _sum_kron_ellipse(data_valid, error_valid, objects, kronrad, scale)
        if not np.isfinite(flux):
            if prev is None:
                return flux, flux_err, kronrad, scale
            warnings.warn("The optimised Kron aperture might not fit in the image!", ApertureWarning)
            return prev[0], prev[1], kronrad, prev[2]
        if prev is not None:
            gain = flux - prev[0]
            # noise of the flux in the added ring (independent of that of the previous aperture)
            gain_err = np.sqrt(max(flux_err**2 - prev[1] ** 2, 0.0))
            if gain < max(gain_err, eps * np.abs(flux)):
                return flux, flux_err, kronrad, scale
        prev = (flux, flux_err, scale)

    warnings.warn(
        f"The optimised Kron aperture reached the maximum scale ({MAX_KRON_SCALE} times the Kron radius) "
        "without converging (e.g. residual sky or a very extended galaxy), so the flux might be overestimated.",
        ApertureWarning,
    )
    return prev[0], prev[1], kronrad, prev[2]


# curve-of-growth aperture (``aperture_method="cog"``): see cog_aperture()
APERTURE_METHODS = ("kron", "cog")
COG_SNR_STOP = 2.0  # the profile is measured out to where its signal-to-noise ratio falls below this value
COG_N_ANNULI = 60  # number of elliptical annuli of the profile (logarithmically spaced)
COG_TAIL_RADIUS = 8.0  # the Sérsic tail is integrated out to this number of effective radii
COG_MIN_RADIUS_FWHM = 2.0  # minimum radius of the aperture (and of the Sérsic fit), in units of the PSF FWHM
# contamination by neighbours is only looked for in annuli with lower signal-to-noise ratios (the structures of
# bright galaxies, e.g. spiral arms, make the mean of their annuli significantly larger than the median)
COG_CONTAMINATION_SNR = 10.0
# without a Sérsic tail, the aperture is flagged only if smaller than this number of half-light radii (of the
# curve of growth): larger apertures include most of the flux (e.g. ~95% within 4 effective radii for n = 2.5)
COG_NO_TAIL_RADIUS = 4.0
# minimum uncertainty of the Sérsic tail, as a fraction of the tail (the profiles of galaxies are not exactly
# Sérsic profiles; from galaxies injected in PanSTARRS images, the error of the tail is ~80% of the tail)
COG_TAIL_MIN_ERR = 0.8
# without a Sérsic tail, the uncertainty of the flux beyond the aperture is the flux in this number of outermost
# annuli of the aperture (~1.2 times the scatter of the missing flux, from galaxies injected in PanSTARRS images)
COG_NO_TAIL_ANNULI = 2


def _profile(
    data: np.ndarray, r_ell: np.ndarray, edges: np.ndarray, sky_noise: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Surface-brightness profile in elliptical annuli (pixels without data are NaN or zero).

    Returns the sigma-clipped (3 sigma, from the MAD) mean of each annulus, its uncertainty (from the scatter of
    the pixels, including the noise of the source, and at least the sky noise), the number of valid pixels, the
    area (all pixels) and the median. The clipping rejects the pixels of neighbours not masked (e.g. halos of
    bright stars) covering part of an annulus.
    """
    idx = np.digitize(r_ell.ravel(), edges) - 1
    vals = data.ravel()
    nbins = len(edges) - 1
    inside = (idx >= 0) & (idx < nbins)
    valid = inside & np.isfinite(vals) & (vals != 0)
    area = np.bincount(idx[inside], minlength=nbins)[:nbins]
    order = np.argsort(idx[valid], kind="stable")
    sorted_idx, sorted_vals = idx[valid][order], vals[valid][order]
    bounds = np.searchsorted(sorted_idx, np.arange(nbins + 1))
    mean, err, median = np.full(nbins, np.nan), np.full(nbins, np.inf), np.full(nbins, np.nan)
    nvalid = np.zeros(nbins, dtype=int)
    for i in range(nbins):
        v = sorted_vals[bounds[i] : bounds[i + 1]]
        if len(v) == 0:
            continue
        med = np.median(v)
        median[i] = med
        mad = 1.4826 * np.median(np.abs(v - med))
        if len(v) >= 10 and mad > 0:
            v = v[np.abs(v - med) < 3 * mad]
        mean[i] = np.mean(v)
        nvalid[i] = len(v)
        scatter = np.std(v) if len(v) >= 10 else sky_noise
        err[i] = max(scatter, sky_noise) / np.sqrt(len(v))
    return mean, err, nvalid, area, median


def _sersic_sb(r: np.ndarray, log_ie: float, re: float, n: float) -> np.ndarray:
    """Sérsic surface-brightness profile."""
    return 10**log_ie * np.exp(-_bn(n) * ((r / re) ** (1 / n) - 1))


def _sersic_flux_between(params: np.ndarray, q: float, r_in: float, r_out: float) -> float:
    """Flux of an elliptical Sérsic profile between two (semi-major-axis) radii."""
    log_ie, re, n = params
    bn = _bn(n)
    total = 2 * np.pi * q * n * 10**log_ie * re**2 * np.exp(bn) * bn ** (-2 * n) * gamma(2 * n)
    return total * (gammainc(2 * n, bn * (r_out / re) ** (1 / n)) - gammainc(2 * n, bn * (r_in / re) ** (1 / n)))


def cog_aperture(
    data: np.ndarray, objects: np.ndarray, kronrad: float, sky_noise: float, fwhm_pix: float, label: str = ""
) -> dict:
    """Curve-of-growth (CoG) aperture of a galaxy, with a Sérsic tail.

    The surface-brightness profile of the galaxy is measured in elliptical annuli (with the centre, axis ratio and
    orientation of the detected galaxy), using sigma-clipped means. The aperture extends out to the radius where
    the profile is no longer significant (signal-to-noise ratio below ``COG_SNR_STOP``), or where, in the
    outskirts (beyond the Kron radius and below 20% of the peak surface brightness), the profile rises, flattens
    (logarithmic slope flatter than -0.5 in two consecutive annuli, e.g. residual sky) or is contaminated by a
    neighbour (clipped mean significantly above the median, in annuli with signal-to-noise ratios below
    ``COG_CONTAMINATION_SNR``); the aperture radius is at least
    ``COG_MIN_RADIUS_FWHM`` times the PSF FWHM. The rules use the median of the annuli, robust to neighbours.
    If the profile is still significant at the edge of the image (or of the region with data), the aperture
    might not include all the flux (with a warning).

    The flux beyond the aperture is estimated by fitting a Sérsic profile to the annuli between
    ``COG_MIN_RADIUS_FWHM`` times the PSF FWHM and the aperture radius, and integrating it from the aperture
    radius out to ``COG_TAIL_RADIUS`` effective radii. The tail is given as a fraction of the flux within the
    aperture, with its uncertainty (from the covariance of the fit, and at least ``COG_TAIL_MIN_ERR`` times the
    tail). No tail is added if the fit fails, the Sérsic index reaches its limits (0.3 or 8), the tail is not
    constrained (uncertainty larger than the tail) or is larger than the flux within the aperture; then, the
    uncertainty of the flux beyond the aperture is the flux in the ``COG_NO_TAIL_ANNULI`` outermost annuli of the
    aperture (``tail_frac_err``, with ``tail_frac`` = 0), and the aperture is flagged (with a warning) only if its
    radius is smaller than ``COG_NO_TAIL_RADIUS`` half-light radii (from the curve of growth within the aperture).

    Parameters
    ----------
    data: Image data (background subtracted, if needed), with NaN for the pixels without data.
    objects: Galaxy detected with :func:`sep.extract()` (one object).
    kronrad: Kron radius (in units of the semi-major axis of the object).
    sky_noise: Noise of the sky (background RMS).
    fwhm_pix: PSF FWHM in pixels.
    label: Description used in the warnings (e.g. survey and filter).

    Returns
    -------
    cog: Dictionary with the aperture radius (semi-major axis, in pixels; ``r_meas``), the fraction of the flux
        added by the tail and its uncertainty (``tail_frac`` and ``tail_frac_err``; see above), the
        parameters of the Sérsic fit (``sersic_n`` and ``sersic_re``, in pixels; NaN if not used), and whether the
        aperture might not include all the flux of the galaxy (``flag``, with a warning).
    """
    x0, y0 = float(objects["x"][0]), float(objects["y"][0])
    q = float(objects["b"][0] / objects["a"][0])
    theta = float(objects["theta"][0])
    ny, nx = data.shape
    # largest radius fully inside the image
    rmax = 0.98 * min(x0, y0, nx - 1 - x0, ny - 1 - y0)
    if rmax <= 1:
        raise ValueError(f"{label}: the galaxy is at the edge of the image, so the CoG aperture cannot be measured.")
    r_ell = elliptical_radius(data.shape, x0, y0, q, theta)
    edges = np.concatenate([[0.0], np.geomspace(min(0.5, rmax / 2), rmax, COG_N_ANNULI)])
    mean, err, nvalid, area, median = _profile(data, r_ell, edges, sky_noise)
    mean = np.where(nvalid > 0, mean, 0.0)
    cum = np.cumsum(mean * area)  # flux within the outer edge of each annulus
    r_out = edges[1:]
    r_mid = np.sqrt(edges[:-1] * edges[1:])
    err_med = 1.2533 * err  # uncertainty of the median

    # stopping rules (with the median profile)
    with np.errstate(invalid="ignore", divide="ignore"):
        snr = median / err_med
        rise = np.zeros(len(mean), dtype=bool)
        rise[1:] = median[1:] - median[:-1] > 2 * np.sqrt(err_med[1:] ** 2 + err_med[:-1] ** 2)
        log_i = np.log(np.where(median > 0, median, np.nan))
        slope = np.full(len(mean), np.nan)
        slope[3:] = (log_i[3:] - log_i[:-3]) / (np.log(r_mid[3:]) - np.log(np.maximum(r_mid[:-3], 1e-3)))
        flat = (slope > -0.5) | ~np.isfinite(slope)
        # (in two consecutive annuli, e.g. residual sky, not a local plateau of the profile such as a ring)
        flat[:-1] &= flat[1:]
        contaminated = ((mean - median) > 2 * err_med) & (snr < COG_CONTAMINATION_SNR)
        r_kron = kronrad * float(objects["a"][0])
        outskirts = (median < 0.2 * np.nanmax(median)) & (r_mid > r_kron)
    # annuli mostly without data (e.g. the edge of the region with data)
    no_data = nvalid < 0.5 * np.maximum(area, 1)
    stop = (
        no_data | (snr < COG_SNR_STOP) | ~np.isfinite(snr) | ((rise | flat | contaminated) & outskirts)
    ) & (r_out > COG_MIN_RADIUS_FWHM * fwhm_pix)
    stops = np.where(stop)[0]
    cog = {"tail_frac": 0.0, "tail_frac_err": 0.0, "sersic_n": np.nan, "sersic_re": np.nan, "flag": False}
    if len(stops) == 0 or no_data[stops[0]]:
        i_meas = len(r_out) - 1 if len(stops) == 0 else max(stops[0] - 1, 0)
        cog["flag"] = True
        warnings.warn(
            f"{label}: the profile of the galaxy is still significant at the edge of the image, so the CoG "
            "aperture might not include all the flux of the galaxy (larger image needed).",
            ApertureWarning,
        )
    else:
        i_meas = max(stops[0] - 1, 0)
    r_meas, f_meas = float(r_out[i_meas]), float(cum[i_meas])
    cog["r_meas"] = r_meas

    # Sérsic tail
    sel = (r_mid > COG_MIN_RADIUS_FWHM * fwhm_pix) & (np.arange(len(r_mid)) <= i_meas) & (nvalid > 0)
    reason = "too few annuli to fit a Sérsic profile"
    if sel.sum() >= 4 and f_meas > 0:
        lower, upper = [-10.0, 0.3, 0.3], [10.0, 20 * rmax, 8.0]
        p0 = [np.log10(max(mean[sel][0], 1e-10)), max(r_meas / 3, 0.31), 1.5]
        p0[0] = float(np.clip(p0[0], -9.9, 9.9))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                params, cov = curve_fit(
                    _sersic_sb, r_mid[sel], mean[sel], sigma=err[sel], absolute_sigma=True, p0=p0,
                    bounds=(lower, upper), maxfev=20000,
                )
        except (RuntimeError, ValueError) as exc:
            params, reason = None, f"the Sérsic fit failed ({exc})"
        if params is not None:
            def tail(p):
                return _sersic_flux_between(p, q, r_meas, max(COG_TAIL_RADIUS * p[1], r_meas))

            tail_flux = tail(params)
            # uncertainty of the tail from the covariance of the fit
            tail_err = np.inf
            if np.all(np.isfinite(cov)):
                rng = np.random.default_rng(0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    draws = rng.multivariate_normal(params, cov, 300)
                draws = draws[np.all((draws > lower) & (draws < upper), axis=1)]
                if len(draws) > 20:
                    tail_err = float(np.std([tail(d) for d in draws]))
            if params[2] > 7.9 or params[2] < 0.31:
                reason = f"the Sérsic index reached its limits (n = {params[2]:.2f})"
            elif not (np.isfinite(tail_err) and tail_err < tail_flux):
                reason = "the Sérsic tail is not constrained"
            elif tail_flux > f_meas:
                reason = "the Sérsic tail is larger than the flux within the aperture"
            else:
                reason = None
                # (at least COG_TAIL_MIN_ERR of the tail: the covariance of the fit does not include the mismatch
                # between the profile of the galaxy and the Sérsic profile)
                tail_err = max(tail_err, COG_TAIL_MIN_ERR * tail_flux)
                cog.update(
                    tail_frac=float(tail_flux / f_meas), tail_frac_err=float(tail_err / f_meas),
                    sersic_n=float(params[2]), sersic_re=float(params[1]),
                )
    elif f_meas <= 0:
        reason = "the flux within the aperture is not positive"
    if reason is not None and f_meas > 0:
        # uncertainty of the (unknown) flux beyond the aperture: the flux in the outermost annuli
        outer = abs(float(cum[i_meas] - cum[max(i_meas - COG_NO_TAIL_ANNULI, 0)]))
        cog["tail_frac_err"] = outer / f_meas
    if reason is not None:
        # half-light radius of the curve of growth within the aperture
        # (first radius enclosing half the flux; the curve of growth is not always monotonic)
        r_half = float(r_out[np.argmax(cum[: i_meas + 1] >= 0.5 * f_meas)]) if f_meas > 0 else np.inf
        if not r_meas >= COG_NO_TAIL_RADIUS * r_half:
            cog["flag"] = True
            warnings.warn(
                f"{label}: no Sérsic tail added to the CoG aperture ({reason}) and the aperture radius is smaller "
                f"than {COG_NO_TAIL_RADIUS:g} half-light radii, so the flux might be underestimated.",
                ApertureWarning,
            )
    return cog


def apply_cog_tail(flux: float, flux_err: float, tail_frac: float, tail_frac_err: float) -> tuple[float, float]:
    """Adds the Sérsic tail (as a fraction of the flux within the aperture; see :func:`cog_aperture`) to the flux
    of the CoG aperture, with its uncertainty."""
    total = flux * (1 + tail_frac)
    total_err = np.sqrt((flux_err * (1 + tail_frac)) ** 2 + (flux * tail_frac_err) ** 2)
    return total, total_err


def extract_aperture(
    name: str,
    host_ra: float,
    host_dec: float,
    filt: str,
    survey: str,
    ra: float = None,
    dec: float = None,
    bkg_sub: bool = False,
    threshold: float = 10,
    use_mask: bool = True,
    optimize_kronrad: bool = False,
    eps: float = 0.0001,
    gal_dist_thresh: float = -1,
    deblend_cont: float = 0.005,
    save_plots: bool = True,
    save_aperture_params: bool = True,
    save_input: bool = True,
    aperture_method: str = "kron",
) -> tuple[np.ndarray, WCS, float, float, bool]:
    """Calculates the aperture for a given galaxy.

    Parameters
    ----------
    name: Name of the object to find the path of the fits file.
    host_ra: Host-galaxy right ascension of the galaxy in degrees.
    host_dec: Host-galaxy declination of the galaxy in degrees.
    filt: Filter to use to load the fits file. List is commonly used for coadds.
    survey: Survey to use for the zero-points and pixel scale.
    ra: Right ascension of an object, in degrees. Used for plotting the position of the object.
    dec: Declination of an object, in degrees. Used for plotting the position of the object.
    bkg_sub: If `True`, the image gets background subtracted. By default, only
        the images that need it get background subtracted (WISE, 2MASS and
        VISTA).
    threshold: Threshold used by `sep.extract()` to extract objects.
    use_mask: If `True`, the masked fits files are used. These must have
        been created beforehand.
    optimize_kronrad: If `True`, the Kron-radius scale is optimized, increasing the aperture
        size until the flux gained is consistent with the noise (see :func:`optimize_aperture`).
        Otherwise (default), the standard Kron scale of 2.5 is used.
    eps: Minimum fractional change in flux used to stop the optimisation of the Kron-radius
        scale (if the flux gained is not already within the noise). A value of 0.0001 means
        0.01% change in flux.
    gal_dist_thresh: Distance in arcsec to crossmatch the galaxy coordinates with a detected object,
        where the object nearest to the galaxy position is considered as the galaxy (within
        the given threshold). If no objects are found within the given distance threshold,
        the galaxy is considered as not found and a warning is printed. If a non-positive value
        is given (default option), the closest detected object is considered as the galaxy only if
        the galaxy coordinates are within its extent (see :func:`hostphot.processing.objects_detection.extract_objects`).
    deblend_cont : Minimum contrast ratio used for object deblending. Default is 0.005.
        To entirely disable deblending, set to 1.0.
    save_plots: If `True`, the mask and galaxy aperture figures are saved.
    save_aperture_params: If `True`, the aperture parameters are saved into a csv file.
    save_input: Whether to save the input parameters.
    aperture_method: Aperture used for the global photometry: ``"kron"`` (default), the Kron aperture, or
        ``"cog"``, the curve-of-growth aperture with a Sérsic tail (see :func:`cog_aperture`). The CoG aperture
        is also given as a scaled Kron ellipse (``scale * kronrad``), with the same shape.

    Returns
    -------
    gal_obj: Galaxy object.
    wcs: Image's WCS.
    kronrad: Kron radius.
    scale: Scale for the Kron radius.
    flip: Whether to flip the orientation of the aperture.
    """
    input_params = locals()  # dictionary
    gal_obj, wcs, kronrad, scale, flip, _ = _extract_aperture(**input_params)
    return gal_obj, wcs, kronrad, scale, flip


def _check_aperture_method(aperture_method: str, optimize_kronrad: bool) -> None:
    if aperture_method not in APERTURE_METHODS:
        raise ValueError(f"Invalid aperture method '{aperture_method}' (valid options: {APERTURE_METHODS}).")
    if aperture_method == "cog" and optimize_kronrad:
        raise ValueError("`optimize_kronrad=True` can only be used with `aperture_method='kron'`.")


def _extract_aperture(
    name: str,
    host_ra: float,
    host_dec: float,
    filt: str,
    survey: str,
    ra: float = None,
    dec: float = None,
    bkg_sub: bool = False,
    threshold: float = 10,
    use_mask: bool = True,
    optimize_kronrad: bool = False,
    eps: float = 0.0001,
    gal_dist_thresh: float = -1,
    deblend_cont: float = 0.005,
    save_plots: bool = True,
    save_aperture_params: bool = True,
    save_input: bool = True,
    aperture_method: str = "kron",
) -> tuple[np.ndarray, WCS, float, float, bool, dict]:
    """As :func:`extract_aperture`, also returning the details of the aperture (method and Sérsic tail of the
    CoG aperture)."""
    input_params = locals()  # dictionary
    _check_aperture_method(aperture_method, optimize_kronrad)
    # save input parameters
    if save_input is True:
        inputs_file = Path(workdir, name, survey, "input_aperture_parameters.csv")
        store_input(input_params, inputs_file)
    # initial checks
    check_survey_validity(survey)
    check_work_dir(workdir)
    obj_dir = Path(workdir, name)
    if use_mask:
        suffix = "_masked"
    else:
        suffix = ""
    if isinstance(filt, list):
        filt = "".join(f for f in filt)
    fits_file = obj_dir / survey / f"{survey}_{filt}{suffix}.fits"

    # image information
    with fits.open(fits_file, memmap=False) as hdu_file:
        _ = [extension.data for extension in hdu_file]  # (loaded before closing the file)
        hdu = remove_nan(hdu_file)
    header = hdu[0].header
    data = hdu[0].data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        wcs = WCS(header, naxis=2)

    data = data.astype(np.float64)
    bkg = _background(data)
    # background subtraction, if needed
    if (bkg_sub is None and survey in bkg_surveys) or bkg_sub is True:
        # background estimated with the sources (including the galaxy) masked
        data_sub = data - masked_background(data, wcs, host_ra, host_dec, label=f"{survey} {filt}")
    else:
        data_sub = np.copy(data)

    # extract galaxy
    gal_obj, _ = extract_objects(
        data_sub,
        bkg.globalrms,
        host_ra,
        host_dec,
        threshold,
        wcs,
        gal_dist_thresh,
        deblend_cont,
    )
    if gal_obj is None:
        raise ValueError(
            f"The galaxy was not detected in {survey} {filt} (threshold={threshold} sigma). "
            "Try a lower `threshold` or check the coordinates."
        )
    aperture_info = {"method": aperture_method, "tail_frac": 0.0, "tail_frac_err": 0.0}
    if aperture_method == "cog":
        kronrad = _kron_radius(data_sub, gal_obj)
        fwhm = image_psf_fwhm(header, survey, filt)
        pixel_scale = float(np.sqrt(abs(np.linalg.det(wcs.pixel_scale_matrix))) * 3600)  # arcsec
        if np.isfinite(fwhm) and fwhm > 0:
            fwhm_pix = fwhm / pixel_scale
        else:
            fwhm_pix = estimate_fwhm_pix(data_sub, bkg.globalrms, ~np.isfinite(data_sub))
        # theta must be in the range [-pi/2, pi/2] (as for the Kron aperture)
        gal_obj["theta"] = wrap_theta(gal_obj["theta"])
        with collect_flags() as aperture_flags:
            cog = cog_aperture(data_sub, gal_obj, kronrad, bkg.globalrms, fwhm_pix, label=f"{survey} {filt}")
        # the CoG aperture is given as a scaled Kron ellipse (same shape)
        scale = cog["r_meas"] / (float(gal_obj["a"][0]) * kronrad)
        aperture_flag = "aperture" in aperture_flags
        aperture_info.update(
            r_meas=cog["r_meas"], sersic_n=cog["sersic_n"], sersic_re=cog["sersic_re"],
            tail_frac=cog["tail_frac"], tail_frac_err=cog["tail_frac_err"],
        )
    elif optimize_kronrad:
        _exptime = get_effective_gain(header, survey, filt)  # for the Poisson noise of the sources
        with collect_flags() as aperture_flags:
            opt_res = optimize_aperture(data_sub, bkg.rms(), _exptime, gal_obj, eps)
        _, _, kronrad, scale = opt_res
        aperture_flag = "aperture" in aperture_flags
    else:
        scale = 2.5
        kronrad = _kron_radius(data_sub, gal_obj)
        aperture_flag = False
    aperture_info["flag"] = aperture_flag

    if save_plots is True:
        outfile = obj_dir / survey / f"global_{survey}_{filt}.jpg"
        title = rf"{name}: {survey}-${filt}$"
        plot_detected_objects(
            hdu,
            gal_obj,
            scale * kronrad,
            ra,
            dec,
            host_ra,
            host_dec,
            True,
            title,
            outfile,
        )

    if survey in flipped_surveys:
        flip = True
    else:
        flip = False

    if save_aperture_params is True:
        gal_df = pd.DataFrame(gal_obj)
        # ellipse on the sky, to use the aperture in other images without this one
        gal_df = add_sky_ellipses(gal_df, gal_obj, wcs)
        gal_df["kronrad"] = kronrad
        gal_df["scale"] = scale
        gal_df["aperture_flag"] = aperture_flag  # (see optimize_aperture() and cog_aperture())
        gal_df["method"] = aperture_method
        if aperture_method == "cog":
            for key in ["r_meas", "sersic_n", "sersic_re", "tail_frac", "tail_frac_err"]:
                gal_df[key] = aperture_info[key]
        gal_df["flip"] = flip
        gal_df["filt"] = filt
        gal_df["survey"] = survey

        # calculate the directional light-radius distance
        if ra is not None and dec is not None:
            gal_pixel_x = gal_obj["x"][0]
            gal_pixel_y = gal_obj["y"][0]
            gal_world = wcs.wcs_pix2world(gal_pixel_x, gal_pixel_y, 0)
            gal_coord = SkyCoord(ra=gal_world[0] * u.deg, dec=gal_world[1] * u.deg, frame="icrs")
            sn_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
            separation = sn_coord.separation(gal_coord).arcsec  # transient-galaxy separation

            a_scaled = gal_obj["a"][0] * scale * kronrad
            b_scaled = gal_obj["b"][0] * scale * kronrad
            theta = gal_obj["theta"][0]

            sn_pixel = wcs.wcs_world2pix(ra, dec, 0)
            dx = sn_pixel[0] - gal_pixel_x
            dy = sn_pixel[1] - gal_pixel_y
            phi = np.arctan2(dy, dx)

            # The ellipse radius in the direction to the SN is calculated using the formula: 
            # r(φ) = (a*b) / sqrt((b*cos(φ-θ))2 + (a*sin(φ-θ))2), where ɸ is the angle from 
            # the galaxy center to the SN position, and θ is the ellipse position angle. 
            ellipse_radius_in_dir = (a_scaled * b_scaled) / np.sqrt(
                (b_scaled * np.cos(phi - theta))**2 + (a_scaled * np.sin(phi - theta))**2
            )
            
            pixel_scale = proj_plane_pixel_scales(wcs)[0] * 3600  # deg → arcsec
            ellipse_radius_arcsec = ellipse_radius_in_dir * pixel_scale

            # directional light-radius distance (separation in units of the aperture radius in that direction)
            directional_light_radius_norm = separation / ellipse_radius_arcsec

            gal_df["sep_arcsec"] = separation
            gal_df["d_dlr"] = directional_light_radius_norm
            gal_df["a_scaled"] = a_scaled
            gal_df["b_scaled"] = b_scaled

        outfile = obj_dir / survey / f"aperture_parameters_{filt}.csv"
        gal_df.to_csv(outfile, index=False)
    hdu.close()

    return gal_obj, wcs, kronrad, scale, flip, aperture_info


def load_aperture_params(
    name: str, filt: str | list, survey: str
) -> tuple[np.ndarray, WCS, float, float, bool]:
    """Loads previously saved aperture parameters.

    Parameters
    ----------
    name: Name of the object to find the path of the aperture-parameters file.
    filt: Name of the filter used for the aperture parameters. Coadds are
        also valid.
    survey: Survey name to be used.

    Returns
    -------
    gal_obj: Galaxy object.
    wcs: WCS of the reference image (``None`` if it is not available; only needed for parameters saved without
        the sky ellipse, i.e. before v4.0.0).
    kronrad: Kron radius.
    scale: Scale for the Kron radius.
    flip: Whether to flip the orientation of the aperture.
    """
    gal_obj, wcs, kronrad, scale, flip, _ = _load_aperture_params(name, filt, survey)
    return gal_obj, wcs, kronrad, scale, flip


def _load_aperture_params(
    name: str, filt: str | list, survey: str
) -> tuple[np.ndarray, WCS, float, float, bool, dict]:
    """As :func:`load_aperture_params`, also returning the details of the aperture (method and Sérsic tail of the
    CoG aperture)."""
    if isinstance(filt, list):
        filt = "".join(f for f in filt)
    obj_dir = Path(workdir, name)
    aper_params_file = obj_dir / survey / f"aperture_parameters_{filt}.csv"
    if not aper_params_file.is_file():
        raise FileNotFoundError(
            f"No aperture parameters found for {survey} {filt} of {name} ({aper_params_file}). Run the global "
            "photometry of the reference filter(s) and survey first with `save_aperture_params=True`."
        )
    gal_df = pd.read_csv(aper_params_file)

    # split parameters
    kronrad = gal_df.pop("kronrad").values[0]
    scale = gal_df.pop("scale").values[0]  # remove host-galaxy row
    flip = gal_df.pop("flip").values[0]
    # (files saved before the CoG aperture was added only have Kron apertures)
    method = gal_df.pop("method").values[0] if "method" in gal_df else "kron"
    aperture_info = {"method": method, "tail_frac": 0.0, "tail_frac_err": 0.0}
    for key in ["r_meas", "sersic_n", "sersic_re", "tail_frac", "tail_frac_err"]:
        if key in gal_df:
            aperture_info[key] = float(gal_df.pop(key).values[0])
    if "aperture_flag" in gal_df and bool(gal_df.pop("aperture_flag").values[0]):
        warnings.warn(
            f"The aperture of the reference ({survey} {filt}) might not include all the flux of the galaxy "
            "(see optimize_aperture() and cog_aperture()).",
            ApertureWarning,
        )
    # remove unused columns
    _ = gal_df.pop("filt")
    _ = gal_df.pop("survey")
    # DataFrame to structured/record array
    gal_obj = gal_df.to_records(index=False)  # (no extra "index" field, e.g. when saved again)

    # WCS of the reference image (only needed for parameters saved without the sky ellipse, i.e. before v4.0.0):
    # the aperture parameters were measured on the image trimmed by remove_nan(), so the same trimming is applied
    fits_file = obj_dir / survey / f"{survey}_{filt}.fits"
    wcs = None
    if fits_file.is_file():
        with fits.open(fits_file) as hdu:
            hdu = remove_nan(hdu)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", AstropyWarning)
                wcs = WCS(hdu[0].header, naxis=2)
    elif not has_sky_ellipses(gal_obj):
        raise FileNotFoundError(
            f"The aperture parameters of {survey} {filt} of {name} were saved without the sky ellipse (before "
            f"v4.0.0), so the reference image is needed ({fits_file})."
        )

    return gal_obj, wcs, kronrad, scale, flip, aperture_info


def _photometry(
    name: str,
    host_ra: float,
    host_dec: float,
    filt: str,
    survey: str,
    ra: Optional[float] = None,
    dec: Optional[float] = None,
    bkg_sub: Optional[bool] = None,
    threshold: float = 10,
    use_mask: bool = True,
    correct_extinction: bool = False,
    optimize_kronrad: bool = False,
    eps: float = 0.0001,
    gal_dist_thresh: float = -1,
    deblend_cont: float = 0.005,
    common_aperture: bool = False,
    ref_filt: Optional[str] = None,
    ref_survey: Optional[str] = None,
    save_plots: bool = True,
    save_aperture_params: bool = False,
    return_flags: bool = False,
    aperture_method: str = "kron",
) -> tuple[float, float, float, float, float]:
    """Calculates the global aperture photometry of a galaxy using
    the Kron flux (or the curve-of-growth flux; see ``aperture_method``).

    **Note:** the galaxy must be ideally centred in the image.

    Parameters
    ----------
    name: Name of the object to find the path of the fits file.
    host_ra: Host-galaxy right ascension of the galaxy in degrees.
    host_dec: Host-galaxy declination of the galaxy in degrees.
    filt: Filter to use to load the fits file.
    survey: Survey to use for the zero-points and pixel scale.
    ra: Right ascension of an object, in degrees. Used for plotting the position of the object.
    dec: Declination of an object, in degrees. Used for plotting the position of the object.
    bkg_sub: If ``True``, the image gets background subtracted. By default, only
        the images that need it get background subtracted (WISE, 2MASS and
        VISTA).
    threshold: Threshold used by `sep.extract()` to extract objects.
    use_mask: If `True`, the masked fits files are used. These must have
        been created beforehand.
    correct_extinction: If `True`, corrects for Milky-Way extinction using the recalibrated dust maps
        by Schlafly & Finkbeiner (2011) and the extinction law from Fitzpatrick (1999). Not applied by default.
    optimize_kronrad: If `True`, the Kron-radius scale is optimized, increasing the aperture
        size until the flux gained is consistent with the noise (see :func:`optimize_aperture`).
        Otherwise (default), the standard Kron scale of 2.5 is used.
    eps: Minimum fractional change in flux used to stop the optimisation of the Kron-radius
        scale (if the flux gained is not already within the noise). A value of 0.0001 means
        0.01% change in flux.
    gal_dist_thresh: Distance in arcsec to crossmatch the galaxy coordinates with a detected object,
        where the object nearest to the galaxy position is considered as the galaxy (within
        the given threshold). If no objects are found within the given distance threshold,
        the galaxy is considered as not found and a warning is printed. If a non-positive value
        is given (default option), the closest detected object is considered as the galaxy only if
        the galaxy coordinates are within its extent (see :func:`hostphot.processing.objects_detection.extract_objects`).
    deblend_cont : Minimum contrast ratio used for object deblending. Default is 0.005.
        To entirely disable deblending, set to 1.0.
    common_aperture: Whether to use common aperture photometry using the reference filter(s) and survey.
    ref_filt: Reference filter (or coadd filters) from which to use the aperture parameters. Note that the parameters
        need to have been previously saved with ``save_aperture_params=True``.
    ref_survey: Reference survey from which to use the aperture parameters. Note that the parameters
        need to have been previously saved with ``save_aperture_params=True``.
    save_plots: If `True`, the mask and galaxy aperture figures are saved.
    save_aperture_params: Whether to save the aperture parameters.
    return_flags: If ``True``, the quality flags (see :mod:`hostphot.flags`; ``True`` if the issue was found,
        with a warning) are returned in a dictionary (``flags``), as a sixth value.
    aperture_method: Aperture used: ``"kron"`` (default), the Kron aperture (see ``optimize_kronrad``), or
        ``"cog"``, the curve-of-growth aperture with a Sérsic tail (see :func:`cog_aperture`). With a common
        aperture, it must be the method of the reference aperture, and the Sérsic tail of the reference (as a
        fraction of the flux within the aperture) is applied to all the filters.

    Returns
    -------
    mag: Aperture magnitude.
    mag_err: Error on the aperture magnitude.
    flux: Aperture flux.
    flux_err: Total flux error on the aperture flux.
    zp: Zeropoint.
    details: For Herschel (or with ``return_flags=True``), dictionary with details of the measurement
        (e.g. units, conversion factor to Jy/pixel and blank-aperture noise for Herschel, and the quality
        flags), as a sixth value.
    """
    # initial checks
    check_survey_validity(survey)
    check_work_dir(workdir)
    _check_aperture_method(aperture_method, optimize_kronrad)
    obj_dir = Path(workdir, name)
    if use_mask:
        suffix = "_masked"
    else:
        suffix = ""
    fits_file = obj_dir / survey / f"{survey}_{filt}{suffix}.fits"

    # image information
    with fits.open(fits_file, memmap=False) as hdu_file:
        _ = [extension.data for extension in hdu_file]  # (loaded before closing the file)
        hdu = remove_nan(hdu_file)
    saturated = saturated_mask(hdu)  # e.g. HSC (None if not available)
    header = hdu[0].header
    data = hdu[0].data
    _exptime = get_effective_gain(header, survey, filt)  # for the Poisson noise of the sources
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        wcs = WCS(header, naxis=2)

    data = data.astype(np.float64)
    bkg = _background(data)
    # background subtraction, if needed
    back_info = None
    if (bkg_sub is None and survey in bkg_surveys) or bkg_sub is True:
        # background estimated with the sources (including the galaxy) masked
        back, back_info = masked_background(
            data, wcs, host_ra, host_dec, label=f"{survey} {filt}", return_info=True
        )
        data_sub = data - back
    else:
        data_sub = np.copy(data)

    if common_aperture is True and (ref_filt is None or ref_survey is None):
        raise ValueError(
            "Reference filter(s) 'ref_filt' and survey 'ref_survey' must be given for common aperture photometry"
        )
    if (ref_filt is None) | (ref_survey is None) | (common_aperture is False):
        # independent photometry for each filter
        gal_obj, wcs, kronrad, scale, _, aperture_info = _extract_aperture(
            name,
            host_ra,
            host_dec,
            filt,
            survey,
            ra=ra,
            dec=dec,
            bkg_sub=bkg_sub,
            threshold=threshold,
            use_mask=use_mask,
            optimize_kronrad=optimize_kronrad,
            eps=eps,
            gal_dist_thresh=gal_dist_thresh,
            deblend_cont=deblend_cont,
            save_plots=save_plots,
            save_aperture_params=save_aperture_params,
            save_input=False,  # (the inputs of the photometry are saved by multi_band_phot())
            aperture_method=aperture_method,
        )
    else:
        # common aperture photometry
        gal_obj, master_wcs, kronrad, scale, flip2, aperture_info = _load_aperture_params(
            name, ref_filt, ref_survey
        )
        if aperture_info["method"] != aperture_method:
            raise ValueError(
                f"The reference aperture ({ref_survey} {ref_filt}) was measured with "
                f"aperture_method='{aperture_info['method']}', not '{aperture_method}'."
            )
        # adapt ellipses between surveys
        if survey in flipped_surveys:
            flip = True
        else:
            flip = False
        if flip == flip2:
            flip_ = False
        else:
            flip_ = True
        if has_sky_ellipses(gal_obj):
            # (exact for any orientation of the images, without the reference image)
            gal_obj, conv_factor = objects_from_sky(gal_obj, wcs)
        else:
            # parameters saved before v4.0.0: through the reference image
            gal_obj, conv_factor = adapt_aperture(gal_obj, master_wcs, wcs, flip_)

    aperture_name = "CoG" if aperture_method == "cog" else "Kron"
    # reliability of the subtracted background in the aperture (with a warning if unreliable)
    if back_info is not None:
        x, y, a, b, theta = _kron_ellipse(gal_obj, kronrad, scale)
        aperture_pixels = (
            EllipticalAperture((x, y), a, b, theta=theta).to_mask(method="center").to_image(data.shape) > 0
        )
        background_flag(data, back, back_info, aperture_pixels, f"{survey} {filt} ({aperture_name} aperture)")

    if survey == "Herschel":
        flux, flux_err, herschel_details = herschel_kron_flux(
            data_sub, header, wcs, gal_obj, kronrad, scale, filt
        )
        # as for the other surveys, the flux is not measured if the aperture lacks too much data
        x, y, a, b, theta = _kron_ellipse(gal_obj, kronrad, scale)
        kron_aperture = EllipticalAperture((x, y), a, b, theta=theta)
        usable, missing_fraction = check_global_coverage(
            data_sub, kron_aperture, f"{survey} {filt} ({aperture_name} aperture)"
        )
        if not usable:
            flux, flux_err = np.nan, np.nan
        else:
            flux, flux_err = correct_missing_flux(flux, flux_err, missing_fraction)
        if aperture_method == "cog":
            flux, flux_err = apply_cog_tail(flux, flux_err, aperture_info["tail_frac"], aperture_info["tail_frac_err"])
            herschel_details["snr"] = flux / flux_err if flux_err > 0 else np.nan
        zp = 3631.0
        if flux > 0 and flux_err > 0:
            mag = -2.5 * np.log10(flux / zp)
            mag_err = np.abs(2.5 * flux_err / (flux * np.log(10)))
        else:
            if flux <= 0:
                warn_non_detection(survey, filt, flux)
            mag = np.nan
            mag_err = np.nan
        herschel_details["magnitude_system"] = "AB"
        herschel_details["zeropoint_unit"] = "Jy"

        if correct_extinction is True:
            # as for the other surveys (negligible in the far infrared)
            A_ext = calc_extinction(filt, survey, host_ra, host_dec)
            mag -= A_ext
            flux *= 10 ** (0.4 * A_ext)
            flux_err *= 10 ** (0.4 * A_ext)  # (the magnitude error is unchanged)

        if save_plots is True:
            outfile = obj_dir / survey / f"global_{survey}_{filt}.jpg"
            filt_ = filt.replace("_", "-")
            title = f"{name}: {survey}-${filt_}$"
            plot_detected_objects(
                hdu,
                gal_obj,
                scale * kronrad,
                ra,
                dec,
                host_ra,
                host_dec,
                title=title,
                outfile=outfile,
            )
        hdu.close()
        return mag, mag_err, flux, flux_err, zp, herschel_details

    # get Kron flux
    invvar = get_invvar_map(hdu, survey)
    x, y, a, b, theta = _kron_ellipse(gal_obj, kronrad, scale)
    label = f"{survey} {filt} ({aperture_name} aperture)"
    pixel_scale = float(np.sqrt(abs(np.linalg.det(wcs.pixel_scale_matrix))) * 3600)  # arcsec
    # the outer parts of the Kron aperture contain little flux, so the missing flux is checked (and added)
    kron_aperture = EllipticalAperture((x, y), a, b, theta=theta)
    usable, missing_fraction = False, 0.0
    if check_saturation(saturated, kron_aperture, label):
        usable, missing_fraction = check_global_coverage(data_sub, kron_aperture, label)
    if not usable:
        flux, flux_err = np.nan, np.nan
    elif invvar is not None:
        flux, flux_err = extract_invvar_kron_flux(
            data_sub,
            invvar,
            gal_obj,
            kronrad,
            scale,
            pixel_scale=pixel_scale,
            label=label,
        )
    else:
        flux, flux_err = kron_flux(
            data_sub, bkg.rms(), _exptime, gal_obj, kronrad, scale
        )
        if survey in blank_floor_surveys:
            flux_err = blank_kron_floor(
                data_sub, flux_err, gal_obj, kronrad, scale, pixel_scale=pixel_scale, label=label,
                residual_background=floor_residual_background(survey, filt),
            )
    # flux in the pixels without data
    flux, flux_err = correct_missing_flux(flux, flux_err, missing_fraction)
    if aperture_method == "cog":
        # flux beyond the aperture (Sérsic tail)
        flux, flux_err = apply_cog_tail(flux, flux_err, aperture_info["tail_frac"], aperture_info["tail_frac_err"])

    # aperture area of the (scaled) Kron ellipse
    ap_area = kron_aperture_area(gal_obj, kronrad, scale)
    mag, mag_err, flux, flux_err, zp = magnitude_calculation(
        flux,
        flux_err,
        survey,
        filt,
        ap_area,
        header,
        bkg.globalrms,
    )

    if correct_extinction is True:
        A_ext = calc_extinction(filt, survey, host_ra, host_dec)
        mag -= A_ext
        flux *= 10 ** (0.4 * A_ext)
        flux_err *= 10 ** (0.4 * A_ext)  # (the magnitude error is unchanged)

    if save_plots is True:
        outfile = obj_dir / survey / f"global_{survey}_{filt}.jpg"
        filt_ = filt.replace("_", "-")
        title = f"{name}: {survey}-${filt_}$"
        plot_detected_objects(
            hdu,
            gal_obj,
            scale * kronrad,
            ra,
            dec,
            host_ra,
            host_dec,
            title=title,
            outfile=outfile,
        )
    hdu.close()

    return mag, mag_err, flux, flux_err, zp, {}


@functools.wraps(_photometry)
def photometry(*args, **kwargs) -> tuple:
    # the quality flags are found from the warnings raised during the photometry (see hostphot.flags) and from
    # the header of the image (flags set when downloading or masking it)
    params = inspect.signature(_photometry).bind(*args, **kwargs)
    params.apply_defaults()
    with collect_flags() as flags:
        mag, mag_err, flux, flux_err, zp, details = _photometry(*params.args, **params.kwargs)
    p = params.arguments
    suffix = "_masked" if p["use_mask"] else ""
    fits_file = Path(workdir, p["name"], p["survey"], f"{p['survey']}_{p['filt']}{suffix}.fits")
    flags |= header_flags(fits.getheader(fits_file))
    if p["return_flags"]:
        details["flags"] = {flag: flag in flags for flag in GLOBAL_FLAGS}
        return mag, mag_err, flux, flux_err, zp, details
    if details:
        return mag, mag_err, flux, flux_err, zp, details  # (Herschel)
    return mag, mag_err, flux, flux_err, zp


def multi_band_phot(
    name: str,
    host_ra: float,
    host_dec: float,
    filters: Optional[str | list] = None,
    survey: str = "PanSTARRS",
    ra: Optional[float] = None,
    dec: Optional[float] = None,
    bkg_sub: Optional[bool] = None,
    threshold: float = 10,
    use_mask: bool = True,
    correct_extinction: bool = False,
    common_aperture: bool = False,
    ref_filt: Optional[str | list] = None,
    ref_survey: Optional[str | list] = None,
    optimize_kronrad: bool = False,
    eps: float = 0.0001,
    gal_dist_thresh: float = -1,
    deblend_cont: float = 0.005,
    save_plots: bool = True,
    save_aperture_params: bool = True,
    save_results: bool = True,
    raise_exception: bool = True,
    save_input: bool = True,
    aperture_method: str = "kron",
):
    """Calculates multi-band aperture photometry of the host galaxy
    for an object.

    Parameters
    ----------
    name: Name of the object to find the path of the fits file.
    host_ra: Host-galaxy right ascension of the galaxy in degrees.
    host_dec: Host-galaxy declination of the galaxy in degrees.
    filters: Filters to use to load the fits files. If `None` use all
        the filters of the given survey.
    survey: Survey to use for the zero-points and pixel scale.
    ra: Right ascension of an object, in degrees. Used for plotting the position of the object.
    dec: Declination of an object, in degrees. Used for plotting the position of the object.
    bkg_sub: If ``True``, the image gets background subtracted. By default, only
        the images that need it get background subtracted (WISE, 2MASS and
        VISTA).
    threshold: Threshold used by :func:`sep.extract()` to extract objects.
    use_mask: If ``True``, the masked fits files are used. These must have
        been created beforehand.
    correct_extinction: If ``True``, the magnitudes are corrected for Milky-Way extinction (not applied by default).
    common_aperture: Whether to use common aperture photometry using the reference filter(s) and survey.
    ref_filt: Reference filter(s) used for common aperture photometry.
    ref_survey: Reference survey used for common aperture photometry. By default uses the same
        as the input 'survey'.
    optimize_kronrad: If `True`, the Kron-radius scale is optimized, increasing the aperture
        size until the flux gained is consistent with the noise (see :func:`optimize_aperture`).
        Otherwise (default), the standard Kron scale of 2.5 is used.
    eps: Minimum fractional change in flux used to stop the optimisation of the Kron-radius
        scale (if the flux gained is not already within the noise). A value of 0.0001 means
        0.01% change in flux.
        when optimizing the Kron radius.
    gal_dist_thresh: Distance in arcsec to crossmatch the galaxy coordinates with a detected object,
        where the object nearest to the galaxy position is considered as the galaxy (within
        the given threshold). If no objects are found within the given distance threshold,
        the galaxy is considered as not found and a warning is printed. If a non-positive value
        is given (default option), the closest detected object is considered as the galaxy only if
        the galaxy coordinates are within its extent (see :func:`hostphot.processing.objects_detection.extract_objects`).
    deblend_cont : Minimum contrast ratio used for object deblending. Default is 0.005.
        To entirely disable deblending, set to 1.0.
    save_plots: If ``True``, the mask and galaxy aperture figures are saved.
    save_aperture_params: Whether to save the aperture parameters.
    save_results: If ``True``, the magnitudes are saved into a csv file.
    raise_exception: If ``True``, an exception is raised if the photometry fails for any filter.
    save_input: Whether to save the input parameters.
    aperture_method: Aperture used: ``"kron"`` (default), the Kron aperture (see ``optimize_kronrad``), or
        ``"cog"``, the curve-of-growth aperture with a Sérsic tail (see :func:`cog_aperture`). With a common
        aperture, it must be the method of the reference aperture, and the Sérsic tail of the reference (as a
        fraction of the flux within the aperture) is applied to all the filters.

    Returns
    -------
    phot_df: DataFrame with the object's photometry and other info.

    Examples
    --------
    >>> from hostphot.photometry import global_photometry as gp
    >>> name = 'SN2004eo'
    >>> host_ra, host_dec = 308.2092, 9.92755  # coords of host galaxy of SN2004eo
    >>> ra, dec =  308.22579, 9.92853 # coords of SN2004eo
    >>> results = gp.multi_band_phot(name, 
                                     host_ra, 
                                     host_dec, 
                                     survey=survey, 
                                     ra=ra, 
                                     dec=dec, 
                                     use_mask=True,
                                     common_aperture=True, 
                                     ref_filt='riz',
                                     ref_survey='PanSTARRS', 
                                     save_plots=True)
    """
    input_params = locals()  # dictionary
    check_survey_validity(survey)
    if filters is None:
        if survey in ["HST", "JWST"]:
            raise ValueError(f"For {survey}, the filter needs to be specified!")
        filters = get_default_filters(survey)
    else:
        filters = normalize_filters(filters, survey)
        check_filters_validity(filters, survey)

    # save input parameters
    if save_input is True:
        inputs_file = Path(workdir, name, survey, "input_global_photometry.csv")
        store_input(input_params, inputs_file)

    results_dict = {
        "name": name,
        "host_ra": host_ra,
        "host_dec": host_dec,
        "survey": survey,
        "hostphot_version": __version__,
    }

    if ref_survey is None:
        ref_survey = survey
    if common_aperture is True and ref_filt is None:
        raise ValueError("The reference filter(s) 'ref_filt' must be given for common aperture photometry")

    for filt in filters:
        try:
            phot_res = photometry(
                name,
                host_ra,
                host_dec,
                filt,
                survey,
                ra=ra,
                dec=dec,
                bkg_sub=bkg_sub,
                threshold=threshold,
                use_mask=use_mask,
                correct_extinction=correct_extinction,
                optimize_kronrad=optimize_kronrad,
                eps=eps,
                gal_dist_thresh=gal_dist_thresh,
                deblend_cont=deblend_cont,
                common_aperture=common_aperture,
                ref_filt=ref_filt,
                ref_survey=ref_survey,
                save_plots=save_plots,
                save_aperture_params=save_aperture_params,
                return_flags=True,
                aperture_method=aperture_method,
            )
            mag, mag_err, flux, flux_err, zp = phot_res[:5]
            extra_results = phot_res[5] if len(phot_res) > 5 else {}
            results_dict[filt] = mag
            results_dict[f"{filt}_err"] = mag_err
            results_dict[f"{filt}_flux"] = flux
            results_dict[f"{filt}_flux_err"] = flux_err
            results_dict[f"{filt}_zeropoint"] = zp
            # AB magnitude, flux density (microjanskys) and upper limit, the same for all the surveys
            for key, value in ab_photometry(survey, filt, flux, flux_err, zp).items():
                results_dict[f"{filt}_{key}"] = value
            # quality flags (e.g. unreliable subtracted background, partial coverage, calibration)
            for flag, value in extra_results.pop("flags", {}).items():
                results_dict[f"{filt}_{flag}_flag"] = value
            for extra_key, extra_value in extra_results.items():
                results_dict[f"{filt}_{extra_key}"] = extra_value
            # provenance of the image
            image_header = fits.getheader(Path(workdir, name, survey, f"{survey}_{filt}.fits"))
            image_id, download_date = image_provenance(image_header, survey)
            results_dict[f"{filt}_image_id"] = image_id
            results_dict[f"{filt}_download_date"] = download_date
        except Exception as exc:
            if raise_exception is True:
                raise Exception(f"{filt}-band: {exc}")
            else:
                results_dict[filt] = np.nan
                results_dict[f"{filt}_err"] = np.nan
                results_dict[f"{filt}_flux"] = np.nan
                results_dict[f"{filt}_flux_err"] = np.nan
                results_dict[f"{filt}_zeropoint"] = np.nan
                for key in ["AB", "flux_uJy", "flux_uJy_err", "AB_ulim"]:
                    results_dict[f"{filt}_{key}"] = np.nan
                for flag in GLOBAL_FLAGS:
                    results_dict[f"{filt}_{flag}_flag"] = np.nan
                results_dict[f"{filt}_image_id"] = ""
                results_dict[f"{filt}_download_date"] = ""

    phot_df = pd.DataFrame({key: [val] for key, val in results_dict.items()})
    if save_results is True:
        outfile = Path(workdir, name, survey, "global_photometry.csv")
        phot_df.to_csv(outfile, index=False)

    return phot_df
