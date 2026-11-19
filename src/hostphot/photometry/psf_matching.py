"""PSF matching of the images for the local photometry.

The flux within a small aperture depends on the resolution of the image: for apertures comparable to the PSF, part
of the light of the region is spread outside the aperture, while light of the surroundings is spread into it. To
compare (or combine in an SED) local fluxes from images of different resolution, the images can be convolved to a
common, broader PSF (``psf_match`` in :func:`hostphot.photometry.local_photometry.multi_band_phot`).

- The PSF FWHM of each image is taken from its header, if available (e.g. KiDS, UKIDSS, UHS, VISTA), or measured
  from Gaussian fits to the isolated, unsaturated Gaia DR3 stars in the image (:func:`measure_fwhm`); if there are
  fewer than ``MIN_STARS``, the typical value of the filter is used (with a warning and the ``psf_match`` flag).
  The seeing of ground-based surveys varies by tens of per cent from image to image, so a typical value is not
  accurate enough for the matching.
- The images are convolved with a Gaussian kernel of width ``sqrt(FWHM_target^2 - FWHM_image^2)``. For an unWISE
  W1 or W2 target, whose PSF has strong wings (e.g. only ~87% of the light of a star within 11 arcsec in W1, while
  a Gaussian of the same FWHM has all of it), the kernel is instead computed between a Gaussian of the FWHM of the
  image and the unWISE PSF model (Meisner & Schlafly; azimuthally averaged, both scan directions, stretched to the
  FWHM measured in the unWISE image), as a Gaussian kernel leaves the matched fluxes ~6-7% too bright.
- Images whose PSF is already broader than the target are not convolved (with a warning and the ``psf_match`` flag).

The convolution correlates the noise on the scale of the target PSF, which is included in the uncertainties by the
blank-aperture floor of the local photometry (:func:`hostphot.photometry.photometry_utils.local_blank_floor`).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import sep
from astropy.convolution import Gaussian2DKernel, convolve_fft
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.optimize import curve_fit

from hostphot.flags import PSFMatchWarning
from hostphot.photometry.photometry_utils import PSF_FWHM_KEYWORDS
from hostphot.surveys_utils import survey_psf_fwhm

SIG2FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))
# minimum number of stars to measure the FWHM of an image
MIN_STARS = 5
# minimum peak signal-to-noise ratio of the stars used
MIN_STAR_SNR = 30.0
# surveys whose own PSF has strong wings, not modelled by the matching kernels
WINGED_SURVEYS = ("Spitzer", "WISE", "unWISE", "GALEX")
# unWISE bands with a PSF model (kernel from the model for an unWISE target in these bands)
UNWISE_MODEL_BANDS = ("W1", "W2")
UNWISE_PIXEL_SCALE = 2.75  # arcsec
# radius of the unWISE matching kernel (arcsec)
UNWISE_KERNEL_RADIUS = 60.0
# targets broader than this (arcsec) include light of faint and neighbouring sources spread by the PSF (confusion)
CONFUSION_FWHM = 5.0
_PSF_MODELS_DIR = Path(__file__).resolve().parent.parent / "psf_models"


@dataclass(frozen=True)
class PSFTarget:
    """Target PSF of the matching: FWHM (arcsec) and, if it is the PSF of an image, its survey and filter."""

    fwhm: float
    survey: Optional[str] = None
    filt: Optional[str] = None

    @property
    def unwise_model(self) -> bool:
        """Whether the kernel is computed from the unWISE PSF model (unWISE W1 or W2 target)."""
        return self.survey == "unWISE" and self.filt in UNWISE_MODEL_BANDS


def _pixel_scale(header: fits.Header) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        return float(np.mean(proj_plane_pixel_scales(WCS(header, naxis=2))) * 3600)


def _gaussian(coords, amp, x0, y0, sigma, const):
    x, y = coords
    return amp * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2)) + const


def _fit_star(stamp: np.ndarray, guess_sigma: float) -> tuple[float, float, bool]:
    """Circular Gaussian (plus constant) fit to a stamp centred on a star: (sigma in pixels, peak S/N, valid);
    saturated (flat-topped) cores are not valid."""
    ny, nx = stamp.shape
    y, x = np.indices(stamp.shape, dtype=float)
    edge = np.concatenate([stamp[0], stamp[-1], stamp[:, 0], stamp[:, -1]])
    const0 = np.median(edge)
    amp0 = stamp[ny // 2 - 1 : ny // 2 + 2, nx // 2 - 1 : nx // 2 + 2].max() - const0
    if not amp0 > 0:
        return np.nan, np.nan, False
    p0 = [amp0, nx / 2 - 0.5, ny / 2 - 0.5, guess_sigma, const0]
    bounds = ([0, nx / 2 - 3, ny / 2 - 3, 0.3, -np.inf], [np.inf, nx / 2 + 2, ny / 2 + 2, 4 * guess_sigma + 3, np.inf])
    try:
        popt, _ = curve_fit(_gaussian, (x.ravel(), y.ravel()), stamp.ravel(), p0=p0, bounds=bounds, maxfev=5000)
    except (RuntimeError, ValueError):
        return np.nan, np.nan, False
    resid = stamp - _gaussian((x, y), *popt)
    noise = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    snr = popt[0] / noise if noise > 0 else np.inf
    xc, yc = int(round(popt[1])), int(round(popt[2]))
    core = stamp[max(yc - 1, 0) : yc + 2, max(xc - 1, 0) : xc + 2]
    flat = core.size > 0 and (core.max() - popt[4]) < 0.85 * popt[0]
    return float(popt[3]), float(snr), not flat


def measure_fwhm(
    data: np.ndarray, header: fits.Header, stars, guess: float, min_snr: float = MIN_STAR_SNR
) -> tuple[float, float, int]:
    """Measures the PSF FWHM of an image from its stars.

    Circular Gaussian fits to the isolated (no other detected source within the stamp), unsaturated stars with a
    peak signal-to-noise ratio above ``min_snr``; the median FWHM, after a 3-sigma clipping.

    Parameters
    ----------
    data: Image data.
    header: Image header (with the WCS).
    stars: Coordinates of the stars (``SkyCoord``; e.g. Gaia DR3 stars).
    guess: Initial guess of the FWHM in arcsec (sets the size of the stamps).
    min_snr: Minimum peak signal-to-noise ratio of the stars.

    Returns
    -------
    fwhm: Median FWHM in arcsec (NaN if no star is valid).
    nmad: Normalised median absolute deviation of the FWHMs of the stars.
    n_stars: Number of stars used.
    """
    if stars is None or len(stars) == 0:
        return np.nan, np.nan, 0
    scale = _pixel_scale(header)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        img_wcs = WCS(header, naxis=2)
    guess_sigma = max(guess / scale / SIG2FWHM, 0.5)
    half = int(np.ceil(max(2 * guess_sigma * SIG2FWHM, 5)))
    filled = np.ascontiguousarray(np.nan_to_num(np.asarray(data, dtype=np.float64)))
    try:
        bkg = sep.Background(filled)
        back = bkg.back()
        objects = sep.extract(filled - back, 5, err=bkg.globalrms, minarea=5)
    except Exception:
        return np.nan, np.nan, 0
    x, y = img_wcs.world_to_pixel(stars)
    fwhms = []
    for xs, ys in zip(np.atleast_1d(x), np.atleast_1d(y)):
        if not (np.isfinite(xs) and np.isfinite(ys)):
            continue
        xi, yi = int(round(float(xs))), int(round(float(ys)))
        if not (half < xi < data.shape[1] - half - 1 and half < yi < data.shape[0] - half - 1):
            continue
        window = (slice(yi - half, yi + half + 1), slice(xi - half, xi + half + 1))
        stamp = np.asarray(data[window], dtype=float)
        if not np.all(np.isfinite(stamp)):
            continue
        if np.sum(np.hypot(objects["x"] - xs, objects["y"] - ys) < 1.2 * half) > 1:
            continue  # not isolated
        sigma, snr, valid = _fit_star(stamp - np.median(back[window]), guess_sigma)
        if valid and np.isfinite(sigma) and snr >= min_snr:
            fwhms.append(sigma * SIG2FWHM * scale)
    fwhms = np.array(fwhms)
    if len(fwhms) == 0:
        return np.nan, np.nan, 0
    for _ in range(3):  # (binaries, galaxies or blends)
        median = np.median(fwhms)
        nmad = 1.4826 * np.median(np.abs(fwhms - median))
        keep = np.abs(fwhms - median) < 3 * max(nmad, 0.02 * median)
        if keep.all():
            break
        fwhms = fwhms[keep]
    median = float(np.median(fwhms))
    return median, float(1.4826 * np.median(np.abs(fwhms - median))), len(fwhms)


def _gaia_stars(header: fits.Header, shape: tuple):
    """Gaia DR3 stars in the image (``None`` if the query failed)."""
    from hostphot.processing.objects_detection import find_gaia_objects

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        img_wcs = WCS(header, naxis=2)
    ny, nx = shape
    centre = img_wcs.pixel_to_world((nx - 1) / 2, (ny - 1) / 2)
    radius = 0.5 * np.hypot(nx, ny) * _pixel_scale(header) / 3600
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # (cached in memory only: a file would replace the cache of the masking, centred on the host)
        stars = find_gaia_objects(round(centre.ra.deg, 3), round(centre.dec.deg, 3), rad=float(radius) * 1.05)
    if len(stars) == 0 and any("Gaia" in str(w.message) for w in caught):
        return None
    return stars


@lru_cache(maxsize=256)
def _measured_fwhm(path: str, mtime: float, survey: str, filt: str) -> tuple[float, str, int]:
    with fits.open(path, memmap=False) as hdul:
        data, header = np.asarray(hdul[0].data, dtype=float), hdul[0].header.copy()
    for keyword in PSF_FWHM_KEYWORDS:
        try:
            value = float(header[keyword])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value) and value > 0:
            return value, "header", 0
    typical = survey_psf_fwhm(survey, filt)
    guess = typical if np.isfinite(typical) else 3 * _pixel_scale(header)
    stars = _gaia_stars(header, data.shape)
    fwhm, _, n_stars = measure_fwhm(data, header, stars, guess)
    if n_stars >= MIN_STARS:
        return fwhm, "stars", n_stars
    return typical, "typical", n_stars


def image_fwhm(image_file: str | Path, survey: str, filt: str) -> tuple[float, str]:
    """PSF FWHM (arcsec) of an image for the matching: from its header (``PSF_FWHM_KEYWORDS``), measured from its
    Gaia DR3 stars (:func:`measure_fwhm`), or the typical value of the filter (with a warning and the ``psf_match``
    flag), and its source (``"header"``, ``"stars"`` or ``"typical"``). The image should not be masked (the stars
    are removed in the masked images)."""
    path = Path(image_file)
    fwhm, source, n_stars = _measured_fwhm(str(path), path.stat().st_mtime, survey, filt)
    if source == "typical":
        warnings.warn(
            f"{survey} {filt}: the PSF FWHM could not be measured from stars ({n_stars} valid Gaia stars, at least "
            f"{MIN_STARS} needed), so the typical value of the filter ({fwhm:.3g} arcsec) is used for the PSF "
            "matching, which is less accurate.",
            PSFMatchWarning,
        )
    return fwhm, source


def resolve_target(psf_match, name: str, filters: list, survey: str, workdir: str | Path) -> PSFTarget:
    """Target PSF of the matching.

    Parameters
    ----------
    psf_match: Target: FWHM in arcsec (Gaussian kernel); ``(survey, filter)`` of an image of the object (its FWHM,
        measured as for the other images; the unWISE PSF model is used for unWISE W1 and W2), or ``"broadest"``,
        the broadest PSF among the given filters.
    name: Name of the object.
    filters: Filters of the photometry (for ``"broadest"``).
    survey: Survey of the photometry.
    workdir: Working directory.

    Returns
    -------
    target: Target PSF.
    """
    obj_dir = Path(workdir, name)
    if isinstance(psf_match, PSFTarget):
        return psf_match
    if isinstance(psf_match, (int, float, np.floating, np.integer)) and not isinstance(psf_match, bool):
        if not psf_match > 0:
            raise ValueError(f"The target FWHM of the PSF matching must be positive ({psf_match}).")
        return PSFTarget(float(psf_match))
    if isinstance(psf_match, str) and psf_match == "broadest":
        best = None
        for filt in filters:
            image_file = obj_dir / survey / f"{survey}_{filt}.fits"
            if not image_file.is_file():
                continue
            fwhm, _ = image_fwhm(image_file, survey, filt)
            if np.isfinite(fwhm) and (best is None or fwhm > best.fwhm):
                best = PSFTarget(fwhm, survey, filt)
        if best is None:
            raise ValueError(f"The broadest PSF could not be found: no {survey} images of {name} with a known FWHM.")
        return best
    if isinstance(psf_match, (tuple, list)) and len(psf_match) == 2:
        target_survey, target_filt = psf_match
        image_file = obj_dir / target_survey / f"{target_survey}_{target_filt}.fits"
        if image_file.is_file():
            fwhm, _ = image_fwhm(image_file, target_survey, target_filt)
        else:
            fwhm = survey_psf_fwhm(target_survey, target_filt)
            warnings.warn(
                f"No {target_survey} {target_filt} image of {name} found, so the typical PSF FWHM of the filter "
                f"({fwhm:.3g} arcsec) is used as the target of the PSF matching.",
                PSFMatchWarning,
            )
        if not np.isfinite(fwhm):
            raise ValueError(f"The PSF FWHM of {target_survey} {target_filt} is not known.")
        return PSFTarget(float(fwhm), target_survey, target_filt)
    raise ValueError(
        f"Not a valid psf_match ({psf_match!r}): a FWHM in arcsec, a (survey, filter) tuple or 'broadest'."
    )


@lru_cache(maxsize=4)
def _unwise_profile(band: str) -> tuple[np.ndarray, np.ndarray]:
    """Azimuthally averaged unWISE PSF model (both scan directions): radii (arcsec) and surface brightness
    (normalised to a total flux of 1, per arcsec^2)."""
    model = fits.getdata(_PSF_MODELS_DIR / f"unwise_psf_model_{band.lower()}.fits").astype(float)
    model = (model + model[::-1, ::-1]) / 2.0
    model /= model.sum()
    centre = (model.shape[0] - 1) / 2
    y, x = np.indices(model.shape)
    r = np.hypot(x - centre, y - centre)
    edges = np.arange(0, r.max() + 0.5, 0.5)
    index = np.digitize(r.ravel(), edges)
    counts = np.bincount(index)
    valid = counts > 0
    profile = np.bincount(index, model.ravel())[valid] / counts[valid]
    radii = np.bincount(index, r.ravel())[valid] / counts[valid]
    return radii * UNWISE_PIXEL_SCALE, profile / UNWISE_PIXEL_SCALE**2


def _profile_image(radii: np.ndarray, profile: np.ndarray, scale: float, half: int, oversample: int = 5) -> np.ndarray:
    """PSF image on a grid of the given pixel scale (arcsec) from a radial profile (pixel-integrated, normalised)."""
    n = 2 * half + 1
    y, x = (np.indices((n * oversample, n * oversample)) + 0.5) / oversample - n / 2
    image = np.interp(np.hypot(x, y) * scale, radii, profile, right=0.0)
    image = image.reshape(n, oversample, n, oversample).sum(axis=(1, 3))
    return image / image.sum()


def _gaussian_image(fwhm: float, scale: float, half: int, oversample: int = 5) -> np.ndarray:
    sigma = fwhm / SIG2FWHM
    n = 2 * half + 1
    y, x = (np.indices((n * oversample, n * oversample)) + 0.5) / oversample - n / 2
    image = np.exp(-((np.hypot(x, y) * scale) ** 2) / (2 * sigma**2))
    image = image.reshape(n, oversample, n, oversample).sum(axis=(1, 3))
    return image / image.sum()


@lru_cache(maxsize=4)
def unwise_model_fwhm(band: str) -> float:
    """FWHM (arcsec) of the unWISE PSF model measured as for the stars (:func:`measure_fwhm`)."""
    guess = survey_psf_fwhm("unWISE", band)
    guess_sigma = guess / UNWISE_PIXEL_SCALE / SIG2FWHM
    half = int(np.ceil(max(2 * guess_sigma * SIG2FWHM, 5)))
    stamp = _profile_image(*_unwise_profile(band), UNWISE_PIXEL_SCALE, half, oversample=1)
    sigma, _, _ = _fit_star(stamp / stamp.max() * 1e4, guess_sigma)
    return sigma * SIG2FWHM * UNWISE_PIXEL_SCALE


def matching_kernel(fwhm: float, target: PSFTarget, scale: float) -> np.ndarray:
    """Matching kernel (normalised) from an image with a PSF of the given FWHM (arcsec) to the target, on a grid
    of the given pixel scale (arcsec)."""
    if target.unwise_model:
        try:  # photutils >= 3.0
            from photutils.psf_matching import SplitCosineBellWindow, make_kernel
        except ImportError:
            from photutils.psf.matching import SplitCosineBellWindow
            from photutils.psf.matching import create_matching_kernel as make_kernel

        half = int(np.ceil(UNWISE_KERNEL_RADIUS / scale))
        radii, profile = _unwise_profile(target.filt)
        stretch = target.fwhm / unwise_model_fwhm(target.filt)  # core of the model matched to the measured FWHM
        kernel = make_kernel(
            _gaussian_image(fwhm, scale, half), _profile_image(radii * stretch, profile, scale, half),
            window=SplitCosineBellWindow(alpha=0.0, beta=0.3),
        )
    else:
        sigma = np.sqrt(target.fwhm**2 - fwhm**2) / SIG2FWHM / scale
        size = int(2 * np.ceil(4 * sigma) + 1)
        kernel = Gaussian2DKernel(sigma, x_size=size, y_size=size).array
    return kernel / kernel.sum()


def match_psf(
    data: np.ndarray, header: fits.Header, fwhm: float, target: PSFTarget, survey: str, label: str = ""
) -> tuple[np.ndarray, bool]:
    """Convolves an image to the target PSF.

    Parameters
    ----------
    data: Image data.
    header: Image header (with the WCS).
    fwhm: PSF FWHM of the image (arcsec).
    target: Target PSF.
    survey: Survey of the image.
    label: Description of the image for the warnings (e.g. survey and filter).

    Returns
    -------
    data: Matched image (the input image if it was not matched).
    matched: Whether the image was convolved.
    """
    if not np.isfinite(fwhm) or fwhm >= target.fwhm:
        warnings.warn(
            f"{label}: the PSF of the image (FWHM {fwhm:.3g} arcsec) is not narrower than the target of the PSF "
            f"matching ({target.fwhm:.3g} arcsec), so the image is not matched.",
            PSFMatchWarning,
        )
        return data, False
    use_target = target
    if survey in WINGED_SURVEYS:
        warnings.warn(
            f"{label}: the PSF of {survey} images has strong wings, which are not modelled by the matching kernel, "
            "so the matched fluxes are less accurate.",
            PSFMatchWarning,
        )
        # (adding the wings of the unWISE model to those of the image over-smooths it: a Gaussian kernel is used)
        use_target = PSFTarget(target.fwhm)
    if target.fwhm >= CONFUSION_FWHM:
        warnings.warn(
            f"{label}: with a target PSF of {target.fwhm:.3g} arcsec, the light of faint and neighbouring sources "
            "spread by the PSF contributes to the aperture (confusion), especially in deep or crowded images; "
            "it is only partly included in the uncertainty.",
            UserWarning,
        )
    kernel = matching_kernel(fwhm, use_target, _pixel_scale(header))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        matched = convolve_fft(
            np.asarray(data, dtype=np.float64), kernel, nan_treatment="interpolate", preserve_nan=True,
            normalize_kernel=False, boundary="fill", fill_value=np.nan, allow_huge=True,
        )
    return np.ascontiguousarray(matched), True
