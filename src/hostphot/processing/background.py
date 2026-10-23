import warnings
from typing import Optional

import numpy as np
import sep
from scipy.ndimage import distance_transform_edt, gaussian_filter, zoom
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

# fraction of masked pixels above which the background is considered unreliable
# (e.g. the galaxy fills most of the image)
MASKED_FRACTION_WARNING = 0.8
# minimum fraction of unmasked pixels for a spatially varying background
MIN_UNMASKED_FRACTION = 0.02
# larger images are binned for the background estimation (much faster; the background is smooth)
MAX_PIXELS = 4e6
# FWHM used when it cannot be measured (too few bright, compact sources): typical ground-based seeing,
# but at most 10 pixels (as for PanSTARRS), so it is not too large for images with small pixels (e.g. HST
# and JWST, with FWHM of ~2 pixels)
FALLBACK_FWHM_ARCSEC = 2.5
FALLBACK_FWHM_MAX_PIX = 10.0


def _safe_mesh(npix: int, mesh: int) -> int:
    """Returns a background mesh size close to ``mesh`` that does not leave a strip of 1 or 2 pixels at
    the image edge (``npix % mesh``), where :func:`sep.Background` gives unstable values (e.g. +17 sigma)."""
    for delta in [0, 1, -1, 2, -2, 3, -3, 4, -4]:
        size = mesh + delta
        if size >= 4 and (npix % size == 0 or npix % size >= 3 or size >= npix):
            return size
    return mesh


def sep_background(data: np.ndarray, mask: Optional[np.ndarray] = None, bw: int = 64, bh: int = 64, **kwargs):
    """:func:`sep.Background` with mesh sizes adjusted to the image size (see :func:`_safe_mesh`)."""
    ny, nx = data.shape
    return sep.Background(data, mask=mask, bw=_safe_mesh(nx, bw), bh=_safe_mesh(ny, bh), **kwargs)


def _set_pixstack(size: int) -> None:
    """Scales the sep buffer of pixels above the detection threshold with the image size (with a cap)."""
    sep.set_extract_pixstack(int(min(max(300000, size), 1e7)))


def _ellipse_mask(
    shape: tuple, x0: float, y0: float, a: float, b: float, theta: float
) -> np.ndarray:
    """Returns a boolean mask with the pixels inside an ellipse."""
    yy, xx = np.indices(shape)
    dx, dy = xx - x0, yy - y0
    cos, sin = np.cos(theta), np.sin(theta)
    u = dx * cos + dy * sin
    v = -dx * sin + dy * cos
    return (u / a) ** 2 + (v / b) ** 2 <= 1


def _estimate_fwhm(
    sub: np.ndarray, objects: np.ndarray, rms: float, pixel_scale: float, minarea: int
) -> float:
    """Estimates the FWHM (in pixels) of an image from bright, compact sources (see ``FALLBACK_FWHM_ARCSEC``
    and ``FALLBACK_FWHM_MAX_PIX`` for the default)."""
    fwhm_pix = min(FALLBACK_FWHM_ARCSEC / pixel_scale, FALLBACK_FWHM_MAX_PIX)
    good = (objects["flag"] == 0) & (objects["peak"] > 20 * rms) & (objects["npix"] > minarea)
    if good.sum() >= 5:
        half_radius, flags = sep.flux_radius(
            sub,
            objects["x"][good],
            objects["y"][good],
            6.0 * objects["a"][good],
            0.5,
            normflux=objects["flux"][good],
            subpix=5,
        )
        half_radius = half_radius[(flags == 0) & (half_radius > 0)]
        if len(half_radius) >= 5:
            fwhm_pix = 2.0 * np.percentile(half_radius, 25)
    return float(np.clip(fwhm_pix, 1.0, 10.0 / pixel_scale))


def source_mask(
    data: np.ndarray,
    pixel_scale: float,
    host_xy: Optional[tuple[float, float]] = None,
    back: Optional[np.ndarray] = None,
    rms: Optional[float] = None,
    nsigma: float = 3.0,
    minarea: int = 5,
    dilate_fwhm: float = 2.5,
    host_factor: float = 2.0,
    host_thresh: float = 10.0,
    host_smooth_arcsec: float = 2.0,
) -> tuple[np.ndarray, dict]:
    """Creates a mask of the sources of an image (including the host galaxy) for the background estimation.

    The sources are detected at ``nsigma`` and their footprints dilated by ``dilate_fwhm`` times the FWHM.
    The host galaxy is additionally masked with its Kron ellipse (2.5 Kron radii) enlarged by ``host_factor``,
    detected on a smoothed image, so that its faint, undetected wings are also masked.

    Parameters
    ----------
    data: Image data.
    pixel_scale: Pixel scale in arcsec.
    host_xy: Pixel coordinates of the host galaxy.
    back: Initial background of the image (calculated without mask if not given).
    rms: Initial background noise of the image.
    nsigma: Detection threshold of the sources. Lower values also mask positive noise fluctuations
        (especially with correlated noise), biasing the background low.
    minarea: Minimum number of pixels of the sources.
    dilate_fwhm: Dilation of the source footprints in units of the FWHM.
    host_factor: Enlargement factor of the Kron ellipse of the host galaxy.
    host_thresh: Detection threshold of the host galaxy (on the smoothed image).
    host_smooth_arcsec: Minimum FWHM of the smoothing kernel used to detect the host galaxy.

    Returns
    -------
    mask: Mask of the sources (``True`` for masked pixels, including pixels without data).
    info: Information about the mask (FWHM in pixels, semi-major axis of the host mask in pixels,
        number of sources and masked fraction).
    """
    nan_mask = ~np.isfinite(data)
    filled = np.ascontiguousarray(np.where(nan_mask, 0.0, data), dtype=np.float64)
    if back is None or rms is None:
        bkg = sep_background(filled, mask=nan_mask)
        back, rms = bkg.back(), bkg.globalrms
    sub = np.ascontiguousarray(filled - back)

    _set_pixstack(data.size)
    objects, segmap = sep.extract(
        sub, nsigma, err=rms, mask=nan_mask, minarea=minarea, segmentation_map=True
    )
    fwhm_pix = _estimate_fwhm(sub, objects, rms, pixel_scale, minarea)
    # dilate the footprints of the sources
    radius = dilate_fwhm * fwhm_pix
    if radius > 0 and (segmap > 0).any():
        mask = distance_transform_edt(segmap == 0) <= radius
    else:
        mask = segmap > 0

    host_a = np.nan
    if host_xy is not None:
        # detection of the host galaxy on a smoothed image, to get its full extent
        smooth_fwhm = max(fwhm_pix, host_smooth_arcsec / pixel_scale)
        smoothed = gaussian_filter(np.where(nan_mask, 0.0, sub), smooth_fwhm / 2.3548)
        smoothed = np.ascontiguousarray(smoothed)
        smoothed_rms = sigma_clipped_stats(
            smoothed[~nan_mask & (segmap == 0)], sigma=3.0, maxiters=5
        )[2]
        if np.isfinite(smoothed_rms) and smoothed_rms > 0:
            host_objs, host_seg = sep.extract(
                smoothed,
                host_thresh,
                err=smoothed_rms,
                mask=nan_mask,
                minarea=minarea,
                segmentation_map=True,
                filter_kernel=None,
            )
            if len(host_objs) > 0:
                hx, hy = float(host_xy[0]), float(host_xy[1])
                iy, ix = int(round(hy)), int(round(hx))
                inside = 0 <= iy < host_seg.shape[0] and 0 <= ix < host_seg.shape[1]
                if inside and host_seg[iy, ix] > 0:
                    idx = host_seg[iy, ix] - 1
                else:
                    idx = int(np.argmin(np.hypot(host_objs["x"] - hx, host_objs["y"] - hy)))
                obj = host_objs[idx : idx + 1]
                kronrad, _ = sep.kron_radius(
                    smoothed, obj["x"], obj["y"], obj["a"], obj["b"], obj["theta"], 6.0
                )
                kronrad = float(kronrad[0]) if np.isfinite(kronrad[0]) and kronrad[0] > 0 else 3.5
                a = host_factor * 2.5 * kronrad * float(obj["a"][0])
                b = host_factor * 2.5 * kronrad * float(obj["b"][0])
                host_a = a
                mask |= _ellipse_mask(
                    data.shape, obj["x"][0], obj["y"][0], a, b, obj["theta"][0]
                )
    mask |= nan_mask
    info = {
        "fwhm_pix": fwhm_pix,
        "host_a_pix": host_a,
        "nobj": len(objects),
        "masked_frac": float(mask.mean()),
    }
    return mask, info


def _masked_background(
    data: np.ndarray,
    wcs: WCS,
    host_ra: Optional[float] = None,
    host_dec: Optional[float] = None,
    mesh_arcsec: float = 30.0,
    niter: int = 2,
    label: str = "",
) -> np.ndarray:
    """Estimates the background of an image with the sources (including the host galaxy) masked.

    A spatially varying background is estimated with :func:`sep.Background` using a mesh of
    ``mesh_arcsec`` (in arcsec, so it is the same for all surveys), after masking the sources
    (see :func:`source_mask`). The mask is refined iteratively, using the previous background.

    Parameters
    ----------
    data: Image data.
    wcs: WCS of the image.
    host_ra: Right ascension of the host galaxy in degrees.
    host_dec: Declination of the host galaxy in degrees.
    mesh_arcsec: Size of the background mesh in arcsec.
    niter: Number of iterations.
    label: Description of the image used in the warnings (e.g. survey and filter).

    Returns
    -------
    back: Background of the image.
    """
    pixel_scale = float(np.mean(proj_plane_pixel_scales(wcs)) * 3600)  # arcsec per pixel
    host_xy = None
    if host_ra is not None and host_dec is not None:
        host_xy = wcs.world_to_pixel_values(host_ra, host_dec)
        if not np.all(np.isfinite(host_xy)):
            host_xy = None  # e.g. position outside the valid range of the WCS projection

    nan_mask = ~np.isfinite(data)
    filled = np.ascontiguousarray(np.where(nan_mask, 0.0, data), dtype=np.float64)
    ny, nx = data.shape
    mesh = int(np.clip(int(round(mesh_arcsec / pixel_scale)), 16, max(nx, ny)))

    back, rms = None, None
    for _ in range(niter):
        mask, info = source_mask(data, pixel_scale, host_xy=host_xy, back=back, rms=rms)
        good = ~mask
        constant = good.sum() < MIN_UNMASKED_FRACTION * data.size
        if constant:
            # (almost) no sky left: constant background from the remaining unmasked pixels
            pixels = filled[good] if good.sum() > 100 else filled[~nan_mask]
            median = sigma_clipped_stats(pixels, sigma=3.0, maxiters=10)[1]
            back = np.full(data.shape, median)
        else:
            back = sep_background(filled, mask=mask, bw=mesh, bh=mesh, fw=3, fh=3).back()
            if not np.any(back):
                # sep returns zero everywhere if all the meshes are mostly masked: constant background instead
                constant = True
                median = sigma_clipped_stats(filled[good], sigma=3.0, maxiters=10)[1]
                back = np.full(data.shape, median)
        residuals = (filled - back)[good] if good.sum() > 100 else (filled - back)[~nan_mask]
        rms = sigma_clipped_stats(residuals, sigma=3.0, maxiters=10)[2]

    if info["masked_frac"] > MASKED_FRACTION_WARNING or constant:
        size_arcmin = max(nx, ny) * pixel_scale / 60
        msg = (
            f"{label}: {100 * info['masked_frac']:.0f}% of the image is masked for the background "
            "estimation (e.g. the galaxy fills most of the image), so the background (and the photometry "
            "of large apertures) might be unreliable."
        )
        if np.isfinite(info["host_a_pix"]):
            # cutout of at least ~4-5 times the Kron semi-major axis of the galaxy
            host_kron_arcmin = info["host_a_pix"] / 2 * pixel_scale / 60
            msg += (
                f" Consider downloading a larger image (currently {size_arcmin:.1f} arcmin; "
                f"at least ~{max(4.5 * host_kron_arcmin, size_arcmin):.0f} arcmin is recommended)."
            )
        else:
            msg += f" Consider downloading a larger image (currently {size_arcmin:.1f} arcmin)."
        if constant:
            msg += " A constant background was used, as very few sky pixels were left unmasked."
        warnings.warn(msg)

    return back


def _bin_image(data: np.ndarray, wcs: WCS, factor: int) -> tuple[np.ndarray, WCS]:
    """Bins an image by averaging blocks of ``factor`` x ``factor`` pixels (ignoring NaNs)."""
    ny, nx = data.shape
    padded = np.pad(data, ((0, (-ny) % factor), (0, (-nx) % factor)), constant_values=np.nan)
    blocks = padded.reshape(padded.shape[0] // factor, factor, padded.shape[1] // factor, factor)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # blocks without data
        binned = np.nanmean(blocks, axis=(1, 3))
    binned_wcs = wcs.deepcopy()
    binned_wcs.wcs.crpix = (wcs.wcs.crpix - 0.5) / factor + 0.5
    if binned_wcs.wcs.has_cd():
        binned_wcs.wcs.cd = wcs.wcs.cd * factor
    else:
        binned_wcs.wcs.cdelt = wcs.wcs.cdelt * factor
    return binned, binned_wcs


def masked_background(
    data: np.ndarray,
    wcs: WCS,
    host_ra: Optional[float] = None,
    host_dec: Optional[float] = None,
    mesh_arcsec: float = 30.0,
    niter: int = 2,
    label: str = "",
) -> np.ndarray:
    """Estimates the background of an image with the sources (including the host galaxy) masked.

    A spatially varying background is estimated with :func:`sep.Background` using a mesh of
    ``mesh_arcsec`` (in arcsec, so it is the same for all surveys), after masking the sources
    (see :func:`source_mask`). The mask is refined iteratively, using the previous background.
    Images with more than ``MAX_PIXELS`` pixels (e.g. JWST or HST mosaics) are binned for the
    estimation, and the (smooth) background is interpolated back to the original pixels.

    Parameters
    ----------
    data: Image data.
    wcs: WCS of the image.
    host_ra: Right ascension of the host galaxy in degrees.
    host_dec: Declination of the host galaxy in degrees.
    mesh_arcsec: Size of the background mesh in arcsec.
    niter: Number of iterations.
    label: Description of the image used in the warnings (e.g. survey and filter).

    Returns
    -------
    back: Background of the image.
    """
    factor = int(np.ceil(np.sqrt(data.size / MAX_PIXELS)))
    if factor <= 1:
        return _masked_background(data, wcs, host_ra, host_dec, mesh_arcsec, niter, label)
    binned, binned_wcs = _bin_image(data, wcs, factor)
    back = _masked_background(binned, binned_wcs, host_ra, host_dec, mesh_arcsec, niter, label)
    return zoom(back, factor, order=1)[: data.shape[0], : data.shape[1]]
