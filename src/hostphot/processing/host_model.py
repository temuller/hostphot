"""Masking with the host galaxy treated separately (``create_mask(method="host_model")``).

The host galaxy is modelled with an elliptical Sersic profile, so that its own structure (star-forming clumps,
spiral arms, bars) is not masked, while the sources on top of it (stars and overlapping galaxies) are:

1. Geometry of the host: centre at the host coordinates; axis ratio and position angle from the second moments of
   the point-symmetric minimum image, ``min(I(x), I(2 x0 - x))``, which removes the asymmetric contaminants (stars
   and neighbours without a counterpart at the mirrored position).
2. Sersic profile (plus a constant) fitted to the median surface brightness in elliptical annuli (the median is
   robust to neighbours covering part of an annulus), without the pixels much brighter (significantly and by more than
   a factor of 2) than their mirrored pixel with respect to the host centre (neighbours and stars). The PSF-dominated centre uses the measured median
   profile.
3. Host footprint: the region where the model is above ``host_threshold`` times the background RMS.
4. Outside the footprint, the detected sources are masked as usual (ellipses scaled by ``r``).
5. Inside the footprint, sources are detected on the residual image (image - model) and only masked if they are
   Gaia stars (with significant proper motion or parallax), or extended (larger than ``extended_factor`` times the
   PSF), brighter than the host model over their area (``contrast``), and above ``structure_sigma`` times the
   scatter of the residuals at their elliptical radius (the structure of the host itself), e.g. an overlapping
   galaxy. Star-forming clumps are point-like, and spiral arms and bars are typical of the residuals at their
   radius, so they are not masked.
6. The masked pixels are replaced with the host model (plus the fitted constant) and random noise with the
   background RMS (with a fixed seed, so the results are reproducible).

Validation:

* 40 large star-forming spirals (effective radii of 10-40 arcsec; 116 Legacy Survey, PanSTARRS and SDSS r-band
  images): the fraction of the host light within 2 effective radii that is masked (excluding Gaia stars) goes from
  16-18% on average with the ellipses (10 sigma; more than half of the light in 19 images) to ~2% (none with more
  than half; the largest values, 40-50%, are a bright galaxy overlapping the host, which is correctly masked).
* 72 galaxies of different types (large and small ellipticals, edge-on discs, small discs, dwarf and low surface
  brightness galaxies, and hosts in crowded fields at low Galactic latitude; Legacy Survey and PanSTARRS r-band
  images): the masked fraction of the host light goes from 4-23% on average (depending on the type) with the ellipses
  to 0-2% (except for small hosts, the same with both methods), and the hosts with more than 10% of their light
  masked from 15 to 3 (Legacy Survey) and from 12 to 2 (PanSTARRS). The fit failed for none of them.
* 120 simulated galaxies (Sersic n = 0.5-4, effective radii of 1.5-12 arcsec, axis ratios of 0.25-0.9, r = 14-19.5
  mag; half with a bright star nearby) with neighbours, injected in a blank and a crowded PanSTARRS field: within 3
  effective radii, the flux of the masked and filled image differs from the true flux by -0.02 +/- 0.10 mag (median
  and NMAD; 42% of the galaxies off by more than 0.1 mag), against -0.06 +/- 0.32 mag (68%) with the ellipses and
  the Gaussian-kernel filling.

Limitations: hosts with a low surface brightness (mean within the effective radius fainter than ~24.5 mag/arcsec^2
in PanSTARRS) are poorly recovered with both methods, strongly asymmetric hosts can have part of their own light
masked (and replaced with the symmetric model), and the ellipse of a bright neighbour just outside the footprint can
still cover the host (its light is then replaced with the model).
"""
import warnings

import numpy as np
import sep
from scipy.ndimage import binary_dilation, label, map_coordinates
from scipy.optimize import curve_fit
from scipy.special import gammaincinv

FILL_SEED = 0  # seed of the random noise added to the filled pixels (reproducible masked images)
RESIDUAL_DEBLEND_CONT = 0.001  # deblending of the residual sources (adjacent clumps are split)
MASK_GROWTH_FWHM = 1.5  # growth of the masks of the residual sources (in units of the PSF FWHM)
GAIA_MATCH_FWHM = 2.0  # cross-match radius of the Gaia stars (in units of the PSF FWHM)
# pixels above their mirrored pixel (with respect to the host centre) by this significance and by this factor are not
# used for the host profile (stars and neighbours; the asymmetries of the host itself, e.g. spiral arms or dust lanes,
# are usually smaller)
ASYMMETRY_SIGMA = 3.0
ASYMMETRY_RATIO = 2.0


def _bn(n: float) -> float:
    return gammaincinv(2 * n, 0.5)


def _sersic(r: np.ndarray, ie: float, re: float, n: float, c: float) -> np.ndarray:
    return ie * np.exp(-_bn(n) * ((r / re) ** (1 / n) - 1)) + c


def elliptical_radius(shape: tuple, x0: float, y0: float, q: float, theta: float) -> np.ndarray:
    """Semi-major-axis radius (in pixels) of each pixel for an ellipse with axis ratio ``q`` and position angle
    ``theta`` (radians, counter-clockwise from the x axis)."""
    y, x = np.indices(shape)
    dx, dy = x - x0, y - y0
    c, s = np.cos(theta), np.sin(theta)
    return np.sqrt((dx * c + dy * s) ** 2 + ((-dx * s + dy * c) / q) ** 2)


def _extract(data: np.ndarray, rms: float, threshold: float, deblend_cont: float, nan_mask: np.ndarray):
    sep.set_extract_pixstack(int(min(max(300000, data.size), 1e7)))
    sep.set_sub_object_limit(4096)
    return sep.extract(np.ascontiguousarray(data), threshold, err=rms, mask=nan_mask, minarea=5,
                       deblend_cont=deblend_cont, segmentation_map=True)


def symmetric_minimum(data: np.ndarray, x0: float, y0: float) -> np.ndarray:
    """Point-symmetric minimum image, ``min(I(x), I(2 x0 - x))``, which removes the asymmetric contaminants
    (stars and neighbours, unless they have a counterpart at the mirrored position). Pixels mirrored outside the
    image are NaN."""
    y, x = np.indices(data.shape, dtype=float)
    mirror = map_coordinates(data, [2 * y0 - y, 2 * x0 - x], order=1, mode="constant", cval=np.nan)
    return np.fmin(data, mirror)


def host_geometry(data: np.ndarray, rms: float, x0: float, y0: float) -> tuple[float, float, float]:
    """Axis ratio, position angle (radians) and semi-major axis (second moment, pixels) of the host, from the
    symmetric minimum image above 2 sigma, in the connected region containing the host position."""
    sym = symmetric_minimum(data, x0, y0)
    lab, _ = label(np.nan_to_num(sym, nan=0.0) > 2 * rms)
    xi, yi = int(round(x0)), int(round(y0))
    region_id = lab[yi, xi]
    if region_id == 0 or (lab == region_id).sum() < 10:
        return 1.0, 0.0, 3.0
    y, x = np.nonzero(lab == region_id)
    w = sym[y, x]
    dx, dy = x - x0, y - y0
    sxx, syy, sxy = (w * dx**2).sum() / w.sum(), (w * dy**2).sum() / w.sum(), (w * dx * dy).sum() / w.sum()
    theta = 0.5 * np.arctan2(2 * sxy, sxx - syy)
    t = np.sqrt(((sxx - syy) / 2) ** 2 + sxy**2)
    a2, b2 = (sxx + syy) / 2 + t, (sxx + syy) / 2 - t
    q = float(np.clip(np.sqrt(max(b2, 0.0) / a2), 0.2, 1.0))
    return q, float(theta), float(np.sqrt(a2))


def fit_host_model(data: np.ndarray, rms: float, x0: float, y0: float, fwhm_pix: float) -> dict:
    """Fits a Sersic profile (plus a constant) to the median elliptical surface-brightness profile of the host.

    Returns
    -------
    host: Dictionary with the geometry (``x``, ``y``, ``q``, ``theta``), the Sersic parameters (``ie``, ``re``
        in pixels, ``n``), the fitted constant (``sky``), the model image without the constant (``model``) and
        the elliptical radius of each pixel (``r``). Raises ``RuntimeError`` if the fit fails.
    """
    q, theta, a_host = host_geometry(data, rms, x0, y0)
    r = elliptical_radius(data.shape, x0, y0, q, theta)
    rmax = min(float(r.max()), 30 * a_host)
    edges = np.concatenate([[0], np.geomspace(0.5, rmax, 70)])
    # pixels with a significant excess over the mirrored pixel (with respect to the host centre) are not used for
    # the profile: neighbours and stars, which can dominate the annuli of small or faint hosts
    sym = symmetric_minimum(data, x0, y0)
    with np.errstate(invalid="ignore"):
        excess = data - sym
        asymmetric = (excess > ASYMMETRY_SIGMA * np.sqrt(2) * rms) & (data > ASYMMETRY_RATIO * sym)
    finite = np.isfinite(data) & ~asymmetric
    idx = np.digitize(r[finite], edges) - 1
    vals = data[finite]
    nb = len(edges) - 1
    keep = (idx >= 0) & (idx < nb)
    idx, vals = idx[keep], vals[keep]
    order = np.argsort(idx, kind="stable")
    sorted_idx, sorted_vals = idx[order], vals[order]
    bounds = np.searchsorted(sorted_idx, np.arange(nb + 1))
    med, err = np.full(nb, np.nan), np.full(nb, np.nan)
    for i in range(nb):
        v = sorted_vals[bounds[i]:bounds[i + 1]]
        if len(v) >= 5:
            med[i] = np.median(v)
            # uncertainty of the median (scatter of the annulus, at least the background noise)
            err[i] = 1.2533 * max(1.4826 * np.median(np.abs(v - med[i])), rms) / np.sqrt(len(v))
    rc = np.sqrt(edges[:-1] * edges[1:])
    rc[0] = 0.25
    ok = np.isfinite(med)
    fit = ok & (rc > 1.5 * fwhm_pix)
    if fit.sum() < 5:
        raise RuntimeError("not enough points in the host profile")
    re0 = max(a_host * 1.5, 2 * fwhm_pix)
    ie0 = max(float(np.interp(re0, rc[ok], med[ok])), rms)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params, _ = curve_fit(_sersic, rc[fit], med[fit], p0=[ie0, re0, 1.5, 0.0], sigma=err[fit],
                              bounds=([0, 0.5, 0.3, -np.inf], [np.inf, 20 * rmax, 8, np.inf]), maxfev=20000)
    ie, re, n, sky = (float(p) for p in params)
    model = _sersic(r, ie, re, n, 0.0)
    # PSF-dominated centre: measured median profile
    inner = r < 2 * fwhm_pix
    model[inner] = np.interp(r[inner], rc[ok], med[ok]) - sky
    return {"x": x0, "y": y0, "q": q, "theta": theta, "ie": ie, "re": re, "n": n, "sky": sky, "model": model, "r": r}


def estimate_fwhm_pix(data: np.ndarray, rms: float, nan_mask: np.ndarray) -> float:
    """Rough PSF FWHM (pixels) from the most compact bright detections, for images without a known PSF FWHM."""
    objs, _ = _extract(np.nan_to_num(data), rms, 20, 0.005, nan_mask)
    objs = objs[objs["flag"] == 0]
    if len(objs) < 3:
        return 3.0
    return float(2.3548 * np.percentile(objs["a"], 20))


def host_model_mask(
    data: np.ndarray,
    rms: float,
    x0: float,
    y0: float,
    fwhm_pix: float,
    objects: np.ndarray,
    gaia_xy: tuple[np.ndarray, np.ndarray],
    threshold: float,
    r: float = 6,
    host_threshold: float = 3.0,
    contrast: float = 3.0,
    structure_sigma: float = 3.0,
    extended_factor: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Mask with the host galaxy treated separately (see the module documentation).

    Parameters
    ----------
    data: Image data (NaN for pixels without data).
    rms: Background RMS.
    x0, y0: Host position (pixels).
    fwhm_pix: PSF FWHM (pixels).
    objects: Sources detected in the image (excluding the host), from :func:`sep.extract()`.
    gaia_xy: Pixel coordinates of the Gaia stars.
    threshold: Detection threshold of the residual sources (in units of ``rms``).
    r: Scale of the ellipses of the sources outside the host footprint (and of the Gaia stars inside it).
    host_threshold: The host footprint is where the host model is above ``host_threshold * rms``.
    contrast: Minimum ratio between the residual flux of a non-stellar source inside the footprint and the flux
        of the host model over the same area for the source to be masked.
    structure_sigma: Minimum mean residual of a non-stellar source inside the footprint, in units of the scatter
        of the residuals at its elliptical radius, for the source to be masked.
    extended_factor: Minimum size of a non-stellar source inside the footprint (second-moment semi-major axis),
        in units of that of the PSF, for the source to be masked.

    Returns
    -------
    mask: Masked pixels.
    filled: Image with the masked pixels replaced by the host model (plus the fitted constant) and noise.
    info: Host model parameters, footprint and number of masked sources.
    """
    nan_mask = ~np.isfinite(data)
    host = fit_host_model(data, rms, x0, y0, fwhm_pix)
    model, rr = host["model"], host["r"]
    footprint = model > host_threshold * rms
    if footprint.any():
        # simply connected footprint: elliptical region out to the extent of the pixels above the threshold
        footprint = rr < np.percentile(rr[footprint], 99)
    mask = np.zeros(data.shape, dtype=bool)
    gx, gy = gaia_xy
    a_psf = fwhm_pix / 2.3548

    # outside the footprint: ellipses of the sources
    outside = np.zeros(len(objects), dtype=bool)
    if len(objects):
        xi = np.clip(np.round(objects["x"]).astype(int), 0, data.shape[1] - 1)
        yi = np.clip(np.round(objects["y"]).astype(int), 0, data.shape[0] - 1)
        outside = ~footprint[yi, xi]
        if outside.any():
            out = objects[outside]
            sep.mask_ellipse(mask, out["x"], out["y"], out["a"], out["b"], out["theta"], r=r)
            # PSF size from the Gaia stars outside the footprint, if enough
            if len(gx):
                dist = np.array([np.min(np.hypot(gx - x, gy - y)) for x, y in zip(out["x"], out["y"])])
                stars = (dist < fwhm_pix) & (out["peak"] > 20 * rms) & (out["flag"] == 0)
                if stars.sum() >= 2:
                    a_psf = float(np.median(out["a"][stars]))

    # inside the footprint: sources on the residual image
    resid = data - host["sky"] - model
    robjs, rseg = _extract(np.nan_to_num(resid), rms, threshold, RESIDUAL_DEBLEND_CONT, nan_mask)
    # scatter of the residuals in elliptical annuli (structure of the host: spiral arms, clumps, bars, ...)
    edges = np.concatenate([[0], np.geomspace(1, rr.max() + 1, 40)])
    ridx = np.clip(np.digitize(rr, edges) - 1, 0, len(edges) - 2)
    structure = np.full(len(edges) - 1, rms)
    for j in range(len(edges) - 1):
        v = resid[(ridx == j) & footprint & ~nan_mask]
        if len(v) >= 20:
            structure[j] = max(1.4826 * np.median(np.abs(v - np.median(v))), rms)
    inner_mask = np.zeros(data.shape, dtype=bool)
    star_mask = np.zeros(data.shape, dtype=bool)
    n_inside, n_masked, n_stars = 0, 0, 0
    if len(robjs):
        xi = np.clip(np.round(robjs["x"]).astype(int), 0, data.shape[1] - 1)
        yi = np.clip(np.round(robjs["y"]).astype(int), 0, data.shape[0] - 1)
        for i in np.where(footprint[yi, xi])[0]:
            n_inside += 1
            is_star = bool(len(gx)) and np.min(np.hypot(gx - robjs["x"][i], gy - robjs["y"][i])) < GAIA_MATCH_FWHM * fwhm_pix
            segment = rseg == i + 1
            ratio = np.nansum(resid[segment]) / max(np.nansum(model[segment]), 1e-30)
            extended = robjs["a"][i] > extended_factor * a_psf
            significant = np.nanmean(resid[segment]) > structure_sigma * structure[ridx[yi[i], xi[i]]]
            if is_star:
                # only the star (its residual segment can include part of the host, e.g. a spiral arm): circle out
                # to where twice a Gaussian PSF with the peak of the star falls to the background noise
                sigma_psf = fwhm_pix / 2.3548
                radius = 2 * sigma_psf * np.sqrt(2 * np.log(max(robjs["peak"][i] / rms, np.e)))
                sep.mask_ellipse(star_mask, [robjs["x"][i]], [robjs["y"][i]], [radius], [radius], [0.0], r=1)
                n_masked += 1
                n_stars += 1
            elif extended and significant and ratio > contrast:
                inner_mask |= segment
                n_masked += 1
    if inner_mask.any():
        inner_mask = binary_dilation(inner_mask, iterations=max(int(round(MASK_GROWTH_FWHM * fwhm_pix)), 1))
    mask |= inner_mask | star_mask
    mask &= ~nan_mask

    rng = np.random.default_rng(FILL_SEED)
    filled = data.copy()
    filled[mask] = model[mask] + host["sky"] + rng.normal(0.0, rms, mask.sum())
    info = {key: host[key] for key in ["x", "y", "q", "theta", "ie", "re", "n", "sky"]}
    info.update(footprint=footprint, outside=outside, n_outside=int(outside.sum()), n_inside=n_inside, n_masked_inside=n_masked,
                n_stars_inside=n_stars, fwhm_pix=fwhm_pix)
    return mask, filled, info
