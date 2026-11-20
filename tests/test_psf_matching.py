"""Offline tests of the PSF matching (hostphot.photometry.psf_matching) and of the blank-aperture floor of the local
photometry (correlated noise of matched images): FWHM measured from stars, Gaussian and unWISE kernels, targets,
and the local photometry with ``psf_match`` on the PanSTARRS test images (Gaia mocked)."""
import warnings
from unittest import mock

import numpy as np
import pytest
import sep
from astropy.convolution import convolve_fft
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from tests.helpers import install_test_images
from hostphot._constants import workdir
from hostphot.flags import PSFMatchWarning
from hostphot.photometry import psf_matching as pm
from hostphot.photometry import local_photometry as lp
from hostphot.photometry import photometry_utils as pu
from hostphot.processing import objects_detection as od

SCALE = 0.25  # arcsec / pixel
N = 400


def _header(scale: float = SCALE, n: int = N) -> fits.Header:
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [150.0, 2.0]
    w.wcs.crpix = [(n + 1) / 2, (n + 1) / 2]
    w.wcs.cdelt = [-scale / 3600, scale / 3600]
    return w.to_header()


def _star_field(fwhm: float, scale: float = SCALE, n: int = N, noise: float = 1.0, seed: int = 1):
    """Image with a grid of Gaussian stars of the given FWHM (arcsec), their coordinates and the header."""
    rng = np.random.default_rng(seed)
    header = _header(scale, n)
    y, x = np.indices((n, n), dtype=float)
    data = rng.normal(0, noise, (n, n))
    sigma = fwhm / pm.SIG2FWHM / scale
    xy = [(px, py) for px in np.arange(40, n - 39, 64.0) for py in np.arange(40, n - 39, 64.0)]
    for px, py in xy:
        data += 2000 * np.exp(-((x - px) ** 2 + (y - py) ** 2) / (2 * sigma**2))
    coords = WCS(header).pixel_to_world([p[0] for p in xy], [p[1] for p in xy])
    return data, header, coords


def test_measure_fwhm():
    for fwhm in [1.0, 1.6, 3.0]:
        data, header, stars = _star_field(fwhm)
        measured, nmad, n_stars = pm.measure_fwhm(data, header, stars, guess=1.2)
        assert n_stars >= pm.MIN_STARS
        assert abs(measured - fwhm) / fwhm < 0.03
    # no stars
    assert pm.measure_fwhm(data, header, SkyCoord([], [], unit="deg"), 1.0)[2] == 0


def test_gaussian_matching_reaches_target():
    data, header, stars = _star_field(1.1)
    target = pm.PSFTarget(3.0)
    matched, ok = pm.match_psf(data, header, 1.1, target, "PanSTARRS", label="test")
    assert ok
    assert abs(pm.measure_fwhm(matched, header, stars, guess=3.0)[0] - 3.0) < 0.06
    # the flux is preserved
    x, y = WCS(header).world_to_pixel(stars)
    before, _, _ = sep.sum_circle(np.ascontiguousarray(data), x, y, 40.0)
    after, _, _ = sep.sum_circle(np.ascontiguousarray(matched), x, y, 40.0)
    assert np.allclose(after, before, rtol=0.01)
    # images already broader than the target are not matched (flagged)
    with pytest.warns(PSFMatchWarning, match="not narrower"):
        same, ok = pm.match_psf(data, header, 3.5, target, "PanSTARRS", label="test")
    assert not ok and same is data


def test_unwise_kernel_reproduces_model():
    """A star matched with the unWISE kernel has the encircled energy of the (stretched) unWISE W1 model, which a
    Gaussian of the same FWHM does not (strong wings)."""
    radii, profile = pm._unwise_profile("W1")
    target = pm.PSFTarget(6.1, "unWISE", "W1")
    assert target.unwise_model and not pm.PSFTarget(6.1, "unWISE", "W3").unwise_model
    scale, half = 0.5, 120
    star = pm._gaussian_image(1.1, scale, half)
    kernel = pm.matching_kernel(1.1, target, scale)
    assert abs(kernel.sum() - 1) < 1e-6
    matched = convolve_fft(star, kernel, normalize_kernel=False)
    stretch = target.fwhm / pm.unwise_model_fwhm("W1")
    model = pm._profile_image(radii * stretch, profile, scale, half)
    gauss = pm._gaussian_image(target.fwhm, scale, half)
    for r in [4.0, 8.0, 16.0]:
        ee = [sep.sum_circle(np.ascontiguousarray(im), [half], [half], [r / scale])[0][0] for im in (matched, model, gauss)]
        assert abs(ee[0] - ee[1]) < 0.02
    assert ee[2] - ee[1] > 0.03  # (at 16": the Gaussian has no wings)


def test_resolve_target(tmp_path):
    assert pm.resolve_target(2.5, "x", ["r"], "PanSTARRS", tmp_path) == pm.PSFTarget(2.5)
    for bad in [-1, "worst", ("PanSTARRS",)]:
        with pytest.raises(ValueError):
            pm.resolve_target(bad, "x", ["r"], "PanSTARRS", tmp_path)
    # (survey, filter) without the image: typical FWHM of the filter, flagged
    with pytest.warns(PSFMatchWarning, match="typical PSF FWHM"):
        target = pm.resolve_target(("unWISE", "W1"), "x", ["r"], "PanSTARRS", tmp_path)
    assert target.unwise_model and target.fwhm == pytest.approx(6.1)
    # broadest PSF among the images of the filters
    obj = tmp_path / "x" / "VISTA"
    obj.mkdir(parents=True)
    for filt, fwhm in [("J", 0.8), ("Ks", 1.3)]:
        header = _header()
        header["PSF_FWHM"] = fwhm
        fits.PrimaryHDU(np.zeros((10, 10)), header=header).writeto(obj / f"VISTA_{filt}.fits")
    assert pm.resolve_target("broadest", "x", ["J", "Ks"], "VISTA", tmp_path) == pm.PSFTarget(1.3, "VISTA", "Ks")


def test_local_floor_correlated_noise():
    """In an image with noise correlated on a large scale (e.g. PSF-matched), the local floor matches the scatter
    of independent apertures, while the previous floor (nearby apertures, fine background mesh) underestimates it."""
    rng = np.random.default_rng(3)
    n = 1200
    sigma_corr = 8.0  # pixels (FWHM ~19 pixels)
    white = rng.normal(0, 1, (n, n))
    k = np.exp(-np.add.outer(np.arange(-40, 41) ** 2, np.arange(-40, 41) ** 2) / (2 * sigma_corr**2))
    data = convolve_fft(white, k / k.sum())
    mask = np.zeros(data.shape, dtype=bool)
    radius = 6.0
    # truth: scatter of apertures on a grid (separated by > 6 correlation lengths)
    g = np.arange(60, n - 60, 60.0)
    gx, gy = np.meshgrid(g, g)
    flux, _, _ = sep.sum_circle(np.ascontiguousarray(data), gx.ravel(), gy.ravel(), radius)
    truth = pu._robust_rms(flux)
    fwhm = sigma_corr * pm.SIG2FWHM
    # (each estimate scatters by ~20%: averaged over several positions)
    centres = [(600, 600), (300, 300), (900, 400), (400, 900)]
    new = np.mean([pu.local_blank_floor(data, 0.0, x, y, radius, fwhm=fwhm, source_mask=mask) for x, y in centres])
    new_bkg = np.mean([pu.local_blank_floor(data, 0.0, x, y, radius, fwhm=fwhm, source_mask=mask,
                                            background_subtracted=True) for x, y in centres])
    old = np.mean([pu.blank_aperture_floor(data, 0.0, x, y, radius, source_mask=mask) for x, y in centres])
    assert 0.8 < new / truth < 1.25
    assert 0.8 < new_bkg / truth < 1.25
    assert old / truth < 0.75


def _no_gaia(*args, **kwargs):
    return SkyCoord([], [], unit="deg")


def test_local_photometry_psf_match():
    name = "offline_psf_match"
    install_test_images(name, "PanSTARRS", "gr")
    ra, dec, z = 50.52379, -15.40089, 0.007482
    pm._measured_fwhm.cache_clear()
    kwargs = dict(survey="PanSTARRS", filters="gr", ap_radii=[1.0, 3.0], ap_units="arcsec", use_mask=False,
                  save_plots=False, save_results=False, save_input=False)
    with mock.patch.object(od, "find_gaia_objects", side_effect=_no_gaia):
        plain = lp.multi_band_phot(name, ra, dec, z, **kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            matched = lp.multi_band_phot(name, ra, dec, z, psf_match=3.0, **kwargs)
            broad = lp.multi_band_phot(name, ra, dec, z, psf_match="broadest", **kwargs)
            narrow = lp.multi_band_phot(name, ra, dec, z, psf_match=0.5, **kwargs)
    assert np.isnan(plain["psf_match_fwhm"][0]) and not plain["g_psf_match_flag"][0]
    assert matched["psf_match_fwhm"][0] == 3.0
    for filt in "gr":
        assert matched[f"{filt}_psf_fwhm"][0] == 3.0
        # no Gaia stars: typical FWHM of the filter, flagged
        assert matched[f"{filt}_psf_match_flag"][0]
        # the 1" aperture loses flux when the light is spread by the broader PSF; the errors include the correlated
        # noise (they do not decrease)
        assert matched[f"{filt}_1.0_flux_uJy"][0] < plain[f"{filt}_1.0_flux_uJy"][0]
        assert matched[f"{filt}_3.0_flux_uJy_err"][0] >= 0.7 * plain[f"{filt}_3.0_flux_uJy_err"][0]
        # target narrower than the image: not matched (flagged), the image FWHM is kept
        assert narrow[f"{filt}_psf_fwhm"][0] == pytest.approx(plain[f"{filt}_psf_fwhm"][0])
        assert narrow[f"{filt}_psf_match_flag"][0]
        assert narrow[f"{filt}_1.0_flux_uJy"][0] == pytest.approx(plain[f"{filt}_1.0_flux_uJy"][0])
    # broadest: the g band (typical FWHM 1.31") is the target; the g image is not convolved
    assert broad["psf_match_fwhm"][0] == pytest.approx(1.31)
    assert broad["g_1.0_flux_uJy"][0] == pytest.approx(plain["g_1.0_flux_uJy"][0])
    assert not broad["g_psf_match_flag"][0]  # (the target image itself is not flagged)
    assert broad["r_psf_match_flag"][0]  # (no Gaia stars: typical FWHM)
    assert broad["r_1.0_flux_uJy"][0] < plain["r_1.0_flux_uJy"][0]
