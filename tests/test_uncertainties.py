import unittest
import numpy as np
import sep
from astropy.io import fits

from hostphot.photometry.photometry_utils import uncertainty_calculation
from hostphot.photometry.global_photometry import kron_flux, kron_aperture_area
from hostphot.photometry.local_photometry import extract_aperture_flux
from photutils.aperture import CircularAperture, EllipticalAperture, aperture_photometry
import warnings
from hostphot.photometry.photometry_utils import (
    blank_aperture_floor,
    magnitude_calculation,
    irac_extended_source_correction,
    aperture_missing_fraction,
    check_aperture_coverage,
    elliptical_missing_flux_fraction,
    blank_kron_floor,
    _extrapolated_blank_rms,
    get_source_mask,
    get_effective_gain,
    magnitude_calculation,
)


class TestUncertainties(unittest.TestCase):
    def test_JWST_zp_uncertainty(self):
        # a typical NIRCam short-wavelength zeropoint (MAGZP ~ 27.5)
        header = fits.Header()
        header["MAGZP"] = -6.10 - 2.5 * np.log10(2.3e-14)
        header["XPOSURE"] = 1000.0
        flux = 1e6
        # with no aperture error, only the 1% flux calibration error remains
        flux_err = uncertainty_calculation(flux, 0.0, "JWST", "NIRCam_F090W", header=header)
        mag_err = 2.5 / np.log(10) * flux_err / flux
        np.testing.assert_allclose(mag_err, 2.5 / np.log(10) * 0.01, rtol=1e-6)


    def test_kron_aperture_area(self):
        # synthetic elliptical Gaussian galaxy
        y, x = np.mgrid[0:201, 0:201]
        data = 1000 * np.exp(-(((x - 100) / 8) ** 2 + ((y - 100) / 5) ** 2) / 2)
        objects = sep.extract(data, thresh=10)
        objects = objects[np.argmax(objects["flux"])][np.newaxis]
        kronrad, _ = sep.kron_radius(
            data, objects["x"], objects["y"], objects["a"], objects["b"], objects["theta"], 6.0
        )
        kronrad, scale = kronrad[0], 2.5
        # the flux of a unit image within the Kron aperture equals the aperture area
        unit = np.ones_like(data)
        unit_flux, _ = kron_flux(unit, unit, 1.0, objects, kronrad, scale)
        ap_area = kron_aperture_area(objects, kronrad, scale)
        np.testing.assert_allclose(ap_area, unit_flux, rtol=1e-3)
        # and is much larger than the bare (unscaled) sep ellipse
        assert ap_area > 5 * np.pi * objects["a"][0] * objects["b"][0]


    def test_WISE_noise_scaling(self):
        # background-dominated noise must grow as sqrt(aperture area), not linearly
        header = fits.Header()
        header["BAND"] = 1
        header["MAGZPUNC"] = 0.0
        header["NAXIS1"] = header["NAXIS2"] = 1000
        flux = 1e6
        errs = []
        for ap_area in [1e3, 4e3]:
            flux_err = uncertainty_calculation(flux, 0.0, "WISE", "W1", ap_area, header, bkg_rms=1.0)
            errs.append(2.5 / np.log(10) * flux_err / flux)
        np.testing.assert_allclose(errs[1] / errs[0], 2.0, rtol=0.01)


    def test_blank_aperture_floor(self):
        # a large bright galaxy on white noise: the floor must measure the noise, not the galaxy
        rng = np.random.default_rng(42)
        y, x = np.mgrid[0:301, 0:301]
        galaxy = 500 * np.exp(-np.hypot(x - 150, y - 150) / 25)
        sigma = 1.0
        data = galaxy + rng.normal(0, sigma, galaxy.shape)
        radius = 4.0
        # for an aperture centred on the galaxy, with the blank apertures landing around it
        flux_err = blank_aperture_floor(data, 0.0, 150, 150, radius)
        expected = sigma * np.sqrt(np.pi * radius**2)  # uncorrelated noise
        # (the floor is estimated from 80 apertures, so it has a ~10-15% scatter)
        np.testing.assert_allclose(flux_err, expected, rtol=0.4)
        # a precomputed source mask gives the same result
        np.testing.assert_allclose(
            blank_aperture_floor(data, 0.0, 150, 150, radius, source_mask=get_source_mask(data, host_xy=(150, 150))), flux_err
        )
        # a larger error is kept
        assert blank_aperture_floor(data, 10 * expected, 150, 150, radius) == 10 * expected


    def test_Spitzer_flux_err_units(self):
        # the flux uncertainty must be in the same units as the flux (MJy/sr)
        for filt, instrument, flux_unc in [("IRAC.1", "IRAC", 0.03), ("MIPS.1", "MIPS", 0.04)]:
            header = fits.Header()
            header["INSTRUME"] = instrument
            header["CHNLNUM"] = 1
            header["EFCONV"] = 0.1
            header["EXPTIME"] = 100.0
            header["MAGZP"] = 21.58
            flux, flux_err = 1000.0, 10.0  # S/N = 100
            _, mag_err, flux_out, flux_err_out, _ = magnitude_calculation(
                flux, flux_err, "Spitzer", filt, 100.0, header, 0.05
            )
            k = 2.5 / np.log(10)
            expected = np.hypot(k * flux_err / flux, k * flux_unc)
            np.testing.assert_allclose(mag_err, expected, rtol=1e-6)
            np.testing.assert_allclose(flux_err_out / flux_out, expected / k, rtol=1e-6)


    def test_IRAC_extended_source_correction(self):
        # values from Table 8.1 of the IRAC Instrument Handbook (Section 8.2.2)
        for filt, radius, expected in [("IRAC.1", 30, 0.934), ("IRAC.3", 30, 0.857), ("IRAC.4", 100, 0.754)]:
            corr, corr_err = irac_extended_source_correction(filt, radius)
            np.testing.assert_allclose(corr, expected, atol=1e-3)
            np.testing.assert_allclose(corr_err, 0.1 * abs(1 - corr))
        # not applied for small apertures nor for MIPS
        assert irac_extended_source_correction("IRAC.4", 5.0) == (1.0, 0.0)
        assert irac_extended_source_correction("MIPS.1", 30.0) == (1.0, 0.0)
        # applied to the flux in the photometry (30" radius aperture with 0.6"/pixel)
        header = fits.Header()
        header["INSTRUME"], header["CHNLNUM"], header["EFCONV"] = "IRAC", 4, 0.2
        header["EXPTIME"], header["MAGZP"] = 100.0, 21.58
        ap_area = np.pi * (30 / 0.6) ** 2
        mag, _, flux_out, _, _ = magnitude_calculation(1000.0, 10.0, "Spitzer", "IRAC.4", ap_area, header, 0.05)
        corr, _ = irac_extended_source_correction("IRAC.4", 30.0)
        np.testing.assert_allclose(flux_out, 1000.0 * corr)
        np.testing.assert_allclose(mag, 21.58 - 2.5 * np.log10(1000.0 * corr))


    def test_aperture_coverage(self):
        data = np.ones((101, 101))
        aperture = CircularAperture((50, 50), r=10)
        # full coverage: usable, no warning
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertTrue(check_aperture_coverage(data, aperture, "test"))
        # half of the aperture without data
        half = data.copy()
        half[:, :50] = np.nan
        # (the edge of the data is half a pixel from the aperture centre)
        self.assertTrue(0.45 < aperture_missing_fraction(half, aperture) < 0.5)
        with self.assertWarns(UserWarning):
            self.assertFalse(check_aperture_coverage(half, aperture, "test"))
        # aperture partially outside the image counts as missing
        self.assertGreater(aperture_missing_fraction(data, CircularAperture((2, 50), r=10)), 0.3)
        # a few missing pixels (within the 5% tolerance): usable, with a warning
        few = data.copy()
        few[50, 41:46] = np.nan  # 5 pixels out of ~314
        with self.assertWarns(UserWarning):
            self.assertTrue(check_aperture_coverage(few, aperture, "test"))
        # the flux is then measured from the remaining pixels (not NaN)
        flux, flux_err = extract_aperture_flux(few, np.ones_like(few), 1.0, 50, 50, 10)
        self.assertTrue(np.isfinite(flux) and np.isfinite(flux_err))
        np.testing.assert_allclose(flux, np.pi * 100 - 5, rtol=0.01)


    def test_source_mask_fills_holes(self):
        # ring galaxy: the faint centre is inside the galaxy and must not be used as blank sky
        rng = np.random.default_rng(1)
        y, x = np.mgrid[0:201, 0:201]
        rad = np.hypot(x - 100, y - 100)
        data = 50 * np.exp(-0.5 * ((rad - 40) / 5) ** 2) + rng.normal(0, 1, (201, 201))
        mask = get_source_mask(data)
        self.assertTrue(mask[100, 100])
        self.assertTrue(mask[90:111, 90:111].all())


    def test_sdss_uncertainties(self):
        # SDSS images are in nanomaggies: the effective gain is gain / NMGY (e-/nanomaggy)
        header = fits.Header({"CAMCOL": 4, "RUN": 4192, "EXPTIME": 53.9, "NMGY": 0.005})
        np.testing.assert_allclose(get_effective_gain(header, "SDSS", "r"), 4.76 / 0.005)
        # the magnitude error is given by the measured flux error (no extra terms in wrong units),
        # plus the calibration uncertainty (0.01 mag in r)
        for flux, flux_err in [(1.0, 0.3), (10.0, 0.3), (100.0, 1.0)]:
            mag_err = magnitude_calculation(flux, flux_err, "SDSS", "r", 100.0, header, 0.03)[1]
            np.testing.assert_allclose(mag_err, np.hypot(1.0857 * flux_err / flux, 0.01), rtol=0.01)


    def test_2mass_uncertainties(self):
        # background-dominated source: error = 2 * k_z * sigma * sqrt(area) (area in 1" coadd pixels),
        # plus the 1% calibration uncertainty
        area, sigma, flux = 100.0, 1.0, 1000.0
        noise = 2 * 1.7 * sigma * np.sqrt(area)
        mag_err = magnitude_calculation(flux, 1.0, "2MASS", "J", area, fits.Header({"MAGZP": 20.9}), sigma)[1]
        expected = np.hypot(1.0857 * np.sqrt(noise**2 + flux / 60) / flux, 0.01)
        np.testing.assert_allclose(mag_err, expected, rtol=1e-3)
        # a larger aperture error (e.g. from the blank-aperture floor) is kept
        mag_err = magnitude_calculation(flux, 100.0, "2MASS", "J", area, fits.Header({"MAGZP": 20.9}), sigma)[1]
        np.testing.assert_allclose(mag_err, np.hypot(1.0857 * 0.1, 0.01), rtol=1e-3)


    def test_kron_missing_flux(self):
        # galaxy whose Kron aperture is cut by the image edge
        y, x = np.mgrid[0:301, 0:301]
        galaxy = 1000 * np.exp(-np.hypot(x - 150, y - 150) / 6)
        aperture = EllipticalAperture((150, 150), 60, 60, theta=0)
        total = aperture.to_mask("exact").multiply(galaxy).sum()
        for offset, usable in [(20, True), (10, False)]:
            data = galaxy + np.random.default_rng(0).normal(0, 1, galaxy.shape)
            data[:, 150 + offset:] = np.nan  # image edge
            true_missing = 1 - aperture.to_mask("exact").multiply(np.where(np.isfinite(data), galaxy, 0)).sum() / total
            estimated = elliptical_missing_flux_fraction(data, aperture)
            np.testing.assert_allclose(estimated, true_missing, atol=0.015)
            self.assertGreater(aperture_missing_fraction(data, aperture), 0.25)  # large missing area
            with self.assertWarns(UserWarning):
                self.assertEqual(check_aperture_coverage(data, aperture, "test", flux_weighted=True), usable)


    def test_blank_floor_correlated_noise(self):
        # correlated noise (e.g. coadded images): the floor must not be biased low by masking noise peaks
        from scipy.ndimage import gaussian_filter
        rng = np.random.default_rng(7)
        data = gaussian_filter(rng.normal(0, 1, (600, 600)), 2.0)
        radius = 6.0
        # true scatter of apertures at random positions
        positions = rng.uniform(30, 570, (400, 2))
        fluxes = np.array(aperture_photometry(data, CircularAperture(positions, radius))["aperture_sum"])
        true_scatter = 1.4826 * np.median(np.abs(fluxes - np.median(fluxes)))
        floor = blank_aperture_floor(data, 0.0, 300, 300, radius)
        np.testing.assert_allclose(floor, true_scatter, rtol=0.25)


    def test_extrapolated_kron_floor(self):
        # correlated noise: the noise of large apertures is extrapolated from smaller ones
        from scipy.ndimage import gaussian_filter
        rng = np.random.default_rng(11)
        data = gaussian_filter(rng.normal(0, 1, (1200, 1200)), 2.0)
        radius = 40.0
        positions = rng.uniform(60, 1140, (400, 2))
        fluxes = np.array(aperture_photometry(data, CircularAperture(positions, radius))["aperture_sum"])
        true_scatter = 1.4826 * np.median(np.abs(fluxes - np.median(fluxes)))
        source_mask = np.zeros(data.shape, dtype=bool)
        extrapolated = _extrapolated_blank_rms(data, 600, 600, np.pi * radius**2, source_mask)
        np.testing.assert_allclose(extrapolated, true_scatter, rtol=0.35)
        # Kron aperture too large for blank apertures to fit in a small image: extrapolated, with a warning
        small = data[:300, :300]
        objects = np.zeros(1, dtype=[("x", float), ("y", float), ("a", float), ("b", float), ("theta", float)])
        objects["x"], objects["y"], objects["a"], objects["b"] = 150, 150, 20, 15
        with self.assertWarnsRegex(UserWarning, "extrapolated"):
            flux_err = blank_kron_floor(small, 0.0, objects, kronrad=2.0, scale=2.5)
        self.assertGreater(flux_err, 0)


    def test_splus_calibration_uncertainty(self):
        # zero-point uncertainties of Almeida-Fernandes et al. (2022), added in quadrature
        header = fits.Header({"MAGZP": 22.0, "EXPTIME": 100.0, "TEXPOSED": 100.0, "GAIN": 1.0, "HIERARCH OAJ QC NCNOISE": 1.0})
        for filt, zp_unc in [("u", 0.025), ("F395", 0.025), ("F378", 0.015), ("r", 0.010), ("F660", 0.010)]:
            mag_err = magnitude_calculation(1e5, 1.0, "SPLUS", filt, 100.0, header, 0.1)[1]
            np.testing.assert_allclose(mag_err, zp_unc, rtol=0.01)


if __name__ == "__main__":
    unittest.main()
