"""Unit tests of the photometry helpers: error propagation of each survey, zeropoints and offsets, gains,
coverage of the global apertures, blank-aperture floors and network helpers."""
import threading
import time
import unittest
import warnings
from unittest import mock

import numpy as np
from astropy.io import fits
from photutils.aperture import CircularAperture, EllipticalAperture

from hostphot import utils as hp_utils
from hostphot.flags import CoverageWarning, ErrorFloorWarning
from hostphot.photometry import photometry_utils as pu
from astropy.wcs import WCS
from tests.test_cutouts_offline import DEC, RA, fake_image

SURVEY_FILTERS = {
    "DES": "g", "PanSTARRS": "r", "SDSS": "u", "GALEX": "NUV", "2MASS": "J", "WISE": "W1", "unWISE": "W2",
    "LegacySurvey": "z", "Spitzer": "MIPS.1", "VISTA": "Z", "UKIDSS": "Y", "UHS": "J", "SkyMapper": "g",
    "SPLUS": "F378", "JWST": "NIRCam_F090W", "HSC": "r", "KiDS": "u", "Euclid": "VIS",
}


class TestErrorPropagation(unittest.TestCase):
    def test_uncertainty_calculation(self):
        header = fits.Header({"MAGZRR": 0.01, "ZPTERR": 0.02, "MAGZPUNC": 0.006, "NAXIS1": 100, "NAXIS2": 100,
                              "EXPTIME": 100.0, "GAIN": 4.0})
        # (the exposure time and gain keywords are survey specific)
        stubs = mock.patch.multiple(pu, get_image_exptime=mock.Mock(return_value=100.0),
                                    get_image_gain=mock.Mock(return_value=4.0))
        stubs.start()
        self.addCleanup(stubs.stop)
        for survey, filt in SURVEY_FILTERS.items():
            with self.subTest(survey=survey):
                flux, flux_err = 1000.0, 10.0
                total = pu.uncertainty_calculation(flux, flux_err, survey, filt, ap_area=50.0, header=header,
                                                   bkg_rms=0.1)
                self.assertTrue(np.isfinite(total))
                # calibration terms only add uncertainty (WISE: the largest between its noise model and the
                # aperture error, plus the zeropoint uncertainty)
                self.assertGreaterEqual(total, flux_err * 0.999)
        # WISE: the noise model is used if larger than the aperture error
        small = pu.uncertainty_calculation(1000.0, 1e-3, "WISE", "W1", ap_area=50.0, header=header, bkg_rms=0.1)
        formula = np.sqrt(min(13.772 * 4, 50.0) * 50.0 * (1 + np.pi / 2 * 50.0 / 1e4)) * 0.1  # (F_corr <= N_A)
        self.assertAlmostEqual(small / np.hypot(formula, 1000.0 * 0.006 / 1.0857), 1.0, delta=0.01)
        with self.assertRaises(Exception):
            pu.uncertainty_calculation(1.0, 0.1, "Unknown", "x", header=header)

    def test_hst_zeropoint_error(self):
        header = fits.Header({"INSTRUME": "WFC3", "DETECTOR": "UVIS", "FILTER": "F275W", "EXPTIME": 100.0})
        err = pu.get_HST_err("WFC3_UVIS_F275W", header)
        self.assertTrue(np.isfinite(err) and err >= 0)

    def test_gains_and_corrections(self):
        self.assertEqual(pu.sdss_gain("u", 2, 1200), 1.825)
        self.assertEqual(pu.sdss_gain("r", 3, 100), 4.72)
        header = fits.Header({"CAMCOL": 3, "RUN": 100, "NMGY": 0.005})
        self.assertAlmostEqual(pu.get_effective_gain(header, "SDSS", "r"), 4.72 / 0.005)
        self.assertEqual(pu.get_effective_gain(fits.Header(), "Herschel", "PACS100"), np.inf)
        self.assertEqual(pu.get_effective_gain(fits.Header({"GAIN": 25.0}), "Spitzer", "IRAC.1"), 25.0)
        self.assertEqual(pu.get_effective_gain(fits.Header({"EFFGAIN": 3.0}), "JWST", "NIRCam_F090W"), 3.0)
        # IRAC extended-source correction (not applied to small apertures)
        self.assertEqual(pu.irac_extended_source_correction("IRAC.1", 5.0), (1.0, 0.0))
        correction, err = pu.irac_extended_source_correction("IRAC.1", 30.0)
        self.assertLess(correction, 1.0)
        self.assertAlmostEqual(err, 0.1 * (1 - correction))
        self.assertEqual(pu.irac_extended_source_correction("MIPS.1", 30.0), (1.0, 0.0))

    def test_psf_and_non_detection(self):
        self.assertEqual(pu.image_psf_fwhm(fits.Header({"PSF_FWHM": 1.3}), "PanSTARRS", "r"), 1.3)
        self.assertTrue(np.isfinite(pu.image_psf_fwhm(fits.Header({"PSF_FWHM": "bad"}), "PanSTARRS", "r")))
        with self.assertWarnsRegex(UserWarning, "not detected"):
            pu.warn_non_detection("PanSTARRS", "r", -5.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pu.warn_non_detection("PanSTARRS", "r", np.nan)  # (no warning)

    def test_integrate_filter(self):
        wave = np.linspace(4000, 6000, 201)
        flat = np.ones_like(wave)
        response = np.where((wave > 4500) & (wave < 5500), 1.0, 0.0)
        for response_type in ["photon", "energy"]:
            flux = pu.integrate_filter(wave, flat * 2.0, wave, response, response_type=response_type)
            self.assertAlmostEqual(flux, 2.0, places=2)
        with self.assertRaises(AssertionError):
            pu.integrate_filter(wave, flat, wave, response, response_type="other")

    def test_invvar_map(self):
        image = np.ones((5, 5))
        self.assertIsNone(pu.get_invvar_map(fits.HDUList([fits.PrimaryHDU(image)]), "LegacySurvey"))
        hdu = fits.HDUList([fits.PrimaryHDU(image), fits.ImageHDU(image * 4)])
        np.testing.assert_array_equal(pu.get_invvar_map(hdu, "unWISE"), image * 4)
        self.assertIsNone(pu.get_invvar_map(hdu, "HSC"))  # no VARIANCE extension
        self.assertIsNone(pu.get_invvar_map(hdu, "PanSTARRS"))
        variance = np.full((5, 5), 0.25)
        variance[0, 0] = 0
        hdu = fits.HDUList([fits.PrimaryHDU(image), fits.ImageHDU(variance, name="VARIANCE")])
        invvar = pu.get_invvar_map(hdu, "Euclid")
        self.assertEqual(invvar[0, 0], 0)
        self.assertEqual(invvar[1, 1], 4)


class TestCoverageAndFloors(unittest.TestCase):
    @staticmethod
    def galaxy(shape=(200, 200), re=15.0) -> np.ndarray:
        yy, xx = np.indices(shape)
        r = np.hypot(xx - 100, (yy - 100) / 0.7)
        return 100 * np.exp(-r / re)

    def test_global_coverage(self):
        data = self.galaxy()
        aperture = EllipticalAperture((100, 100), 60, 42, theta=0.0)
        self.assertEqual(pu.check_global_coverage(data, aperture, "test"), (True, 0.0))
        cut = data.copy()
        cut[:, 145:] = np.nan  # outskirts without data: little flux missing
        with self.assertWarnsRegex(CoverageWarning, "added to the measured flux"):
            usable, missing = pu.check_global_coverage(cut, aperture, "test")
        self.assertTrue(usable)
        self.assertGreater(missing, 0)
        # the filled-in flux recovers the true flux (symmetric galaxy)
        weights = aperture.to_mask("exact")
        measured = np.nansum(weights.cutout(cut) * weights.data)
        full = np.nansum(weights.cutout(data) * weights.data)
        corrected, corrected_err = pu.correct_missing_flux(measured, 1.0, missing)
        self.assertAlmostEqual(corrected / full, 1.0, delta=0.01)
        self.assertGreater(corrected_err, 1.0)
        self.assertTrue(np.isnan(pu.correct_missing_flux(np.nan, 1.0, 0.05)[0]))  # (NaN fluxes are kept)
        # half of the galaxy without data: too much flux missing
        cut[:, 100:] = np.nan
        with self.assertWarnsRegex(UserWarning, "tolerance"):
            usable, _ = pu.check_global_coverage(cut, aperture, "test")
        self.assertFalse(usable)
        # local apertures: missing area
        with self.assertWarns(UserWarning):
            self.assertFalse(pu.check_aperture_coverage(cut, CircularAperture((100, 100), 10), "test"))

    def test_saturation(self):
        flags = np.zeros((50, 50), dtype=bool)
        aperture = CircularAperture((25, 25), 5)
        self.assertTrue(pu.check_saturation(None, aperture, "test"))
        self.assertTrue(pu.check_saturation(flags, aperture, "test"))
        flags[25, 25] = True
        with self.assertWarnsRegex(UserWarning, "without measured data"):
            self.assertFalse(pu.check_saturation(flags, aperture, "test"))

    def test_blank_floor_near_far(self):
        # sky noise, plus faint structure around the source (e.g. the outer disc of a galaxy filling the image),
        # below the detection threshold: the local blank apertures are replaced by the distant ones
        rng = np.random.default_rng(1)
        noise = rng.normal(0, 1, (600, 600))
        yy, xx = np.indices(noise.shape)
        r = np.hypot(xx - 300, yy - 300)
        structure = 0.8 * np.sin(r / 3.0) * (r < 150)  # ring-like structure near the source
        source_mask = np.zeros(noise.shape, dtype=bool)
        source_mask[290:311, 290:311] = True
        clean, replaced = pu._blank_aperture_rms(noise, 300, 300, 8.0, source_mask=source_mask)
        self.assertFalse(replaced)
        rms, replaced = pu._blank_aperture_rms(noise + structure, 300, 300, 8.0, source_mask=source_mask,
                                               residual_background=False)
        self.assertTrue(replaced)
        self.assertAlmostEqual(rms / clean, 1.0, delta=0.4)
        with self.assertWarns(ErrorFloorWarning):
            err = pu.blank_aperture_floor(noise + structure, 0.0, 300, 300, 8.0, source_mask=source_mask,
                                          residual_background=False, label="test")
        self.assertAlmostEqual(err, rms)
        # without blank sky: no floor (with a warning)
        with self.assertWarns(ErrorFloorWarning):
            self.assertEqual(pu.blank_aperture_floor(noise, 2.0, 300, 300, 8.0,
                                                     source_mask=np.ones(noise.shape, dtype=bool)), 2.0)
        self.assertEqual(pu._apply_floor(3.0, None), 3.0)
        # the row-run check of large apertures gives the same blank apertures as the pixel-by-pixel one
        noisy = noise + structure
        noisy[50:120, 400:470] = np.nan  # region without data
        for make_aperture, half_size in [(lambda x, y: CircularAperture((x, y), r=12.0), 12.0),
                                         (lambda x, y: EllipticalAperture((x, y), 20.0, 9.0, theta=0.7), 20.0)]:
            reference = pu._blank_rms(noisy, 300, 300, make_aperture, half_size, 2.5 * half_size, 6 * half_size,
                                      source_mask=source_mask, residual_background=False)
            with mock.patch.object(pu, "_VECTORISED_MAX_PIXELS", 0):
                large = pu._blank_rms(noisy, 300, 300, make_aperture, half_size, 2.5 * half_size, 6 * half_size,
                                      source_mask=source_mask, residual_background=False)
            self.assertEqual(reference, large)
        self.assertFalse(pu.floor_residual_background("HST", "NICMOS_F160W"))
        self.assertTrue(pu.floor_residual_background("HST", "WFC3_IR_F160W"))

    @staticmethod
    def kron_object(x=300.0, y=300.0, a=5.0, b=4.0, theta=0.3) -> np.ndarray:
        dtype = [("x", "f8"), ("y", "f8"), ("a", "f8"), ("b", "f8"), ("theta", "f8")]
        return np.array([(x, y, a, b, theta)], dtype=dtype)

    def test_sky_level_error(self):
        self.assertEqual(pu._sky_level_error({}, 1.0), 0.0)
        self.assertEqual(pu._sky_level_error({"raw_rms": 5.0, "offset": 1.0, "offset_err": 1.0}, None), 0.0)
        stats = {"raw_rms": 5.0, "offset": 10.0, "offset_err": 6.0}
        self.assertAlmostEqual(pu._sky_level_error(stats, 3.0), np.sqrt(4.0**2 + 8.0**2))
        # residual sky offset (much smaller than the pixel noise): included in the uncertainty of large apertures
        rng = np.random.default_rng(2)
        noise = rng.normal(0, 1, (600, 600))
        objects = self.kron_object()
        area = np.pi * 10 * 8  # (Kron radius 1, scale 2)
        clean = pu.blank_kron_floor(noise, 0.0, objects, 1.0, 2.0)
        self.assertLess(clean, 1.3 * np.sqrt(area))  # (white noise, no sky offset: ~ the floor)
        offset = 0.5
        err = pu.blank_kron_floor(noise + offset, 0.0, objects, 1.0, 2.0)
        self.assertAlmostEqual(err / np.hypot(clean, offset * area), 1.0, delta=0.15)
        # same statistics with the row-run check of large apertures
        stats = []
        for max_pixels in [1000, 0]:
            stats.append({})
            with mock.patch.object(pu, "_VECTORISED_MAX_PIXELS", max_pixels):
                pu._blank_elliptical_rms(noise + offset, 300, 300, 10, 8, 0.3, sky_stats=stats[-1],
                                         source_mask=np.zeros(noise.shape, dtype=bool))
        self.assertEqual(stats[0], stats[1])
        # aperture larger than the image allows: the offset is scaled from smaller apertures
        yy, xx = np.indices((160, 160))
        small = noise[:160, :160] + offset + 50 * np.exp(-np.hypot(xx - 80, yy - 80) / 4)  # (host masked)
        with self.assertWarnsRegex(ErrorFloorWarning, "do not fit"):
            err = pu.blank_kron_floor(small, 0.0, self.kron_object(80, 80, 20, 16), 1.0, 2.0)
        self.assertAlmostEqual(err / (offset * np.pi * 40 * 32), 1.0, delta=0.3)

    def test_sparse_background(self):
        from hostphot.processing.background import SparseBackground, sep_background, sparse_pixel_rms

        rng = np.random.default_rng(4)
        # shallow photon-counting image (~0.02 sky photons per pixel in 155 s, in counts per second, minus a smooth
        # background map, as GALEX): the MAD-based RMS collapses, so the noise from the variance in boxes is used
        lam, exptime = 0.02, 155.0
        yy, xx = np.indices((400, 400))
        sparse = rng.poisson(lam, (400, 400)) / exptime - lam / exptime * (1 + 0.05 * np.sin(xx / 50))
        sparse[200:205, 200:205] += 0.5  # (a source)
        bkg = sep_background(sparse)
        self.assertIsInstance(bkg, SparseBackground)
        self.assertAlmostEqual(bkg.globalrms / (np.sqrt(lam) / exptime), 1.0, delta=0.1)
        self.assertEqual(bkg.rms().shape, sparse.shape)
        self.assertTrue(np.all(bkg.rms() == bkg.globalrms))
        self.assertEqual(bkg.back().shape, sparse.shape)  # (other attributes from sep.Background)
        # Gaussian noise (and a gradient): unchanged
        gauss = rng.normal(0, 1, (400, 400)) + np.linspace(0, 20, 400)[None, :]
        bkg = sep_background(gauss)
        self.assertNotIsInstance(bkg, SparseBackground)
        flat = rng.normal(0, 1, (400, 400))
        self.assertAlmostEqual(sparse_pixel_rms(flat) / sep_background(flat).globalrms, 1.0, delta=0.1)

    def test_cog_tail_uncertainty(self):
        from hostphot.photometry import global_photometry as gp

        rng = np.random.default_rng(3)
        yy, xx = np.indices((400, 400))
        r = np.hypot(xx - 200, (yy - 200) / 0.8)
        objects = self.kron_object(200, 200, 8, 6.4, 0.0)
        for profile, noise in [(np.exp(-((r / 12) ** 0.25)), 0.02), (np.exp(-r / 6), 0.01)]:
            data = 100 * profile / profile.max() + rng.normal(0, noise, r.shape)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cog = gp.cog_aperture(data, objects, 2.5, noise, 3.0, label="test")
            self.assertGreater(cog["tail_frac_err"], 0)
            if cog["tail_frac"] > 0:
                self.assertGreaterEqual(cog["tail_frac_err"], gp.COG_TAIL_MIN_ERR * cog["tail_frac"] * (1 - 1e-9))


class TestNetworkHelpers(unittest.TestCase):
    def tearDown(self):
        hp_utils.set_network_timeout(300)

    def test_call_with_timeout(self):
        start = time.time()
        with self.assertRaisesRegex(TimeoutError, "test: no response within 1 s"):
            hp_utils.call_with_timeout(threading.Event().wait, 5, timeout=1, label="test")
        self.assertLess(time.time() - start, 3)
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) < 2:
                raise ConnectionError("temporary")
            return "ok"

        with mock.patch.object(hp_utils.time, "sleep"):
            self.assertEqual(hp_utils.call_with_timeout(flaky), "ok")
            with self.assertRaises(ValueError):
                hp_utils.call_with_timeout(mock.Mock(side_effect=ValueError("bad")), attempts=2)
        hp_utils.set_network_timeout(10)
        self.assertEqual(hp_utils.network_timeout(), 10)
        self.assertEqual(hp_utils.network_timeout(large=True), 10 * hp_utils.LARGE_DOWNLOAD_FACTOR)

    def test_open_fits_from_url(self):
        import gzip
        import io

        buffer = io.BytesIO()
        fits.PrimaryHDU(np.ones((3, 3))).writeto(buffer)
        for content in [buffer.getvalue(), gzip.compress(buffer.getvalue())]:
            with mock.patch.object(hp_utils.requests, "get") as get:
                get.return_value.content = content
                hdu = hp_utils.open_fits_from_url("https://example.org/image.fits")
            np.testing.assert_array_equal(hdu[0].data, np.ones((3, 3)))
            self.assertEqual(get.call_args.kwargs["timeout"], hp_utils.NETWORK_TIMEOUT)

    def test_read_fits_header(self):
        import gzip
        import io

        from astropy.wcs import WCS

        buffer = io.BytesIO()
        header = WCS(naxis=2).to_header()
        header["EXPTIME"] = 12.5
        fits.PrimaryHDU(np.zeros((300, 300)), header=header).writeto(buffer)
        for content in [buffer.getvalue(), gzip.compress(buffer.getvalue())]:
            chunks = [content[i:i + 1000] for i in range(0, len(content), 1000)]
            with mock.patch.object(hp_utils.requests, "get") as get:
                response = get.return_value.__enter__.return_value
                response.iter_content.return_value = iter(chunks)
                read = hp_utils.read_fits_header("https://example.org/image.fits")
            self.assertEqual(read["EXPTIME"], 12.5)
            self.assertEqual(read["NAXIS1"], 300)
            self.assertTrue(get.call_args.kwargs["stream"])
        with mock.patch.object(hp_utils.requests, "get", side_effect=ConnectionError("down")):
            self.assertIsNone(hp_utils.read_fits_header("https://example.org/image.fits"))

    def test_header_coverage(self):
        image = fake_image(RA, DEC, 100, 1.0)
        img_wcs = WCS(image[0].header)
        for ra, size in [(RA, 60), (RA + 40 / 3600, 60), (RA + 0.5, 60)]:
            self.assertAlmostEqual(hp_utils.header_coverage(image[0].header, ra, DEC, size),
                                   hp_utils.image_coverage(np.ones((100, 100)), img_wcs, ra, DEC, size))
        self.assertEqual(hp_utils.header_coverage(fits.Header(), RA, DEC, 60), 0.0)


if __name__ == "__main__":
    unittest.main()


class TestDustAndImageHelpers(unittest.TestCase):
    def test_extinction(self):
        # the dust maps (downloaded from GitHub) are replaced by a constant E(B-V)
        from hostphot.photometry import dust

        sfd = mock.Mock()
        sfd.ebv.return_value = 0.1
        with mock.patch.object(dust, "_download_dustmaps"), mock.patch.object(
            dust.sfdmap, "SFDMap", return_value=sfd
        ), mock.patch.dict(dust._sfd_maps, clear=True):
            a_r = dust.calc_extinction("r", "PanSTARRS", 10.0, -20.0)
            a_g = dust.calc_extinction("g", "PanSTARRS", 10.0, -20.0)
            self.assertGreater(a_g, a_r)
            self.assertAlmostEqual(a_r, 0.1 * 2.3, delta=0.2)  # A_r / E(B-V) ~ 2.3-2.6
            # BASS+MzLS filters for the north of the Legacy Survey
            self.assertGreater(dust.calc_extinction("g", "LegacySurvey", 180.0, 60.0), 0)
            wave = np.linspace(4000, 9000, 50)
            for law in ["ccm89", "odonnell94", "calzetti00", "fm07"]:
                dereddened = dust.deredden(wave, np.ones(50), 10.0, -20.0, reddening_law=law)
                self.assertTrue(np.all(dereddened > 1))
            with self.assertRaises(AssertionError):
                dust.deredden(wave, np.ones(50), 10.0, -20.0, reddening_law="other")

    def test_image_header_helpers(self):
        from hostphot.photometry import image_utils as iu

        self.assertEqual(iu.get_image_readnoise(fits.Header({"HIERARCH CELL.READNOISE": 5.0}), "PanSTARRS"), 5.0)
        self.assertEqual(iu.get_image_readnoise(fits.Header(), "DES"), 7.0)
        self.assertEqual(iu.get_image_readnoise(fits.Header({"INSTRUME": "IRAC", "CHNLNUM": 2}), "Spitzer"), 12.0)
        self.assertEqual(iu.get_image_readnoise(fits.Header({"INSTRUME": "MIPS"}), "Spitzer"), 40.0)
        self.assertEqual(iu.get_image_readnoise(fits.Header({"READNOIS": 25.0}), "UHS"), 25.0)
        self.assertEqual(iu.get_image_readnoise(fits.Header(), "GALEX"), 0.0)
        self.assertAlmostEqual(iu.wrap_theta(np.pi * 0.75), -np.pi * 0.25)

    def test_trim_and_saturation_mask(self):
        from astropy.coordinates import SkyCoord
        from astropy.wcs import WCS
        from hostphot.processing.cleaning import saturated_mask, trim_images

        w = WCS(naxis=2)
        w.wcs.crpix, w.wcs.crval = [50.5, 50.5], [10.0, 0.0]
        w.wcs.cdelt, w.wcs.ctype = [-1 / 3600, 1 / 3600], ["RA---TAN", "DEC--TAN"]
        hdu = fits.HDUList([fits.PrimaryHDU(np.ones((100, 100)), header=w.to_header())])
        trimmed = trim_images([hdu], SkyCoord(10.0, 0.0, unit="deg"), 20)[0]
        self.assertEqual(trimmed[0].data.shape, (20, 20))
        # saturated pixels (NO_DATA bit of the HSC mask plane)
        mask = np.zeros((100, 100), dtype=np.int32)
        mask[5, 5] = 1 << 8
        mask_hdu = fits.ImageHDU(mask, name="MASK")
        self.assertIsNone(saturated_mask(fits.HDUList([fits.PrimaryHDU(np.ones((100, 100))), mask_hdu])))
        mask_hdu.header["MP_NO_DATA"] = 8
        flags = saturated_mask(fits.HDUList([fits.PrimaryHDU(np.ones((100, 100))), mask_hdu]))
        self.assertTrue(flags[5, 5])
        self.assertEqual(flags.sum(), 1)

    def test_uhs(self):
        from hostphot.cutouts import uhs

        with mock.patch.object(uhs, "get_WFCAM_images", return_value=[None]) as wfcam:
            self.assertEqual(uhs.get_UHS_images(10.0, 0.0, filters="J"), [None])
        self.assertEqual(wfcam.call_args.kwargs["survey"], "UHS")
