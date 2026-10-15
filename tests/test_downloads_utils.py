import shutil
import socket
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from astropy.io import fits
import pandas as pd
from astropy.table import Table

from hostphot._constants import workdir
from hostphot.cutouts import download_images
from hostphot.cutouts import panstarrs
from hostphot.cutouts import splus
from hostphot.cutouts import legacysurvey
from hostphot.cutouts import vista
from hostphot.cutouts import hsc
from hostphot.cutouts.vista import get_VISTA_images
from hostphot.surveys_utils import normalize_filters, load_yml, filters_file
from hostphot.utils import store_input, image_has_data, open_fits_from_urls
from hostphot import utils as hp_utils
from hostphot.cutouts import wise
from hostphot.photometry.photometry_utils import extract_filter


class TestDownloadsUtils(unittest.TestCase):
    def test_normalize_filters(self):
        # single filters with multi-character names are not split into characters
        self.assertEqual(normalize_filters("FUV", "GALEX"), ["FUV"])
        self.assertEqual(normalize_filters("Ks", "2MASS"), ["Ks"])
        self.assertEqual(normalize_filters("W1", "WISE"), ["W1"])
        self.assertEqual(normalize_filters("WFC3_UVIS_F275W", "HST"), ["WFC3_UVIS_F275W"])
        # concatenation of single-character filters
        self.assertEqual(normalize_filters("grizy", "PanSTARRS"), ["g", "r", "i", "z", "y"])
        self.assertEqual(normalize_filters("JH", "2MASS"), ["J", "H"])
        # lists are kept as they are
        self.assertEqual(normalize_filters(["r", "g"], "PanSTARRS"), ["r", "g"])
        self.assertIsNone(normalize_filters(None, "PanSTARRS"))

    def test_PS1_urls_order(self):
        # the PS1 service returns the filters sorted from blue to red, and only those with images
        table = Table(
            {
                "filter": ["g", "r"],
                "filename": ["g_image.fits", "r_image.fits"],
            }
        )
        with mock.patch.object(panstarrs, "query_ps1", return_value=table):
            urls = panstarrs.get_PS1_urls(0.0, 0.0, 1, ["r", "i", "g"])
        self.assertEqual(len(urls), 3)
        self.assertTrue(urls[0].endswith("r_image.fits"))
        self.assertIsNone(urls[1])  # no image for the i band
        self.assertTrue(urls[2].endswith("g_image.fits"))

    def test_HST_existing_image(self):
        # with overwrite=False, an existing HST image is not downloaded again (no crash)
        name, filt = "HST_existing_test", "WFC3_UVIS_F275W"
        survey_dir = Path(workdir, name, "HST")
        survey_dir.mkdir(parents=True, exist_ok=True)
        try:
            fits.PrimaryHDU(data=np.zeros((5, 5))).writeto(survey_dir / f"HST_{filt}.fits", overwrite=True)
            with mock.patch("hostphot.cutouts.hst.get_HST_images") as get_images:
                download_images(name, 10.0, 10.0, survey="HST", filters=filt, overwrite=False, save_input=False)
            get_images.assert_not_called()
        finally:
            shutil.rmtree(Path(workdir, name))

    def test_HST_filters_required(self):
        with self.assertRaises(ValueError):
            download_images("HST_no_filters_test", 10.0, 10.0, survey="HST", save_input=False)

    def test_VISTA_no_coverage(self):
        # no VISTA coverage in the far north: one None per filter (not False)
        self.assertEqual(get_VISTA_images(10.0, 80.0, 1, ["J", "H"]), [None, None])

    def test_extract_filter(self):
        wave, trans = extract_filter("WFC3_UVIS_F275W", "HST")
        self.assertGreater(len(wave), 0)
        # Legacy Survey without version defaults to DECam
        wave, trans = extract_filter("g", "LegacySurvey")
        wave_decam, _ = extract_filter("g", "LegacySurvey", "DECam")
        np.testing.assert_array_equal(wave, wave_decam)


    def test_store_input_changed_columns(self):
        # file written by an older version (without 'timestamp'), then appended to without header
        tmp_dir = Path(tempfile.mkdtemp())
        inputs_file = tmp_dir / "input_cutouts.csv"
        try:
            inputs_file.write_text(
                "name,ra,survey\n"
                "obj,10.0,PanSTARRS\n"
                "obj,10.0,PanSTARRS,2025-01-01T00:00:00+00:00\n"
            )
            store_input({"name": "obj", "ra": 11.0, "survey": "DES", "size": 3}, inputs_file)
            df = pd.read_csv(inputs_file)  # must be readable
            self.assertEqual(list(df.columns), ["name", "ra", "survey", "size", "timestamp"])
            self.assertEqual(len(df), 3)
            self.assertTrue(pd.isna(df.timestamp[0]))
            self.assertEqual(df.timestamp[1], "2025-01-01T00:00:00+00:00")
            self.assertTrue(pd.isna(df["size"][1]))
            self.assertEqual(df.survey[2], "DES")
            # appending with the same columns keeps working
            store_input({"name": "obj", "ra": 12.0, "survey": "DES", "size": 3}, inputs_file)
            self.assertEqual(len(pd.read_csv(inputs_file)), 4)
        finally:
            shutil.rmtree(tmp_dir)


    def test_filters_config(self):
        # every filter of every survey must have the keys used by the photometry
        required = ["zeropoint", "pixel_scale", "pixel_units", "magnitude_system", "AB_offset"]
        for survey, filters in load_yml(filters_file).items():
            for filt, config in filters.items():
                missing = [key for key in required if key not in config]
                self.assertEqual(missing, [], f"{survey} {filt} is missing {missing}")


    def test_splus_urls(self):
        # the service lists broad-band filters in upper case, plus weight maps
        table = pd.DataFrame(
            {
                "obs_bandpass": ["G", "G", "F660", "F660", "G", "G"],
                "access_format": ["image/fits"] * 6,
                "access_url": ["T1_G_swpweight.fz", "T1_G_swp.fz", "T1_F660_swp.fz", "T1_F660_swpweight.fz",
                               "T2_G_swp.fz", "T2_G_swpweight.fz"],
                # tile T1 is nearer to the position than T2
                "s_ra": [30.1, 30.1, 30.1, 30.1, 30.9, 30.9],
                "s_dec": [0.3] * 6,
            }
        )
        with mock.patch.object(splus.sia, "SIAService") as service:
            service.return_value.search.return_value = table
            with self.assertWarns(UserWarning):  # no image for 'r'
                urls = splus.get_SPLUS_urls(30.0, 0.3, 0.05, ["g", "F660", "r"])
        self.assertEqual(urls, ["T1_G_swp.fz", "T1_F660_swp.fz", None])


    def test_legacysurvey_single_band(self):
        # a single band is returned by the service as a 2D image (not a cube)
        image = np.arange(100, dtype=np.float32).reshape(10, 10)
        response = fits.HDUList([fits.PrimaryHDU(image), fits.ImageHDU(image * 2)])
        with mock.patch.object(legacysurvey, "open_fits_from_url", return_value=response):
            hdu_list = legacysurvey.get_LegacySurvey_images(150.0, 2.0, size=0.04, filters="r")
        self.assertEqual(len(hdu_list), 1)
        np.testing.assert_array_equal(hdu_list[0][0].data, image)
        np.testing.assert_array_equal(hdu_list[0][1].data, image * 2)

    def test_vista_timeout(self):
        # an unresponsive archive raises a clear error instead of hanging
        with mock.patch.object(vista.urllib.request, "urlopen", side_effect=socket.timeout("timed out")):
            with self.assertRaisesRegex(TimeoutError, "did not respond"):
                vista.get_VISTA_images(10.0, 0.0, size=3, filters="J")


    def test_hsc_coadd_type(self):
        # coadds without the local sky subtraction (which removes the extended light of galaxies)
        with mock.patch.object(hsc.requests, "get") as get:
            get.return_value.content = b""
            hsc._request_cutout(150.0, 2.0, 0.01, 0.01, "HSC-R", "pdr3_wide", ("user", "password"))
        self.assertEqual(get.call_args.kwargs["params"]["type"], "coadd/bg")


    def test_image_has_data(self):
        rng = np.random.default_rng(0)
        sky = rng.normal(0, 1, (200, 200))
        self.assertTrue(image_has_data(sky))
        sky_with_star = sky.copy()
        sky_with_star[100, 100] = 1e6  # bright source (inflates the standard deviation)
        self.assertTrue(image_has_data(sky_with_star))
        # sparse real counts (zero-count pixels in between)
        counts = np.zeros((200, 200))
        counts[rng.integers(0, 200, 6000), rng.integers(0, 200, 6000)] = 1e-3
        self.assertTrue(image_has_data(counts))
        # (almost) all zeros: no data (e.g. outside the footprint)
        zero_filled = np.zeros((200, 200))
        zero_filled[:20, :10] = rng.normal(0, 1, (20, 10))
        self.assertFalse(image_has_data(zero_filled))
        # no data: constant, or only uniform compression noise
        self.assertFalse(image_has_data(np.full((200, 200), 3.0)))
        self.assertFalse(image_has_data(rng.uniform(-0.02, 0.02, (200, 200))))
        self.assertFalse(image_has_data(np.full((10, 10), np.nan)))

    def test_failed_download_warns(self):
        # failed downloads are retried once and reported (they used to fail silently)
        with mock.patch.object(hp_utils, "open_fits_from_url", side_effect=ConnectionError("down")) as opener:
            with self.assertWarnsRegex(UserWarning, "could not be downloaded"):
                hdus = open_fits_from_urls(["https://example.org/image.fits", None])
        self.assertEqual(hdus, [None, None])
        self.assertEqual(opener.call_count, 2)

    def test_wise_bad_response(self):
        # an unexpected response from the image service raises an error instead of returning no images
        with mock.patch.object(wise.requests, "get") as get:
            get.return_value.text = "<html>Service unavailable</html>"
            with self.assertRaisesRegex(ConnectionError, "Unexpected response"):
                wise.get_WISE_images(150.0, 2.0, size=3, filters=["W1"])


    def test_zeros_to_nan(self):
        from hostphot.processing.cleaning import zeros_to_nan
        data = np.random.default_rng(1).normal(0, 1, (200, 200))
        data[:, 150:] = 0  # region without data (10000 pixels)
        data[10:15, 10:15] = 0  # small group of zero pixels (kept)
        out = zeros_to_nan(data)
        self.assertTrue(np.isnan(out[:, 150:]).all())
        self.assertFalse(np.isnan(out[:, :150]).any())
        # integer images are also handled
        ints = np.ones((100, 100), dtype=int)
        ints[:, 50:] = 0
        self.assertTrue(np.isnan(zeros_to_nan(ints)[:, 50:]).all())


    def test_splus_tile_coverage(self):
        # the nearest tile is skipped if it does not cover the position (zero weight)
        from astropy.wcs import WCS
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crval, wcs.wcs.crpix, wcs.wcs.cdelt = [30.0, 0.3], [50.5, 50.5], [-0.55 / 3600, 0.55 / 3600]
        header = wcs.to_header()
        header.update({"OBJECT": "STRIPE82_0044", "FILTER": "G", "TEXPOSED": 120.0})
        science = np.ones((100, 100))
        no_coverage = fits.HDUList([fits.PrimaryHDU(np.zeros((100, 100)), header)])
        weight = np.ones((100, 100))
        weight[:, :20] = 0  # part without coverage
        coverage = fits.HDUList([fits.PrimaryHDU(weight, header)])
        files = {
            "T1_G_swp.fz": fits.HDUList([fits.PrimaryHDU(science, header)]), "T1_G_swpweight.fz": no_coverage,
            "T2_G_swp.fz": fits.HDUList([fits.PrimaryHDU(science, header)]), "T2_G_swpweight.fz": coverage,
        }
        candidates = [[("T1_G_swp.fz", "T1_G_swpweight.fz"), ("T2_G_swp.fz", "T2_G_swpweight.fz")]]
        with mock.patch.object(splus, "_get_SPLUS_candidates", return_value=candidates), \
                mock.patch.object(splus, "open_fits_from_url", side_effect=lambda url: files[url]):
            hdu = splus.get_SPLUS_images(30.0, 0.3, size=1, filters=["g"])[0]
        data = hdu[0].data
        self.assertTrue(np.isnan(data[:, :20]).all())  # no coverage in tile T2
        self.assertTrue(np.isfinite(data[:, 20:]).all())
        self.assertIn("MAGZP", hdu[0].header)


if __name__ == "__main__":
    unittest.main()
