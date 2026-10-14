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
from hostphot.cutouts.vista import get_VISTA_images
from hostphot.surveys_utils import normalize_filters, load_yml, filters_file
from hostphot.utils import store_input
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
                "obs_bandpass": ["G", "G", "F660", "F660"],
                "access_format": ["image/fits"] * 4,
                "access_url": ["G_swpweight.fz", "G_swp.fz", "F660_swp.fz", "F660_swpweight.fz"],
            }
        )
        with mock.patch.object(splus.sia, "SIAService") as service:
            service.return_value.search.return_value = table
            with self.assertWarns(UserWarning):  # no image for 'r'
                urls = splus.get_SPLUS_urls(30.0, 0.3, 0.05, ["g", "F660", "r"])
        self.assertEqual(urls, ["G_swp.fz", "F660_swp.fz", None])


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


if __name__ == "__main__":
    unittest.main()
