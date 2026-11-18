"""Offline tests of the main workflows on the PanSTARRS test images (host of SN 2002fk): masking with the host
model (Gaia mocked), global photometry with the CoG aperture and common apertures, plots and SEDs. Each test
uses its own object name, as the tests can run in parallel."""
import unittest
import warnings
from pathlib import Path
from unittest import mock

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

from tests.helpers import install_test_images
from hostphot._constants import workdir
from hostphot.flags import FlagWarning
from hostphot.processing import coadd_images, create_mask
from hostphot.processing import objects_detection as od
from hostphot.photometry import global_photometry as gp
from hostphot.photometry import local_photometry as lp
from hostphot.photometry.sed_plotting import plot_sed

HOST_RA, HOST_DEC = 50.527333, -15.400056
SN_RA, SN_DEC, Z = 50.52379, -15.40089, 0.007482
SURVEY = "PanSTARRS"


def _gaia_table(ra: float, dec: float, rad: float) -> Table:
    """Synthetic Gaia DR3 query result: two stars (significant proper motion) and a non-stellar source."""
    return Table(
        {
            "ra": [ra + 0.01, ra - 0.012, ra + 0.002],
            "dec": [dec + 0.01, dec - 0.008, dec],
            "parallax_over_error": [1.0, 20.0, 0.5],
            "pmra": [30.0, 1.0, 0.1],
            "pmra_error": [0.5, 0.5, 0.5],
            "pmdec": [-10.0, 1.0, 0.1],
            "pmdec_error": [0.5, 0.5, 0.5],
        }
    )


class TestOfflineWorkflows(unittest.TestCase):
    def setUp(self):
        od._gaia_cache.clear()

    def test_host_model_masking(self):
        name = "offline_host_model"
        install_test_images(name, SURVEY)
        Path(workdir, name, od.GAIA_CACHE_FILE).unlink(missing_ok=True)  # (from previous runs)
        coadd_images(name, "riz", SURVEY)
        with mock.patch.object(od, "_query_gaia_esa", side_effect=_gaia_table) as query:
            create_mask(name, HOST_RA, HOST_DEC, "riz", SURVEY, method="host_model", crossmatch=True,
                        save_mask_params=True, save_plots=True)
            for filt in "gr":
                create_mask(name, HOST_RA, HOST_DEC, filt, SURVEY, method="host_model", ref_filt="riz",
                            ref_survey=SURVEY, save_plots=False)
        # the Gaia sources are queried once and cached for the object
        self.assertEqual(query.call_count, 1)
        self.assertTrue(Path(workdir, name, od.GAIA_CACHE_FILE).is_file())
        survey_dir = Path(workdir, name, SURVEY)
        for filt in ["riz", "g", "r"]:
            masked = fits.getdata(survey_dir / f"{SURVEY}_{filt}_masked.fits")
            original = fits.getdata(survey_dir / f"{SURVEY}_{filt}.fits")
            self.assertEqual(masked.shape, original.shape)
            self.assertFalse(np.allclose(masked, original, equal_nan=True))  # some sources masked
            self.assertTrue(np.all(np.isfinite(masked) | ~np.isfinite(original)))

    def test_masking_gaia_failure(self):
        # all the Gaia sources fail: the mask is still created, with the mask_nogaia flag in the header
        name = "offline_gaia_failure"
        install_test_images(name, SURVEY, "r")
        with mock.patch.object(od, "_query_gaia_esa", side_effect=ConnectionError("down")), mock.patch.object(
            od, "_query_gaia_mirror", side_effect=ConnectionError("down")
        ):
            with self.assertWarnsRegex(FlagWarning, "Gaia DR3 query failed"):
                create_mask(name, HOST_RA, HOST_DEC, "r", SURVEY, crossmatch=True, save_plots=False)
        header = fits.getheader(Path(workdir, name, SURVEY, f"{SURVEY}_r_masked.fits"))
        self.assertTrue(header["HPFNOGAI"])

    def test_global_cog_and_common_aperture(self):
        name = "offline_cog"
        install_test_images(name, SURVEY)
        kwargs = dict(survey=SURVEY, use_mask=False, save_plots=False, save_input=False, raise_exception=True)
        kron = gp.multi_band_phot(name, HOST_RA, HOST_DEC, filters="gr", optimize_kronrad=True, **kwargs)
        cog = gp.multi_band_phot(name, HOST_RA, HOST_DEC, filters="gr", aperture_method="cog",
                                 save_aperture_params=True, **kwargs)
        for filt in "gr":
            # CoG and optimised Kron magnitudes of a bright galaxy agree
            self.assertLess(abs(cog[filt][0] - kron[filt][0]), 0.15)
        params = pd.read_csv(Path(workdir, name, SURVEY, "aperture_parameters_r.csv"))
        self.assertEqual(params["method"].values[0], "cog")
        self.assertGreater(params["r_meas"].values[0], 0)
        # common aperture from the r band (same method)
        common = gp.multi_band_phot(name, HOST_RA, HOST_DEC, filters="giz", common_aperture=True, ref_filt="r",
                                    ref_survey=SURVEY, aperture_method="cog", **kwargs)
        self.assertTrue(all(np.isfinite(common[filt][0]) for filt in "giz"))
        # the common aperture must use the same method as the reference
        with self.assertRaises(Exception):
            gp.multi_band_phot(name, HOST_RA, HOST_DEC, filters="g", common_aperture=True, ref_filt="r",
                               ref_survey=SURVEY, aperture_method="kron", **kwargs)
        # the aperture is transferred from its sky ellipse: the reference image is not needed, and parameters saved
        # without the sky ellipse (before v4.0.0) use the reference image, with the same result
        for col in ["ra", "dec", "a_arcsec", "b_arcsec", "pa_deg"]:
            self.assertIn(col, params)
        ref_file = Path(workdir, name, SURVEY, f"{SURVEY}_r.fits")
        ref_copy = ref_file.with_suffix(".bak")
        ref_file.rename(ref_copy)
        try:
            no_ref = gp.multi_band_phot(name, HOST_RA, HOST_DEC, filters="g", common_aperture=True, ref_filt="r",
                                        ref_survey=SURVEY, aperture_method="cog", **kwargs)
        finally:
            ref_copy.rename(ref_file)
        self.assertAlmostEqual(no_ref["g"][0], common["g"][0], places=6)
        params_file = Path(workdir, name, SURVEY, "aperture_parameters_r.csv")
        params.drop(columns=["a_arcsec", "b_arcsec", "pa_deg"]).to_csv(params_file, index=False)
        old = gp.multi_band_phot(name, HOST_RA, HOST_DEC, filters="g", common_aperture=True, ref_filt="r",
                                 ref_survey=SURVEY, aperture_method="cog", **kwargs)
        self.assertAlmostEqual(old["g"][0], common["g"][0], places=3)

    def test_global_partial_coverage(self):
        # part of the galaxy without data: the missing flux is added (within the tolerance), with the coverage flag
        name = "offline_coverage"
        install_test_images(name, SURVEY, "r")
        kwargs = dict(filters="r", survey=SURVEY, use_mask=False, save_plots=False, save_input=False,
                      save_results=False, raise_exception=True)
        full = gp.multi_band_phot(name, HOST_RA, HOST_DEC, **kwargs)
        image_file = Path(workdir, name, SURVEY, f"{SURVEY}_r.fits")
        with fits.open(image_file, mode="update") as hdul:
            data = hdul[0].data
            ny, nx = data.shape
            data[:, : int(nx * 0.43)] = np.nan  # the edge of the image cuts into the outskirts of the galaxy
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cut = gp.multi_band_phot(name, HOST_RA, HOST_DEC, **kwargs)
        messages = [str(w.message) for w in caught]
        if np.isfinite(cut["r"][0]):
            self.assertTrue(any("added to the measured flux" in m for m in messages), messages)
            self.assertTrue(bool(cut["r_coverage_flag"][0]))
            self.assertLess(abs(cut["r"][0] - full["r"][0]), 0.1)
        else:
            self.assertTrue(any("tolerance" in m for m in messages), messages)

    def test_extract_aperture_background_default(self):
        # extract_aperture() subtracts the background by default for the surveys that need it (bkg_surveys), as
        # the photometry and the masking do
        name = "offline_aperture_bkg"
        install_test_images(name, SURVEY, "r")
        image_file = Path(workdir, name, SURVEY, f"{SURVEY}_r.fits")
        with fits.open(image_file, mode="update") as hdul:
            hdul[0].data += 300.0  # sky level left in the image
        kwargs = dict(filt="r", survey=SURVEY, use_mask=False, save_plots=False, save_aperture_params=False,
                      save_input=False)
        with mock.patch.object(gp, "bkg_surveys", [SURVEY]):
            default = gp.extract_aperture(name, HOST_RA, HOST_DEC, **kwargs)
            subtracted = gp.extract_aperture(name, HOST_RA, HOST_DEC, bkg_sub=True, **kwargs)
        self.assertEqual(default[0]["a"][0], subtracted[0]["a"][0])
        self.assertEqual(default[2], subtracted[2])  # Kron radius

    def test_local_photometry_plots_and_sed(self):
        name = "offline_sed"
        install_test_images(name, SURVEY)
        lp.multi_band_phot(name, SN_RA, SN_DEC, Z, survey=SURVEY, ap_radii=[1, 2], use_mask=False, save_plots=True,
                           save_results=True, save_input=True, raise_exception=True)
        gp.multi_band_phot(name, HOST_RA, HOST_DEC, survey=SURVEY, use_mask=False, save_plots=True,
                           save_results=True, save_input=True, raise_exception=True)
        obj_dir = Path(workdir, name)
        self.assertTrue((obj_dir / SURVEY / "local_photometry.csv").exists())
        self.assertTrue((obj_dir / SURVEY / "global_photometry.csv").exists())
        for phot_type, radius in [("global", None), ("local", 1)]:
            outfile = obj_dir / f"sed_{phot_type}.png"
            plot_sed(name, phot_type=phot_type, z=Z, radius=radius, outfile=outfile)
            self.assertTrue(outfile.exists())
        plot_sed(name, "global", include=[SURVEY], save_plot=False, plot_flux=True)
        plot_sed(name, "global", exclude=["2MASS"], save_plot=False)
        with self.assertRaises(ValueError):
            plot_sed(name, "global", include=[SURVEY], exclude=["2MASS"])
        self.assertIsNone(plot_sed(name, "global", include=["2MASS"]))  # no photometry


class TestGaiaCache(unittest.TestCase):
    def setUp(self):
        od._gaia_cache.clear()
        self.cache_dir = Path(workdir, "offline_gaia_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / od.GAIA_CACHE_FILE).unlink(missing_ok=True)

    def test_cache_and_star_selection(self):
        with mock.patch.object(od, "_query_gaia_esa", side_effect=_gaia_table) as query:
            stars = od.find_gaia_objects(10.0, -5.0, rad=0.1, cache_dir=self.cache_dir)
            self.assertEqual(len(stars), 2)  # the source without significant proper motion or parallax is excluded
            od.find_gaia_objects(10.0, -5.0, rad=0.05, cache_dir=self.cache_dir)  # smaller radius: cached
            self.assertEqual(query.call_count, 1)
            self.assertAlmostEqual(query.call_args.args[2], od.GAIA_MIN_QUERY_RADIUS)
            od.find_gaia_objects(10.0, -5.0, rad=0.3, cache_dir=self.cache_dir)  # larger radius: queried again
            self.assertEqual(query.call_count, 2)
        # new session: read from the file
        od._gaia_cache.clear()
        with mock.patch.object(od, "_query_gaia_esa", side_effect=AssertionError("should not query")):
            self.assertEqual(len(od.find_gaia_objects(10.0, -5.0, rad=0.2, cache_dir=self.cache_dir)), 2)
        # the sources outside the requested radius are excluded (the stars are ~0.014 deg away)
        with mock.patch.object(od, "_query_gaia_esa", side_effect=_gaia_table):
            self.assertEqual(len(od.find_gaia_objects(10.0, -5.0, rad=0.005)), 0)

    def test_mirror_fallback(self):
        with mock.patch.object(od, "_query_gaia_esa", side_effect=ConnectionError("maintenance")), mock.patch.object(
            od, "_query_gaia_mirror", side_effect=lambda ra, dec, rad, url, table: _gaia_table(ra, dec, rad)
        ) as mirror:
            stars = od.find_gaia_objects(20.0, 5.0)
        self.assertEqual(len(stars), 2)
        self.assertEqual(mirror.call_count, 1)
        self.assertEqual(mirror.call_args.args[3], od.GAIA_MIRRORS[0][0])

    def test_all_sources_fail(self):
        with mock.patch.object(od, "_query_gaia_esa", side_effect=ConnectionError("down")), mock.patch.object(
            od, "_query_gaia_mirror", side_effect=ConnectionError("down")
        ) as mirror:
            with self.assertWarnsRegex(FlagWarning, "Gaia DR3 query failed"):
                stars = od.find_gaia_objects(30.0, 5.0, cache_dir=self.cache_dir)
        self.assertEqual(len(stars), 0)
        self.assertEqual(mirror.call_count, len(od.GAIA_MIRRORS))
        self.assertNotIn((30.0, 5.0), od._gaia_cache)  # failures are not cached
        self.assertFalse((self.cache_dir / od.GAIA_CACHE_FILE).exists())

    def test_mirror_query(self):
        # partner data centre: parallax over error computed; an empty result is an error (not cached)
        result = Table({"ra": [1.0], "dec": [2.0], "parallax": [2.0], "parallax_error": [0.5], "pmra": [1.0],
                        "pmra_error": [0.1], "pmdec": [1.0], "pmdec_error": [0.1]})
        with mock.patch("pyvo.dal.TAPService") as service:
            service.return_value.run_sync.return_value.to_table.return_value = result
            table = od._query_gaia_mirror(1.0, 2.0, 0.15, "https://example.org/tap", "gaiadr3.gaia_source")
            self.assertEqual(table.colnames, od.GAIA_COLUMNS)
            self.assertAlmostEqual(table["parallax_over_error"][0], 4.0)
            service.return_value.run_sync.return_value.to_table.return_value = result[:0]
            with self.assertRaisesRegex(ValueError, "empty result"):
                od._query_gaia_mirror(1.0, 2.0, 0.15, "https://example.org/tap", "gaiadr3.gaia_source")

    def test_esa_query(self):
        with mock.patch("astroquery.gaia.Gaia") as gaia:
            gaia.launch_job_async.return_value.get_results.return_value = _gaia_table(1.0, 2.0, 0.1)
            table = od._query_gaia_esa(1.0, 2.0, 0.15)
        self.assertEqual(len(table), 3)
        self.assertIn("CIRCLE('ICRS', 1.0, 2.0, 0.15)", gaia.launch_job_async.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
