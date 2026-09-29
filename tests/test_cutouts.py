import pytest
import warnings
import unittest
import requests
from pathlib import Path
from hostphot.cutouts import download_images, set_HST_image, set_JWST_image
from pyvo.dal import DALServiceError

class TestHostPhot(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestHostPhot, self).__init__(*args, **kwargs)
        # object used for most surveys
        self.sn_name = "2002fk"
        self.ra = 50.527333
        self.dec = -15.400056

    def test_cutouts_PanSTARRS(self):
        try:
            download_images(
                self.sn_name, self.ra, self.dec, overwrite=True, survey="PanSTARRS"
            )
        except requests.exceptions.ConnectionError as e:
            warnings.warn(f"Connection error for PanSTARRS: {e}", RuntimeWarning)
            pytest.skip(f"Connection error for PanSTARRS: {e}")

    def test_cutouts_DES(self):
        try:
            download_images(
                self.sn_name, self.ra, self.dec, overwrite=True, survey="DES"
            )
        except requests.exceptions.ConnectionError as e:
            warnings.warn(f"Connection error for DES: {e}", RuntimeWarning)
            pytest.skip(f"Connection error for DES: {e}")

    def test_cutouts_SDSS(self):
        try:
            download_images(
                self.sn_name, self.ra, self.dec, overwrite=True, survey="SDSS"
            )
        except requests.exceptions.ConnectionError as e:
            warnings.warn(f"Connection error for SDSS: {e}", RuntimeWarning)
            pytest.skip(f"Connection error for SDSS: {e}")

    def test_cutouts_GALEX(self):
        try:
            download_images(
                self.sn_name, self.ra, self.dec, overwrite=True, survey="GALEX"
            )
        except requests.exceptions.ConnectionError as e:
            warnings.warn(f"Connection error for GALEX: {e}", RuntimeWarning)
            pytest.skip(f"Connection error for GALEX: {e}")

    def test_cutouts_WISE(self):
        try:
            download_images(
                self.sn_name, self.ra, self.dec, overwrite=True, survey="WISE"
            )
        except requests.exceptions.ConnectionError as e:
            warnings.warn(f"Connection error for WISE: {e}", RuntimeWarning)
            pytest.skip(f"Connection error for WISE: {e}")

    def test_cutouts_2MASS(self):
        try:
            download_images(
                self.sn_name, self.ra, self.dec, overwrite=True, survey="2MASS"
            )
        except requests.exceptions.ConnectionError as e:
            warnings.warn(f"Connection error for 2MASS: {e}", RuntimeWarning)
            pytest.skip(f"Connection error for 2MASS: {e}")

    def test_cutouts_unWISE(self):
        try:
            for version in ["neo1", "neo2", "allwise"]:
                download_images(
                    self.sn_name,
                    self.ra,
                    self.dec,
                    overwrite=True,
                    survey="unWISE",
                    version=version,
                )
        except requests.exceptions.ConnectionError as e:
            warnings.warn(f"Connection error for unWISE: {e}", RuntimeWarning)
            pytest.skip(f"Connection error for unWISE: {e}")

    def test_cutouts_LegacySurvey(self):
        try:
            download_images(
                self.sn_name,
                self.ra,
                self.dec,
                overwrite=True,
                survey="LegacySurvey",
            )
        except requests.exceptions.ConnectionError as e:
            warnings.warn(f"Connection error for LegacySurvey: {e}", RuntimeWarning)
            pytest.skip(f"Connection error for LegacySurvey: {e}")

    def test_cutouts_Spitzer(self):
        try:
            name = "Spitzer_test"
            ra, dec = 52.158591, -27.891113
            download_images(name, ra, dec, overwrite=True, survey="Spitzer")
        except requests.exceptions.ConnectionError as e:
            warnings.warn(f"Connection error for Spitzer: {e}", RuntimeWarning)
            pytest.skip(f"Connection error for Spitzer: {e}")

    def test_cutouts_VISTA(self):
        try:
            name = "VISTA_test"
            # use different coordinates for each survey as they don't overlap
            surveys = {
                "VHS": [120, -60],
                "VIDEO": [36.1, -5],
                "VIKING": [220.5, 0.0],
            }
            for version, coords in surveys.items():
                ra, dec = coords
                #try:
                download_images(
                    name,
                    ra,
                    dec,
                    overwrite=True,
                    survey="VISTA",
                    version=version,
                )
        except requests.exceptions.ConnectionError as e:
            warnings.warn(f"Connection error for VISTA: {e}", RuntimeWarning)
            pytest.skip(f"Connection error for VISTA: {e}")

    def test_cutouts_SkyMapper(self):
        try:
            download_images(
                self.sn_name,
                self.ra,
                self.dec,
                overwrite=True,
                survey="SkyMapper",
            )
        except (DALServiceError, requests.exceptions.ConnectionError) as e:
            warnings.warn(f"SkyMapper service failed: {e}", RuntimeWarning)
            pytest.skip(f"SkyMapper service unavailable: {e}")

    def test_cutouts_SPLUS(self):
        try:
            name = "SPLUS_test"
            ra, dec = 0.6564206, -0.3740297
            download_images(name, ra, dec, overwrite=True, survey="SPLUS")
        except requests.exceptions.ConnectionError as e:
            warnings.warn(f"Connection error for SPLUS: {e}", RuntimeWarning)
            pytest.skip(f"Connection error for SPLUS: {e}")

    def test_cutouts_UKIDSS(self):
        try:
            name = "UKIDSS_test"
            ra, dec = 359.5918320, +0.1964120
            download_images(name, ra, dec, overwrite=True, survey="UKIDSS")
        except requests.exceptions.ConnectionError as e:
            warnings.warn(f"Connection error for UKIDSS: {e}", RuntimeWarning)
            pytest.skip(f"Connection error for UKIDSS: {e}")
        
    def test_cutouts_HST(self):
        name = "HST_test"
        filt = 'WFC3_UVIS_F275W'
        file = "tests/hst_16741_2v_wfc3_uvis_f275w_iepo2v_drc.fits"
        if Path(file).exists():
            # this test only runs locally as the file is too large
            set_HST_image(file, filt, name)
        
    def test_cutouts_JWST(self):
        name = "JWST_test"
        filt = 'NIRCam_F090W'
        file = "tests/jw01685-c1008_t004_nircam_clear-f090w_i2d.fits"
        if Path(file).exists():
            # this test only runs locally as the file is too large
            set_JWST_image(file, filt, name)

    def test_cutouts_HSC(self):
        try:
            name = "HSC_test"
            # GAMA09H field — well within HSC PDR3 Wide footprint
            ra, dec = 150.0, 2.2
            download_images(
                name, ra, dec, overwrite=True, survey="HSC", version="pdr3_wide"
            )
        except ValueError as e:
            # HSC_SSP_USERNAME / HSC_SSP_PASSWORD not set
            warnings.warn(f"HSC credentials not set: {e}", RuntimeWarning)
            pytest.skip(f"HSC credentials not configured: {e}")
        except requests.exceptions.ConnectionError as e:
            warnings.warn(f"Connection error for HSC: {e}", RuntimeWarning)
            pytest.skip(f"Connection error for HSC: {e}")

    def test_cutouts_Herschel(self):
        """Test that Herschel download shows beta warning."""
        import warnings
        from hostphot.cutouts import download_images
        from hostphot.surveys_utils import check_survey_validity

        name = "Herschel_test"
        ra = 148.969
        dec = 69.683

        # Test that check_survey_validity shows the warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_survey_validity("Herschel")

            # Check that the beta warning was shown
            herschel_warning = any(
                "Herschel" in str(warning.message) and "beta" in str(warning.message).lower()
                for warning in w
            )
            self.assertTrue(herschel_warning, "Herschel beta warning should be shown")

    def test_download_images_normalizes_negative_ra(self):
        """Regression test for issue #17.

        A negative right ascension must be wrapped into the valid [0, 360)
        degree range before it is handed to a survey service. IRSA's WISE
        SIA endpoint rejects negative RA, which previously surfaced as a
        confusing downstream ``AttributeError``. This test mocks the survey
        download function so it runs offline and only checks the RA handling.
        """
        import tempfile
        from unittest import mock
        import hostphot.cutouts.downloads as downloads

        negative_ra = -15.072453
        dec = -0.013717
        with tempfile.TemporaryDirectory() as tmp_workdir, mock.patch(
            "hostphot.cutouts.wise.get_WISE_images", return_value=None
        ) as mocked_get, mock.patch.object(downloads, "workdir", tmp_workdir):
            download_images(
                "issue17_negative_ra",
                negative_ra,
                dec,
                survey="WISE",
                filters=["W1"],
                save_input=False,
            )

        self.assertTrue(
            mocked_get.called, "the survey download function was never reached"
        )
        passed_ra = mocked_get.call_args[0][0]
        self.assertGreaterEqual(passed_ra, 0.0)
        self.assertLess(passed_ra, 360.0)
        self.assertAlmostEqual(passed_ra, negative_ra % 360)


if __name__ == "__main__":
    unittest.main()
