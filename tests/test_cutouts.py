import unittest
import warnings
from hostphot.cutouts import download_images


class TestHostPhot(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestHostPhot, self).__init__(*args, **kwargs)
        # object used for most surveys
        self.sn_name = "2002fk"
        self.ra = 50.527333
        self.dec = -15.400056

    def test_cutouts_PS1(self):
        download_images(
            self.sn_name, self.ra, self.dec, overwrite=True, survey="PS1"
        )

    def test_cutouts_DES(self):
        download_images(
            self.sn_name, self.ra, self.dec, overwrite=True, survey="DES"
        )

    def test_cutouts_SDSS(self):
        download_images(
            self.sn_name, self.ra, self.dec, overwrite=True, survey="SDSS"
        )

    def test_cutouts_GALEX(self):
        download_images(
            self.sn_name, self.ra, self.dec, overwrite=True, survey="GALEX"
        )

    def test_cutouts_WISE(self):
        download_images(
            self.sn_name, self.ra, self.dec, overwrite=True, survey="WISE"
        )

    def test_cutouts_2MASS(self):
        download_images(
            self.sn_name, self.ra, self.dec, overwrite=True, survey="2MASS"
        )

    def test_cutouts_unWISE(self):
        for version in ["neo1", "neo2", "allwise"]:
            download_images(
                self.sn_name,
                self.ra,
                self.dec,
                overwrite=True,
                survey="unWISE",
                version=version,
            )

    def test_cutouts_LegacySurvey(self):
        download_images(
            self.sn_name,
            self.ra,
            self.dec,
            overwrite=True,
            survey="LegacySurvey",
        )

    def test_cutouts_Spitzer(self):
        name = "Spitzer_test"
        ra, dec = 52.158591, -27.891113
        download_images(name, ra, dec, overwrite=True, survey="Spitzer")

    def test_cutouts_VISTA(self):
        name = "VISTA_test"
        # use different coordinates for each survey as they don't overlap
        surveys = {
            "VHS": [120, -60],
            "VIDEO": [36.1, -5],
            "VIKING": [220.5, 0.0],
        }

        
        for version, coords in surveys.items():
            ra, dec = coords
            try:
                download_images(
                    name,
                    ra,
                    dec,
                    overwrite=True,
                    survey="VISTA",
                    version=version,
                )
            except Exception as exc:
                warnings.warn(
                    "The VISTA SCIENCE ARCHIVE might be having issues..."
                )
                print(f"Skipping the test for {version}...")
                print(exc)

    def test_cutouts_SkyMapper(self):
        download_images(
            self.sn_name,
            self.ra,
            self.dec,
            overwrite=True,
            survey="SkyMapper",
        )

    def test_cutouts_SPLUS(self):
        name = "SPLUS_test"
        ra, dec = 0.6564206, -0.3740297
        download_images(name, ra, dec, overwrite=True, survey="SPLUS")

    def test_cutouts_UKIDSS(self):
        name = "UKIDSS_test"
        ra, dec = 359.5918320, +0.1964120
        download_images(name, ra, dec, overwrite=True, survey="UKIDSS")


if __name__ == "__main__":
    unittest.main()
