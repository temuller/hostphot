import unittest
from pathlib import Path
from hostphot.cutouts import download_images, set_HST_image, set_JWST_image


class TestHostPhot(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestHostPhot, self).__init__(*args, **kwargs)
        # object used for most surveys
        self.sn_name = "2002fk"
        self.ra = 50.527333
        self.dec = -15.400056

    def test_cutouts_PanSTARRS(self):
        download_images(
            self.sn_name, self.ra, self.dec, overwrite=True, survey="PanSTARRS"
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
            #try:
            download_images(
                name,
                ra,
                dec,
                overwrite=True,
                survey="VISTA",
                version=version,
            )

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


if __name__ == "__main__":
    unittest.main()
