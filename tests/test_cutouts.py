import unittest
from hostphot.cutouts import download_images


class TestHostPhot(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestHostPhot, self).__init__(*args, **kwargs)
        self.sn_name = "2002fk"
        self.ra = 50.527333
        self.dec = -15.400056

    def test_cutouts_PS1(self):
        download_images(
            self.sn_name, self.ra, self.dec,
            overwrite=True, survey="PS1"
        )

    def test_cutouts_DES(self):
        download_images(
            self.sn_name, self.ra, self.dec,
            overwrite=True, survey="DES"
        )

    def test_cutouts_SDSS(self):
        download_images(
            self.sn_name, self.ra, self.dec,
            overwrite=True, survey="SDSS"
        )

    def test_cutouts_GALEX(self):
        download_images(
            self.sn_name, self.ra, self.dec,
            overwrite=True, survey="GALEX"
        )

    def test_cutouts_WISE(self):
        download_images(
            self.sn_name, self.ra, self.dec,
            overwrite=True, survey="WISE"
        )

    def test_cutouts_2MASS(self):
        download_images(
            self.sn_name, self.ra, self.dec,
            overwrite=True, survey="2MASS"
        )

    def test_cutouts_unWISE(self):
        for survey in ["unWISEneo1", "unWISEneo2", "unWISEallwise"]:
            download_images(
                self.sn_name, self.ra, self.dec,
                overwrite=True, survey=survey
            )


if __name__ == "__main__":
    unittest.main()
