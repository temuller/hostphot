import unittest
from hostphot.cutouts import download_images


class TestHostPhot(unittest.TestCase):
    def test_cutouts(self):
        sn_name = "2002fk"
        ra = 50.527333
        dec = -15.400056
        size = 3  # arcmin

        for survey in ["PS1", "DES", "SDSS", "GALEX", "WISE", "2MASS"]:
            download_images(
                sn_name, ra, dec, overwrite=True, size=size, survey=survey
            )


if __name__ == "__main__":
    unittest.main()
