import unittest
from hostphot.cutouts import download_images
from hostphot.interactive_aperture import InteractiveAperture


class TestHostPhot(unittest.TestCase):
    def test_interactivity(self):
        sn_name = "SN2004eo"
        ra = 308.22579
        dec = 9.92853
        survey = "PS1"

        download_images(sn_name, ra, dec, survey=survey)
        obj = InteractiveAperture(sn_name, masked=False)


if __name__ == "__main__":
    unittest.main()
