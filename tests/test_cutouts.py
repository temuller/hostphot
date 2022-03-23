import unittest
from hostphot.cutouts import download_multiband_images

class TestHostPhot(unittest.TestCase):

    def test_cutouts(self):
        sn_name = 'SN2004ey'
        ra = 327.282542
        dec = 0.444222
        size = 100

        for survey in ['PS1', 'DES', 'SDSS']:
            download_multiband_images(sn_name, ra, dec,
                                    size=size, survey=survey)

if __name__ == '__main__':
    unittest.main()
