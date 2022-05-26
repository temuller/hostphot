import unittest
from hostphot.cutouts import download_images

class TestHostPhot(unittest.TestCase):

    def test_cutouts(self):
        sn_name = 'SN2004ey'
        ra = 327.28254
        dec = 0.44422
        size = 100

        for survey in ['PS1', 'DES', 'SDSS']:
            download_images(sn_name, ra, dec, overwrite=True,
                        size=size, survey=survey)

if __name__ == '__main__':
    unittest.main()
