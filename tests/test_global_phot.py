import unittest
from hostphot.cutouts import download_multiband_images
from hostphot.global_photometry import multi_global_photometry

class TestHostPhot(unittest.TestCase):

    def test_global_phot(self):
        sn_name = 'SN2006hx'
        ra = 18.488792
        dec = 0.371667
        z = 0.045500
        survey = 'PS1'

        download_multiband_images(sn_name, ra, dec, survey=survey)
        multi_global_photometry([sn_name], [ra], [dec], survey=survey)

if __name__ == '__main__':
    unittest.main()
