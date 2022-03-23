import unittest
from hostphot.cutouts import download_multiband_images
from hostphot.local_photometry import multi_local_photometry

class TestHostPhot(unittest.TestCase):

    def test_local_phot(self):
        sn_name = 'SN2006hx'
        ra = 18.488792
        dec = 0.371667
        z = 0.045500
        survey = 'PS1'

        download_multiband_images(sn_name, ra, dec, survey=survey)
        multi_local_photometry([sn_name], [ra], [dec], [z],
                                ap_radius=4, survey=survey)

if __name__ == '__main__':
    unittest.main()
