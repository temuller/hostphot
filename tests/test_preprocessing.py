import unittest
from hostphot.cutouts import download_images
from hostphot.coadd import coadd_images
from hostphot.image_masking import create_mask

class TestHostPhot(unittest.TestCase):

    def test_preprocessing(self):
        # coadd
        coadd_filters = 'riz'
        survey = 'PS1'
        name = 'SN2004eo'

        download_images(sn_name, ra, dec, survey=survey)
        coadd_images(name, coadd_filters, survey)

        # masking
        host_ra, host_dec = 308.2092, 9.92755  # coods of host galaxy of SN2004eo
        coadd_mask_params = create_mask(name, host_ra, host_dec,
                                         filt=coadd_filters, survey=survey,
                                         extract_params=True)

        for filt in 'grizy':
            create_mask(name, host_ra, host_dec, filt, survey=survey,
                        common_params=coadd_mask_params)


if __name__ == '__main__':
    unittest.main()
