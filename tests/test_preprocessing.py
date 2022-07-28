import unittest
from hostphot.cutouts import download_images
from hostphot.coadd import coadd_images
from hostphot.image_masking import create_mask


class TestHostPhot(unittest.TestCase):
    def test_preprocessing(self):
        coadd_filters = "riz"
        survey = "PS1"
        sn_name = "2002fk"
        host_ra = 50.527333
        host_dec = -15.400056

        download_images(sn_name, host_ra, host_dec, survey=survey)

        # coadd
        coadd_images(sn_name, coadd_filters, survey)

        # masking
        coadd_mask_params = create_mask(
            sn_name,
            host_ra,
            host_dec,
            filt=coadd_filters,
            survey=survey,
            extract_params=True,
            crossmatch=True,
        )

        # apply mask on single-filter images
        for filt in "grizy":
            create_mask(
                sn_name,
                host_ra,
                host_dec,
                filt,
                survey=survey,
                common_params=coadd_mask_params,
            )


if __name__ == "__main__":
    unittest.main()
