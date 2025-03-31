import unittest
from hostphot.cutouts import download_images
from hostphot.processing import coadd_images
from hostphot.processing import create_mask


class TestHostPhot(unittest.TestCase):
    def test_processing(self):
        coadd_filters = "riz"
        survey = "PanSTARRS"
        sn_name = "2002fk"
        host_ra = 50.527333
        host_dec = -15.400056

        download_images(sn_name, host_ra, host_dec, survey=survey, overwrite=False)

        # coadd
        coadd_images(sn_name, coadd_filters, survey)

        # masking
        create_mask(
            sn_name,
            host_ra,
            host_dec,
            filt=coadd_filters,
            survey=survey,
            save_mask_params=True,
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
                ref_filt=coadd_filters,
                ref_survey=survey,
            )


if __name__ == "__main__":
    unittest.main()
