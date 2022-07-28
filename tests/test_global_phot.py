import unittest
from hostphot.cutouts import download_images
import hostphot.global_photometry as gp


class TestHostPhot(unittest.TestCase):
    def test_global_phot(self):
        sn_name = "2002fk"
        host_ra = 50.527333
        host_dec = -15.400056
        survey = "PS1"

        download_images(sn_name, host_ra, host_dec, survey=survey)
        gp.multi_band_phot(
            sn_name,
            host_ra,
            host_dec,
            survey=survey,
            use_mask=False,
            common_aperture=False,
            optimize_kronrad=True,
            save_plots=True,
        )


if __name__ == "__main__":
    unittest.main()
