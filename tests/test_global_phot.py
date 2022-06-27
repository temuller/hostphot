import unittest
from hostphot.cutouts import download_images
import hostphot.global_photometry as gp


class TestHostPhot(unittest.TestCase):
    def test_global_phot(self):
        sn_name = "SN2004eo"
        host_ra = 308.2092
        host_dec = 9.92755
        z = 0.0157
        survey = "PS1"

        download_images(sn_name, host_ra, host_dec, survey=survey)
        gp.multi_band_phot(
            sn_name,
            host_ra,
            host_dec,
            survey=survey,
            use_mask=False,
            common_aperture=False,
            optimze_kronrad=True,
            save_plots=True,
        )


if __name__ == "__main__":
    unittest.main()
