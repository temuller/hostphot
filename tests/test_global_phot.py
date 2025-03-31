import unittest
import numpy as np
from hostphot.cutouts import download_images
from hostphot.photometry import global_photometry as gp


class TestHostPhot(unittest.TestCase):
    def test_global_phot(self):
        sn_name = "2002fk"
        host_ra = 50.527333
        host_dec = -15.400056
        survey = "PanSTARRS"

        download_images(sn_name, host_ra, host_dec, survey=survey, overwrite=False)
        phot = gp.multi_band_phot(
            sn_name,
            host_ra,
            host_dec,
            survey=survey,
            use_mask=False,
            common_aperture=False, 
            optimize_kronrad=True,
            save_plots=False,
            raise_exception=True,
        )
        mags = [phot[filt][0] for filt in "griz"]
        # griz reference magnitudes compared to Blast
        ref_mags = [11.684, 11.353, 11.215, 11.073]

        err_msg = (
            "Large difference between calculated and reference magnitudes"
        )
        np.testing.assert_allclose(mags, ref_mags, rtol=0.05, err_msg=err_msg)


if __name__ == "__main__":
    unittest.main()
