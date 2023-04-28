import unittest
import numpy as np
from hostphot.cutouts import download_images
import hostphot.global_photometry as gp


class TestHostPhot(unittest.TestCase):
    def test_global_phot(self):
        sn_name = "2002fk"
        host_ra = 50.527333
        host_dec = -15.400056
        survey = "PS1"

        download_images(sn_name, host_ra, host_dec, survey=survey)
        phot = gp.multi_band_phot(
            sn_name,
            host_ra,
            host_dec,
            survey=survey,
            use_mask=False,
            common_aperture=False,
            optimize_kronrad=True,
            save_plots=True,
            raise_exception=True
        )
        mags = [phot[filt] for filt in "griz"]
        # griz SIMBAD reference magnitudes of the host galaxy of SN 2002fk
        ref_mags = [12.155, 11.508, 11.205, 10.979]

        err_msg = (
            "Large difference between calculated and reference magnitudes"
        )
        np.testing.assert_allclose(mags, ref_mags, rtol=0.05, err_msg=err_msg)


if __name__ == "__main__":
    unittest.main()
