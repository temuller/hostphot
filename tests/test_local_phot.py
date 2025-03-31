import unittest
import numpy as np
from hostphot.cutouts import download_images
from hostphot.photometry import local_photometry as lp


class TestHostPhot(unittest.TestCase):
    def test_local_phot(self):
        sn_name = "2002fk"
        ra = 50.52379
        dec = -15.40089
        z = 0.007482
        survey = "PanSTARRS"

        ap_radii = 4  # in units of kpc
        download_images(sn_name, ra, dec, survey=survey, overwrite=False)
        phot = lp.multi_band_phot(
            sn_name,
            ra,
            dec,
            z,
            survey=survey,
            filters="grizy",
            ap_radii=ap_radii,
            use_mask=False,
            save_plots=False,
            raise_exception=True,
        )
        mags = [phot[filt][0] for filt in ["g_4", "r_4", "i_4", "z_4"]]
        # pre-calculated magnitudes
        ref_mags = [12.26, 11.89, 11.72, 11.57]

        err_msg = (
            "Large difference between calculated and reference magnitudes"
        )
        np.testing.assert_allclose(mags, ref_mags, rtol=0.03, err_msg=err_msg)


if __name__ == "__main__":
    unittest.main()
