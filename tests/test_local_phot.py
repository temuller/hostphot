import unittest
from hostphot.cutouts import download_images
import hostphot.local_photometry as lp


class TestHostPhot(unittest.TestCase):
    def test_local_phot(self):
        sn_name = "SN2011fe"
        ra = 210.77420833333332
        dec = 54.273719444444446
        z = 0.0015972039982078844
        survey = "PS1"

        ap_radii = [3, 4]  # in units of kpc
        download_images(sn_name, ra, dec, survey=survey)
        lp.multi_band_phot(
            sn_name,
            ra,
            dec,
            z,
            survey="PS1",
            filters="g",
            ap_radii=ap_radii,
            use_mask=False,
            save_plots=True,
        )


if __name__ == "__main__":
    unittest.main()
