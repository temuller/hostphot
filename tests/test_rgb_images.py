import unittest
from hostphot.cutouts import download_images
from hostphot.rgb_images import (
    create_RGB_image,
    get_PS1_RGB_image,
    get_SDSS_RGB_image,
)


class TestHostPhot(unittest.TestCase):
    def test_create_RGB_image(self):
        sn_name = "2002fk"
        ra = 50.527333
        dec = -15.400056
        download_images(sn_name, ra, dec, survey="PS1")

        images_dir = "images/2002fk"
        for scaling in ["linear", "sqrt", "log", "asinh"]:
            create_RGB_image(images_dir=images_dir, scaling=scaling)

    def test_get_PS1_RGB_image(self):
        ra, dec = 50.527333, -15.400056
        get_PS1_RGB_image("PS1.jpg", ra=ra, dec=dec)

    #def test_get_SDSS_RGB_image(self):
    #    ra, dec = 50.527333, -15.400056
    #    get_SDSS_RGB_image("SDSS.jpg", ra=ra, dec=dec)


if __name__ == "__main__":
    unittest.main()
