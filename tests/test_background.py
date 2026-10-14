import unittest

import numpy as np
from astropy.wcs import WCS
from photutils.aperture import CircularAperture, aperture_photometry

from hostphot.processing.background import masked_background


def _wcs(size: int, pixel_scale: float = 0.5) -> WCS:
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [150.0, 2.0]
    wcs.wcs.crpix = [size / 2 + 0.5, size / 2 + 0.5]
    wcs.wcs.cdelt = [-pixel_scale / 3600, pixel_scale / 3600]
    return wcs


class TestBackground(unittest.TestCase):
    def test_galaxy_light_is_kept(self):
        # extended galaxy on a sky with a gradient: the galaxy light must not be subtracted
        size = 960
        y, x = np.mgrid[0:size, 0:size]
        centre = size / 2
        galaxy = 50 * np.exp(-np.hypot(x - centre, y - centre) / 20)
        sky = 100 + 0.01 * (x - centre)
        data = galaxy + sky + np.random.default_rng(0).normal(0, 1, (size, size))
        back = masked_background(data, _wcs(size), 150.0, 2.0)
        aperture = CircularAperture((centre, centre), 120)
        true_flux = aperture_photometry(galaxy, aperture)["aperture_sum"][0]
        flux = aperture_photometry(data - back, aperture)["aperture_sum"][0]
        np.testing.assert_allclose(flux, true_flux, rtol=0.02)

    def test_blank_sky(self):
        # pure noise on a constant sky: unbiased background, with fluctuations close to the expected
        # noise of each mesh (~1/60 for a 30" mesh of 60 pixels)
        size = 400
        data = 10 + np.random.default_rng(1).normal(0, 1, (size, size))
        back = masked_background(data, _wcs(size))
        self.assertLess(abs(np.median(back) - 10), 0.01)
        self.assertLess(np.std(back), 0.03)

    def test_galaxy_filling_the_image(self):
        # a warning recommends a larger image
        size = 360
        y, x = np.mgrid[0:size, 0:size]
        data = 50 * np.exp(-np.hypot(x - size / 2, y - size / 2) / 40) + 100
        data += np.random.default_rng(2).normal(0, 1, (size, size))
        with self.assertWarnsRegex(UserWarning, "larger image"):
            masked_background(data, _wcs(size), 150.0, 2.0, label="test")


    def test_large_image_is_binned(self):
        # images above MAX_PIXELS are binned: the (smooth) background must still be recovered
        size = 2100
        y, x = np.mgrid[0:size, 0:size]
        sky = 100 + 0.002 * (x - size / 2)
        data = sky + np.random.default_rng(3).normal(0, 1, (size, size))
        data[:50, :50] = np.nan
        back = masked_background(data, _wcs(size, pixel_scale=0.1))
        self.assertEqual(back.shape, data.shape)
        self.assertLess(np.nanmedian(np.abs(back - sky)), 0.05)


if __name__ == "__main__":
    unittest.main()
