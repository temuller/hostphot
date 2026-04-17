import unittest
import numpy as np
import os
from astropy.io import fits

from hostphot.photometry import global_photometry as gp


class TestHerschel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create a mock Herschel image for testing."""
        cls.test_name = "Herschel_test"
        cls.host_ra = 148.969
        cls.host_dec = 69.683

        # Create test directory
        cls.test_dir = f"images/{cls.test_name}/Herschel"
        os.makedirs(cls.test_dir, exist_ok=True)

        # Create mock M82-like image (1300 Jy total)
        size = 300
        center = 150
        total_flux = 1300  # Jy (matching catalog)
        sigma = 40  # pixels

        data = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                r = np.sqrt((i - center) ** 2 + (j - center) ** 2)
                data[i, j] = (
                    total_flux
                    * np.exp(-r / sigma)
                    / (2 * np.pi * sigma**2)
                )

        # Create FITS header
        header = fits.Header()
        header["NAXIS"] = 2
        header["NAXIS1"] = size
        header["NAXIS2"] = size
        header["CTYPE1"] = "RA---TAN"
        header["CTYPE2"] = "DEC--TAN"
        header["CRVAL1"] = cls.host_ra
        header["CRVAL2"] = cls.host_dec
        header["CRPIX1"] = center
        header["CRPIX2"] = center
        header["CDELT1"] = -3.5 / 3600  # PACS100 pixel scale
        header["CDELT2"] = 3.5 / 3600
        header["BUNIT"] = "Jy/pixel"
        header["FILTER"] = "PACS100"
        header["SURVEY"] = "Herschel"

        # Save FITS
        fits.writeto(
            f"{cls.test_dir}/Herschel_PACS100.fits",
            data,
            header,
            overwrite=True,
        )

    def test_herschel_photometry(self):
        """Test Herschel global photometry returns reasonable flux."""
        result = gp.photometry(
            self.test_name,
            self.host_ra,
            self.host_dec,
            filt="PACS100",
            survey="Herschel",
            ra=self.host_ra,
            dec=self.host_dec,
            use_mask=False,
            optimize_kronrad=True,
            save_plots=False,
        )

        mag, mag_err, flux, flux_err, zp, details = result

        # Check flux is reasonable (within ~30% of catalog for M82)
        # Catalog: 1200-1400 Jy, our test image has ~1300 Jy
        self.assertGreater(flux, 900, "Flux too low")
        self.assertLess(flux, 1600, "Flux too high")

        # Check flux error is reasonable (should be ~5%)
        self.assertLess(flux_err / flux, 0.1, "Flux error too large")

        # Check SNR is positive
        self.assertGreater(flux / flux_err, 0, "SNR should be positive")

        # Check details are present
        self.assertIn("flux_unit", details)
        self.assertEqual(details["flux_unit"], "Jy")

    def test_herschel_multi_band_phot(self):
        """Test Herschel multi-band photometry."""
        result = gp.multi_band_phot(
            self.test_name,
            self.host_ra,
            self.host_dec,
            survey="Herschel",
            filters="PACS100",
            use_mask=False,
            optimize_kronrad=True,
            save_plots=False,
            save_results=False,
        )

        # Check result contains expected columns
        self.assertIn("PACS100", result.columns)
        self.assertIn("PACS100_flux", result.columns)
        self.assertIn("PACS100_flux_err", result.columns)


if __name__ == "__main__":
    unittest.main()