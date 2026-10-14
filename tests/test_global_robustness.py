import shutil
import unittest
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from hostphot._constants import workdir
from hostphot.photometry import global_photometry as gp

RA, DEC = 150.0, 2.0


def _write_image(name: str, data: np.ndarray) -> None:
    """Writes a synthetic PanSTARRS-like image centred on (RA, DEC)."""
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [RA, DEC]
    wcs.wcs.crpix = [data.shape[1] / 2 + 0.5, data.shape[0] / 2 + 0.5]
    wcs.wcs.cdelt = [-0.25 / 3600, 0.25 / 3600]
    header = wcs.to_header()
    header["EXPTIME"] = 100.0
    header["HIERARCH CELL.GAIN"] = 1.0
    header["HIERARCH CELL.READNOISE"] = 1.0
    out_dir = Path(workdir, name, "PanSTARRS")
    out_dir.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(data.astype(np.float32), header).writeto(
        out_dir / "PanSTARRS_r.fits", overwrite=True
    )


class TestGlobalRobustness(unittest.TestCase):
    def setUp(self):
        self.names = []

    def tearDown(self):
        for name in self.names:
            shutil.rmtree(Path(workdir, name), ignore_errors=True)

    def _photometry(self, name: str):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return gp.photometry(
                name, RA, DEC, "r", "PanSTARRS",
                use_mask=False, correct_extinction=False, save_plots=False,
            )

    def test_nan_pixels_near_galaxy(self):
        # galaxy with a few NaN pixels close to it: no crash (the Kron radius is NaN-safe)
        name = "_test_nan_galaxy"
        self.names.append(name)
        rng = np.random.default_rng(1)
        y, x = np.mgrid[0:301, 0:301]
        data = 2000 * np.exp(-np.hypot(x - 150, y - 150) / 6) + rng.normal(0, 1, (301, 301))
        data[150:153, 175:178] = np.nan
        _write_image(name, data)
        mag = self._photometry(name)[0]
        self.assertTrue(np.isfinite(mag))

    def test_galaxy_not_detected(self):
        # pure noise: clear error instead of a TypeError
        name = "_test_no_galaxy"
        self.names.append(name)
        data = np.random.default_rng(2).normal(0, 1, (301, 301))
        _write_image(name, data)
        with self.assertRaisesRegex(ValueError, "not detected"):
            self._photometry(name)


if __name__ == "__main__":
    unittest.main()
