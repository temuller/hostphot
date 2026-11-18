import unittest
import pytest
from tests.helpers import install_test_images
from hostphot.processing import coadd_images
from hostphot.processing import create_mask


class TestHostPhot(unittest.TestCase):
    sn_name = "2002fk"
    host_ra = 50.527333
    host_dec = -15.400056
    survey = "PanSTARRS"
    coadd_filters = "riz"

    def _run_processing(self, crossmatch: bool) -> None:
        install_test_images(self.sn_name, self.survey)  # offline test images (3 arcmin cutouts)

        # coadd
        coadd_images(self.sn_name, self.coadd_filters, self.survey)

        # masking
        create_mask(
            self.sn_name,
            self.host_ra,
            self.host_dec,
            filt=self.coadd_filters,
            survey=self.survey,
            save_mask_params=True,
            crossmatch=crossmatch,
        )

        # apply mask on single-filter images
        for filt in "grizy":
            create_mask(
                self.sn_name,
                self.host_ra,
                self.host_dec,
                filt,
                survey=self.survey,
                ref_filt=self.coadd_filters,
                ref_survey=self.survey,
            )

    def test_processing(self):
        self._run_processing(crossmatch=False)

    @pytest.mark.network
    def test_processing_crossmatch(self):
        # cross-match of the detected sources with the Gaia and TIC catalogues
        self._run_processing(crossmatch=True)


def test_save_tight_figure(tmp_path):
    """The figures are saved as with bbox_inches="tight" (identical files)."""
    import aplpy
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from astropy.wcs import WCS
    from hostphot.utils import save_tight_figure

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [50.5, -15.4]
    wcs.wcs.crpix = [50.5, 50.5]
    wcs.wcs.cdelt = [-0.25 / 3600, 0.25 / 3600]
    data = np.random.default_rng(1).normal(size=(100, 100))
    hdu = fits.PrimaryHDU(data, header=wcs.to_header())
    figure = plt.figure(figsize=(10, 4))
    for i in range(2):
        fig = aplpy.FITSFigure(hdu, figure=figure, subplot=(1, 2, i + 1))
        fig.show_grayscale(vmin=-3, vmax=3)
        fig.set_title("Title")
        fig.add_label(1.3, 0.5, "label outside the axes", relative=True)  # extends the bounding box
    for ext in ["jpg", "png"]:
        figure.savefig(tmp_path / f"ref.{ext}", bbox_inches="tight")
        save_tight_figure(figure, tmp_path / f"new.{ext}")
        assert (tmp_path / f"ref.{ext}").read_bytes() == (tmp_path / f"new.{ext}").read_bytes()
    plt.close(figure)


if __name__ == "__main__":
    unittest.main()
