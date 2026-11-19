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


def test_host_model_keeps_central_bar():
    """The host's own bar (underestimated by the single Sersic profile) is not masked, while an extended galaxy
    on top of the host is."""
    import numpy as np
    from hostphot.processing.host_model import host_model_mask

    rng = np.random.default_rng(1)
    n = 601
    y, x = np.indices((n, n))
    x0 = y0 = 300.0
    disc = 40 * np.exp(-np.hypot(x - x0, y - y0) / 40)
    bar = 200 * np.exp(-0.5 * (((x - x0) / 60) ** 2 + ((y - y0) / 4) ** 2))
    neighbour = 300 * np.exp(-0.5 * ((x - 380) ** 2 + (y - 300) ** 2) / 6**2)
    data = disc + bar + neighbour + rng.normal(0, 1, (n, n))
    no_objects = np.zeros(0, dtype=[("x", float), ("y", float), ("a", float), ("b", float), ("theta", float),
                                    ("peak", float), ("flag", int)])
    mask, filled, info = host_model_mask(data, 1.0, x0, y0, 4.0, no_objects, (np.array([]), np.array([])), 10)
    assert not mask[300, 300] and not mask[300, 330]  # centre and bar
    assert mask[300, 380]  # neighbour



def test_orient_axis_labels():
    """Images rotated by 90 degrees (RA along the y axis) get their axis labels swapped; others are unchanged."""
    import aplpy
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from astropy.wcs import WCS
    from hostphot.utils import orient_axis_labels

    for rotated in [False, True]:
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crval = [334.57, 40.56]
        wcs.wcs.crpix = [50.5, 50.5]
        scale = 0.4 / 3600
        wcs.wcs.cd = [[0, -scale], [scale, 0]] if rotated else [[-scale, 0], [0, scale]]
        hdu = fits.PrimaryHDU(np.zeros((100, 100)), header=wcs.to_header())
        figure = plt.figure()
        fig = aplpy.FITSFigure(hdu, figure=figure)
        assert orient_axis_labels(fig) is rotated
        ra, dec = fig.ax.coords[0], fig.ax.coords[1]
        x_coord, y_coord = (dec, ra) if rotated else (ra, dec)
        assert x_coord.get_axislabel_position() == ["b"] and y_coord.get_axislabel_position() == ["l"]
        assert "RA" in ra.get_axislabel() and "Dec" in dec.get_axislabel()
        fig.tick_labels.hide_y()  # refers to the coordinate along the y axis
        assert not y_coord.ticklabels.get_visible() and x_coord.ticklabels.get_visible()
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
