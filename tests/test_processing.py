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


if __name__ == "__main__":
    unittest.main()
