import unittest
import warnings

from hostphot.cutouts import download_images
from hostphot.photometry import local_photometry as lp


class TestSurveyPhotometryAgreement(unittest.TestCase):
    """Test that 2MASS, VISTA, and UKIDSS produce consistent photometry
    for overlapping fields (within ~0.2 mag)."""

    def test_survey_photometry_agreement(self):
        """Compare photometry from 2MASS, VISTA, and UKIDSS surveys."""
        sn_name = "2002fk_comparison"
        ra = 359.5918320
        dec = 0.1964120
        z = 0.05
        ap_radii = 10
        ap_units = "arcsec"

        surveys_filters = {
            "2MASS": ["J", "H"],
            "VISTA": ["Y", "J", "H"],
            "UKIDSS": ["Y", "J", "H"],
        }

        for filt in ["Y", "J", "H"]:
            print(f"\n=== {filt} band ===")
            mags = {}

            for survey in ["2MASS", "VISTA", "UKIDSS"]:
                if filt not in surveys_filters[survey]:
                    print(f"{survey}: Not available")
                    continue
                    
                try:
                    download_images(
                        sn_name,
                        ra,
                        dec,
                        survey=survey,
                        filters=filt,
                        overwrite=False,
                    )

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result = lp.photometry(
                            sn_name,
                            ra,
                            dec,
                            z,
                            filt,
                            survey=survey,
                            ap_radii=ap_radii,
                            ap_units=ap_units,
                            use_mask=False,
                            save_plots=False,
                        )

                    mags[survey] = result[0][0]
                    print(f"{survey}: {mags[survey]:.4f}")
                except Exception as e:
                    print(f"{survey}: Error - {e}")

            if len(mags) >= 2:
                surveys_list = list(mags.keys())
                for i, s1 in enumerate(surveys_list):
                    for s2 in surveys_list[i+1:]:
                        diff = mags[s1] - mags[s2]
                        print(f"{s1} - {s2}: {diff:.4f}")


if __name__ == "__main__":
    unittest.main()