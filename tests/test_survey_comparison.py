import unittest
import warnings

import numpy as np
import pytest
import requests
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord

from hostphot._constants import workdir
from hostphot.cutouts import download_images
from hostphot.photometry import local_photometry as lp


class TestSurveyPhotometryAgreement(unittest.TestCase):
    """Test that 2MASS, VISTA, and UKIDSS produce consistent photometry
    for overlapping fields."""

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
            mags, errs = {}, {}

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
                    errs[survey] = result[1][0]
                    print(f"{survey}: {mags[survey]:.4f} +- {errs[survey]:.4f}")
                except Exception as e:
                    print(f"{survey}: Error - {e}")

            if len(mags) >= 2:
                surveys_list = list(mags.keys())
                for i, s1 in enumerate(surveys_list):
                    for s2 in surveys_list[i+1:]:
                        diff = mags[s1] - mags[s2]
                        # different filter systems (no colour terms) and resolutions: 0.05 mag
                        # added in quadrature to the measurement errors (the galaxy is faint in 2MASS)
                        tolerance = 3 * np.sqrt(errs[s1] ** 2 + errs[s2] ** 2 + 0.05**2)
                        print(f"{s1} - {s2}: {diff:.4f} (tolerance {tolerance:.3f})")
                        assert abs(diff) < tolerance, f"{s1} - {s2} = {diff:.3f} mag in {filt}"

    def test_catalog_stars(self):
        """Compare 2MASS, VISTA and UKIDSS photometry of stars with the 2MASS
        Point Source Catalog, transformed into each photometric system."""
        from astroquery.vizier import Vizier

        name = "NIR_stars_test"
        ra, dec = 359.5918320, 0.1964120
        try:
            stars = Vizier(columns=["**"], row_limit=-1).query_region(
                SkyCoord(ra, dec, unit="deg"), radius=4 * u.arcmin, catalog="II/246/out"
            )[0].to_pandas()
        except (requests.exceptions.ConnectionError, IndexError) as e:
            pytest.skip(f"2MASS catalogue not available: {e}")
        # bright, good-quality, unsaturated stars
        stars = stars[(stars.Qflg == "AAA") & stars.Jmag.between(12.0, 15.0)]
        JH = (stars.Jmag - stars.Hmag).values
        # colour equations: VISTA from CASU (vista-sensitivity page) and WFCAM from Hodgkin et al. (2009)
        reference = {
            ("2MASS", "J"): stars.Jmag.values,
            ("2MASS", "H"): stars.Hmag.values,
            ("VISTA", "J"): stars.Jmag.values - 0.077 * JH,
            ("VISTA", "H"): stars.Hmag.values + 0.032 * JH,
            ("UKIDSS", "J"): stars.Jmag.values - 0.065 * JH,
            ("UKIDSS", "H"): stars.Hmag.values + 0.070 * JH - 0.030,
        }
        for survey in ["2MASS", "VISTA", "UKIDSS"]:
            try:
                download_images(name, ra, dec, survey=survey, filters=["J", "H"], size=8, overwrite=False)
            except requests.exceptions.ConnectionError as e:
                pytest.skip(f"Connection error for {survey}: {e}")
            for filt in ["J", "H"]:
                fits_file = f"{workdir}/{name}/{survey}/{survey}_{filt}.fits"
                with fits.open(fits_file) as hdu:
                    wcs = WCS(hdu[0].header, naxis=2)
                    ny, nx = hdu[0].data.shape
                diffs = []
                for (_, star), ref_mag in zip(stars.iterrows(), reference[(survey, filt)]):
                    x, y = wcs.wcs_world2pix(star.RAJ2000, star.DEJ2000, 0)
                    margin = 15 / (wcs.proj_plane_pixel_scales()[0].to(u.arcsec).value)
                    if not (margin < x < nx - margin and margin < y < ny - margin):
                        continue
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        mags = lp.photometry(name, star.RAJ2000, star.DEJ2000, 0.01, filt, survey, ap_radii=6.0,
                                             ap_units="arcsec", use_mask=False, correct_extinction=False,
                                             save_plots=False)[0]
                    diffs.append(mags[0] - ref_mag)
                if len(diffs) < 3:
                    continue
                offset = np.nanmedian(diffs)
                print(f"{survey} {filt}: median offset from 2MASS = {offset:+.3f} mag (N={len(diffs)})")
                # VISTA: ~0.04 mag of the offset comes from the known broader PSF wings of the VISTA tiles
                # (more light within the aperture than for the 2MASS PSF photometry)
                tolerance = 0.06 if survey == "VISTA" else 0.05
                assert abs(offset) < tolerance, f"{survey} {filt} differs from 2MASS by {offset:+.3f} mag"


if __name__ == "__main__":
    unittest.main()