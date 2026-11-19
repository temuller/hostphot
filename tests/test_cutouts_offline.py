"""Offline tests of the survey downloaders: the archive queries and downloads are replaced by synthetic responses
(tables and images with a WCS), so that the selection of the images, cutouts and header keywords are tested
without the archives."""
import gzip
import io
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
import warnings
from unittest import mock

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.io.votable import from_table
from astropy.table import Table
from astropy.wcs import WCS

from hostphot.flags import CalibrationWarning, FallbackImageWarning
from hostphot.cutouts import des, euclid, galex, herschel, hsc, kids, panstarrs, sdss, skymapper, spitzer, twomass
from hostphot.cutouts import ukidss, vista, wise
from hostphot.cutouts import downloads
from hostphot._constants import workdir

RA, DEC = 150.0, 2.0


def fake_image(ra: float, dec: float, npix: int, scale: float, value: float = 1.0, **keywords) -> fits.HDUList:
    """FITS image centred at (ra, dec), with ``npix`` pixels of ``scale`` arcsec, noise around ``value`` and the
    given header keywords."""
    w = WCS(naxis=2)
    w.wcs.crpix = [(npix + 1) / 2, (npix + 1) / 2]
    w.wcs.crval = [ra, dec]
    w.wcs.cdelt = [-scale / 3600, scale / 3600]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    rng = np.random.default_rng(int(abs(ra * 1000 + dec * 100)) % 2**32)
    header = w.to_header()
    for key, val in keywords.items():
        header[key] = val
    data = (value + rng.normal(0, 0.1, (npix, npix))).astype(np.float32)
    return fits.HDUList([fits.PrimaryHDU(data=data, header=header)])


def votable_bytes(table: Table) -> bytes:
    buffer = io.BytesIO()
    from_table(table).to_xml(buffer)
    return buffer.getvalue()


class TestSDSS(unittest.TestCase):
    def test_best_frame(self):
        frames = Table({"run": [1, 2, 3], "rerun": [301] * 3, "camcol": [1, 2, 3], "field": [10, 20, 30],
                        # frame 1: position near the edge; frames 2 and 3: cutout fits, 3 has better quality
                        "raMin": [RA - 0.01, RA - 0.1, RA - 0.1], "raMax": [RA + 0.2, RA + 0.1, RA + 0.1],
                        "decMin": [DEC - 0.1, DEC - 0.1, DEC - 0.1], "decMax": [DEC + 0.1, DEC + 0.1, DEC + 0.1],
                        "quality": [3, 2, 3]})
        with mock.patch.object(sdss, "SDSS") as service:
            service.query_sql.return_value = frames
            best = sdss._best_frame(RA, DEC, 3, 17)
            self.assertEqual(list(best["run"]), [3, 2, 1])
            service.query_sql.return_value = None
            self.assertIsNone(sdss._best_frame(RA, DEC, 3, 17))

    def test_frame_selection(self):
        frames = Table({"run": [1, 2], "rerun": [301, 301], "camcol": [1, 1], "field": [1, 2]})
        # the first frame does not contain the position (tilted frame), the second does
        far = fake_image(RA + 0.3, DEC, 400, 0.396)
        near = fake_image(RA + 0.01, DEC, 2048, 0.396)
        with mock.patch.object(sdss, "_best_frame", return_value=frames), mock.patch.object(sdss, "SDSS") as service:
            service.get_images.side_effect = [[far], [near]]
            hdu_list = sdss.get_SDSS_images(RA, DEC, size=2, filters="r")
        self.assertIs(hdu_list[0], near)
        self.assertEqual(hdu_list[0][0].data.shape, (303, 303))  # 2 arcmin
        self.assertEqual(hdu_list[0][0].header["HPDATAV"], "dr17")  # default data release
        # no frame contains the position
        with mock.patch.object(sdss, "_best_frame", return_value=frames[:1]), mock.patch.object(
            sdss, "SDSS"
        ) as service:
            service.get_images.return_value = [fake_image(RA + 0.3, DEC, 400, 0.396)]
            with self.assertWarnsRegex(UserWarning, "outside the SDSS footprint"):
                self.assertIsNone(sdss.get_SDSS_images(RA, DEC, size=2, filters="r"))
        with self.assertRaises(AssertionError):
            sdss.get_SDSS_images(RA, DEC, version="dr5")

    def test_closest_object_fallback(self):
        objects = Table({"ra": [RA + 0.001, RA], "dec": [DEC, DEC], "run": [5, 6]})
        with mock.patch.object(sdss, "SDSS") as service:
            service.query_region.side_effect = [None, objects]
            frame = sdss._closest_object_frame(sdss.SkyCoord(RA, DEC, unit="deg"), 17)
        self.assertEqual(frame["run"][0], 6)


class TestTwoMASS(unittest.TestCase):
    def test_tile_selection(self):
        content = "".join(
            f"<TD><![CDATA[https://irsa.ipac.caltech.edu/{band}i_tile{i}.fits]]></TD>"
            for band in "jhk" for i in (1, 2)
        ).encode()
        # tile 1 does not cover the cutout; tile 2 does
        tiles = {f"{band}i_tile1": fake_image(RA + 0.1, DEC, 512, 1.0) for band in "jhk"}
        tiles.update({f"{band}i_tile2": fake_image(RA, DEC, 512, 1.0) for band in "jhk"})

        def opener(urls, **kwargs):
            return [tiles[url.rsplit("/", 1)[-1].replace(".fits", "")] for url in urls]

        with mock.patch.object(twomass.requests, "get") as get, mock.patch.object(
            twomass, "open_fits_from_urls", side_effect=opener
        ):
            get.return_value.content = content
            hdu_list = twomass.get_2MASS_images(RA, DEC, size=3, filters=["J", "Ks"])
        self.assertEqual(len(hdu_list), 2)
        for hdu in hdu_list:
            self.assertEqual(hdu[0].data.shape, (180, 180))
            self.assertAlmostEqual(hdu[0].header["CRVAL1"], RA)


class TestDES(unittest.TestCase):
    def test_urls_and_images(self):
        table = pd.DataFrame({
            "obs_bandpass": ["g", "g", "g", "r"],
            "proctype": ["Stack"] * 4,
            "prodtype": ["image", "image", "weight", "image"],
            "exptime": [90.0, 450.0, 450.0, 300.0],
            "access_url": [f"https://des.org/{n}?SIZE=0.5,0.5&POS=1" for n in ["g1", "g2", "g2w", "r1"]],
        })
        with mock.patch.object(des.sia, "SIAService") as service:
            service.return_value.search.return_value = table
            urls, weights = des.get_DES_urls(RA, DEC, 0.05, ["g", "r", "i"])
        self.assertTrue(urls[0].startswith("https://des.org/g2?SIZE=0.05"))  # longest exposure, requested size
        self.assertTrue(weights[0].startswith("https://des.org/g2w"))
        self.assertIsNone(weights[1])
        self.assertEqual(urls[2:], [None])
        image, weight = fake_image(RA, DEC, 100, 0.263), fake_image(RA, DEC, 100, 0.263, value=4.0)
        with mock.patch.object(des, "get_DES_urls", return_value=(["a", None], ["aw", None])), mock.patch.object(
            des, "open_fits_from_urls", return_value=[image, weight, None, None]
        ):
            hdu_list = des.get_DES_images(RA, DEC, size=0.5, filters="gr")
        self.assertEqual(len(hdu_list[0]), 2)  # image and weight map
        self.assertIsNone(hdu_list[1])
        with mock.patch.object(des.sia, "SIAService") as service:
            service.return_value.search.return_value = []
            with self.assertWarnsRegex(UserWarning, "outside the DES footprint"):
                self.assertEqual(des.get_DES_urls(RA, DEC, 0.05, ["g"]), (None, None))


class TestSkyMapper(unittest.TestCase):
    def test_urls_and_zeropoint(self):
        table = pd.DataFrame({
            "band": ["g", "g", "g", "r"],
            "format": ["image/fits"] * 4,
            "exptime": [100.0, 100.0, 20.0, 100.0],
            "filesize": [10, 10, 10, 10],
            "mean_fwhm": [2.5, 2.0, 1.5, 2.2],
            "get_fits": ["g_bad_seeing", "g_best", "g_short", "r_best"],
        })
        with mock.patch.object(skymapper.sia, "SIAService") as service:
            service.return_value.search.return_value = table
            urls = skymapper.get_SkyMapper_urls(RA, DEC, 0.05, ["g", "r", "i"])
        self.assertEqual(urls, ["g_best", "r_best", None])
        with mock.patch.object(skymapper, "get_SkyMapper_urls", return_value=["g_best", None]), mock.patch.object(
            skymapper, "open_fits_from_urls", return_value=[fake_image(RA, DEC, 50, 0.5, ZPAPPROX=25.3), None]
        ):
            hdu_list = skymapper.get_SkyMapper_images(RA, DEC, size=0.5, filters="gr")
        self.assertEqual(hdu_list[0][0].header["MAGZP"], 25.3)
        self.assertIsNone(hdu_list[1])


class TestGALEX(unittest.TestCase):
    def test_image_selection(self):
        obs = Table({"project": ["AIS", "MIS"], "t_exptime": [100.0, 1500.0],
                     "dataURL": ["https://galex/AIS_1-nd-int.fits.gz", "https://galex/MIS_1-nd-int.fits.gz"]})
        # the deepest survey (MIS) has an image too far from the position; AIS covers it
        far = fake_image(RA + 1.0, DEC, 200, 1.5, RA_CENT=RA + 1.0, DEC_CENT=DEC)
        near = fake_image(RA + 0.05, DEC, 400, 1.5, RA_CENT=RA + 0.05, DEC_CENT=DEC)
        with mock.patch.object(galex, "Observations") as observations, mock.patch.object(
            galex, "open_fits_from_urls", side_effect=[[far], [near]]
        ) as opener:
            observations.query_criteria.return_value = obs
            hdu_list = galex.get_GALEX_images(RA, DEC, size=3, filters="NUV")
        self.assertIs(hdu_list[0], near)
        self.assertEqual(hdu_list[0][0].data.shape, (120, 120))
        self.assertEqual(hdu_list[0][0].header["HPDATAV"], "AIS")  # survey of the image used
        self.assertTrue(opener.call_args_list[0].args[0][0].endswith("MIS_1-nd-intbgsub.fits.gz"))


class TestKiDS(unittest.TestCase):
    def test_tiles_and_zeropoint(self):
        csv = "dp_id,filter,t_exptime\nKIDS_r_1,r_SDSS,1800\nKIDS_r_2,r_SDSS,900\nKIDS_g_1,g_SDSS,900\n"
        r1 = fake_image(RA, DEC, 120, 0.2, PHOTZP=0.0, DMAG=0.05)
        r2 = fake_image(RA, DEC, 120, 0.2, PHOTZP=0.0, DMAG=0.02)
        g1 = fake_image(RA, DEC, 120, 0.2, PHOTZP=0.0)  # without the DR5 correction

        def opener(urls, **kwargs):
            return [{"KIDS_r_1": r1, "KIDS_r_2": r2, "KIDS_g_1": g1}[url.split("?")[2].split("&")[0]] for url in urls]

        with mock.patch.object(kids.requests, "get") as get, mock.patch.object(
            kids, "open_fits_from_urls", side_effect=opener
        ):
            get.return_value.text = csv
            with self.assertWarns(CalibrationWarning):
                hdu_list = kids.get_KiDS_images(RA, DEC, size=0.3, filters="rgi")
        self.assertAlmostEqual(hdu_list[0][0].header["MAGZP"], 0.05)  # deepest tile
        self.assertAlmostEqual(hdu_list[1][0].header["MAGZP"], 0.0)
        self.assertIsNone(hdu_list[2])
        with mock.patch.object(kids.requests, "get") as get:
            get.return_value.text = "dp_id,filter,t_exptime\n"
            with self.assertWarnsRegex(UserWarning, "outside the KiDS footprint"):
                self.assertEqual(kids.get_KiDS_images(RA, DEC, filters="r"), [None])


class TestEuclid(unittest.TestCase):
    def test_query_and_images(self):
        rows = Table({
            "energy_bandpassname": ["VIS", "VIS", "VIS", "VIS", "g", "VIS"],
            "instrument_name": ["VIS", "VIS", "VIS", "VIS", "MegaCam", "VIS"],
            "dataproduct_subtype": ["science", "noise", "science", "noise", "science", "flag"],
            "obs_id": ["T1_x", "T1_x", "T2_x", "T2_x", "T1_x", "T1_x"],
            "access_url": ["sci1", "rms1", "sci2", "rms2", "ext", "flag"],
        })
        with mock.patch.object(euclid.requests, "get") as get:
            get.return_value.content = votable_bytes(rows)
            mosaics = euclid.query_euclid_mosaics(RA, DEC, version="q1")
            self.assertEqual(get.call_count, 1)
            # DR1 (default): wide and deep collections queried, tiles kept separately
            dr1 = euclid.query_euclid_mosaics(RA, DEC)
            self.assertEqual(get.call_count, 3)
            self.assertEqual([c.kwargs["params"]["COLLECTION"] for c in get.call_args_list[1:]],
                             euclid._COLLECTIONS["dr1"])
        self.assertEqual([m["tile"] for m in mosaics], ["T1", "T2"])
        self.assertEqual(len(dr1), 4)
        # cutouts from the IRSA cutout service, with the path of the mosaic
        self.assertEqual(
            euclid._cutout_url("https://irsacloud.ipac.caltech.edu/ibe/data/euclid/dr1/MER/1/VIS/a.fits", 10.0, -5.0, 0.01),
            "https://irsacloud.ipac.caltech.edu/ibe/cutout?ra=10.0&dec=-5.0&size=0.01&path=euclid/dr1/MER/1/VIS/a.fits",
        )
        with self.assertRaises(ValueError):
            euclid.query_euclid_mosaics(RA, DEC, version="dr9")
        # both tiles cover the cutout; T2 is deeper (lower RMS); a masked pixel (RMS 1e16) becomes NaN
        sci1, sci2 = fake_image(RA, DEC, 60, 0.1, MAGZERO=24.4), fake_image(RA, DEC, 60, 0.1, MAGZERO=24.5)
        rms1, rms2 = fake_image(RA, DEC, 60, 0.1, value=2.0), fake_image(RA, DEC, 60, 0.1, value=1.0)
        rms2[0].data[0, 0] = 1e16
        hdus = {"sci1": sci1, "rms1": rms1, "sci2": sci2, "rms2": rms2}
        with mock.patch.object(euclid, "query_euclid_mosaics", return_value=mosaics), mock.patch.object(
            euclid, "open_fits_from_urls", side_effect=lambda urls, **kw: [hdus[url.split("path=")[1]] for url in urls]
        ):
            hdu_list = euclid.get_Euclid_images(RA, DEC, size=0.1, filters=["VIS", "H"])
        self.assertEqual(hdu_list[0][0].header["TILEID"], "T2")
        self.assertEqual(hdu_list[0][0].header["HPDATAV"], "dr1")  # default data release
        self.assertEqual(hdu_list[0][0].header["MAGZP"], 24.5)
        self.assertTrue(np.isnan(hdu_list[0][0].data[0, 0]))
        self.assertEqual(hdu_list[0][1].name, "VARIANCE")
        self.assertIsNone(hdu_list[1])
        with mock.patch.object(euclid, "query_euclid_mosaics", return_value=[]):
            with self.assertWarnsRegex(UserWarning, "No Euclid images"):
                self.assertEqual(euclid.get_Euclid_images(RA, DEC, filters="VIS"), [None])


class TestUKIDSS(unittest.TestCase):
    @staticmethod
    def frame(ra: float, magzpt: float, texptime: float = 100.0) -> fits.HDUList:
        image = fake_image(ra, DEC, 300, 0.4)
        primary = fits.PrimaryHDU(header=fits.Header({"AMSTART": 1.2, "AMEND": 1.4, "CD1_1": 1.0}))
        ext = fits.ImageHDU(data=image[0].data, header=image[0].header)
        ext.header.update({"MAGZPT": magzpt, "MAGZRR": 0.02, "GAIN": 4.2, "READNOIS": 25.0, "EXTINCT": 0.05,
                           "TEXPTIME": texptime, "SEEING": 2.0})
        return fits.HDUList([primary, ext])

    def test_frame_selection(self):
        urls = ["https://wsa/get?file=/a/w1_st.fit&band=J", "https://wsa/get?file=/a/w2_dp.fit&band=J",
                "https://wsa/get?file=/a/w3_st.fit&band=H2", "https://wsa/get?file=/a/w4_st.fit&band=H"]
        frames = {"w1_st": self.frame(RA, 25.0), "w2_dp": self.frame(RA, 25.5, texptime=5000.0),
                  "w4_st": fits.HDUList([fits.PrimaryHDU()])}  # frame without the image extension

        def opener(url_list, **kwargs):
            return [frames[url.split("/a/")[1].split(".fit")[0]] for url in url_list]

        with mock.patch.object(ukidss, "Ukidss") as service, mock.patch.object(
            ukidss, "open_fits_from_urls", side_effect=opener
        ) as opened:
            service.return_value.get_image_list.return_value = urls
            with self.assertWarnsRegex(UserWarning, "without image data"):
                hdu_list = ukidss.get_UKIDSS_images(RA, DEC, size=1, filters="JH")
        # the deep stack is downloaded first and covers the cutout: the other J frame is not downloaded
        self.assertEqual(opened.call_args_list[0].args[0][0], urls[1])
        header = hdu_list[0][0].header
        self.assertAlmostEqual(header["MAGZP"], 25.5 - 0.05 * 0.3)
        self.assertAlmostEqual(header["PSF_FWHM"], 0.8)
        self.assertEqual(header["TEXPTIME"], 5000.0)
        self.assertIsNone(hdu_list[1])
        self.assertTrue(ukidss._is_deep_stack(urls[1]))
        self.assertFalse(ukidss._is_deep_stack(urls[0]))


class TestSpitzer(unittest.TestCase):
    def test_seip_and_sha_fallback(self):
        text = ("https://irsa/IRAC.1.mosaic.fits https://irsa/IRAC.2.mosaic.fits other.txt")
        # IRAC.1: SEIP covers the cutout; IRAC.2: the SEIP mosaic only covers part of it, an SHA mosaic all of it
        seip1 = fake_image(RA, DEC, 200, 0.6, ZPAB=21.58, EXPTIME=100.0)
        seip2 = fake_image(RA + 0.012, DEC, 100, 0.6, ZPAB=21.58, EXPTIME=100.0)
        sha2 = fake_image(RA, DEC, 300, 0.6, GAIN=3.7, EXPTIME=10.4, FLUXCONV=0.1469, AORKEY=1234)
        seip = {"https://irsa/IRAC.1.mosaic.fits": seip1, "https://irsa/IRAC.2.mosaic.fits": seip2}
        with mock.patch.object(spitzer.requests, "get") as get, mock.patch.object(
            spitzer, "open_fits_from_urls", side_effect=lambda urls, **kw: [seip.get(u, sha2) for u in urls]
        ), mock.patch.object(spitzer, "query_sha_mosaics", return_value={"IRAC.1": [], "IRAC.2": ["sha2"]}), \
                mock.patch.object(spitzer, "read_fits_header", return_value=sha2[0].header):
            get.return_value.text = text
            with self.assertWarns(FallbackImageWarning):
                hdu_list = spitzer.get_Spitzer_images(RA, DEC, size=1, filters=["IRAC.1", "IRAC.2"])
        self.assertEqual(hdu_list[0][0].header["ARCHIVE"], "SEIP")
        self.assertEqual(hdu_list[0][0].header["MAGZP"], 21.58)
        self.assertEqual(hdu_list[1][0].header["ARCHIVE"], "SHA")
        self.assertTrue(hdu_list[1][0].header["HPFFALLB"])
        self.assertAlmostEqual(hdu_list[1][0].header["GAIN"], 3.7 * 10.4 / 0.1469)
        self.assertEqual(hdu_list[1][0].data.shape, (100, 100))

    def test_sha_sorted_by_footprint(self):
        # mosaics: not containing the cutout, containing part of it, containing all of it, unreadable header
        headers = {
            "out": fake_image(RA + 1.0, DEC, 100, 0.6)[0].header,
            "part": fake_image(RA + 0.012, DEC, 100, 0.6)[0].header,
            "full": fake_image(RA, DEC, 300, 0.6)[0].header,
            "unknown": None,
        }
        with mock.patch.object(spitzer, "read_fits_header", side_effect=lambda url: headers[url]):
            urls = spitzer._sort_sha_by_footprint(RA, DEC, 60, ["out", "part", "unknown", "full"])
        self.assertEqual(urls, ["full", "unknown", "part"])
        # the first mosaic is downloaded first; as it only covers part of the cutout, the headers of the others
        # are read and the one containing the cutout is downloaded next (the one not containing it is not)
        images = {"part": fake_image(RA + 0.012, DEC, 100, 0.6, GAIN=3.7, EXPTIME=10.4, FLUXCONV=0.1469),
                  "full": fake_image(RA, DEC, 300, 0.6, GAIN=3.7, EXPTIME=10.4, FLUXCONV=0.1469)}
        downloaded = []

        def download(urls, **kw):
            downloaded.extend(urls)
            return [images.get(url) for url in urls]

        with mock.patch.object(spitzer, "read_fits_header", side_effect=lambda url: headers[url]) as read, \
                mock.patch.object(spitzer, "open_fits_from_urls", side_effect=download):
            hdu, coverage = spitzer.get_SHA_image(RA, DEC, 60, ["part", "out", "full"])
        self.assertEqual(downloaded, ["part", "full"])
        self.assertEqual(read.call_count, 2)
        self.assertGreaterEqual(coverage, 0.99)
        # the first mosaic covers the cutout: no headers are read
        downloaded.clear()
        with mock.patch.object(spitzer, "read_fits_header") as read, \
                mock.patch.object(spitzer, "open_fits_from_urls", side_effect=download):
            spitzer.get_SHA_image(RA, DEC, 60, ["full", "part"])
        self.assertEqual(downloaded, ["full"])
        read.assert_not_called()

    def test_sha_query(self):
        rows = Table({"dataproduct_subtype": ["science", "science", "science", "uncertainty"],
                      "access_url": ["a_maic.fits", "b_maic.fits", "c_cbcd.fits", "d_munc.fits"],
                      "t_exptime": [10.0, 100.0, 50.0, 100.0],
                      "energy_bandpassname": ["IRAC1", "IRAC1", "IRAC1", "IRAC1"]})
        with mock.patch.object(spitzer.requests, "get") as get:
            get.return_value.content = votable_bytes(rows)
            urls = spitzer.query_sha_mosaics(RA, DEC, ["IRAC.1", "MIPS.1"])
        self.assertEqual(urls, {"IRAC.1": ["b_maic.fits", "a_maic.fits"], "MIPS.1": []})


class TestHerschel(unittest.TestCase):
    @staticmethod
    def write_map(path: Path, ra: float, wavelength: float, instrument: str, bunit: str, npix: int = 200,
                  scale: float = 3.2) -> None:
        image = fake_image(ra, DEC, npix, scale)
        primary = fits.PrimaryHDU(header=fits.Header({"INSTRUME": instrument, "WAVELNTH": wavelength,
                                                      "TELESCOP": "Herschel"}))
        ext = fits.ImageHDU(data=image[0].data, header=image[0].header, name="image")
        ext.header["BUNIT"] = bunit
        path.parent.mkdir(parents=True, exist_ok=True)
        fits.HDUList([primary, ext]).writeto(path, overwrite=True)

    def test_get_images(self):
        observations = Table({
            "observation_id": [1001, 2002], "instrument_oid": [1, 2], "observing_mode_oid": [1, 4],
            "target_name": ["NGC", "NGC"], "ra": [RA, RA], "dec": [DEC, DEC], "duration": [3600.0, 1800.0],
            # (polygon vertices in radians)
            "polygon_fov": ["{" + ",".join(f"({np.radians(RA + dx)}, {np.radians(DEC + dy)})"
                                           for dx, dy in [(-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1)]) + "}"] * 2,
        })

        def download_data(retrieval_type, observation_id, instrument_name, filename, download_dir, **kwargs):
            # tarball with the maps of the observation (and a point-source map, which is not used)
            source = Path(tempfile.mkdtemp())
            if instrument_name == "PACS":
                self.write_map(source / "level2_5/HPPJSMAPB/hpacs_HPPJSMAPB_green.fits", RA, 100.0, "PACS", "Jy/pixel")
                self.write_map(source / "level2_5/HPPPMAPB/hpacs_HPPPMAPB_green.fits", RA, 100.0, "PACS", "Jy/pixel")
                self.write_map(source / "level2_5/HPPJSMAPR/hpacs_HPPJSMAPR.fits", RA, 160.0, "PACS", "Jy/pixel")
            else:
                self.write_map(source / "level2_5/extdPSW/hspire_extdPSW.fits", RA, 250.0, "SPIRE", "MJy/sr", 100, 6.0)
            tarball = Path(download_dir) / f"{filename}.tar"
            with tarfile.open(tarball, "w") as tar:
                tar.add(source, arcname=".")
            return str(tarball)

        with mock.patch.object(herschel, "HSA") as hsa:
            hsa.query_hsa_tap.return_value = observations
            hsa.download_data.side_effect = download_data
            with tempfile.TemporaryDirectory() as out:
                hdu_list = herschel.get_Herschel_images(RA, DEC, size=3, filters=["PACS100", "PACS160", "SPIRE250"],
                                                        output_dir=out)
                self.assertTrue((Path(out) / "hsa_selected_maps.csv").exists())
        self.assertEqual([hdu[0].header["FILTER"] for hdu in hdu_list], ["PACS100", "PACS160", "SPIRE250"])
        self.assertEqual(hdu_list[0][0].header["OBSID"], "1001")
        self.assertIn("HPPJSMAPB", hdu_list[0][0].header["SRCFILE"])  # extended-emission map, not the point-source map
        self.assertEqual(hdu_list[0][0].data.shape, (56, 56))  # 3 arcmin at 3.2 arcsec/pixel
        # only the final maps are downloaded (they include the instrument maps)
        self.assertEqual(hsa.download_data.call_count, 2)
        # no observations
        with mock.patch.object(herschel, "HSA") as hsa:
            hsa.query_hsa_tap.return_value = observations[:0]
            self.assertEqual(herschel.get_Herschel_images(RA, DEC, filters="SPIRE250"), [None])

    def test_helpers(self):
        self.assertTrue(np.isnan(herschel._footprint_area_deg2("{(1, 2)}")))
        vertices = [(10.0, 0.0), (11.0, 0.0), (11.0, 1.0), (10.0, 1.0)]  # (in order)
        square = "{" + ",".join(f"({np.radians(r)}, {np.radians(d)})" for r, d in vertices) + "}"
        self.assertAlmostEqual(herschel._footprint_area_deg2(square), 1.0, places=2)
        header = fits.Header({"INSTRUME": "PACS"})
        self.assertEqual(herschel._candidate_filter(Path("x/HPPJSMAPB_blue.fits"), header), "PACS70")
        self.assertEqual(herschel._candidate_filter(Path("x/HPPJSMAPB_green.fits"), header), "PACS100")
        self.assertEqual(herschel._candidate_filter(Path("x/HPPJSMAPR.fits"), header), "PACS160")
        self.assertIsNone(herschel._candidate_filter(Path("x/other.fits"), fits.Header({"INSTRUME": "HIFI"})))
        self.assertLess(herschel._score_candidate(Path("x/HPPJSMAPB_coverage.fits"), "PACS100", header), 0)
        with self.assertRaises(PermissionError), mock.patch.object(herschel, "HSA") as hsa:
            hsa.download_data.side_effect = RuntimeError("Proprietary data")
            herschel._download_product(1, "PACS", None, Path(tempfile.mkdtemp()))
        # unsafe paths in the tarballs are rejected
        tmp = Path(tempfile.mkdtemp())
        tarball = tmp / "bad.tar"
        (tmp / "file.txt").write_text("x")
        with tarfile.open(tarball, "w") as tar:
            tar.add(tmp / "file.txt", arcname="../outside.txt")
        with self.assertRaisesRegex(RuntimeError, "Unsafe path"):
            herschel._safe_extract_tarball(tarball, tmp / "extract")


class TestVISTA(unittest.TestCase):
    @staticmethod
    def tile(ra: float, magzrr: float, exptime: float = 10.0) -> fits.HDUList:
        image = fake_image(ra, DEC, 300, 0.34)
        primary = fits.PrimaryHDU(header=fits.Header({"HIERARCH ESO TEL AIRM START": 1.1,
                                                      "HIERARCH ESO TEL AIRM END": 1.3}))
        ext = fits.ImageHDU(data=image[0].data, header=image[0].header)
        ext.header.update({"MAGZPT": 23.8, "MAGZRR": magzrr, "EXPTIME": exptime, "EXTINCT": 0.05,
                           "SEEING": 2.5, "NJITTER": 5})
        return fits.HDUList([primary, ext])

    def test_tile_selection(self):
        links = ["http://vsa/getFImage?file=t1&band=J", "http://vsa/getFImage?file=t2&band=J",
                 "http://vsa/getFImage?file=t3&band=J"]
        tiles = {"t1": self.tile(RA + 0.05, 0.01),  # does not cover the cutout
                 "t2": self.tile(RA, 0.2),  # covers it, badly calibrated
                 "t3": self.tile(RA, 0.02)}  # covers it, well calibrated

        def opener(urls, **kwargs):
            return [tiles[url.split("file=")[1].split("&")[0]] for url in urls]

        with mock.patch.object(vista, "contains_coords", return_value=True), mock.patch.object(
            vista, "_query_links", return_value=links
        ), mock.patch.object(vista, "open_fits_from_urls", side_effect=opener):
            hdu_list = vista.get_VISTA_images(RA, DEC, size=1, filters="J", version="VHS")
            header = hdu_list[0][0].header
            self.assertEqual(header["MAGZRR"], 0.02)
            self.assertAlmostEqual(header["MAGZP"], 23.8 - 0.05 * 0.2)
            self.assertAlmostEqual(header["PSF_FWHM"], 2.5 * 0.34)
            self.assertEqual(header["TEXPTIME"], 50.0)
            self.assertEqual(header["HPDATAV"], "VHS")
            # survey covering the position (the deepest one), if not given
            self.assertEqual(vista.get_VISTA_images(RA, DEC, size=1, filters="J")[0][0].header["HPDATAV"], "VIDEO")
            # only badly calibrated tiles: used with a warning (calib flag)
            tiles["t3"] = self.tile(RA, 0.3)
            with self.assertWarns(CalibrationWarning):
                hdu_list = vista.get_VISTA_images(RA, DEC, size=1, filters="J", version="VHS")
            self.assertTrue(hdu_list[0][0].header["HPFCALIB"])
        with self.assertRaises(ValueError):
            vista.get_VISTA_images(RA, DEC, filters="J", version="UKIDSS")
        with mock.patch.object(vista, "contains_coords", return_value=False):
            self.assertEqual(vista.get_VISTA_images(RA, DEC, filters="JH"), [None, None])
        self.assertEqual(vista._latest_version(["a/mosaics/x_v9/f", "a/mosaics/x_v10/f"]), ["a/mosaics/x_v10/f"])


class TestWISE(unittest.TestCase):
    def test_wise(self):
        csv = ("energy_bandpassname,access_url,t_exptime\nW1,https://irsa/w1.fits?x=1,100\n"
               "W2,https://irsa/w2.fits?x=1,90\n")
        with mock.patch.object(wise.requests, "get") as get, mock.patch.object(
            wise, "open_fits_from_urls", return_value=[fake_image(RA, DEC, 400, 1.375), None]
        ):
            get.return_value.text = csv
            hdu_list = wise.get_WISE_images(RA, DEC, size=3, filters=["W1", "W3"])
        self.assertEqual(hdu_list[0][0].header["EXPTIME"], "100")
        self.assertEqual(hdu_list[0][0].header["HPSOURCE"], "w1.fits")
        self.assertIsNone(hdu_list[1])

    def test_unwise(self):
        # tarball with two coadd tiles in W1 (the deeper one, with the larger inverse variance, is used)
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            for tile, invvar in [("1497p015", 1.0), ("1500p015", 4.0)]:
                for kind, value in [("img-m", 1.0), ("invvar-m", invvar)]:
                    content = io.BytesIO()
                    fake_image(RA, DEC, 100, 2.75, value=value).writeto(content)
                    data = content.getvalue()
                    name = f"unwise-{tile}-w1-{kind}.fits"
                    if kind == "invvar-m":
                        data, name = gzip.compress(data), name + ".gz"
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
        with mock.patch.object(wise.requests, "get") as get:
            get.return_value.content = buffer.getvalue()
            with self.assertWarnsRegex(UserWarning, "only has W1 and W2"):
                hdu_list = wise.get_unWISE_images(RA, DEC, size=3, filters=["W1", "W3"], version="neo7")
        self.assertIn("1500p015", hdu_list[0][0].header["HPSOURCE"])
        self.assertEqual(hdu_list[0][0].header["HPDATAV"], "neo7")
        self.assertEqual(hdu_list[0][1].name, "INVVAR")
        self.assertIsNone(hdu_list[1])
        with self.assertRaises(AssertionError):
            wise.get_unWISE_images(RA, DEC, version="neo9")


class TestHSC(unittest.TestCase):
    def test_images(self):
        image = fake_image(RA, DEC, 50, 0.168)
        cutout = fits.HDUList([
            fits.PrimaryHDU(header=fits.Header({"FLUXMAG0": 10 ** (27 / 2.5) * 10, "OBJECT": "x"})),
            fits.ImageHDU(data=image[0].data, header=image[0].header),
            fits.ImageHDU(data=np.zeros((50, 50), dtype=np.int32)),
            fits.ImageHDU(data=np.ones((50, 50), dtype=np.float32)),
        ])
        with mock.patch.dict(os.environ, {"HSC_SSP_USERNAME": "user", "HSC_SSP_PASSWORD": "password"}), mock.patch.object(
            hsc, "_request_cutout", side_effect=[cutout, None]
        ):
            hdu_list = hsc.get_HSC_images(RA, DEC, size=0.1, filters=["r", "i"])
        self.assertAlmostEqual(hdu_list[0][0].header["MAGZP"], 29.5)
        self.assertEqual([h.name for h in hdu_list[0][1:]], ["VARIANCE", "MASK"])
        self.assertEqual(hdu_list[0][0].header["HPDATAV"], "pdr3_wide")  # default version
        self.assertIsNone(hdu_list[1])
        with mock.patch.dict(os.environ, {"HSC_SSP_USERNAME": "", "HSC_SSP_PASSWORD": ""}):
            with self.assertRaisesRegex(ValueError, "credentials"):
                hsc._get_credentials()


class TestPanSTARRSAndDownloads(unittest.TestCase):
    def test_panstarrs_images(self):
        table = Table({"filter": ["r", "g"], "filename": ["r.fits", "g.fits"]})
        with mock.patch.object(panstarrs, "query_ps1", return_value=table), mock.patch.object(
            panstarrs, "open_fits_from_urls", side_effect=lambda urls: [fake_image(RA, DEC, 10, 0.25) for _ in urls]
        ) as opener:
            hdu_list = panstarrs.get_PanSTARRS_images(RA, DEC, size=0.04, filters="gir")
        self.assertEqual(len(opener.call_args.args[0]), 2)
        self.assertIsNone(hdu_list[1])
        self.assertIsNotNone(hdu_list[2])

    def test_download_images(self):
        name = "offline_downloads"
        good = fake_image(RA, DEC, 100, 0.25)
        empty = fits.HDUList([fits.PrimaryHDU(data=np.zeros((100, 100), dtype=np.float32), header=good[0].header)])
        elsewhere = fake_image(RA + 1.0, DEC, 100, 0.25)  # no data at the position of the target
        with mock.patch.object(panstarrs, "get_PanSTARRS_images", return_value=[good, None, empty, elsewhere]):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                downloads.download_images(name, RA - 360, DEC, survey="PanSTARRS", size=0.4, filters="grizy"[:4])
        messages = " ".join(str(w.message) for w in caught)
        self.assertIn("No PanSTARRS image was downloaded for filter r", messages)
        self.assertIn("contains no data", messages)
        self.assertIn("no data at the position of the target", messages)
        survey_dir = Path(workdir, name, "PanSTARRS")
        header = fits.getheader(survey_dir / "PanSTARRS_g.fits")
        self.assertEqual(header["HPSURVEY"], "PanSTARRS")
        self.assertNotIn("HPDATAV", header)  # PanSTARRS has no versions
        self.assertFalse((survey_dir / "PanSTARRS_i.fits").exists())
        # existing images are not downloaded again
        with mock.patch.object(panstarrs, "get_PanSTARRS_images") as getter:
            self.assertIsNone(downloads.download_images(name, RA, DEC, survey="PanSTARRS", filters="g", overwrite=False))
            getter.assert_not_called()
        # no images found
        with mock.patch.object(panstarrs, "get_PanSTARRS_images", return_value=None):
            with self.assertWarnsRegex(UserWarning, "No PanSTARRS images were found"):
                downloads.download_images(name, RA, DEC, survey="PanSTARRS", filters="y")
        with self.assertRaises(ValueError):
            downloads.download_images(name, RA, DEC, survey="HST")


def test_add_provenance_keeps_used_version():
    """The survey version written by the download function (the one actually used) is kept."""
    from hostphot.utils import add_provenance, set_data_version

    header = fits.Header()
    add_provenance(header, "VISTA", size=3)
    assert "HPDATAV" not in header
    add_provenance(header, "VISTA", version="vhs")
    assert header["HPDATAV"] == "vhs"
    set_data_version(header, "VHS")
    add_provenance(header, "VISTA", version="vhs")
    assert header["HPDATAV"] == "VHS"


if __name__ == "__main__":
    unittest.main()
