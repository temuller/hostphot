import tarfile
import requests
import numpy as np
from pathlib import Path

import extinction
from sfdmap2 import sfdmap
from astropy.coordinates import SkyCoord
from astropy import units as u

import hostphot
from hostphot.photometry.photometry_utils import integrate_filter, extract_filter

import warnings


def _download_dustmaps():
    """Downloads the dust maps for extinction calculation if they are not found
    locally.
    """
    mapsdir = Path(hostphot.__path__[0])

    # check if files already exist locally
    dust_files = [
        mapsdir.joinpath("sfddata-master", rf"SFD_dust_4096_{sky}gp.fits")
        for sky in ["n", "s"]
    ]
    mask_files = [
        mapsdir.joinpath("sfddata-master", rf"SFD_mask_4096_{sky}gp.fits")
        for sky in ["n", "s"]
    ]
    maps_files = dust_files + mask_files
    existing_files = [file.is_file() for file in maps_files]

    if not all(existing_files) == True:
        # download dust maps
        sfdmaps_url = "https://github.com/kbarbary/sfddata/archive/master.tar.gz"
        response = requests.get(sfdmaps_url)

        master_tar = Path("master.tar.gz")
        with open(master_tar, "wb") as file:
            file.write(response.content)

        # extract tar file under mapsdir directory
        tar = tarfile.open(master_tar)
        tar.extractall(mapsdir)
        tar.close()
        master_tar.unlink()  # remove file


def deredden(
    wave: np.ndarray,
    flux: np.ndarray,
    ra: float,
    dec: float,
    scaling: float = 0.86,
    reddening_law: str = "fitzpatrick99",
    r_v: float = 3.1,
) -> np.ndarray:
    """Dereddens the given spectrum, given a right ascension and declination or :math:`E(B-V)`.

    Parameters
    ----------
    wave: Wavelength values.
    flux: Flux density values.
    ra:  Right ascension in degrees.
    dec: Declination in degrees.
    scaling: Calibration of the Milky Way dust maps. Either ``0.86``
        for the Schlafly & Finkbeiner (2011) recalibration or ``1.0`` for the original
        dust map of Schlegel, Fikbeiner & Davis (1998).
    reddening_law: Reddening law. The options are: ``ccm89`` (Cardelli, Clayton & Mathis 1989),
        ``odonnell94`` (O'Donnell 1994), ``fitzpatrick99`` (Fitzpatrick 1999), ``calzetti00``
        (Calzetti 2000) and ``fm07`` (Fitzpatrick & Massa 2007 with :math:`R_V` = 3.1.)
    r_v: Total-to-selective extinction ratio (:math:`R_V`)

    Returns
    -------
    deredden_flux : array
        Deredden flux values.
    """
    _download_dustmaps()

    hostphot_path = Path(hostphot.__path__[0])
    dustmaps_dir = hostphot_path / "sfddata-master"

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        m = sfdmap.SFDMap(mapdir=dustmaps_dir, scaling=scaling)
        ebv = m.ebv(ra, dec)  # RA and DEC in degrees
        a_v = r_v * ebv

    rl_list = ["ccm89", "odonnell94", "fitzpatrick99", "calzetti00", "fm07"]
    assert (
        reddening_law in rl_list
    ), f"Choose one of the available reddening laws: {rl_list}"

    if reddening_law == "ccm89":
        ext = extinction.ccm89(wave, a_v, r_v)
    elif reddening_law == "odonnell94":
        ext = extinction.odonnell94(wave, a_v, r_v)
    elif reddening_law == "fitzpatrick99":
        ext = extinction.fitzpatrick99(wave, a_v, r_v)
    elif reddening_law == "calzetti00":
        ext = extinction.calzetti00(wave, a_v, r_v)
    elif reddening_law == "fm07":
        ext = extinction.fm07(wave, a_v)

    deredden_flux = extinction.remove(ext, flux)

    return deredden_flux


def calc_extinction(
    filt: str,
    survey: str,
    ra: float,
    dec: float,
    scaling: float = 0.86,
    reddening_law: str = "fitzpatrick99",
    r_v: float = 3.1,
):
    """Calculates the extinction for a given filter, right ascension and declination
    or `E(B-V)`.

    Parameters
    ----------
    filter_wave: Filter's wavelength range.
    filter_response: Filter's response function.
    ra: Right ascension.
    dec: Declinationin degrees.
    scaling: Calibration of the Milky Way dust maps. Either ``0.86``
        for the Schlafly & Finkbeiner (2011) recalibration or ``1.0`` for the original
        dust map of Schlegel, Fikbeiner & Davis (1998).
    reddening_law: Reddening law. The options are: ``ccm89`` (Cardelli, Clayton & Mathis 1989),
        ``odonnell94`` (O'Donnell 1994), ``fitzpatrick99`` (Fitzpatrick 1999), ``calzetti00``
        (Calzetti 2000) and ``fm07`` (Fitzpatrick & Massa 2007 with :math:`R_V` = 3.1.)
    r_v: Total-to-selective extinction ratio (:math:`R_V`)

    Returns
    -------
    A_ext: Extinction value in magnitudes.
    """
    if survey == "LegacySurvey":
        # https://datalab.noirlab.edu/ls/bass.php
        # declination above ~32 and above the galactic plane
        gal_coords = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame="icrs")
        if (dec > 32.375) and (gal_coords.b.value > 0):
            version = "BASS+MzLS"
        else:
            version = "DECam"
    else:
        version = None

    # extract transmission function for the given filter+survey
    filter_wave, filter_response = extract_filter(filt, survey, version)

    # calculate extinction
    flux = 100
    dereddened_flux = deredden(filter_wave, flux, ra, dec, scaling, reddening_law, r_v)

    f1 = integrate_filter(filter_wave, flux, filter_wave, filter_response)
    f2 = integrate_filter(filter_wave, dereddened_flux, filter_wave, filter_response)
    A_ext = -2.5 * np.log10(f1 / f2)

    return A_ext
