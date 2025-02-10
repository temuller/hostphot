from mocpy import MOC
import astropy.units as u
from astropy.coordinates import SkyCoord
import warnings

import hostphot
from pathlib import Path
mapsdir = Path(hostphot.__path__[0], 'moc/maps')

    
def contains_coords(ra: float, dec: float, survey: str) -> bool:
    """Checks whether the given coordinates overlap with a survey footprint.

    Parameters
    ----------
    ra: right ascension in degrees.
    dec: declination in degrees.
    survey: survey name.

    Returns
    -------
    overlap: whether the coordinates overlap.
    """
    # load MOC map
    moc_file = [file for file in mapsdir.glob(f"{survey}*.fits")][0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        moc_map = MOC.load(moc_file)
    # check intersection with given coordinates
    coord = SkyCoord(ra * u.deg, dec * u.deg)
    overlap = moc_map.contains_skycoords(coord)
    if overlap:
        return True
    else:
        return False