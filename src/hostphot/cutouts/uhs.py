import astropy.units as u
from astropy.io import fits

from hostphot.cutouts.ukidss import get_WFCAM_images


def get_UHS_images(ra: float, dec: float, size: float | u.Quantity = 3, filters: str = "JHK") -> list[fits.HDUList]:
    """Gets UKIRT Hemisphere Survey (UHS) fits images for the given coordinates and filters.

    The images are WFCAM stacks from the WFCAM Science Archive (data release 3), processed and
    calibrated in the same way as those of UKIDSS.

    Parameters
    ----------
    ra: Right Ascension in degrees.
    dec: Declination in degrees.
    size: Image size. If a float is given, the units are assumed to be arcmin.
    filters: Filters to use. If ``None``, uses ``JHK``.

    Returns
    -------
    fits_files: List of fits images.
    """
    return get_WFCAM_images(ra, dec, size, filters, survey="UHS", database="UHSDR3")
