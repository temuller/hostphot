import numpy as np
from astropy import wcs
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.nddata.utils import Cutout2D

import warnings
from astropy.utils.exceptions import AstropyWarning


def trim_images(
    fits_files: list[fits.ImageHDU], pos: SkyCoord, size: float
) -> list[fits.ImageHDU]:
    """Trims the size of the given fits images.

    Parameters
    ----------
    fits_files: List of fits images.
    pos: The position of the cutout array's center.
    size: Image size in pixels.

    Returns
    -------
    trimmed_fits_files: List of the trimmed fits images.
    """
    trimmed_fits_files = []
    for fits_file in fits_files:
        data = fits_file[0].data.copy()
        header = fits_file[0].header
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AstropyWarning)
            img_wcs = wcs.WCS(header, naxis=2)

        # trim data and update the header with the WCS
        trimmed_data = Cutout2D(data, pos, size, img_wcs, copy=True)
        fits_file[0].data = trimmed_data.data
        header.update(trimmed_data.wcs.to_header())
        header["COMMENT"] = "= Trimmed fits file (hostphot)"
        fits_file[0].header = header
        trimmed_fits_files.append(fits_file)

    return trimmed_fits_files


def remove_nan(hdu: list[fits.ImageHDU]) -> list[fits.ImageHDU]:
    """Remove columns and/or rows which have all NaN values.
    The WCS is updated accordingly.

    Parameters
    ----------
    image: Header Data Unit
        Fits image with header and data.

    Returns
    -------
    trimmed_hdu: Trimmed image.
    """
    header = hdu[0].header
    data = hdu[0].data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        img_wcs = wcs.WCS(header, naxis=2)

    mask = ~np.all(np.isnan(data), axis=0)
    x_size = np.sum(np.ones_like(data[0])[mask])
    x_pos = np.median(np.arange(data.shape[1])[mask])

    mask = ~np.all(np.isnan(data), axis=1)
    y_size = np.sum(np.ones_like(data[:, 0])[mask])
    y_pos = np.median(np.arange(data.shape[0])[mask])

    # trim data and update the header with the WCS
    trimmed_data = Cutout2D(
        data,
        (x_pos, y_pos),
        (y_size, x_size),  # has to be (ny, nx)
        wcs=img_wcs,
        copy=True,
    )
    header.update(trimmed_data.wcs.to_header())
    header["COMMENT"] = "= Trimmed fits file (hostphot)"

    trimmed_hdu = hdu.copy()
    trimmed_hdu[0].data = trimmed_data.data
    trimmed_hdu[0].header = header

    return trimmed_hdu
