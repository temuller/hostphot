import numpy as np
from astropy import wcs
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.nddata.utils import Cutout2D
from scipy.ndimage import label

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


def zeros_to_nan(data: np.ndarray, min_size: int = 1000) -> np.ndarray:
    """Converts large connected regions of exact zeros into NaNs, as they are regions without data
    (e.g. outside the survey footprint), not valid measurements.

    Parameters
    ----------
    data: Image data.
    min_size: Minimum number of connected pixels for a region of zeros to be considered without data.

    Returns
    -------
    data: Image data with the regions without data as NaN.
    """
    zeros = data == 0
    if zeros.sum() < min_size:
        return data
    labels, _ = label(zeros)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0  # pixels that are not zero
    no_data = sizes[labels] >= min_size
    if not no_data.any():
        return data
    data = data.astype(np.float64)  # also for integer images
    data[no_data] = np.nan
    return data


# mask-plane bits of pixels without measured data, e.g. saturated in all the exposures of a coadd
# (HSC; their values are interpolated). Pixels only flagged as interpolated (INTRP) are detector
# defects (e.g. bad columns or cosmic rays), with reasonable interpolated values, so they are kept
SATURATED_MASK_BITS = ["NO_DATA"]


def saturated_mask(hdu: list[fits.ImageHDU]) -> np.ndarray | None:
    """Returns the mask of the pixels without measured data (e.g. saturated and interpolated) from the
    mask plane of an image (``MASK`` extension with the bit definitions in ``MP_*`` keywords, as in
    HSC images), if available.

    Parameters
    ----------
    hdu: Image HDU list.

    Returns
    -------
    mask: Boolean mask (``True`` for the flagged pixels), or ``None`` if there is no mask plane.
    """
    if "MASK" not in hdu or hdu["MASK"].data is None or hdu["MASK"].data.shape != hdu[0].data.shape:
        return None
    header = hdu["MASK"].header
    bits = [header[f"MP_{name}"] for name in SATURATED_MASK_BITS if f"MP_{name}" in header]
    if len(bits) == 0:
        return None
    flag = np.bitwise_or.reduce([1 << int(bit) for bit in bits])
    return (hdu["MASK"].data.astype(np.int64) & flag) > 0


def remove_nan(hdu: list[fits.ImageHDU]) -> list[fits.ImageHDU]:
    """Remove columns and/or rows which have all NaN values (after converting large regions of exact
    zeros, without data, into NaN; see :func:`zeros_to_nan`).
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
    # regions without data filled with zeros (e.g. outside the footprint) are treated as NaN
    data = zeros_to_nan(hdu[0].data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        img_wcs = wcs.WCS(header, naxis=2)

    # first and last columns/rows with data: only the rows and columns without data at the edges
    # are removed (those inside the image, e.g. chip gaps, are kept)
    valid_x = np.where(~np.all(np.isnan(data), axis=0))[0]
    valid_y = np.where(~np.all(np.isnan(data), axis=1))[0]
    if len(valid_x) == 0 or len(valid_y) == 0:
        return hdu  # no data at all
    slices = (slice(valid_y[0], valid_y[-1] + 1), slice(valid_x[0], valid_x[-1] + 1))

    # trim data and update the header with the WCS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        header.update(img_wcs[slices].to_header())
    header["COMMENT"] = "= Trimmed fits file (hostphot)"

    trimmed_hdu = hdu.copy()
    trimmed_hdu[0].data = np.array(data[slices])
    trimmed_hdu[0].header = header

    # trim other HDUs if they exist and have the same shape
    if len(hdu) > 1:
        for i in range(1, len(hdu)):
            if hdu[i].data is not None and hdu[i].data.shape == data.shape:
                trimmed_hdu[i].data = hdu[i].data[slices]

    return trimmed_hdu
