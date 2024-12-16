from pathlib import Path

from astropy.io import fits
from reproject import reproject_interp
from reproject.mosaicking import reproject_and_coadd

from hostphot._constants import workdir

import warnings
from astropy.utils.exceptions import AstropyWarning

def coadd_images(name: str, filters: str | list, survey: str) -> None:
    """Reprojects and coadds images for the choosen filters for
    common-aperture photometry.

    Parameters
    ----------
    name: Name to be used for finding the images locally.
    filters: Filters to use for the coadd image.
    survey: Survey to use as prefix for the images.

    Examples
    --------
    >>> from hostphot.coadd import coadd_images
    >>> name = 'SN2004eo'
    >>> survey = 'PS1'
    >>> coadd_filters = 'riz'
    >>> coadd_images(name, filters=coadd_filters, survey=survey)  # creates a new fits file
    """
    obj_dir = Path(workdir, name)
    fits_files = [
         obj_dir / survey / f"{survey}_{filt}.fits" for filt in filters
    ]
    hdu_list = []
    for fits_file in fits_files:
        fits_image = fits.open(fits_file)
        hdu_list.append(fits_image[0])

    hdu_list = fits.HDUList(hdu_list)
    # use the last image as reference
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        coadd = reproject_and_coadd(
            hdu_list, fits_image[0].header, reproject_function=reproject_interp
        )
    fits_image[0].data = coadd[0]
    if isinstance(filters, list):
        filters = "".join(filt for filt in filters)
    outfile = obj_dir / survey / f"{survey}_{filters}.fits"
    fits_image.writeto(outfile, overwrite=True)
