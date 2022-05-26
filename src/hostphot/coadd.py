import os

from astropy.io import fits
from reproject import reproject_interp
from reproject.mosaicking import reproject_and_coadd

from hostphot._constants import __workdir__

#----------------------------------------
def _choose_workdir(workdir):
    """Updates the work directory.

    Parameters
    ----------
    workdir: str
        Path to the work directory.
    """
    global __workdir__
    __workdir__ = workdir

#----------------------------------------

def coadd_images(name, filters='riz', survey='PS1'):
    """Reprojects and coadds images for the choosen filters for
    common-aperture photometry.

    Parameters
    ----------
    name: str
        Name to be used for finding the images locally.
    filters: str, default ``riz``
        Filters to use for the coadd image.
    survey: str, default ``PS1``
        Survey to use as prefix for the images.
    """
    obj_dir = os.path.join(__workdir__, name)
    fits_files = [os.path.join(obj_dir, f'{survey}_{filt}.fits')
                                                for filt in filters]

    hdu_list = []
    for fits_file in fits_files:
        fits_image = fits.open(fits_file)
        hdu_list.append(fits_image[0])

    hdu_list = fits.HDUList(hdu_list)
    # use the last image as reference
    coadd = reproject_and_coadd(hdu_list, fits_image[0].header,
                                reproject_function=reproject_interp)
    fits_image[0].data = coadd[0]
    outfile = os.path.join(obj_dir, f'{survey}_{filters}.fits')
    fits_image.writeto(outfile, overwrite=True)
