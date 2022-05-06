import os

from astropy.io import fits
import reproject
from reproject.mosaicking import reproject_and_coadd

def coadd_images(name, filters='riz', work_dir='', survey='PS1'):
    """Reprojects and coadds images for the choosen filters for
    common-aperture photometry.

    Parameters
    ----------
    name: str
        Name to be used for finding the images locally.
    filters: str, default `riz`
        Filters to use for the coadd image.
    work_dir: str, default ''
        Working directory where to find the objects'
        directories with the images. Default, current directory.
    survey: str, default `PS1`
        Survey to use as prefix for the images.

    Returns
    -------
    outfile: str
        Path to the fits file with the coadded images.
    """

    init_dir = os.path.abspath('.')
    dir = os.path.join(work_dir, name)
    fits_files = [os.path.join(dir, f'{survey}_{filt}.fits') for filt in filters]

    hdu_list = []
    for fits_file in fits_files:
        fits_image = fits.open(fits_file)
        hdu_list.append(fits_image[0])

    hdu_list = fits.HDUList(hdu_list)
    # use the last image as reference
    coadd = reproject_and_coadd(hdu_list, fits_image[0].header,
                                reproject_function=reproject.reproject_interp)
    fits_image[0].data = coadd[0]
    outfile = os.path.join(sn_dir, f'{survey}_{filters}.fits')
    fits_image.writeto(outfile, overwrite=True)

    return outfile
