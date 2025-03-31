import numpy as np
from pathlib import Path

from astropy import wcs
from astropy.io import fits

from hostphot._constants import workdir
from hostphot.utils import check_work_dir
from hostphot.surveys_utils import check_JWST_filters

import warnings
from astropy.utils.exceptions import AstropyWarning


def update_JWST_header(hdu: fits.hdu.ImageHDU) -> None:
    """Updates the JWST image header with the necessary keywords.

    Parameters
    ----------
    hdu : JWST FITS image.
    """
    # get WCS - SCI is hdu[1] - hostphot always assumes hdu[0] is used, so need to move them over
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        img_wcs = wcs.WCS(hdu[1].header)
    hdu[0].header.update(img_wcs.to_header())
    hdu[0].header["PIXAR_SR"] = hdu[1].header["PIXAR_SR"]
    hdu[0].data = hdu[1].data
    # add zeropoints
    # https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-performance/nircam-absolute-flux-calibration-and-zeropoints
    pixar_sr = hdu[0].header["PIXAR_SR"]
    hdu[0].header["MAGZP"] = (
       -6.10 - (2.5 * np.log10(pixar_sr))
    )

def set_JWST_image(file: str, filt: str, name: str) -> None:
    """Moves a previously downloaded JWST image into the work directory.

    The image's header is updated with the necessary keywords to obtain
    photometry and is also moved under the objects directory inside the
    work directory.

    JWST images take very long to download, so the user might prefer to
    download the images manually and then use this function to include
    the image into the workflow.

    Parameters
    ----------
    file: JWST image to use.
    filt: JWST filter, e.g. ``NIRCam_F150W``.
    name: Object's name.
    """
    # check output directory
    check_work_dir(workdir)
    obj_dir = Path(workdir, name, "JWST")
    if obj_dir.is_dir() is False:
        obj_dir.mkdir(parents=True, exist_ok=True)
    # update header and save file
    hdu = fits.open(file)
    update_JWST_header(hdu)
    outfile = obj_dir / f"JWST_{filt}.fits"
    hdu.writeto(outfile, overwrite=True)