import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import sep
from astropy.io import fits
from astropy import wcs
from astropy.convolution import (Gaussian2DKernel, convolve_fft,
                                 interpolate_replace_nans)

from hostphot._constants import __workdir__
from hostphot.image_cleaning import remove_nan
from hostphot.objects_detect import (extract_objects, find_gaia_objects,
                                        find_catalog_objects, cross_match)
from hostphot.utils import check_survey_validity, pixel2pixel


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
def create_circular_mask(h, w, centre, radius):
    """Creates a circular mask of an image.

    Parameters
    ----------
    h: int
        Image height.
    w: int
        Image width.
    centre: tuple-like
        Centre of the circular mask.
    radius: float
        Radius of the circular mask.

    Returns
    -------
    mask: 2D bool-array
        Circular mask (inside the circle = ``True``).
    """
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - centre[0])**2 + (Y-centre[1])**2)
    mask = dist_from_center <= radius

    return mask

def inside_galaxy(star_center, gal_center, gal_r):
    """Checks whether a star is inside a galaxy.

    Parameters
    ----------
    star_center: tuple-like
        Centre of the star.
    gal_center: tuple-like
        Centre of the galaxy.
    gal_r: float
        Radius to define the galaxy size.

    Returns
    -------
    condition: bool
        ``True`` if the star is inside the galaxy,
        ``False`` otherwise.
    """
    dist_from_center = np.sqrt((star_center[0] - gal_center[0])**2 +
                               (star_center[1] - gal_center[1])**2)
    condition = dist_from_center <= gal_r

    return condition

def mask_image(data, objects, r=5, sigma=20):
    """Masks objects in an image (2D array) by convolving it with
    a 2D Gaussian kernel.

    Parameters
    ----------
    data: 2D array
        Image data.
    objects: array
        Objects extracted with :func:`sep.extract()`.
    r: float, default ``5``
        Scale of the semi-mayor and semi-minor axes
        of the ellipse of the `obejcts`.
    sigma: float, default ``20``
        Standard deviation in pixel units of the 2D Gaussian kernel
        used to convolve the image.

    Returns
    -------
    masked_data: 2D array
        Masked image data.
    """
    mask = np.zeros(data.shape, dtype=bool)
    sep.mask_ellipse(mask, objects['x'], objects['y'],
                     objects['a'], objects['b'],
                     objects['theta'], r=r)

    masked_data = data.copy()
    masked_data[mask] = np.nan
    # mask data by convolving it with a 2D Gaussian kernel
    # with the same sigma in x and y
    kernel = Gaussian2DKernel(sigma)
    masked_data = interpolate_replace_nans(masked_data, kernel,
                                            convolve_fft)

    return masked_data

def create_mask(name, host_ra, host_dec, filt, survey, bkg_sub=False,
                threshold=10, sigma=20, extract_params=False,
                common_params=None, save_plots=True):
    """Calculates the aperture parameters for common aperture.

    Parameters
    ----------
    name: str
        Name of the object to find the path of the fits file.
    host_ra: float
        Host-galaxy Right ascension of the galaxy in degrees.
    host_dec: float
        Host-galaxy Declination of the galaxy in degrees.
    filt: str
        Filter to use to load the fits file.
    survey: str
        Survey to use for the zero-points and correct filter path.
    bkg_sub: bool, default ``False``
        If ``True``, the image gets background subtracted.
    threshold: float, default `10`
        Threshold used by :func:`sep.extract()` to extract objects.
    sigma: float, default ``20``
        Standard deviation in pixel units of the 2D Gaussian kernel
        used to convolve the image.
    extract_params: bool, default ``False``
        If ``True``, returns the parameters listed below.
    common_params: tuple, default ``None``
        Parameters to use for common masking of different filters.
        This are the same as the outputs of this function.
    save_plots: bool, default ``True``
        If ``True``, the mask and galaxy aperture figures are saved.

    Returns
    -------
    **This are only returned if ``extract_params==True``.**
    gal_obj: array
        Galaxy object.
    gal_obj: array
        Non-galaxy objects.
    img_wcs: WCS
        Image's WCS.
    """
    check_survey_validity(survey)

    obj_dir = os.path.join(__workdir__, name)
    fits_file = os.path.join(obj_dir, f'{survey}_{filt}.fits')
    img = fits.open(fits_file)
    img = remove_nan(img)

    header = img[0].header
    data = img[0].data
    img_wcs = wcs.WCS(header, naxis=2)

    data = data.astype(np.float64)
    bkg = sep.Background(data)
    bkg_rms = bkg.globalrms
    if bkg_sub:
        data_sub = np.copy(data - bkg)
    else:
        data_sub = np.copy(data)

    if common_params is None:
        # extract objects
        gal_obj, nogal_objs = extract_objects(data_sub, bkg_rms,
                                              host_ra, host_dec,
                                              threshold, img_wcs)
        # preprocessing: cross-match extracted objects with a catalog
        #cat_coord = find_gaia_objects(host_ra, host_dec, img_wcs)
        cat_coord = find_catalog_objects(host_ra, host_dec, img_wcs)
        nogal_objs = cross_match(nogal_objs, img_wcs, cat_coord)
    else:
        # use objects previously extracted
        # the pixels coordinates are updated accordingly
        gal_obj, nogal_objs, img_wcs0 = common_params
        gal_obj['x'], gal_obj['y'] = pixel2pixel(gal_obj['x'],
                                                gal_obj['y'],
                                                img_wcs0, img_wcs)
        nogal_objs['x'], nogal_objs['y'] = pixel2pixel(nogal_objs['x'],
                                                nogal_objs['y'],
                                                img_wcs0, img_wcs)

    masked_data = mask_image(data_sub, nogal_objs, sigma=sigma)
    img[0].data = masked_data
    outfile = os.path.join(obj_dir, f'masked_{survey}_{filt}.fits')
    img.writeto(outfile, overwrite=True)

    if save_plots:
        outfile = os.path.join(obj_dir,
                                f'masked_{survey}_{filt}.jpg')
        plot_masked_image(data_sub, masked_data,
                            nogal_objs, outfile)

    if extract_params:
        return gal_obj, nogal_objs, img_wcs

def plot_masked_image(data, masked_data, objects, outfile=None):
    """Plots the masked image together with the original image and
    the detected objects.

    Parameters
    ----------
    data: 2D array
        Image data.
    masked_data: 2D array
         Masked image data.
    objects: array
        Objects extracted with :func:`sep.extract()`.
    outfile: str, default ``None``
        If given, path where to save the output figure.
    """
    r = 4  # scale
    fig, ax = plt.subplots(1, 3, figsize=(20, 8))
    m, s = np.nanmean(data), np.nanstd(data)
    for i in range(2):
        ax[i].imshow(data, interpolation='nearest', cmap='gray',
                       vmin=m-s, vmax=m+s, origin='lower')

    for i in range(len(objects)):
        e = Ellipse(xy=(objects['x'][i], objects['y'][i]),
                    width=2*r*objects['a'][i],
                    height=2*r*objects['b'][i],
                    angle=objects['theta'][i] * 180. / np.pi)
        e.set_facecolor('none')
        e.set_edgecolor('red')
        ax[1].add_artist(e)

    ax[2].imshow(masked_data, interpolation='nearest', cmap='gray',
                   vmin=m-s, vmax=m+s, origin='lower')

    ax[0].set_title('Initial Image')
    ax[1].set_title('Detected Objects')
    ax[2].set_title('Masked Image')

    if outfile:
        plt.tight_layout()
        plt.savefig(outfile)
        plt.close(fig)
    else:
        plt.show()
