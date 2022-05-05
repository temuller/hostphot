import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import sep
from astropy.convolution import (Gaussian2DKernel,
                                 interpolate_replace_nans)

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
        Circular mask (inside the circle = `True`).
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
        `True` if the star is inside the galaxy,
        `False` otherwise.
    """
    dist_from_center = np.sqrt((star_center[0] - gal_center[0])**2 +
                               (star_center[1] - gal_center[1])**2)
    condition = dist_from_center <= gal_r

    return condition

def mask_image(data, objects, r=4, sigma=10, plot=False):
    """Masks objects in an image (2D array) by convolving it with
    a 2D Gaussian kernel.

    Parameters
    ----------
    data: 2D array
        Image data.
    objects: array
        Objects extracted with `sep.extract()`.
    r: float
        Scale of the semi-mayor and semi-minor axes
        of the ellipse of the `obejcts`.
    sigma: float
        Standard deviation in pixel units of the 2D Gaussian kernel
        used to convolve the image.
    plot: bool, default `False`
        If `True`, the output is plotted.

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
    masked_data = interpolate_replace_nans(masked_data, kernel)

    return masked_data

def plot_masked_image(data, masked_data, objects):
    """Plots the masked image together with the original image and
    the detected objects.

    Parameters
    ----------
    data: 2D array
        Image data.
    masked_data: 2D array
         Masked image data.
    objects: array
        Objects extracted with `sep.extract()`.
    """
    fig, ax = plt.subplots(1, 3, figsize=(20, 8))
    for i in range(2):
        ax[i].imshow(data, interpolation='nearest', cmap='gray',
                       vmin=m-s, vmax=m+s, origin='lower')

    for i in objs_id:
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
    ax[0].set_title('Detected Objects')
    ax[0].set_title('Masked Image')
    plt.tight_layout()
    plt.savefig(plot_output)
    plt.close(fig)
