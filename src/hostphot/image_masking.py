import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import sep
from astropy.io import fits
from astropy import wcs, units as u
from astropy.coordinates import SkyCoord, concatenate
from astropy.convolution import (
    Gaussian2DKernel,
    convolve_fft,
    interpolate_replace_nans,
)

from hostphot._constants import workdir
from hostphot.image_cleaning import remove_nan
from hostphot.objects_detect import (
    extract_objects,
    find_gaia_objects,
    find_catalog_objects,
    cross_match,
)
from hostphot.utils import (
    check_survey_validity,
    pixel2pixel,
    update_axislabels,
    survey_pixel_scale,
)

import warnings
from astropy.utils.exceptions import AstropyWarning


# ----------------------------------------
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
    sep.mask_ellipse(
        mask,
        objects["x"],
        objects["y"],
        objects["a"],
        objects["b"],
        objects["theta"],
        r=r,
    )

    masked_data = data.copy()
    masked_data[mask] = np.nan
    # mask data by convolving it with a 2D Gaussian kernel
    # with the same sigma in x and y
    kernel = Gaussian2DKernel(sigma)
    masked_data = interpolate_replace_nans(masked_data, kernel, convolve_fft)

    return masked_data


def create_mask(
    name,
    host_ra,
    host_dec,
    filt,
    survey,
    ra=None,
    dec=None,
    bkg_sub=False,
    threshold=15,
    sigma=8,
    crossmatch=False,
    extract_params=False,
    common_params=None,
    save_plots=True,
):
    """Calculates the aperture parameters for common aperture.

    Parameters
    ----------
    name: str
        Name of the object to find the path of the fits file.
    host_ra: float
        Host-galaxy right ascension in degrees.
    host_dec: float
        Host-galaxy declination in degrees.
    filt: str
        Filter to use to load the fits file.
    survey: str
        Survey to use for the zero-points and correct filter path.
    ra: float, default ``None``
       Right ascension of an object, in degrees. Used for plotting.
    dec: float, default ``None``
       Declination of an object, in degrees. Used for plotting.
    bkg_sub: bool, default ``False``
        If ``True``, the image gets background subtracted.
    threshold: float, default `15`
        Threshold used by :func:`sep.extract()` to extract objects.
    sigma: float, default ``8``
        Standard deviation in pixel units of the 2D Gaussian kernel
        used to convolve the image.
    crossmatch: bool, default ``False``
        If ``True``, the detected objects are cross-matched with a
        Gaia catalog.
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
    pixel_scale: float
        Pixel scale for the survey.
    """
    check_survey_validity(survey)

    obj_dir = os.path.join(workdir, name)
    fits_file = os.path.join(obj_dir, f"{survey}_{filt}.fits")
    img = fits.open(fits_file)
    img = remove_nan(img)

    header = img[0].header
    data = img[0].data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        img_wcs = wcs.WCS(header, naxis=2)

    data = data.astype(np.float64)
    bkg = sep.Background(data)
    bkg_rms = bkg.globalrms
    if bkg_sub:
        data_sub = np.copy(data - bkg)
    else:
        data_sub = np.copy(data)

    pixel_scale = survey_pixel_scale(survey)

    if common_params is None:
        # extract objects
        gal_obj, nogal_objs = extract_objects(
            data_sub, bkg_rms, host_ra, host_dec, threshold, img_wcs
        )
        # preprocessing: cross-match extracted objects with a catalog
        # using two Gaia catalogs as they do not always include the
        # same objects.
        if crossmatch:
            cat_coord1 = find_gaia_objects(host_ra, host_dec, img_wcs)
            cat_coord2 = find_catalog_objects(host_ra, host_dec, img_wcs)
            cat_coord = concatenate([cat_coord1, cat_coord2])
            nogal_objs = cross_match(nogal_objs, img_wcs, cat_coord)
    else:
        # use objects previously extracted
        # the pixels coordinates are updated accordingly
        gal_obj, nogal_objs, img_wcs0, pixel_scale0 = common_params
        # make copies to prevent altering the initial values
        gal_obj = gal_obj.copy()
        nogal_objs = nogal_objs.copy()
        scale = pixel_scale0 / pixel_scale

        gal_obj["x"], gal_obj["y"] = pixel2pixel(
            gal_obj["x"], gal_obj["y"], img_wcs0, img_wcs
        )
        gal_obj["a"] *= scale
        gal_obj["b"] *= scale

        nogal_objs["x"], nogal_objs["y"] = pixel2pixel(
            nogal_objs["x"], nogal_objs["y"], img_wcs0, img_wcs
        )
        nogal_objs["a"] *= scale
        nogal_objs["b"] *= scale

    masked_data = mask_image(data_sub, nogal_objs, sigma=sigma)
    img[0].data = masked_data
    outfile = os.path.join(obj_dir, f"masked_{survey}_{filt}.fits")
    img.writeto(outfile, overwrite=True)

    if save_plots:
        outfile = os.path.join(obj_dir, f"masked_{survey}_{filt}.jpg")
        plot_masked_image(
            data_sub, masked_data, nogal_objs, img_wcs, ra, dec, outfile
        )

    if extract_params:
        return gal_obj, nogal_objs, img_wcs, pixel_scale


def plot_masked_image(
    data, masked_data, objects, img_wcs, ra=None, dec=None, outfile=None
):
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
    img_wcs: WCS
        Image's WCS.
    ra: float, default ``None``
       Right ascension of an object, in degrees. Used for plotting the position of the object.
    dec: float, default ``None``
       Declination of an object, in degrees. Used for plotting the position of the object.
    outfile: str, default ``None``
        If given, path where to save the output figure.
    """
    r = 4  # scale
    m, s = np.nanmean(data), np.nanstd(data)

    fig = plt.figure(figsize=(24, 10))
    ax0 = plt.subplot(131, projection=img_wcs)
    ax1 = plt.subplot(132, projection=img_wcs)
    ax2 = plt.subplot(133, projection=img_wcs)
    axes = [ax0, ax1, ax2]

    overlays = []
    for ax in axes:
        update_axislabels(ax)
        ax.imshow(
            data,
            interpolation="nearest",
            cmap="gray",
            vmin=m - s,
            vmax=m + s,
            origin="lower",
        )
    for ax in axes[1:]:
        ax.coords[1].set_axislabel("")
        ax.coords[1].set_ticklabel_visible(False)

    for i in range(len(objects)):
        e = Ellipse(
            xy=(objects["x"][i], objects["y"][i]),
            width=2 * r * objects["a"][i],
            height=2 * r * objects["b"][i],
            angle=objects["theta"][i] * 180.0 / np.pi,
        )
        e.set_facecolor("none")
        e.set_edgecolor("red")
        e.set_linewidth(1.5)
        ax1.add_artist(e)

    ax2.imshow(
        masked_data,
        interpolation="nearest",
        cmap="gray",
        vmin=m - s,
        vmax=m + s,
        origin="lower",
    )

    ax0.set_title("Initial Image", fontsize=24)
    ax1.set_title("Detected Objects", fontsize=24)
    ax2.set_title("Masked Image", fontsize=24)

    if (ra is not None) and (dec is not None):
        coord = SkyCoord(
            ra=ra, dec=dec, unit=(u.degree, u.degree), frame="icrs"
        )
        x, y = img_wcs.world_to_pixel(coord)
        for ax in axes:
            ax.scatter(x, y, marker="*", s=100, c="g", edgecolor="gold")

    if outfile:
        plt.tight_layout()
        plt.savefig(outfile, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
