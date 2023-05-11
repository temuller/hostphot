import os
import pickle
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import aplpy

plt.rcParams["mathtext.fontset"] = "cm"

import sep
from astropy.io import fits
from astropy import wcs
from astropy.coordinates import concatenate
from astropy.convolution import (
    Gaussian2DKernel,
    convolve_fft,
    interpolate_replace_nans,
)

from hostphot._constants import workdir, font_family
from hostphot.image_cleaning import remove_nan
from hostphot.objects_detect import (
    extract_objects,
    find_gaia_objects,
    find_catalog_objects,
    cross_match,
)
from hostphot.utils import (
    check_survey_validity,
    bkg_surveys,
    adapt_aperture,
    suppress_stdout,
)

import warnings
from astropy.utils.exceptions import AstropyWarning


# ----------------------------------------
def mask_image(data, objects, r=6, sigma=20):
    """Masks objects in an image (2D array) by convolving it with
    a 2D Gaussian kernel.

    Parameters
    ----------
    data: ndarray
        Image data.
    objects: array
        Objects extracted with :func:`sep.extract()`.
    r: float, default ``6``
        Scale of the semi-mayor and semi-minor axes
        of the ellipse of the `objects`.
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
    bkg_sub=None,
    threshold=15,
    sigma=8,
    r=6,
    crossmatch=False,
    gal_dist_thresh=-1,
    extract_params=False,
    common_params=None,
    save_plots=True,
    save_mask_params=True,
):
    """Calculates the aperture parameters to mask detected sources.

    Parameters
    ----------
    name: str
        Name of the object to find the path of the fits file.
    host_ra: float
        Host-galaxy right ascension in degrees.
    host_dec: float
        Host-galaxy declination in degrees.
    filt: str or list
        Filter to use to load the fits file. List is commonly used for coadds.
    survey: str
        Survey to use for the zero-points and correct filter path.
    ra: float, default ``None``
       Right ascension of an object, in degrees. Used for plotting.
    dec: float, default ``None``
       Declination of an object, in degrees. Used for plotting.
    bkg_sub: bool, default ``None``
        If ``True``, the image gets background subtracted. By default, only
        the images that need it get background subtracted (WISE, 2MASS and
        VISTA).
    threshold: float, default `15`
        Threshold used by :func:`sep.extract()` to extract objects.
    sigma: float, default ``8``
        Standard deviation in pixel units of the 2D Gaussian kernel
        used to convolve the image.
    r: float, default ``6``
        Scale of the aperture size for the sources to be masked.
    crossmatch: bool, default ``False``
        If ``True``, the detected objects are cross-matched with a
        Gaia catalog.
    crossmatch: bool, default ``False``
        If ``True``, the detected objects are cross-matched with a
        Gaia catalog.
    gal_dist_thresh: float, default ``-1``.
        Distance in arcsec to crossmatch the galaxy coordinates with a detected object,
        where the object nearest to the galaxy position is considered as the galaxy (within
        the given threshold). If no objects are found within the given distance threshold,
        the galaxy is considered as not found and a warning is printed. If a non-positive value
        is given, the threshold is considered as infinite, i.e. the closest detected object is
        considered as the galaxy (default option).
    extract_params: bool, default ``False``
        If ``True``, returns the parameters listed below.
    common_params: tuple, default ``None``
        Parameters to use for common masking of different filters.
        These are the same as the outputs of this function.
    save_plots: bool, default ``True``
        If ``True``, the mask and galaxy aperture figures are saved.
    save_mask_params: bool, default `True`
        If `True`, the extracted mask parameters are saved into a pickle file.

    Returns
    -------
    **This are only returned if ``extract_params==True``.**
    gal_obj: array
        Galaxy object.
    nogal_obj: array
        Non-galaxy objects.
    img_wcs: WCS
        Image's WCS.
    flip: bool
        Whether to flip the orientation of the
        aperture. Only used for DES images.
    """
    check_survey_validity(survey)
    if isinstance(filt, list):
        filt = "".join(f for f in filt)

    obj_dir = os.path.join(workdir, name)
    fits_file = os.path.join(obj_dir, f"{survey}_{filt}.fits")
    hdu = fits.open(fits_file)
    hdu = remove_nan(hdu)

    header = hdu[0].header
    data = hdu[0].data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        img_wcs = wcs.WCS(header, naxis=2)

    data = data.astype(np.float64)
    bkg = sep.Background(data)
    bkg_rms = bkg.globalrms
    if (bkg_sub is None and survey in bkg_surveys) or bkg_sub is True:
        data_sub = np.copy(data - bkg)
    else:
        data_sub = np.copy(data)

    if common_params is None:
        # extract objects
        gal_obj, nogal_objs = extract_objects(
            data_sub,
            bkg_rms,
            host_ra,
            host_dec,
            threshold,
            img_wcs,
            gal_dist_thresh,
        )
        # preprocessing: cross-match extracted objects with a catalog
        # using two Gaia catalogs as they do not always include the
        # same objects.
        if crossmatch:
            cat_coord1 = find_gaia_objects(host_ra, host_dec)
            cat_coord2 = find_catalog_objects(host_ra, host_dec)
            cat_coord = concatenate([cat_coord1, cat_coord2])
            nogal_objs = cross_match(nogal_objs, img_wcs, cat_coord)
    else:
        # use objects previously extracted
        # the aperture/ellipse parameters are updated accordingly
        gal_obj, nogal_objs, master_img_wcs, flip2 = common_params

        if survey == "DES":
            flip = True
        else:
            flip = False
        if flip == flip2:
            flip_ = False
        else:
            flip_ = True

        gal_obj, _ = adapt_aperture(gal_obj, master_img_wcs, img_wcs, flip_)
        nogal_objs, _ = adapt_aperture(nogal_objs, master_img_wcs, img_wcs, flip_)

    masked_data = mask_image(data_sub, nogal_objs, r=r, sigma=sigma)
    masked_hdu = deepcopy(hdu)
    masked_hdu[0].data = masked_data
    outfile = os.path.join(obj_dir, f"masked_{survey}_{filt}.fits")
    masked_hdu.writeto(outfile, overwrite=True)

    if save_plots:
        outfile = os.path.join(obj_dir, f"masked_{survey}_{filt}.jpg")
        title = f"{name}: {survey}-${filt}$"
        plot_masked_image(
            hdu,
            masked_hdu,
            nogal_objs,
            gal_obj,
            r,
            host_ra,
            host_dec,
            ra,
            dec,
            title,
            outfile,
        )

    if survey == "DES":
        flip = True
    else:
        flip = False

    if save_mask_params is True:
        outfile = os.path.join(
            obj_dir, f"{survey}_{filt}_mask_parameters.pickle"
        )
        with open(outfile, "wb") as fp:
            mask_parameters = gal_obj, nogal_objs, img_wcs, flip
            pickle.dump(mask_parameters, fp, protocol=4)

    if extract_params:
        return gal_obj, nogal_objs, img_wcs, flip


def plot_masked_image(
    hdu,
    masked_hdu,
    objects,
    gal_obj=None,
    r=6,
    host_ra=None,
    host_dec=None,
    ra=None,
    dec=None,
    title=None,
    outfile=None,
):
    """Plots the masked image together with the original image and
    the detected objects.

    Parameters
    ----------
    data: ndarray
        Image data.
    masked_data: 2D array
         Masked image data.
    objects: array
        Objects extracted with :func:`sep.extract()`.
    img_wcs: WCS
        Image's WCS.
    gal_obj: array, default ``None``
        Galaxy object.
    r: float, default ``6``
        Scale of the aperture size for the masked sources.
    host_ra: float, default ``None``
       Right ascension of the galaxy, in degrees. Used for plotting the position of the galaxy.
    host_dec: float, default ``None``
       Declination of the galaxy, in degrees. Used for plotting the position of the galaxy.
    ra: float, default ``None``
       Right ascension of an object, in degrees. Used for plotting the position of the object.
    dec: float, default ``None``
       Declination of an object, in degrees. Used for plotting the position of the object.
    title: str, default ``None``
        Title of the image
    outfile: str, default ``None``
        If given, path where to save the output figure.
    """
    figure = plt.figure(figsize=(30, 12))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        fig1 = aplpy.FITSFigure(hdu, figure=figure, subplot=(1, 3, 1))
        fig2 = aplpy.FITSFigure(hdu, figure=figure, subplot=(1, 3, 2))
        fig3 = aplpy.FITSFigure(masked_hdu, figure=figure, subplot=(1, 3, 3))

    fig2.tick_labels.hide_y()
    fig2.axis_labels.hide_y()
    fig3.tick_labels.set_yposition("right")
    fig3.axis_labels.set_yposition("right")

    figures = [fig1, fig2, fig3]
    titles = ["Initial Image", "Detected Sources", "Masked Image"]
    for i, fig in enumerate(figures):
        fig.set_theme("publication")
        fig.set_title(titles[i], **{"family": font_family, "size": 24})
        with suppress_stdout():
            fig.show_grayscale(stretch="arcsinh")

        # ticks
        fig.tick_labels.set_font(**{"family": font_family, "size": 18})
        fig.tick_labels.set_xformat("dd.dd")
        fig.tick_labels.set_yformat("dd.dd")
        fig.ticks.set_length(6)

        fig.axis_labels.set_font(**{"family": font_family, "size": 18})

    # galaxy markers
    fig2.show_markers(
        host_ra,
        host_dec,
        edgecolor="k",
        facecolor="r",
        alpha=0.7,
        marker="P",
        s=200,
        label="Given galaxy",
    )
    fig2.show_markers(
        gal_obj["x"][0],
        gal_obj["y"][0],
        edgecolor="k",
        facecolor="r",
        alpha=0.7,
        marker="X",
        s=200,
        label="Identified galaxy",
        coords_frame="pixel",
    )
    # galaxy pseudo-aperture
    fig2.show_ellipses(
        gal_obj["x"][0],
        gal_obj["y"][0],
        2 * r * gal_obj["a"][0],
        2 * r * gal_obj["b"][0],
        gal_obj["theta"][0] * 180.0 / np.pi,
        coords_frame="pixel",
        linewidth=3,
        edgecolor="r",
    )

    # other sources markers
    fig2.show_ellipses(
        objects["x"],
        objects["y"],
        2 * r * objects["a"],
        2 * r * objects["b"],
        gal_obj["theta"] * 180.0 / np.pi,
        coords_frame="pixel",
        linewidth=2,
        edgecolor="orangered",
        linestyle="dotted",
    )

    # SN marker
    if (ra is not None) and (dec is not None):
        for fig in figures[1:]:
            fig.show_markers(
                ra,
                dec,
                edgecolor="k",
                facecolor="aqua",
                marker="*",
                s=200,
                label="SN",
            )

    fig2.ax.legend(
        fancybox=True, framealpha=1, prop={"size": 20, "family": font_family}
    )

    # title
    length = len(title) - 2
    text = fig1.ax.text(
        0.022 * length,
        0.06,
        title,
        fontsize=28,
        horizontalalignment="center",
        verticalalignment="center",
        transform=fig1.ax.transAxes,
        font=font_family,
    )
    text.set_bbox(
        dict(facecolor="white", edgecolor="white", alpha=0.9, boxstyle="round")
    )

    if outfile:
        plt.savefig(outfile, bbox_inches="tight")
        plt.close(figure)
    else:
        plt.show()
