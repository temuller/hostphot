import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from copy import deepcopy
import matplotlib.pyplot as plt
import aplpy

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
from .cleaning import remove_nan
from .objects_detection import (
    extract_objects,
    find_gaia_objects,
    find_catalog_objects,
    cross_match,
)
from hostphot.utils import suppress_stdout, store_input
from hostphot.photometry.image_utils import adapt_aperture
from hostphot.surveys_utils import (
    check_survey_validity,
    bkg_surveys,
    flipped_surveys,
)

import warnings
from astropy.utils.exceptions import AstropyWarning

plt.rcParams["mathtext.fontset"] = "cm"


def mask_image(
    data: np.ndarray,
    objects: np.ndarray,
    r: float | np.ndarray = 6,
    sigma: float | np.ndarray = 8,
) -> np.ndarray:
    """Masks objects in an image (2D array) by convolving it with
    a 2D Gaussian kernel.

    Parameters
    ----------
    data: Image data.
    objects: Objects extracted with :func:`sep.extract()`.
    r: Scale of the semi-mayor and semi-minor axes
        of the ellipse of the `objects`.
    sigma: Standard deviation in pixel units of the 2D Gaussian kernel
        used to convolve the image.

    Returns
    -------
    masked_data: Masked image data.
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
    name: str,
    host_ra: float,
    host_dec: float,
    filt: str,
    survey: str,
    ra: Optional[float] = None,
    dec: Optional[float] = None,
    bkg_sub: Optional[bool] = None,
    threshold: float = 15,
    sigma: float | np.ndarray = 8,
    r: float | np.ndarray = 6,
    crossmatch: bool = False,
    gal_dist_thresh: float = -1,
    deblend_cont: float = 0.005,
    ref_filt: Optional[str] = None,
    ref_survey: Optional[str] = None,
    save_plots: bool = False,
    save_mask_params: bool = False,
    save_input: bool = True,
) -> None:
    """Calculates the aperture parameters to mask detected sources.

    Parameters
    ----------
    name: Name of the object to find the path of the fits file.
    host_ra: Host-galaxy right ascension in degrees.
    host_dec: Host-galaxy declination in degrees.
    filt: Filter to use to load the fits file. List is commonly used for coadds.
    survey: Survey to use for the zero-points and correct filter path.
    ra: Right ascension of an object, in degrees. Used for plotting.
    dec: Declination of an object, in degrees. Used for plotting.
    bkg_sub: If ``True``, the image gets background subtracted. By default, only
        the images that need it get background subtracted (WISE, 2MASS and
        VISTA).
    threshold: Threshold used by :func:`sep.extract()` to extract objects.
    sigma: Standard deviation in pixel units of the 2D Gaussian kernel
        used to convolve the image.
    r: Scale of the aperture size for the sources to be masked.
    crossmatch: If ``True``, the detected objects are cross-matched with a
        Gaia catalog.
    crossmatch: If ``True``, the detected objects are cross-matched with a
        Gaia catalog.
    gal_dist_thresh: Distance in arcsec to crossmatch the galaxy coordinates with a detected object,
        where the object nearest to the galaxy position is considered as the galaxy (within
        the given threshold). If no objects are found within the given distance threshold,
        the galaxy is considered as not found and a warning is printed. If a non-positive value
        is given, the threshold is considered as infinite, i.e. the closest detected object is
        considered as the galaxy (default option).
    deblend_cont : Minimum contrast ratio used for object deblending. Default is 0.005.
        To entirely disable deblending, set to 1.0.
    ref_filt: Reference filter (or coadd filters) from which to use the mask parameters. Note that the parameters
        need to have been previously saved with ``save_mask_params=True``.
    ref_survey: Reference survey from which to use the mask parameters. Note that the parameters
        need to have been previously saved with ``save_mask_params=True``.
    save_plots: If ``True``, the mask and galaxy aperture figures are saved.
    save_mask_params: If `True`, the extracted mask parameters are saved for later use.
    save_input: Whether to save the input parameters.
    """
    input_params = locals()  # dictionary
    check_survey_validity(survey)
    if isinstance(filt, list):
        filt = "".join(f for f in filt)

    obj_dir = Path(workdir, name)
    fits_file = obj_dir / survey / f"{survey}_{filt}.fits"
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

    # save input parameters
    if save_input is True:
        inputs_file = obj_dir / survey / "input_masking_parameters.csv"
        store_input(input_params, inputs_file)

    if (ref_filt is None) & (ref_survey is None):
        # extract objects
        gal_obj, nongal_objs = extract_objects(
            data_sub,
            bkg_rms,
            host_ra,
            host_dec,
            threshold,
            img_wcs,
            gal_dist_thresh,
            deblend_cont,
        )
        # preprocessing: cross-match extracted objects with a catalog
        # using two Gaia catalogs as they do not always include the
        # same objects.
        if crossmatch:
            cat_coord1 = find_gaia_objects(host_ra, host_dec)
            cat_coord2 = find_catalog_objects(host_ra, host_dec)
            cat_coord = concatenate([cat_coord1, cat_coord2])
            nongal_objs = cross_match(nongal_objs, img_wcs, cat_coord)
    else:
        # use objects previously extracted
        # the aperture/ellipse parameters are updated accordingly
        gal_obj, nongal_objs, master_img_wcs, sigma, r, flip2 = load_mask_params(
            name, ref_filt, ref_survey
        )

        # cross-survey mask: need to adapt some values
        # flipping images
        if survey in flipped_surveys:
            flip = True
        else:
            flip = False
        if flip == flip2:
            flip_ = False
        else:
            flip_ = True
        # adapt sizes
        gal_obj, _ = adapt_aperture(gal_obj, master_img_wcs, img_wcs, flip_)
        nongal_objs, conv_factor = adapt_aperture(
            nongal_objs, master_img_wcs, img_wcs, flip_
        )
        sigma /= conv_factor

    #masked_data = mask_image(data_sub, nongal_objs, r=r, sigma=sigma)
    if len(nongal_objs) > 0:
        masked_data = mask_image(data_sub, nongal_objs, r=r, sigma=sigma)
    else:
        masked_data = data_sub.copy()
    masked_hdu = deepcopy(hdu)
    masked_hdu[0].data = masked_data
    outfile = obj_dir / survey / f"{survey}_{filt}_masked.fits"
    masked_hdu.writeto(outfile, overwrite=True)

    if survey in flipped_surveys:
        flip = True
    else:
        flip = False

    if save_mask_params is True:
        # save detected objects and masking parameters
        objects_df = pd.concat([pd.DataFrame(gal_obj), pd.DataFrame(nongal_objs)])
        objects_df["sigma"] = sigma
        objects_df["r"] = r
        objects_df["flip"] = flip
        objects_df["filt"] = filt
        objects_df["survey"] = survey
        outfile = obj_dir / survey / f"mask_parameters_{filt}.csv"
        objects_df.to_csv(outfile, index=False)

    if save_plots is True:
        outfile = obj_dir / survey / f"{survey}_{filt}_masked.jpg"
        title = fr"{name}: {survey}-${filt}$"
        plot_masked_image(
            hdu,
            masked_hdu,
            gal_obj,
            nongal_objs,
            r,
            host_ra,
            host_dec,
            ra,
            dec,
            title,
            outfile,
        )
    hdu.close()


def load_mask_params(
    name: str, filt: str | list, survey: str
) -> tuple[np.ndarray, np.ndarray, wcs.WCS, np.ndarray, float, bool]:
    """Loads previously saved mask parameters.

    Parameters
    ----------
    name: Name of the object to find the path of the mask-parameters file.
    filt: Name of the filter used to create the mask parameters. Coadds are
        also valid.
    survey: Survey name to be used.

    Returns
    -------
    gal_obj, nongal_objs, img_wcs, sigma, r, flip: Mask parameters.
    """
    if isinstance(filt, list):
        filt = ''.join(f for f in filt)
    obj_dir = Path(workdir, name)
    mask_params_file = obj_dir / survey / f"mask_parameters_{filt}.csv"
    objects_df = pd.read_csv(mask_params_file)
    
    # split parameters
    sigma = objects_df.pop("sigma").values[0]
    r = objects_df.pop("r").values[1:]  # remove host-galaxy row
    flip = objects_df.pop("flip").values[0]
    # remove unused columns
    _ = objects_df.pop("filt")
    _ = objects_df.pop("survey")
    # DataFrame to structured/record array
    objects = objects_df.to_records()
    gal_obj = objects[:1]
    nongal_objs = objects[1:]
    
    # load image WCS
    fits_file = obj_dir / survey / f"{survey}_{filt}.fits"
    hdu = fits.open(fits_file)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        img_wcs = wcs.WCS(hdu[0].header, naxis=2)
        
    return gal_obj, nongal_objs, img_wcs, sigma, r, flip


def plot_masked_image(
    hdu: list[fits.ImageHDU],
    masked_hdu: list[fits.ImageHDU],
    gal_obj: np.ndarray,
    objects: np.ndarray,
    r: float | np.ndarray = 6,
    host_ra: Optional[float] = None,
    host_dec: Optional[float] = None,
    ra: Optional[float] = None,
    dec: Optional[float] = None,
    title: Optional[str] = None,
    outfile: Optional[float] = None,
) -> None:
    """Plots the masked image together with the original image and
    the detected objects.

    Parameters
    ----------
    data: Image FITS file.
    masked_data: Masked image FITS file.
    objects: Objects extracted with :func:`sep.extract()`.
    img_wcs: Image's WCS.
    gal_obj: Galaxy object.
    r: Scale of the aperture size for the masked sources.
    host_ra: Right ascension of the galaxy, in degrees. Used for plotting the position of the galaxy.
    host_dec: Declination of the galaxy, in degrees. Used for plotting the position of the galaxy.
    ra: Right ascension of an object, in degrees. Used for plotting the position of the object.
    dec: Declination of an object, in degrees. Used for plotting the position of the object.
    title: Title of the image.
    outfile: If given, path where to save the output figure.
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
        # ToDo: solve this deprecation warning (Aplpy should do it?)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AstropyWarning)
            fig.axis_labels.set_font(**{"family": font_family, "size": 18})  # this line is giving a warning!
    # galaxy markers
    fig2.show_markers(
        host_ra,
        host_dec,
        edgecolor="k",
        facecolor="m",
        alpha=0.7,
        marker="P",
        s=250,
        label="Given galaxy",
    )
    fig2.show_markers(
        gal_obj["x"][0],
        gal_obj["y"][0],
        edgecolor="k",
        facecolor="r",
        alpha=0.7,
        marker="X",
        s=250,
        label="Identified galaxy",
        coords_frame="pixel",
    )
    # galaxy pseudo-aperture
    """
    fig2.show_ellipses(
        gal_obj["x"][0],
        gal_obj["y"][0],
        2 * r * gal_obj["a"][0],
        2 * r * gal_obj["b"][0],
        gal_obj["theta"][0] * 180.0 / np.pi,
        coords_frame="pixel",
        linewidth=3,
        edgecolor="r",
        linestyle="dotted",
    )
    """
    # other sources apertures
    fig2.show_ellipses(
        objects["x"],
        objects["y"],
        2 * r * objects["a"],
        2 * r * objects["b"],
        objects["theta"] * 180.0 / np.pi,
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
                s=250,
                label="SN",
            )

    fig2.ax.legend(
        fancybox=True, framealpha=1, prop={"size": 20, "family": font_family}
    )
    # show sources indeces
    for i, (x, y) in enumerate(zip(objects["x"], objects["y"])):
        # the host galaxy has index one
        fig2.ax.text(x, y, i+2, fontsize=14, color="orangered")
    # title
    if title is not None:
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
