# Check the following urls for more info about Pan-STARRS:
#
#     https://outerspace.stsci.edu/display/PANSTARRS/PS1+Image+Cutout+Service#PS1ImageCutoutService-ImportantFITSimageformat,WCS,andflux-scalingnotes
#     https://outerspace.stsci.edu/display/PANSTARRS/PS1+Stack+images#PS1Stackimages-Photometriccalibration
#
# For DES:
#
#     https://des.ncsa.illinois.edu/releases/dr1/dr1-docs/processing
#
# For SDSS:
#
#     https://www.sdss.org/dr12/algorithms/fluxcal/#SDSStoAB
#     https://data.sdss.org/datamodel/files/BOSS_PHOTOOBJ/frames/RERUN/RUN/CAMCOL/frame.html
#
# Some parts of this notebook are based on https://github.com/djones1040/PS1_surface_brightness/blob/master/Surface%20Brightness%20Tutorial.ipynb and codes from Lluís Galbany

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt

import aplpy
import sep
from photutils.aperture import aperture_photometry, CircularAperture
from photutils.utils import calc_total_error

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM, Cosmology

from hostphot._constants import workdir, font_family
from hostphot.processing.cleaning import remove_nan
from hostphot.photometry.dust import calc_extinction
from hostphot.utils import check_work_dir, suppress_stdout, store_input
from hostphot.photometry.image_utils import get_image_exptime
from hostphot.photometry.photometry_utils import magnitude_calculation
from hostphot.surveys_utils import (
    get_survey_filters,
    survey_pixel_units,
    check_filters_validity,
    check_survey_validity,
    survey_pixel_scale,
    bkg_surveys,
)

import warnings
from astropy.utils.exceptions import AstropyWarning

from photutils.background import Background2D, SExtractorBackground

plt.rcParams["mathtext.fontset"] = "cm"

# set initial cosmology
H0 = 70
Om0 = 0.3
__cosmo__ = FlatLambdaCDM(H0, Om0)
sep.set_sub_object_limit(1e4)


def choose_cosmology(cosmo: Cosmology) -> None:
    """Updates the cosmology used to calculate the aperture size.

    Parameters
    ----------
    cosmo: Cosmological model. E.g. :func:`FlatLambdaCDM(70, 0.3)`.
    """
    global __cosmo__
    __cosmo__ = cosmo


def calc_aperture_size(z: float, ap_radius: float) -> float:
    """Calculates the size of the aperture in arsec,
    for aperture photometry, given a physical size.

    Parameters
    ----------
    z: Redshift.
    ap_radius: Physical aperture size in kpc.

    Returns
    -------
    radius_arcsec: Aperture size in arcsec.
    """
    ap_radius = ap_radius * u.kpc
    # transverse separations
    transv_sep_per_arcmin = __cosmo__.kpc_proper_per_arcmin(z)
    transv_sep_per_arcsec = transv_sep_per_arcmin.to(u.kpc / u.arcsec)
    radius_arcsec = ap_radius / transv_sep_per_arcsec

    return radius_arcsec.value


def extract_aperture_flux(
    data: np.ndarray, bkg: np.ndarray, exptime: np.ndarray, px: float, py: float, radius: float
) -> tuple[float, float]:
    """Extracts aperture photometry of a single image.

    Parameters
    ----------
    data: Image data in a 2D numpy array.
    bkg: Background error of the images.
    exptime: Exposure time.
    px: x-axis pixel coordinate of the aperture center.
    py: y-axis pixel coordinate of the aperture center.
    radius: Aperture radius in pixels.

    Returns
    -------
    raw_flux: Aperture photometry ("raw" flux).
    raw_flux_err: Uncertainty on the aperture photometry.
    """
    error = calc_total_error(data, bkg, exptime)
     
    aperture = CircularAperture((px, py), r=radius)
    ap_results = aperture_photometry(data, aperture, error=error)
    raw_flux = ap_results["aperture_sum"][0]
    raw_flux_err = ap_results["aperture_sum_err"][0]

    return raw_flux, raw_flux_err


def plot_aperture(
    hdu: list[fits.ImageHDU],
    px: float,
    py: float,
    radius_pix: float,
    title: str,
    outfile: Optional[str | Path] = None,
) -> None:
    """Plots the aperture for the given parameters.

    Parameters
    ----------
    hdu: HDU image.
    px: x-axis center of the aperture in pixels.
    py: y-axis center of the aperture in pixels.
    radius_pix: Aperture radius in pixels.
    title: Title of the figure.
    outfile: If given, path where to save the output figure.
    """
    figure = plt.figure(figsize=(10, 10))
    title, label = title.split("|")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        fig = aplpy.FITSFigure(hdu, figure=figure)
    with suppress_stdout():
        fig.show_grayscale(stretch="arcsinh")

    # plot SN and aperture
    fig.show_markers(
        px,
        py,
        edgecolor="k",
        facecolor="aqua",
        marker="*",
        s=200,
        label="SN",
        coords_frame="pixel",
    )
    fig.show_circles(
        px, py, radius_pix, coords_frame="pixel", linewidth=3, edgecolor="r"
    )
    fig.show_lines([], [], color="r", lw=3, label=label)  # for the legend only

    # ticks
    fig.tick_labels.set_font(**{"family": font_family, "size": 18})
    fig.tick_labels.set_xformat("dd.dd")
    fig.tick_labels.set_yformat("dd.dd")
    fig.ticks.set_length(6)
    # ToDo: solve this deprecation warning (Aplpy should do it?)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        fig.axis_labels.set_font(**{"family": font_family, "size": 18})

    fig.set_title(title, **{"family": font_family, "size": 24})
    fig.set_theme("publication")
    fig.ax.legend(fancybox=True, framealpha=1, prop={"size": 20, "family": font_family})
    # output
    if outfile:
        plt.savefig(outfile, bbox_inches="tight")
        plt.close(figure)
    else:
        plt.show()


def photometry(
    name: str,
    ra: float,
    dec: float,
    z: float,
    filt: str,
    survey: str,
    ap_radii: int | float | list = 1,
    ap_units: str = "kpc",
    bkg_sub: Optional[bool] = None,
    use_mask: bool = True,
    correct_extinction: bool = True,
    save_plots: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Calculates the local aperture photometry in a given radius.

    Parameters
    ----------
    name: Name of the object to find the path of the fits file.
    ra: Right Ascensions in degrees to center the aperture.
    dec: Declinations in degrees to center the aperture.
    z: Redshift of the object to estimate the physical calculate
        of the aperture.
    filt: Filter to use to load the fits file.
    survey: Survey to use for the zero-points and pixel scale.
    ap_radii: Aperture size.
    ap_units: Aperture size units. Either ``kpc`` or ``arcsec``.
    bkg_sub: If ``True``, the image gets background subtracted. By default, only
        the images that need it get background subtracted (WISE, 2MASS and
        VISTA).
    use_mask: If ``True``, the masked fits files are used. These must have
        been created beforehand.
    correct_extinction: If `True`, corrects for Milky-Way extinction using the recalibrated dust maps
        by Schlafly & Finkbeiner (2011) and the extinction law from Fitzpatrick (1999).
    save_plots: If ``True``, the figure with the aperture is saved.

    Returns
    -------
    mags: Aperture magnitudes for the given aperture radii.
    mags_err: Aperture magnitude errors for the given aperture radii.
    fluxes: Aperture flux for the given aperture radii.
    fluxes_err: Aperture flux errors for the given aperture radii.
    zp: Zeropoint.
    """     
    # initial checks
    check_survey_validity(survey)
    check_work_dir(workdir)
    assert ap_units in ["kpc", "arcsec"], "not valid aperture size units"
    obj_dir = Path(workdir, name)
    if use_mask:
        suffix = "_masked"
    else:
        suffix = ""
    fits_file = obj_dir / survey / f"{survey}_{filt}{suffix}.fits"

    # load image information
    hdu = fits.open(fits_file)
    hdu = remove_nan(hdu)
    header = hdu[0].header
    data = hdu[0].data
    exptime = get_image_exptime(header, survey)
    pixel_units = survey_pixel_units(survey, filt)
    if pixel_units == "counts":
        _exptime = 1
    else:
        _exptime = exptime
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        wcs = WCS(header, naxis=2)

    data = data.astype(np.float64)
    #box_size = int(0.1 * np.sqrt(data.size))
    #bkg = SExtractorBackground(sigma_clip=None)
    #bkg = Background2D(data, box_size=box_size, bkg_estimator=bkg)
    bkg = sep.Background(data)
    # background subtraction, if needed
    if (bkg_sub is None and survey in bkg_surveys) or bkg_sub is True:
        data_sub = np.copy(data - bkg)
        #data_sub = np.copy(data - bkg.background)
    else:
        data_sub = np.copy(data)

    # turn float into a list for consistency
    if isinstance(ap_radii, (float, int)):
        ap_radii = [ap_radii]

    pixel_scale = survey_pixel_scale(survey, filt)
    if survey == "UKIDSS" and filt == "J":
        if "LAS" in header["PROJECT"].split("/")[-1]:
            # if the J comes from the LAS, then the observations were micro-stepped,
            # so the pixel size is smaller.
            nustep = header["NUSTEP"]
            pixel_scale /= nustep / 2  # divided by 2 for the two dimensions (x and y)

    mags, mags_err = [], []
    fluxes, fluxes_err = [], []
    px, py = wcs.wcs_world2pix(ra, dec, 0)  # object's coordinates

    for ap_radius in ap_radii:
        # aperture photometry
        if ap_units == "kpc":
            radius_arcsec = calc_aperture_size(z, ap_radius)
        else:
            radius_arcsec = ap_radius
        radius_pix = radius_arcsec / pixel_scale
        flux, flux_err = extract_aperture_flux(data_sub, bkg.rms(), _exptime, px, py, radius_pix)
        #flux, flux_err = extract_aperture_flux(data_sub, bkg.background_rms, _exptime, px, py, radius_pix)

        ap_area = np.pi * (radius_pix**2)
        mag, mag_err, flux, flux_err, zp = magnitude_calculation(
            flux,
            flux_err,
            survey,
            filt,
            ap_area,
            header,
            bkg.globalrms,
            #np.median(bkg.background_rms),
        )

        if correct_extinction is True:
            A_ext = calc_extinction(filt, survey, ra, dec)
            mag -= A_ext
            flux *= 10 ** (0.4 * A_ext)

        mags.append(mag)
        mags_err.append(mag_err)
        fluxes.append(flux)
        fluxes_err.append(flux_err)

        if save_plots:
            outfile = obj_dir / survey / f"local_{survey}_{filt}_{ap_radius}{ap_units}.jpg"
            filt_ = filt.replace("_", "-")
            if ap_units == "kpc":
                title = rf"{name}: {survey}-${filt_}$|r$={ap_radius}$ {ap_units} @ $z={z}$"
            else:
                title = rf"{name}: {survey}-${filt_}$|r$={ap_radius}$ {ap_units}"
            plot_aperture(hdu, px, py, radius_pix, title, outfile)
    hdu.close()

    return mags, mags_err, fluxes, fluxes_err, zp


def multi_band_phot(
    name: str,
    ra: float,
    dec: float,
    z: float,
    filters: Optional[str | list] = None,
    survey: str = "PanSTARRS",
    ap_radii: int | float | list = 1,
    ap_units: str = "kpc",
    bkg_sub: Optional[bool] = None,
    use_mask: bool = True,
    correct_extinction: bool = True,
    save_plots: bool = True,
    save_results: bool = True,
    raise_exception: bool = True,
    save_input: bool = True,
) -> pd.DataFrame:
    """Calculates the local aperture photometry for multiple filters.

    Parameters
    ----------
    name: Name of the object to find the path of the fits file.
    ra: Right Ascensions in degrees to center the aperture.
    dec: Declinations in degrees to center the aperture.
    z: Redshift of the object to estimate the physical calculate
        of the aperture.
    filters: Filters to use to load the fits files. If ``None`` use all
        the filters of the given survey.
    survey: Survey to use for the zero-points and pixel scale.
    ap_radii: Aperture size.
    ap_units: Aperture size units. Either ``kpc`` or ``arcsec``.
    bkg_sub: If ``True``, the image gets background subtracted. By default, only
        the images that need it get background subtracted (WISE, 2MASS and
        VISTA).
    use_mask: If ``True``, the masked fits files are used. These must have
        been created beforehand.
    correct_extinction: If `True`, corrects for Milky-Way extinction using the recalibrated dust maps
        by Schlafly & Finkbeiner (2011) and the extinction law from Fitzpatrick (1999).
    save_plots: If ``True``, the figure with the aperture is saved.
    save_results: If ``True``, the magnitudes are saved into a csv file.
    raise_exception: If ``True``, an exception is raised if the photometry fails for any filter.
    save_input: Whether to save the input parameters.

    Returns
    -------
    phot_df: DataFrame with the object's photometry and other info.

    Examples
    --------
    >>> from hostphot.photometry import local_photometry as lp
    >>> ap_radii = [3, 4]  # aperture radii in units of kpc
    >>> ra, dec =  308.22579, 9.92853 # coords of SN2004eo
    >>> z = 0.0157  # redshift
    >>> survey = 'PanSTARRS'
    >>> results = lp.multi_band_phot(name, 
                                     ra, 
                                     dec, 
                                     z,
                                     survey=survey, 
                                     ap_radii=ap_radii, 
                                     use_mask=True, 
                                     save_plots=True)
    """
    input_params = locals()  # dictionary
    # initial checks
    check_survey_validity(survey)
    if filters is None:
        if survey in ["HST", "JWST"]:
            raise ValueError(f"For {survey}, the filter needs to be specified!")
        filters = get_survey_filters(survey)
    else:
        check_filters_validity(filters, survey)
    #if survey in ["HST", "JWST"]:
    #     filters = [filters]
    assert ap_units in ["kpc", "arcsec"], "not valid aperture size units"

    # save input parameters
    if save_input is True:
        inputs_file = Path(workdir, name, survey, "input_local_photometry.csv")
        store_input(input_params, inputs_file)

    # turn int/float into a list for consistency
    if isinstance(ap_radii, (float, int)):
        ap_radii = [ap_radii]

    results_dict = {
        "name": name,
        "ra": ra,
        "dec": dec,
        "redshift": z,
        "survey": survey,
        "ap_units": ap_units,
    }
    # calculate photometry
    for filt in filters:
        try:
            mags, mags_err, fluxes, fluxes_err, zp = photometry(
                name,
                ra,
                dec,
                z,
                filt,
                survey,
                ap_radii,
                ap_units,
                bkg_sub,
                use_mask,
                correct_extinction,
                save_plots,
            )
            for radius, mag, mag_err, flux, flux_err in zip(
                ap_radii, mags, mags_err, fluxes, fluxes_err
            ):
                results_dict[f"{filt}_{radius}"] = mag
                results_dict[f"{filt}_{radius}_err"] = mag_err
                results_dict[f"{filt}_{radius}_flux"] = flux
                results_dict[f"{filt}_{radius}_flux_err"] = flux_err
            results_dict[f"{filt}_zeropoint"] = zp
        except Exception as exc:
            if raise_exception is True:
                raise Exception(exc)
            else:
                for radius in ap_radii:
                    results_dict[f"{filt}_{radius}"] = np.nan
                    results_dict[f"{filt}_{radius}_err"] = np.nan
                    results_dict[f"{filt}_{radius}_flux"] = np.nan
                    results_dict[f"{filt}_{radius}_flux_err"] = np.nan
                results_dict[f"{filt}_zeropoint"] = np.nan
    # store photometry
    phot_df = pd.DataFrame({key: [val] for key, val in results_dict.items()})
    if save_results is True:
        outfile = Path(workdir, name, survey, "local_photometry.csv")
        phot_df.to_csv(outfile, index=False)

    return phot_df
