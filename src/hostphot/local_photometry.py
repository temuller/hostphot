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

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sep
from photutils.aperture import aperture_photometry, CircularAperture

from astropy.io import fits
from astropy import units as u, wcs
from astropy.cosmology import FlatLambdaCDM

from hostphot._constants import workdir
from hostphot.utils import (
    get_survey_filters,
    check_survey_validity,
    check_filters_validity,
    calc_sky_unc,
    get_image_exptime,
    survey_pixel_scale,
    check_work_dir,
    update_axislabels,
    magnitude_calculation,
    bkg_surveys,
)
from hostphot.image_cleaning import remove_nan
from hostphot.dust import calc_extinction

import warnings
from astropy.utils.exceptions import AstropyWarning

H0 = 70
Om0 = 0.3
__cosmo__ = FlatLambdaCDM(H0, Om0)
sep.set_sub_object_limit(1e4)

# ----------------------------------------
def choose_cosmology(cosmo):
    """Updates the cosmology used to calculate the aperture size.

    Parameters
    ----------
    cosmo: `astropy.cosmology` object
        Cosmological model. E.g. :func:`FlatLambdaCDM(70, 0.3)`.
    """
    global __cosmo__
    __cosmo__ = cosmo


# -------------------------------
def calc_aperture_size(z, ap_radius):
    """Calculates the size of the aperture in arsec,
    for aperture photometry, given a physical size.

    Parameters
    ----------
    z: float
        Redshift.
    ap_radius: float
        Physical aperture size in kpc.

    Returns
    -------
    radius_arcsec: float
        Aperture size in arcsec.
    """
    ap_radius = ap_radius * u.kpc

    # transverse separations
    transv_sep_per_arcmin = __cosmo__.kpc_proper_per_arcmin(z)
    transv_sep_per_arcsec = transv_sep_per_arcmin.to(u.kpc / u.arcsec)

    radius_arcsec = ap_radius / transv_sep_per_arcsec

    return radius_arcsec.value


def extract_aperture_flux(data, error, px, py, radius):
    """Extracts aperture photometry of a single image.

    Parameters
    ----------
    data: array
        Image data in a 2D numpy array.
    error: array
        Errors of ``data``.
    px: float
        x-axis pixel coordinate of the aperture center.
    py: float
        y-axis pixel coordinate of the aperture center.
    radius: float
        Aperture radius in pixels.

    Returns
    -------
    raw_flux: float
        Aperture photometry ("raw" flux).
    raw_flux_err: float
        Uncertainty on the aperture photometry.
    """
    aperture = CircularAperture((px, py), r=radius)
    ap_results = aperture_photometry(data, aperture, error=error)
    raw_flux = ap_results["aperture_sum"][0]
    raw_flux_err = ap_results["aperture_sum_err"][0]

    return raw_flux, raw_flux_err


def plot_aperture(data, px, py, radius_pix, img_wcs, outfile=None):
    """Plots the aperture for the given parameters.

    Parameters
    ----------
    data: ndarray
        Data of an image.
    px: float
        X-axis center of the aperture in pixels.
    py: float
        Y-axis center of the aperture in pixels.
    radius_pix: float
        Aperture radius in pixels.
    img_wcs: WCS
        Image's WCS.
    outfile: str, default ``None``
        If given, path where to save the output figure.
    """
    m, s = np.nanmean(data), np.nanstd(data)

    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(projection=img_wcs)
    update_axislabels(ax)
    ax.scatter(px, py, marker="*", s=140, c="g", edgecolor="gold")

    im = ax.imshow(
        data,
        interpolation="nearest",
        cmap="gray",
        vmin=m - s,
        vmax=m + s,
        origin="lower",
    )

    circle = plt.Circle(
        (px, py), radius_pix, color="r", fill=False, linewidth=1.5
    )
    ax.add_patch(circle)

    if outfile:
        basename = os.path.basename(outfile)
        title = os.path.splitext(basename)[0]
        title = "-".join(part for part in title.split("_"))
        fig.suptitle(title, fontsize=28)
        plt.tight_layout()
        plt.savefig(outfile, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def photometry(
    name,
    ra,
    dec,
    z,
    filt,
    survey,
    ap_radii=1,
    bkg_sub=None,
    use_mask=True,
    correct_extinction=True,
    save_plots=True,
):
    """Calculates the local aperture photometry in a given radius.

    Parameters
    ----------
    name: str
        Name of the object to find the path of the fits file.
    ra: float
        Right Ascensions in degrees to center the aperture.
    dec: float
        Declinations in degrees to center the aperture.
    z: float
        Redshift of the object to estimate the physical calculate
        of the aperture.
    filt: str
        Filter to use to load the fits file.
    survey: str
        Survey to use for the zero-points and pixel scale.
    ap_radii: float or list-like, default ``1``
        Physical size of the aperture in kpc.
    bkg_sub: bool, default ``None``
        If ``True``, the image gets background subtracted. By default, only
        the images that need it get background subtracted (WISE, 2MASS and
        VISTA).
    use_mask: bool, default ``True``
        If ``True``, the masked fits files are used. These must have
        been created beforehand.
    correct_extinction: bool, default `True`
        If `True`, corrects for Milky-Way extinction using the recalibrated dust maps
        by Schlafly & Finkbeiner (2011) and the extinction law from Fitzpatrick (1999).
    save_plots: bool, default ``True``
        If ``True``, the figure with the aperture is saved.

    Returns
    -------
    mags: list
        Aperture magnitudes for the given aperture radii.
    mags_err: list
        Aperture magnitude errors for the given aperture radii.
    fluxes: list
        Aperture flux for the given aperture radii.
    fluxes_err: list
        Aperture flux errors for the given aperture radii.
    """
    check_survey_validity(survey)
    check_work_dir(workdir)
    obj_dir = os.path.join(workdir, name)
    if use_mask:
        suffix = "masked_"
    else:
        suffix = ""
    fits_file = os.path.join(obj_dir, f"{suffix}{survey}_{filt}.fits")

    img = fits.open(fits_file)
    img = remove_nan(img)

    header = img[0].header
    data = img[0].data
    exptime = get_image_exptime(header, survey)

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

    # turn float into a list
    if isinstance(ap_radii, (float, int)):
        ap_radii = [ap_radii]

    mags, mags_err = [], []
    fluxes, fluxes_err = [], []
    px, py = img_wcs.wcs_world2pix(ra, dec, 1)
    pixel_scale = survey_pixel_scale(survey, filt)
    error = calc_sky_unc(data_sub, exptime)

    for ap_radius in ap_radii:
        # aperture photometry
        radius_arcsec = calc_aperture_size(z, ap_radius)
        radius_pix = radius_arcsec / pixel_scale

        flux, flux_err = extract_aperture_flux(
            data_sub, error, px, py, radius_pix
        )

        ap_area = 2 * np.pi * (radius_pix**2)
        mag, mag_err, flux, flux_err = magnitude_calculation(
            flux,
            flux_err,
            survey,
            filt,
            ap_area,
            header,
            bkg_rms,
        )

        # extinction correction is optional
        if correct_extinction:
            A_ext = calc_extinction(filt, survey, ra, dec)
            mag -= A_ext

        mags.append(mag)
        mags_err.append(mag_err)
        fluxes.append(flux)
        fluxes_err.append(flux_err)

        if save_plots:
            outfile = os.path.join(
                obj_dir, f"local_{survey}_{filt}_{ap_radius}kpc.jpg"
            )
            plot_aperture(data_sub, px, py, radius_pix, img_wcs, outfile)

    return mags, mags_err, fluxes, fluxes_err


def multi_band_phot(
    name,
    ra,
    dec,
    z,
    filters=None,
    survey="PS1",
    ap_radii=1,
    bkg_sub=None,
    use_mask=True,
    correct_extinction=True,
    save_plots=True,
    save_results=True,
    raise_exception=False,
):
    """Calculates the local aperture photometry for multiple filters.

    Parameters
    ----------
    name: str
        Name of the object to find the path of the fits file.
    ra: float
        Right Ascensions in degrees to center the aperture.
    dec: float
        Declinations in degrees to center the aperture.
    z: float
        Redshift of the object to estimate the physical calculate
        of the aperture.
    filters: str, default, ``None``
        Filters to use to load the fits files. If ``None`` use all
        the filters of the given survey.
    survey: str, default ``PS1``
        Survey to use for the zero-points and pixel scale.
    ap_radii: float or list-like, default ``1``
        Physical size of the aperture in kpc.
    bkg_sub: bool, default ``None``
        If ``True``, the image gets background subtracted. By default, only
        the images that need it get background subtracted (WISE, 2MASS and
        VISTA).
    use_mask: bool, default ``True``
        If ``True``, the masked fits files are used. These must have
        been created beforehand.
    correct_extinction: bool, default `True`
        If `True`, corrects for Milky-Way extinction using the recalibrated dust maps
        by Schlafly & Finkbeiner (2011) and the extinction law from Fitzpatrick (1999).
    save_plots: bool, default ``True``
        If ``True``, the figure with the aperture is saved.
    save_results: bool, default ``True``
        If ``True``, the magnitudes are saved into a csv file.
    raise_exception: bool, default ``False``
        If ``True``, an exception is raised if the photometry fails for any filter.

    Returns
    -------
    results_dict: dict
        Dictionary with the object's photometry and other info.

    Examples
    --------
    >>> import hostphot.local_photometry as lp
    >>> ap_radii = [3, 4]  # aperture radii in units of kpc
    >>> ra, dec =  308.22579, 9.92853 # coords of SN2004eo
    >>> z = 0.0157  # redshift
    >>> survey = 'PS1'
    >>> results = lp.multi_band_phot(name, ra, dec, z,
                            survey=survey, ap_radii=ap_radii,
                            use_mask=True, save_plots=True)
    """
    check_survey_validity(survey)
    if filters is None:
        if survey=='HST':
            raise ValueError("For HST, the filter needs to be specified!")
        filters = get_survey_filters(survey)
    else:
        check_filters_validity(filters, survey)
    if survey=='HST':
        filters = [filters]

    # turn float into a list
    if isinstance(ap_radii, (float, int)):
        ap_radii = [ap_radii]

    results_dict = {
        "name": name,
        "ra": ra,
        "dec": dec,
        "redshift": z,
        "survey": survey,
    }
    
    for filt in filters:
        try:
            mags, mags_err, fluxes, fluxes_err = photometry(
                name,
                ra,
                dec,
                z,
                filt,
                survey,
                ap_radii,
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
        except Exception as exc:
            if raise_exception is True:
                raise Exception(exc)
            else:
                for radius in ap_radii:
                    results_dict[f"{filt}_{radius}"] = np.nan
                    results_dict[f"{filt}_{radius}_err"] = np.nan
                    results_dict[f"{filt}_{radius}_flux"] = np.nan
                    results_dict[f"{filt}_{radius}_flux_err"] = np.nan

    if save_results is True:
        outfile = os.path.join(workdir, name, f"{survey}_local.csv")
        phot_df = pd.DataFrame(
            {key: [val] for key, val in results_dict.items()}
        )
        phot_df.to_csv(outfile, index=False)

    return results_dict
