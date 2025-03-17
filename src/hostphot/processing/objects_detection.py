import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
import aplpy

from astropy import wcs
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
from astroquery.mast import Catalogs

import sep
from hostphot._constants import font_family
from hostphot.utils import suppress_stdout

import warnings
from astropy.utils.exceptions import AstropyWarning

plt.rcParams["mathtext.fontset"] = "cm"


def extract_objects(
    data: np.ndarray,
    bkg: np.ndarray,
    host_ra: float,
    host_dec: float,
    threshold: float,
    img_wcs: wcs.WCS,
    dist_thresh: float = -1,
    deblend_cont: float = 0.005,
) -> tuple[np.ndarray, np.ndarray]:
    """Extracts objects and their ellipse parameters. The function :func:`sep.extract()`
    is used.

    If there is no detected object within a distance of ``dist_thresh`` from the galaxy
    coordinates, it means that the galaxy was not correctly identified.

    Parameters
    ----------
    data: Image data.
    bkg: Background level of the image.
    host_ra: Host-galaxy Right ascension of the galaxy in degrees.
    host_dec: Host-galaxy Declination of the galaxy in degrees.
    threshold: Source with flux above ``threshold*bkg_rms`` are extracted.
        See :func:`sep.extract()` for more information.
    img_wcs: Image's WCS.
    pixel_scale: Pixel scale, in units of arcsec/pixel, used to convert from pixel units
        to arcseconds.
    dist_thresh: Distance in arcsec to crossmatch the galaxy coordinates with a detected object,
        where the object nearest to the galaxy position is considered as the galaxy (within
        the given threshold). If no objects are found within the given distance threshold,
        the galaxy is considered as not found and a warning is printed. If a non-positive value
        is given, the threshold is considered as infinite, i.e. the closest detected object is
        considered as the galaxy (default option).
    deblend_cont : Minimum contrast ratio used for object deblending. Default is 0.005.
        To entirely disable deblending, set to 1.0.

    Returns
    -------
    gal_obj: Galaxy object extracted.
    nogal_objs: All objects extracted except for the galaxy.
    """
    # extract objects with Source Extractor
    objects = sep.extract(data, threshold, err=bkg, deblend_cont=deblend_cont)

    host_coords = SkyCoord(
        ra=host_ra, dec=host_dec, unit=(u.degree, u.degree), frame="icrs"
    )
    objs_coords = img_wcs.pixel_to_world(objects["x"], objects["y"])
    distances = host_coords.separation(objs_coords).to(u.arcsec)
    dist_arcsec = distances.value

    if dist_thresh <= 0.0:
        dist_thresh = np.inf

    if any(dist_arcsec <= dist_thresh):
        gal_id = np.argmin(dist_arcsec)
        gal_obj = objects[gal_id : gal_id + 1]
    else:
        gal_obj = None
        gal_id = -99
        print("WARNING: the galaxy was no detected")

    objs_id = [i for i in range(len(objects)) if i != gal_id]
    nogal_objs = objects.take(objs_id)
    return gal_obj, nogal_objs


def find_gaia_objects(ra: float, dec: float, rad: float = 0.15) -> SkyCoord:
    """Finds objects using the Gaia DR3 catalog for the given
    coordinates in a given radius.

    Parameters
    ----------
    ra: Right ascension in degrees.
    dec: Declination in degrees.
    rad: Search radius in degrees.

    Returns
    -------
    gaia_coord: Coordinates of the objects found.
    """
    Gaia.MAIN_GAIA_TABLE = "gaiaedr3.gaia_source"
    Gaia.ROW_LIMIT = -1
    coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame="icrs")
    width = u.Quantity(rad, u.deg)
    height = u.Quantity(rad, u.deg)
    try:
        with suppress_stdout():
            gaia_cat = Gaia.query_object_async(
                coordinate=coord, width=width, height=height
            )
    except Exception as exc:
        print(exc)
        print("No objects found with Gaia DR3, switching to DR2")
        Gaia.MAIN_GAIA_TABLE = "gaiadr2.gaia_source"
        gaia_cat = Gaia.query_object_async(coordinate=coord, width=width, height=height)

    gaia_ra = np.array(gaia_cat["ra"].value)
    gaia_dec = np.array(gaia_cat["dec"].value)
    gaia_coord = SkyCoord(
        ra=gaia_ra, dec=gaia_dec, unit=(u.degree, u.degree), frame="icrs"
    )
    return gaia_coord


def find_catalog_objects(ra: float, dec: float, rad: float = 0.15) -> SkyCoord:
    """Finds objects using the TESS image cutouts (Tic) catalog
    for the given coordinates in a given radius.

    **Note:** this catalog includes objects from these sources:
        HIP, TYC, UCAC, TWOMASS, SDSS, ALLWISE, GAIA, APASS, KIC

    Parameters
    ----------
    ra: Right ascension in degrees.
    dec: Declination in degrees.
    rad: Search radius in degrees.

    Returns
    -------
    cat_coord: Coordinates of the objects found.
    """
    coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame="icrs")
    cat_data = Catalogs.query_criteria(
        catalog="Tic", radius=rad, coordinates=coord, objType="STAR"
    )

    cat_ra = np.array(cat_data["ra"].value)
    cat_dec = np.array(cat_data["dec"].value)
    cat_coord = SkyCoord(
        ra=cat_ra, dec=cat_dec, unit=(u.degree, u.degree), frame="icrs"
    )
    return cat_coord


def cross_match(
    objects: np.ndarray, img_wcs: wcs.WCS, coord: SkyCoord, dist_thresh: float = 1.0
) -> None:
    """Cross-matches objects with a given set of coordinates.
    Those with a distance of less than ``dist_thresh`` are selected.

    Parameters
    ----------
    objects: Objects detected with :func:`sep.extract()`.
    img_wcs: WCS of the image from which the objects were extracted.
    coord: Coordinates for the cross-match.
    dist_thresh: Distance in arcsec to crossmatch the objects with
        the given coordinates.
    """
    # coordinates in arcsec
    objs_coord = img_wcs.pixel_to_world(objects["x"], objects["y"])
    objs_ra = objs_coord.ra.to(u.arcsec).value
    objs_dec = objs_coord.dec.to(u.arcsec).value

    cat_ra = np.array(coord.ra.to(u.arcsec).value)
    cat_dec = np.array(coord.dec.to(u.arcsec).value)

    objs_id = []
    for i in range(len(objects)):
        ra_dist = objs_ra[i] - cat_ra
        dec_dist = objs_dec[i] - cat_dec
        dist_arcsec = np.sqrt(ra_dist**2 + dec_dist**2)
        if any(dist_arcsec <= dist_thresh):
            objs_id.append(i)

    objs = objects.take(objs_id)

    return objs


def plot_detected_objects(
    hdu: list[fits.ImageHDU],
    objects: np.ndarray,
    scale: float,
    ra: Optional[float] = None,
    dec: Optional[float] = None,
    host_ra: Optional[float] = None,
    host_dec: Optional[float] = None,
    title: Optional[str] = None,
    outfile: Optional[str] = None,
) -> None:
    """Plots the objects extracted with :func:`sep.extract()``.

    Parameters
    ----------
    hdu: ~astropy.io.fits
        Header Data Unit.
    objects: array
        Objects detected with :func:`sep.extract()`.
    scale: float
        Scale of the ellipse's semi-mayor and semi-minor axes.
    img_wcs: WCS
        Image's WCS.
    ra: float, default ``None``
       Right ascension of an object, in degrees. Used for plotting the position of the object.
    dec: float, default ``None``
       Declination of an object, in degrees. Used for plotting the position of the object.
    host_ra: float, default ``None``
       Right ascension of a galaxy.
    host_dec: float, default ``None``
       Declination of a galaxy.
    title: str, default ``None``
        Title of the image.
    outfile: str, default ``None``
        If given, path where to save the output figure.
    """
    figure = plt.figure(figsize=(10, 10))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        fig = aplpy.FITSFigure(hdu, figure=figure)

    with suppress_stdout():
        fig.show_grayscale(stretch="arcsinh")

    # plot SN marker
    if (ra is not None) and (dec is not None):
        fig.show_markers(
            ra,
            dec,
            edgecolor="k",
            facecolor="aqua",
            marker="*",
            s=250,
            label="SN",
        )

    # plot galaxy marker and aperture
    if (host_ra is not None) and (host_dec is not None):
        fig.show_markers(
            host_ra,
            host_dec,
            edgecolor="k",
            facecolor="m",
            alpha=0.7,
            marker="P",
            s=250,
            label="Galaxy position",
        )
    fig.show_ellipses(
        objects["x"],
        objects["y"],
        2 * scale * objects["a"],
        2 * scale * objects["b"],
        objects["theta"][0] * 180.0 / np.pi,
        coords_frame="pixel",
        linewidth=3,
        edgecolor="r",
    )
    # ticks
    fig.tick_labels.set_font(**{"family": font_family, "size": 18})
    fig.tick_labels.set_xformat("dd.dd")
    fig.tick_labels.set_yformat("dd.dd")
    fig.ticks.set_length(6)
    # other configuration options
    # ToDo: solve this deprecation warning (Aplpy should do it?)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        fig.axis_labels.set_font(**{"family": font_family, "size": 18})
    fig.set_title(title, **{"family": font_family, "size": 24})
    fig.set_theme("publication")
    fig.ax.legend(fancybox=True, framealpha=1, prop={"size": 20, "family": font_family})
    if outfile:
        plt.savefig(outfile, bbox_inches="tight")
        plt.close(figure)
    else:
        plt.show()
