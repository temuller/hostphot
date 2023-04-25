import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import sep
from astroquery.gaia import Gaia
from astroquery.mast import Catalogs
from astropy import units as u
from astropy.coordinates import SkyCoord

from hostphot.utils import update_axislabels


def extract_objects(
    data, err, host_ra, host_dec, threshold, img_wcs, dist_thresh=-1
):
    """Extracts objects and their ellipse parameters. The function :func:`sep.extract()`
    is used.

    If there is no detected object within a distance of ``dist_thresh`` from the galaxy
    coordinates, it means that the galaxy was not correctly identified.

    Parameters
    ----------
    data: ndarray
        Image data.
    err: 2D array
        Background error of the image.
    host_ra: float
        Host-galaxy Right ascension of the galaxy in degrees.
    host_dec: float
        Host-galaxy Declination of the galaxy in degrees.
    threshold: float
        Source with flux above ``threshold*bkg_rms`` are extracted.
        See :func:`sep.extract()` for more information.
    img_wcs: WCS
        Image's WCS.
    pixel_scale: float
        Pixel scale, in units of arcsec/pixel, used to convert from pixel units
        to arcseconds.
    dist_thresh: float, default ``-1``.
        Distance in arcsec to crossmatch the galaxy coordinates with a detected object,
        where the object nearest to the galaxy position is considered as the galaxy (within
        the given threshold). If no objects are found within the given distance threshold,
        the galaxy is considered as not found and a warning is printed. If a non-positive value
        is given, the threshold is considered as infinite, i.e. the closest detected object is
        considered as the galaxy (default option).

    Returns
    -------
    gal_obj: numpy array
        Galaxy object extracted.
    nogal_objs: numpy array
        All objects extracted except for the galaxy.
    """
    # extract objects with Source Extractor
    objects = sep.extract(data, threshold, err=err)

    objs_coords = img_wcs.pixel_to_world(objects["x"], objects["y"])
    objs_ra, objs_dec = objs_coords.ra.value, objs_coords.dec.value
    dist = np.sqrt(
        (objs_ra - host_ra) ** 2 * (np.cos(host_dec * np.pi / 180) ** 2)
        + (objs_dec - host_dec) ** 2
    )  # in degrees
    dist_arcsec = dist * 3600

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


def find_gaia_objects(ra, dec, rad=0.15):
    """Finds objects using the Gaia DR3 catalog for the given
    coordinates in a given radius.

    Parameters
    ----------
    ra: float
        Right ascension in degrees.
    dec: float
        Declination in degrees.
    rad: float, default ``0.15``
        Search radius in degrees.

    Returns
    -------
    gaia_coord: SkyCoord object
        Coordinates of the objects found.
    """
    Gaia.MAIN_GAIA_TABLE = "gaiaedr3.gaia_source"
    Gaia.ROW_LIMIT = -1
    coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame="icrs")
    width = u.Quantity(rad, u.deg)
    height = u.Quantity(rad, u.deg)
    try:
        gaia_cat = Gaia.query_object_async(
            coordinate=coord, width=width, height=height
        )
    except Exception as exc:
        print(exc)
        print("No objects found with Gaia DR3, switching to DR2")
        Gaia.MAIN_GAIA_TABLE = "gaiadr2.gaia_source"
        gaia_cat = Gaia.query_object_async(
            coordinate=coord, width=width, height=height
        )

    gaia_ra = np.array(gaia_cat["ra"].value)
    gaia_dec = np.array(gaia_cat["dec"].value)
    gaia_coord = SkyCoord(
        ra=gaia_ra, dec=gaia_dec, unit=(u.degree, u.degree), frame="icrs"
    )

    return gaia_coord


def find_catalog_objects(ra, dec, rad=0.15):
    """Finds objects using the TESS image cutouts (Tic) catalog
    for the given coordinates in a given radius.

    **Note:** this catalog includes objects from these sources:
        HIP, TYC, UCAC, TWOMASS, SDSS, ALLWISE, GAIA, APASS, KIC

    Parameters
    ----------
    ra: float
        Right ascension in degrees.
    dec: float
        Declination in degrees.
    rad: float, default ``0.15``
        Search radius in degrees.

    Returns
    -------
    cat_coord: SkyCoord object
        Coordinates of the objects found.
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


def cross_match(objects, img_wcs, coord, dist_thresh=1.0):
    """Cross-matches objects with a given set of coordinates.
    Those with a distance of less than ``dist_thresh`` are selected.

    Parameters
    ----------
    objects: array
        Objects detected with :func:`sep.extract()`.
    img_wcs: WCS
        WCS of the image from which the objects were extracted.
    coord: SkyCoord object
        Coordinates for the cross-match.
    dist_thresh: float, default ``1.0``
        Distance in arcsec to crossmatch the objects with
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
    data, objects, scale, img_wcs, ra=None, dec=None, outfile=None
):
    """Plots the objects extracted with :func:`sep.extract()``.

    Parameters
    ----------
    data: ndarray
        Data of an image.
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
    outfile: str, default ``None``
        If given, path where to save the output figure.
    """
    m, s = np.nanmean(data), np.nanstd(data)

    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(projection=img_wcs)
    update_axislabels(ax)

    im = ax.imshow(
        data,
        interpolation="nearest",
        cmap="gray",
        vmin=m - s,
        vmax=m + s,
        origin="lower",
    )

    e = Ellipse(
        xy=(objects["x"][0], objects["y"][0]),
        width=2 * scale * objects["a"][0],
        height=2 * scale * objects["b"][0],
        angle=objects["theta"][0] * 180.0 / np.pi,
    )

    e.set_facecolor("none")
    e.set_edgecolor("red")
    e.set_linewidth(1.5)
    ax.add_artist(e)

    # plot SN position
    if (ra is not None) and (dec is not None):
        coord = SkyCoord(
            ra=ra, dec=dec, unit=(u.degree, u.degree), frame="icrs"
        )
        px, py = img_wcs.world_to_pixel(coord)
        ax.scatter(px, py, marker="*", s=200, c="g", edgecolor="gold")

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
