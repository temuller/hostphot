import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
import aplpy

from astropy import wcs
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import proj_plane_pixel_scales

import sep
from hostphot.utils import call_with_timeout
from hostphot._constants import font_family
from hostphot.utils import suppress_stdout, add_fields

import warnings

from hostphot.flags import GaiaWarning
from astropy.utils.exceptions import AstropyWarning
from requests.exceptions import HTTPError

plt.rcParams["mathtext.fontset"] = "cm"


# a detected object is considered the host galaxy if the host coordinates are inside its ellipse scaled
# by this factor (roughly its extent at the detection threshold) or within MIN_HOST_DIST arcsec
HOST_EXTENT_FACTOR = 3.0
MIN_HOST_DIST = 2.0


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
        is given (default option), the closest detected object is considered as the galaxy only
        if the galaxy coordinates are within its extent (inside its ellipse scaled by
        ``HOST_EXTENT_FACTOR``, or within ``MIN_HOST_DIST`` arcsec).
    deblend_cont : Minimum contrast ratio used for object deblending. Default is 0.005.
        To entirely disable deblending, set to 1.0.

    Returns
    -------
    gal_obj: Galaxy object extracted.
    nogal_objs: All objects extracted except for the galaxy.
    """
    # extract objects with Source Extractor
    # pixels without data (NaN) are masked
    nan_mask = ~np.isfinite(data)
    data = np.ascontiguousarray(np.where(nan_mask, 0.0, data), dtype=np.float64)
    # large galaxies (e.g. in high-resolution images) can exceed the default buffer of
    # pixels above the threshold (300000), so it is scaled with the image size (with a cap)
    sep.set_extract_pixstack(int(min(max(300000, data.size), 1e7)))
    objects = sep.extract(data, threshold, err=bkg, mask=nan_mask, deblend_cont=deblend_cont)

    host_coords = SkyCoord(
        ra=host_ra, dec=host_dec, unit=(u.degree, u.degree), frame="icrs"
    )
    objs_coords = img_wcs.pixel_to_world(objects["x"], objects["y"])
    # add coordinates and ellipse a/b in degrees
    sx, sy = proj_plane_pixel_scales(img_wcs)  # degrees per pixel
    a_deg, b_deg = objects["a"] * sx, objects["b"] * sy
    objects = add_fields(objects, 
                         names=["ra", "dec", "a_deg", "b_deg"], 
                         dtypes=["f8", "f8", "f8", "f8"], 
                         data=[objs_coords.ra.value, 
                               objs_coords.dec.value,
                               a_deg, 
                               b_deg
                               ]
                         )
    
    distances = host_coords.separation(objs_coords).to(u.arcsec)
    dist_arcsec = distances.value

    if len(objects) == 0:
        matched = False
        gal_id = -99
    else:
        gal_id = int(np.argmin(dist_arcsec))
        if dist_thresh > 0.0:
            matched = dist_arcsec[gal_id] <= dist_thresh
        else:
            # the host position must be within the extent of the nearest detected object: inside its
            # ellipse scaled by HOST_EXTENT_FACTOR (a and b, in pixels, are the second moments of the
            # light distribution), or within MIN_HOST_DIST arcsec (e.g. small offsets of compact galaxies)
            host_x, host_y = img_wcs.world_to_pixel_values(host_ra, host_dec)
            obj = objects[gal_id]
            dx, dy = host_x - obj["x"], host_y - obj["y"]
            cos, sin = np.cos(obj["theta"]), np.sin(obj["theta"])
            u_ = (dx * cos + dy * sin) / (HOST_EXTENT_FACTOR * obj["a"])
            v_ = (-dx * sin + dy * cos) / (HOST_EXTENT_FACTOR * obj["b"])
            matched = (np.hypot(u_, v_) <= 1) or (dist_arcsec[gal_id] <= MIN_HOST_DIST)
    if matched:
        gal_obj = objects[gal_id : gal_id + 1]
    else:
        gal_obj = None
        if gal_id >= 0:
            warnings.warn(
                "The galaxy was not detected: the nearest detected object is "
                f"{dist_arcsec[gal_id]:.1f} arcsec away from the given coordinates (outside its extent)."
            )
        gal_id = -99
        print("WARNING: the galaxy was not detected")

    objs_id = [i for i in range(len(objects)) if i != gal_id]
    nogal_objs = objects.take(objs_id)
    return gal_obj, nogal_objs


GAIA_COLUMNS = ["ra", "dec", "parallax_over_error", "pmra", "pmra_error", "pmdec", "pmdec_error"]
GAIA_MIN_QUERY_RADIUS = 0.15  # minimum radius of the Gaia queries (deg), so that the cached stars can be reused
GAIA_CACHE_FILE = "gaia_dr3_stars.ecsv"  # name of the file with the Gaia stars of a target (see find_gaia_objects())
# Gaia partner data centres (TAP service, table), used if the ESA Gaia archive fails
GAIA_MIRRORS = [
    ("https://gaia.ari.uni-heidelberg.de/tap", "gaiadr3.gaia_source"),
    ("https://gaia.aip.de/tap", "gaiadr3.gaia_source"),
    ("https://dc.g-vo.org/tap", "gaia.dr3lite"),
]
_gaia_cache: dict = {}  # in-memory cache: (ra, dec) -> Gaia DR3 table (with the query radius in its metadata)


def _query_gaia_esa(ra: float, dec: float, rad: float):
    """Queries the Gaia DR3 sources within a radius from the ESA Gaia archive."""
    from astroquery.gaia import Gaia

    query = (
        f"SELECT {', '.join(GAIA_COLUMNS)} FROM gaiadr3.gaia_source "
        f"WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {rad})) = 1"
    )
    with suppress_stdout():
        return call_with_timeout(
            lambda: Gaia.launch_job_async(query).get_results(), label="Gaia DR3 query (ESA)"
        )  # (no row limit)


def _query_gaia_mirror(ra: float, dec: float, rad: float, tap_url: str, table: str):
    """Queries the Gaia DR3 sources within a radius from a Gaia partner data centre (TAP service), with the
    column names of the ESA Gaia archive."""
    import pyvo

    query = (
        "SELECT ra, dec, parallax, parallax_error, pmra, pmra_error, pmdec, pmdec_error "
        f"FROM {table} WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {rad}))"
    )
    result = call_with_timeout(
        lambda: pyvo.dal.TAPService(tap_url).run_sync(query, maxrec=10_000_000).to_table(),
        attempts=1,
        label=f"Gaia DR3 query ({tap_url})",
    )
    # (failed requests could return an empty result instead of an error; Gaia always has sources within the
    # minimum query radius, so an empty result is not used, nor cached)
    if len(result) == 0:
        raise ValueError(f"empty result from {tap_url}")
    with np.errstate(invalid="ignore", divide="ignore"):
        result["parallax_over_error"] = np.ma.filled(
            np.ma.asarray(result["parallax"], dtype=float) / np.ma.asarray(result["parallax_error"], dtype=float),
            np.nan,
        )
    return result[GAIA_COLUMNS]


def _gaia_catalogue(ra: float, dec: float, rad: float, cache_dir: Optional[str] = None):
    """Returns the Gaia DR3 sources within a radius (cached; see :func:`find_gaia_objects`), or ``None`` if
    the queries failed (with a warning)."""
    from pathlib import Path
    from astropy.table import Table

    key = (round(float(ra), 6), round(float(dec), 6))

    def covers(table) -> bool:
        meta = table.meta
        return (round(float(meta.get("ra", np.nan)), 6), round(float(meta.get("dec", np.nan)), 6)) == key and float(
            meta.get("rad", -1)
        ) >= rad

    table = _gaia_cache.get(key)
    if table is None or not covers(table):
        table = None
        cache_file = Path(cache_dir, GAIA_CACHE_FILE) if cache_dir is not None else None
        if cache_file is not None and cache_file.is_file():
            try:
                cached = Table.read(cache_file, format="ascii.ecsv")
                if covers(cached):
                    table = cached
            except Exception:
                pass  # (corrupted file: queried again)
        if table is None:
            query_rad = max(rad, GAIA_MIN_QUERY_RADIUS)
            errors = []
            sources = [("ESA Gaia archive", _query_gaia_esa)] + [
                (url, lambda ra_, dec_, rad_, url=url, tab=tab: _query_gaia_mirror(ra_, dec_, rad_, url, tab))
                for url, tab in GAIA_MIRRORS
            ]
            for source, query in sources:
                try:
                    table = Table(query(ra, dec, query_rad))
                    break
                except Exception as exc:
                    errors.append(f"{source}: {exc!r}"[:300])
            if table is None:
                warnings.warn(
                    f"The Gaia DR3 query failed ({'; '.join(errors)}), so no stars are used for the cross-match.",
                    GaiaWarning,
                )
                return None
            table = table[[c for c in GAIA_COLUMNS]]
            table.meta.update({"ra": key[0], "dec": key[1], "rad": float(query_rad)})
            if cache_file is not None and cache_file.parent.is_dir():
                try:
                    table.write(cache_file, format="ascii.ecsv", overwrite=True)
                except Exception:
                    pass  # (the cache file is optional)
        _gaia_cache[key] = table

    # sources within the requested radius
    coord = SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")
    sources = SkyCoord(ra=np.asarray(table["ra"], dtype=float), dec=np.asarray(table["dec"], dtype=float), unit="deg")
    return table[sources.separation(coord).deg <= rad] if len(table) > 0 else table


def find_gaia_objects(
    ra: float, dec: float, rad: float = 0.15, min_significance: float = 5.0, cache_dir: Optional[str] = None
) -> SkyCoord:
    """Finds stars using the Gaia DR3 catalog for the given coordinates in a given radius.

    Only the sources with a significant proper motion or parallax (above ``min_significance``
    sigma) are returned, as Gaia also detects compact regions of nearby galaxies (e.g. HII regions),
    which are not stars (e.g. ~5-25 times more sources within 1 arcmin of nearby spiral galaxies
    than in the surrounding field, without this selection).

    The ESA Gaia archive is queried (or, if it fails, the Gaia partner data centres, ``GAIA_MIRRORS``), with a radius of at least
    ``GAIA_MIN_QUERY_RADIUS``, and the sources are cached in memory and, if ``cache_dir`` is given (e.g. the
    directory of the object), in a file (``GAIA_CACHE_FILE``), so that the masks of several filters and surveys
    of the same object (and later runs) do not repeat the query.

    Parameters
    ----------
    ra: Right ascension in degrees.
    dec: Declination in degrees.
    rad: Search radius in degrees.
    min_significance: Minimum significance of the proper motion or parallax.
    cache_dir: Directory for the cache file (no file if ``None``).

    Returns
    -------
    gaia_coord: Coordinates of the stars found.
    """
    gaia_cat = _gaia_catalogue(ra, dec, rad, cache_dir)
    if gaia_cat is None or len(gaia_cat) == 0:
        return SkyCoord([], [], unit="deg")

    def values(column):
        return np.ma.filled(np.ma.asarray(gaia_cat[column], dtype=float), np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        pm_significance = np.hypot(values("pmra") / values("pmra_error"), values("pmdec") / values("pmdec_error"))
        is_star = (pm_significance > min_significance) | (values("parallax_over_error") > min_significance)
    gaia_coord = SkyCoord(
        ra=values("ra")[is_star], dec=values("dec")[is_star], unit=(u.degree, u.degree), frame="icrs"
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
    from astroquery.mast import Catalogs
    coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame="icrs")
    cat_data = call_with_timeout(
        Catalogs.query_criteria, catalog="Tic", radius=rad, coordinates=coord, objType="STAR",
        label="TIC (MAST) query",
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
    if len(objects) == 0 or len(coord) == 0:
        return objects[:0]
    # angular separations on the sky (i.e. including cos(dec) and the RA wrap at 0/360 deg)
    objs_coord = img_wcs.pixel_to_world(objects["x"], objects["y"])
    _, sep2d, _ = objs_coord.match_to_catalog_sky(coord)
    objs_id = np.where(sep2d.arcsec <= dist_thresh)[0]

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
    add_centre: bool = False,
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
    if add_centre:
        fig.show_markers(
            objects["x"],
            objects["y"],
            edgecolor="k",
            facecolor="red",
            coords_frame="pixel",
            marker="o",
            s=50,
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
