import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import sep
from astroquery.gaia import Gaia
from astropy import coordinates as coords, units as u, wcs

def extract_objects(image, host_ra, host_dec, threshold, bkg_sub=False):
    """Extracts objects and their ellipse parameters. The function `sep.extract()`
    is used.

    Parameters
    ----------
    image: str
        Fits file.
    host_ra: float
        Host-galaxy Right ascension of the galaxy in degrees.
    host_dec: float
        Host-galaxy Declination of the galaxy in degrees.
    threshold: float
        Source with flux above `threshold*bkg_rms` are extracted.
        See `sep.extract()` for more information.
    bkg_sub: bool, default `False`
        If `True`, the image gets background subtracted.

    Returns
    -------
    gal_obj: numpy array
        Galaxy object extracted.
    nogal_objs: numpy array
        All objects extracted except for the galaxy.
    """
    header = image.header
    data = image.data
    img_wcs = wcs.WCS(header, naxis=2)

    data = data.astype(np.float64)
    bkg = sep.Background(data)
    bkg_rms = bkg.globalrms
    if bkg_sub:
        data_sub = np.copy(data - bkg)
    else:
        data_sub = np.copy(data)

    # extract objects with Source Extractor
    objects = sep.extract(data_sub, threshold, err=bkg_rms)

    gal_coords = coords.SkyCoord(ra=host_ra*u.degree,
                                 dec=host_dec*u.degree)
    gal_x, gal_y = img_wcs.world_to_pixel(gal_coords)

    # find the galaxy
    x_diff = np.abs(objects['x']-gal_x)
    y_diff = np.abs(objects['y']-gal_y)
    dist = np.sqrt(x_diff**2 + y_diff**2)
    gal_id = np.argmin(dist)
    gal_obj = objects[gal_id:gal_id+1]

    objs_id = [i for i in range(len(objects)) if i!=gal_id]
    nogal_objs = objects.take(objs_id)

    return gal_obj, nogal_objs

def find_gaia_objects(ra, dec, img_wcs, rad=0.1):
    """Finds objects from the Gaia DR2 catalogue for the given
    coordinates in a given radius.

    parameters
    ----------
    ra: float
        Right ascension in degrees.
    dec: float
        Declination in degrees.
    img_wcs: WCS object
        WCS of an image.
    rad: float, default `0.1`
        Search radius in degrees.

    Returns
    -------
    gaia_coord: SkyCoor object
        Coordinates of the objects found.
    pix_coords: tuple
        Tuple with the coordinates in pixels of the objects found.
    """
    coord = SkyCoord(ra=host_ra, dec=host_dec,
                          unit=(u.degree, u.degree), frame='icrs')
    width = u.Quantity(rad, u.deg)
    height = u.Quantity(rad, u.deg)
    gaia_cat = Gaia.query_object_async(coordinate=coord,
                                       width=width, height=height)

    gaia_ra = np.array(gaia_cat['ra'].value)
    gaia_dec = np.array(gaia_cat['dec'].value)
    gaia_coord = SkyCoord(ra=gaia_ra, dec=gaia_dec,
                          unit=(u.degree, u.degree), frame='icrs')
    pix_coords = img_wcs.world_to_pixel(gaia_coord)

    return gaia_coord, pix_coords

def cross_match(objects, img_wcs, coord, dist_thresh=1.0):
    """Cross-matches objects with a given set of coordinates.
    Those with a distance of less than `dist_thresh` are selected.

    Parameters
    ----------
    objects: array
        Objects detected with `sep.extract()`.
    img_wcs: WCS
        WCS of the image from which the objects where extracted.
    coord: SkyCoor object
        Coordinates for the cross-match.
    dist_thresh: float, default `1.0`
        Distance threshold.
    """
    # coordinates in arcsec
    objs_coord = img_wcs.pixel_to_world(objects['x'], objects['y'])
    objs_ra = objs_coord.ra.to(u.arcsec).value
    objs_dec = objs_coord.dec.to(u.arcsec).value

    cat_ra = np.array(coord['ra'].to(u.arcsec).value)
    cat_dec = np.array(coord['dec'].to(u.arcsec).value)

    objs_id = []
    for i in range(len(objects)):
        ra_dist = objs_ra[i] - cat_ra
        dec_dist = objs_dec[i] - cat_dec
        dist_arcsec = np.sqrt(ra_dist**2 + dec_dist**2)
        if any(dist_arcsec <= dist_thresh):
            objs_id.append(i)

    objs = objects.take(objs_id)

    return objs

def plot_detected_objects(data, objects, scale=6):
    """Plots the objects extracted with `sep.extract()``.

    Parameters
    ----------
    data: 2D array
        Data of an image.
    objects: array
        Objects detected with `sep.extract()`.
    scale: float, default `6`
        Scale of the ellipse's semi-mayor and semi-minor axes.
    """
    fig, ax = plt.subplots()
    m, s = np.nanmean(data), np.nanstd(data_sub)
    im = ax.imshow(data_sub, interpolation='nearest',
                   cmap='gray',
                   vmin=m-s, vmax=m+s,
                   origin='lower')

    e = Ellipse(xy=(objects['x'][0], objects['y'][0]),
                width=scale*objects['a'][0],
                height=scale*objects['b'][0],
                angle=objects['theta'][0]*180./np.pi)

    e.set_facecolor('none')
    e.set_edgecolor('red')
    ax.add_artist(e)

    plt.tight_layout()
    plt.savefig(plot_output)
    plt.close(fig)
