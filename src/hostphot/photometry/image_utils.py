import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from astropy import wcs
import astropy.units as u
from astropy.io import fits
from photutils.aperture import EllipticalAperture

import hostphot
from hostphot.surveys_utils import check_survey_validity


hostphot_path = Path(hostphot.__path__[0])
config_file = hostphot_path.joinpath("filters", "config.txt")
config_df = pd.read_csv(config_file, sep="\\s+")
plt.rcParams["mathtext.fontset"] = "cm"


def pixel2pixel(
    x1: float, y1: float, img_wcs1: wcs.WCS, img_wcs2: wcs.WCS
) -> tuple[float, float]:
    """Convert the pixel coordinates from one image to another."""
    coord1 = img_wcs1.pixel_to_world(x1, y1)
    x2, y2 = img_wcs2.world_to_pixel(coord1)

    return x2, y2


def get_image_gain(header: fits.Header, survey: str) -> float:
    """Returns the gain from an image's header.

    **Note:** for ``SDSS`` this is assumed to be one
    as it should already be included.

    Parameters
    ----------
    header: Header of an image.
    survey: Survey name: e.g. ``PanSTARRS``, ``GALEX``.

    Returns
    -------
    gain: Gain value.
    """
    check_survey_validity(survey)
    if survey == "PanSTARRS":
        gain = header["HIERARCH CELL.GAIN"]
    elif survey == "DES":
        gain = header["GAIN"]
    elif survey == "SDSS":
        gain = 1.0
    elif survey == "2MASS":
        # the value comes from https://wise2.ipac.caltech.edu/staff/jarrett/2mass/3chan/noise/
        # Note that it is different from the value of
        # https://iopscience.iop.org/article/10.1086/498708/pdf (8 e ADU^-1)
        gain = 10.0
    elif survey == "LegacySurvey":
        gain = 1  # not used
    elif survey == "Spitzer":
        # also in the "EXPGAIN" keyword
        if header["INSTRUME"] == "IRAC":
            # Table 2.4 of IRAC Instrument Handbook
            gain = 3.8  # all filters have similar gain
            gain *= header["EFCONV"]  # convert DN/s to MJy/sr
        elif header["INSTRUME"] == "MIPS":
            # Table 2.4 of MIPS Instrument Handbook
            gain = 5.0
    elif survey == "VISTA":
        # use median from http://casu.ast.cam.ac.uk/surveys-projects/vista/technical/vista-gain
        gain = 4.19
    elif survey == "HST":
        # NICMOS has ADCGAIN instead of CCDGAIN
        gain = header.get("CCDGAIN", header.get("ADCGAIN", 1.0))
    elif survey == "SkyMapper":
        gain = header["GAIN"]
    elif survey == "SPLUS":
        gain = header["GAIN"]
    elif survey in ["UKIDSS", "UHS"]:
        gain = header["GAIN"]
    else:
        gain = 1.0

    return gain


def get_image_readnoise(header: fits.Header, survey: str) -> float:
    """Returns the read noise from an image's header.
    All values are per-pixel values.

    **Note:** for ``SDSS`` this is assumed to be zero
    as it should already be included.

    Parameters
    ----------
    header: Header of an image.
    survey: Survey name: e.g. ``PanSTARRS``, ``GALEX``.

    Returns
    -------
    readnoise: Read noise value.
    """
    check_survey_validity(survey)
    if survey == "PanSTARRS":
        readnoise = header["HIERARCH CELL.READNOISE"]
    elif survey == "DES":
        # see https://arxiv.org/pdf/0810.3600.pdf
        readnoise = 7.0  # electrons per pixel
    elif survey == "SDSS":
        readnoise = 0.0
    elif survey == "2MASS":
        # https://iopscience.iop.org/article/10.1086/498708/pdf
        # 6 combined images
        readnoise = 4.5 * np.sqrt(6)  # not used
    elif survey == "LegacySurvey":
        readnoise = 1.0  # not used
    elif survey == "Spitzer":
        if header["INSTRUME"] == "IRAC":
            # Table 2.3 of IRAC Instrument Handbook
            # very rough average
            readnoise_dict = {1: 16.0, 2: 12.0, 3: 10.0, 4: 8.0}
            channel = header["CHNLNUM"]
            readnoise = readnoise_dict[channel]
        elif header["INSTRUME"] == "MIPS":
            # Table 2.4 of MIPS Instrument Handbook
            readnoise = 40.0
    elif survey == "VISTA":
        # very rough average for all filters in
        # http://casu.ast.cam.ac.uk/surveys-projects/vista/technical/vista-gain
        readnoise = 24.0
    elif survey == "HST":
        # tipically 0.0
        readnoise = header["PCTERNOI"]
    elif survey == "SkyMapper":
        # https://rsaa.anu.edu.au/observatories/instruments/skymapper-instrument
        readnoise = 5  # electrons
    elif survey == "SPLUS":
        readnoise = header["HIERARCH OAJ QC NCNOISE"]
    elif survey in ["UKIDSS", "UHS"]:
        readnoise = header["READNOIS"]
    else:
        readnoise = 0.0

    return readnoise


def get_image_exptime(header: fits.Header, survey: str) -> float:
    """Returns the exposure time from an image's header.

    Parameters
    ----------
    header: Header of an image.
    survey: Survey name: e.g. ``PanSTARRS``, ``GALEX``.

    Returns
    -------
    exptime: Exposure time in seconds.
    """
    check_survey_validity(survey)
    if survey in ["PanSTARRS", "DES", "SDSS", "GALEX", "VISTA", "Spitzer", "HST", "SkyMapper"]:
        exptime = float(header["EXPTIME"])
    elif survey == "WISE":
        # see: https://wise2.ipac.caltech.edu/docs/release/allsky/
        # and https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec1_1.html
        if header["BAND"] in [1, 2]:
            exptime = 7.7
        elif header["BAND"] in [3, 4]:
            exptime = 8.8
    elif survey == "2MASS":
        # https://iopscience.iop.org/article/10.1086/498708/pdf
        exptime = 7.8
    elif survey == "LegacySurvey":
        exptime = 1.0  # ???
    elif survey == "SPLUS":
        exptime = header["TEXPOSED"]
    elif survey in ["UKIDSS", "UHS"]:
        # the pixel values of the WFCAM stacks are per exposure, i.e. the NEXP exposures of each
        # integration are averaged (e.g. UHS H stacks have NEXP = 2: multiplying by NEXP gave
        # magnitudes 0.75 mag too faint compared to the UHS catalogue)
        exptime = float(header["EXP_TIME"])
    elif survey == "JWST":
        exptime = header["XPOSURE"]
    else:
        exptime = 1.0

    return exptime


def wrap_theta(theta: float) -> float:
    """Wraps an ellipse position angle (in radians) into the range [-pi/2, pi/2),
    as expected by sep (an ellipse is unchanged by a rotation of pi)."""
    return (theta + np.pi / 2) % np.pi - np.pi / 2


def adapt_aperture(
    objects: np.ndarray, img_wcs: wcs.WCS, img_wcs2: wcs.WCS, flip: bool = False
) -> tuple[np.ndarray, float]:
    """Changes the aperture parameters to consider differences
    in WCS between surveys.

    The values of ``center``, ``a`` and ``b`` should in pixel
    units. DES images are flipped, so these need to be corrected
    with ``theta -> -theta``, i.e. using ``flip=True``.

    Parameters
    ----------
    objects: Objects with apertures.
    img_wcs: WCS of the image from where the objects were extracted.
    img_wcs2: WCS used to adapt the apertures.
    flip: Whether to flip the orientation of the aperture. Only
        used for DES images.

    Returns
    -------
    objects_: Objects with adapted apertures.
    conv_factor: Convertion factor between the resolutions of the images.
    """
    objects_ = objects.copy()  # avoid modifying the intial objects
    for obj in objects_:
        center = (obj["x"], obj["y"])
        apertures = EllipticalAperture(center, obj["a"], obj["b"], theta=obj["theta"])
        sky_apertures = apertures.to_sky(img_wcs)

        new_apertures = sky_apertures.to_pixel(img_wcs2)
        obj["x"], obj["y"] = new_apertures.positions
        obj["a"] = new_apertures.a
        obj["b"] = new_apertures.b
        # photutils returns theta as an angle (in degrees) in recent versions, or as a float (radians)
        if isinstance(new_apertures.theta, u.Quantity):
            theta = new_apertures.theta.to_value(u.rad)
        else:
            theta = float(new_apertures.theta)

        if flip is True:
            # flip aperture orientation
            theta *= -1
        # sep expects theta in the range [-pi/2, pi/2]
        obj["theta"] = wrap_theta(theta)

    # ratio between ellipse axis of two images
    conv_factor = np.mean(np.copy(objects["a"] / objects_["a"]))

    return objects_, conv_factor


# columns with the ellipses of the objects on the sky (saved with the aperture and mask parameters)
SKY_ELLIPSE_COLUMNS = ["ra", "dec", "a_arcsec", "b_arcsec", "pa_deg"]


def objects_to_sky(objects: np.ndarray, img_wcs: wcs.WCS) -> dict:
    """Ellipses of the objects (``sep`` parameters in pixels: ``x``, ``y``, ``a``, ``b``, ``theta``) on the sky.

    The axes and the position angle (east of north) are measured from the sky positions of the ends of the
    semi-major and semi-minor axes, so they are exact whatever the orientation and handedness of the image (e.g.
    images with east to the right).

    Parameters
    ----------
    objects: Objects (structured array).
    img_wcs: WCS of the image where the objects were measured.

    Returns
    -------
    sky: Arrays of the centres (``ra``, ``dec``, in degrees), semi-axes (``a_arcsec``, ``b_arcsec``) and position
        angle of the semi-major axis (``pa_deg``, east of north).
    """
    x, y = np.asarray(objects["x"], float), np.asarray(objects["y"], float)
    a, b, theta = np.asarray(objects["a"], float), np.asarray(objects["b"], float), np.asarray(objects["theta"], float)
    centre = img_wcs.pixel_to_world(x, y)
    major = img_wcs.pixel_to_world(x + a * np.cos(theta), y + a * np.sin(theta))
    minor = img_wcs.pixel_to_world(x - b * np.sin(theta), y + b * np.cos(theta))
    return {
        "ra": centre.icrs.ra.deg,
        "dec": centre.icrs.dec.deg,
        "a_arcsec": centre.separation(major).arcsec,
        "b_arcsec": centre.separation(minor).arcsec,
        "pa_deg": centre.position_angle(major).deg % 180.0,
    }


def add_sky_ellipses(objects_df: pd.DataFrame, objects: np.ndarray, img_wcs: wcs.WCS) -> pd.DataFrame:
    """Adds (or updates) the sky ellipses of the objects (see :func:`objects_to_sky`) to their DataFrame."""
    for key, values in objects_to_sky(objects, img_wcs).items():
        objects_df[key] = values
    return objects_df


def has_sky_ellipses(objects) -> bool:
    """Whether the objects (structured array or DataFrame) include their sky ellipses."""
    names = objects.dtype.names if hasattr(objects, "dtype") and objects.dtype.names else list(objects)
    return all(col in names for col in SKY_ELLIPSE_COLUMNS)


def objects_from_sky(objects: np.ndarray, img_wcs: wcs.WCS) -> tuple[np.ndarray, float]:
    """Ellipses of the objects in the pixels of an image, from their sky ellipses (see :func:`objects_to_sky`).

    This is used to transfer apertures and masks between images (filters or surveys) without the reference image.
    The ends of the semi-major and semi-minor axes are placed on the sky and converted into pixels, which is exact
    whatever the orientation and handedness of the images.

    Parameters
    ----------
    objects: Objects with their sky ellipses (and pixel ellipses in the reference image).
    img_wcs: WCS of the image where the ellipses are needed.

    Returns
    -------
    objects_: Objects with the ellipses in the pixels of the image (``x``, ``y``, ``a``, ``b``, ``theta``).
    conv_factor: Mean ratio between the semi-major axes in the reference image and in this image (pixel scales).
    """
    from astropy.coordinates import SkyCoord

    objects_ = objects.copy()
    centre = SkyCoord(np.asarray(objects["ra"], float), np.asarray(objects["dec"], float), unit="deg", frame="icrs")
    pa = np.asarray(objects["pa_deg"], float) * u.deg
    major = centre.directional_offset_by(pa, np.asarray(objects["a_arcsec"], float) * u.arcsec)
    minor = centre.directional_offset_by(pa + 90 * u.deg, np.asarray(objects["b_arcsec"], float) * u.arcsec)
    x, y = img_wcs.world_to_pixel(centre)
    xa, ya = img_wcs.world_to_pixel(major)
    xb, yb = img_wcs.world_to_pixel(minor)
    objects_["x"], objects_["y"] = x, y
    objects_["a"] = np.hypot(xa - x, ya - y)
    objects_["b"] = np.hypot(xb - x, yb - y)
    objects_["theta"] = [wrap_theta(t) for t in np.arctan2(ya - y, xa - x)]
    conv_factor = float(np.mean(np.asarray(objects["a"], float) / np.asarray(objects_["a"], float)))
    return objects_, conv_factor
