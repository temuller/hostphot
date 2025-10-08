import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from astropy import wcs
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
    survey: Survey name: e.g. ``PS1``, ``GALEX``.

    Returns
    -------
    gain: Gain value.
    """
    check_survey_validity(survey)
    if survey == "PS1":
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
        gain = header["CCDGAIN"]
    elif survey == "SkyMapper":
        gain = header["GAIN"]
    elif survey == "SPLUS":
        gain = header["GAIN"]
    elif survey == "UKIDSS":
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
    survey: Survey name: e.g. ``PS1``, ``GALEX``.

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
    elif survey == "UKIDSS":
        readnoise = header["READNOIS"]
    else:
        readnoise = 0.0

    return readnoise


def get_image_exptime(header: fits.Header, survey: str) -> float:
    """Returns the exposure time from an image's header.

    Parameters
    ----------
    header: Header of an image.
    survey: Survey name: e.g. ``PS1``, ``GALEX``.

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
    elif survey == "UKIDSS":
        exptime = header["EXP_TIME"] * header["NEXP"]
    elif survey == "JWST":
        exptime = header["XPOSURE"]
    else:
        exptime = 1.0

    return exptime


def correct_HST_aperture(filt: str, ap_area: float, header: fits.Header) -> float:
    """Get the aperture correction for the given configuration.

    see: https://hst-docs.stsci.edu/wfc3dhb/chapter-9-wfc3-data-analysis/9-1-photometry
         https://www.stsci.edu/hst/instrumentation/acs/data-analysis/zeropoints
         https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration/uvis-photometric-calibration
    
    Parameters
    ----------
    filt: HST filter, e.g. ``WFC3_UVIS_F225W``.
    ap_area: Aperture area.
    header: Header of an image.

    Returns
    -------
    correction: Aperture correction (encircled energy fraction).
    """
    # split instrument and filter
    filt_split = filt.split("_")
    filt = filt_split[-1]
    instrument = filt_split[-2]

    if instrument == "UVIS":
        # either UVIS1 or UVIS2
        instrument = header["APERTURE"]
        # some images have a different APERTURE value
        # see: https://hst-docs.stsci.edu/wfc3ihb/chapter-6-uvis-imaging-with-wfc3/6-4-uvis-field-geometry
        # not sure if this is the correct solution
        if instrument == "UVIS-CENTER":
            instrument = "UVIS2"
        if instrument == "UVIS":
            instrument = "UVIS1"

    # assuming circular aperture
    # for an ellipse, this would take the average of the axes
    ap_radius = np.sqrt(ap_area / np.pi)

    # get correction curve
    ac_file = [
        file
        for file in Path(hostphot_path, "filters", "HST").glob(
            f"*{instrument.lower()}_aper*"
        )
    ][0]
    ac_df = pd.read_csv(ac_file)

    # linear interpolation of the aperture correction
    apertures = np.array(
        [
            float(col.replace("APER#", ""))
            for col in ac_df.columns
            if col.startswith("AP")
        ]
    )
    ap_corr = ac_df[ac_df.FILTER == filt].values[0][2:].astype(float)

    cont_apertures = np.arange(0, 9, 0.01)
    cont_ap_corr = np.interp(cont_apertures, apertures, ap_corr)

    # get the closest value
    ind = np.argmin(np.abs(cont_apertures - ap_radius))
    correction = cont_ap_corr[ind]

    return correction


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
        apertures = EllipticalAperture(center, obj["a"], obj["b"], obj["theta"])
        sky_apertures = apertures.to_sky(img_wcs)

        new_apertures = sky_apertures.to_pixel(img_wcs2)
        obj["x"], obj["y"] = new_apertures.positions
        obj["a"] = new_apertures.a
        obj["b"] = new_apertures.b
        # wether theta is a float or not seems to depend on the python version
        if isinstance(new_apertures.theta, float):
            obj["theta"] = new_apertures.theta
        else:
            obj["theta"] = new_apertures.theta.value

        if flip is True:
            # flip aperture orientation
            obj["theta"] *= -1

    # ratio between ellipse axis of two images
    conv_factor = np.mean(np.copy(objects["a"] / objects_["a"]))

    return objects_, conv_factor
