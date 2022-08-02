# based on https://github.com/tremou/RGB_MSUastro

import os
import math
import requests
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import units as u
from astropy.table import Table

from hostphot.utils import survey_pixel_scale


def linear(inputArray, scale_min=None, scale_max=None):
    """Performs linear scaling of the input numpy array.

    Parameters
    ----------
    inputArray: numpy array
            Image data array.
    scale_min: float, default ``None``
            Minimum data value.
    scale_max: float, default ``None``
            Maximum data value.

    Returns
    -------
    imageData: numpy array
            Scaled image data array.
    """
    imageData = np.array(inputArray, copy=True)

    if scale_min == None:
        scale_min = np.nanmin(imageData)
    if scale_max == None:
        scale_max = np.nanmax(imageData)

    imageData = imageData.clip(min=scale_min, max=scale_max)
    imageData = (imageData - scale_min) / (scale_max - scale_min)
    indices = np.where(imageData < 0)
    imageData[indices] = 0.0
    indices = np.where(imageData > 1)
    imageData[indices] = 1.0

    return imageData


def sqrt(inputArray, scale_min=None, scale_max=None):
    """Performs :func:`sqrt` scaling of the input numpy array.

    Parameters
    ----------
    inputArray: numpy array
            Image data array.
    scale_min: float, default ``None``
            Minimum data value.
    scale_max: float, default ``None``
            Maximum data value.

    Returns
    -------
    imageData: numpy array
            Scaled image data array.
    """
    imageData = np.array(inputArray, copy=True)

    if scale_min == None:
        scale_min = np.nanmin(imageData)
    if scale_max == None:
        scale_max = np.nanmax(imageData)

    imageData = imageData.clip(min=scale_min, max=scale_max)
    imageData = imageData - scale_min
    indices = np.where(imageData < 0)
    imageData[indices] = 0.0
    imageData = np.sqrt(imageData)
    imageData = imageData / math.sqrt(scale_max - scale_min)

    return imageData


def log(inputArray, scale_min=None, scale_max=None):
    """Performs :func:`log10` scaling of the input numpy array.

    Parameters
    ----------
    inputArray: numpy array
            Image data array.
    scale_min: float, default ``None``
            Minimum data value.
    scale_max: float, default ``None``
            Maximum data value.

    Returns
    -------
    imageData: numpy array
            Scaled image data array.
    """
    imageData = np.array(inputArray, copy=True)

    if scale_min == None:
        scale_min = np.nanmin(imageData)
    if scale_max == None:
        scale_max = np.nanmax(imageData)

    factor = math.log10(scale_max - scale_min)
    indices0 = np.where(imageData < scale_min)
    indices1 = np.where((imageData >= scale_min) & (imageData <= scale_max))
    indices2 = np.where(imageData > scale_max)
    imageData[indices0] = 0.0
    imageData[indices2] = 1.0
    try:
        imageData[indices1] = np.log10(imageData[indices1]) / factor
    except:
        print("Error on math.log10 for ", (imageData[indices1] - scale_min))

    return imageData


def asinh(inputArray, scale_min=None, scale_max=None, non_linear=2.0):
    """Performs :func:`asinh` scaling of the input numpy array.

    Parameters
    ----------
    inputArray: numpy array
            Image data array.
    scale_min: float, default ``None``
            Minimum data value.
    scale_max: float, default ``None``
            Maximum data value.
    non_linear: float, default ``2.0``
            Non-linearity factor.

    Returns
    -------
    imageData: numpy array
            Scaled image data array.
    """
    imageData = np.array(inputArray, copy=True)

    if scale_min == None:
        scale_min = np.nanmin(imageData)
    if scale_max == None:
        scale_max = np.nanmax(imageData)

    factor = np.arcsinh((scale_max - scale_min) / non_linear)
    indices0 = np.where(imageData < scale_min)
    indices1 = np.where((imageData >= scale_min) & (imageData <= scale_max))
    indices2 = np.where(imageData > scale_max)
    imageData[indices0] = 0.0
    imageData[indices2] = 1.0
    imageData[indices1] = (
        np.arcsinh((imageData[indices1] - scale_min) / non_linear) / factor
    )

    return imageData


def create_RGB_image(
    outfile=None,
    survey="PS1",
    filters="zir",
    images_dir="",
    scaling="linear",
    scaling_params=None,
):
    """Creates an RGB image.

    Parameters
    ----------
    outfile: str, default ``None``
        Output file name. If `None`, no output is saved.
    survey: str, default ``PS1``
        Survey used for the filters.
    filters: str, defaul ``zir``
        Filters used to create the RGB image (in that order). Must be three filters.
    images_dir: str, default `''`
        Directory where to find the images.
    scaling: str
        Type of scaling. Choose between ``linear``, ``sqrt``, ``log`` and ``asinh``.
    scaling_params: dict
        Dictionary with the parameters for the ``scaling`` function. If ``None``,
        use the default values.

    Examples
    --------
    >>> # Example of scaling_params:
    >>> scaling_params = {'R':{'min':r_med/100, 'max':r_med*30},
                                      'G':{'min':g_med/1000, 'max':g_med*150},
                                      'B':{'min':b_med/30, 'max':b_med*100}}
    """

    assert len(filters) == 3, "Three filters should be given."
    scalings = ["linear", "sqrt", "log", "asinh"]
    assert (
        scaling in scalings
    ), f"Not a valid scaling (choose between {scalings})"

    rgb_files = []
    for filt in filters:
        filt_image = os.path.join(images_dir, f"{survey}_{filt}.fits")
        rgb_files.append(filt_image)

    r_img = fits.getdata(rgb_files[0], ext=0)
    r_med = np.nanmedian(r_img)
    g_img = fits.getdata(rgb_files[1], ext=0)
    g_med = np.nanmedian(g_img)
    b_img = fits.getdata(rgb_files[2], ext=0)
    b_med = np.nanmedian(b_img)
    imgs = [r_img, g_img, b_img]

    if scaling_params is None:
        scaling_params = {
            "R": {"min": r_med / 100, "max": r_med * 30},
            "G": {"min": g_med / 1000, "max": g_med * 150},
            "B": {"min": b_med / 30, "max": b_med * 100},
        }
    scaling_func = eval(scaling)  # scaling function

    img = np.zeros((r_img.shape[0], r_img.shape[1], 3), dtype=float)
    for i, filt_img in enumerate(imgs):
        img_scalings = list(scaling_params.values())[i]
        scale_min, scale_max = img_scalings.values()
        img[:, :, i] = scaling_func(
            filt_img, scale_min=scale_min, scale_max=scale_max
        )

    fig = plt.figure(figsize=(12, 12))
    plt.clf()
    plt.imshow(img, origin="lower")
    plt.title(f"Blue = {filters[0]}, Green = {filters[1]}, Red = {filters[2]}")
    if outfile is not None:
        plt.savefig(outfile)


# ---------------------------------------


def get_PS1_url(ra, dec, size=3, filters="grizy", data_format="jpg"):
    """Get URL for the colour image.

    Parameters
    ----------
    ra: float
        Right Ascension in degrees.
    dec: float
        Declination in degrees.
    size: float or ~astropy.units.Quantity, default ``3``
        Image size. If a float is given, the units are assumed to be arcmin.
    filters: str
        Filters to include.
    data_format: str
        Data format (options are ``jpg`` or ``png``).

    Returns
    -------
    url: str
        The image's URL for a colour image.
    """
    assert data_format in ["jpg", "png"], (
        "format must be one of " "'jpg' or 'png'"
    )
    assert len(filters) >= 3, "must choose at least 3 filters"

    pixel_scale = survey_pixel_scale("PS1")
    if isinstance(size, (float, int)):
        size_arcsec = (size * u.arcmin).to(u.arcsec).value
    else:
        size_arcsec = size.to(u.arcsec).value
    size_pixels = int(size_arcsec / pixel_scale)

    # get table with images
    service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    table_url = (
        f"{service}?ra={ra}&dec={dec}&size={size_pixels}&format=fits&"
        f"filters={filters}"
    )
    table = Table.read(table_url, format="ascii")

    # url for colour image
    url = (
        "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
        f"ra={ra}&dec={dec}&size={size_pixels}&format={data_format}"
        f"&output_size={size_pixels}"
    )

    # sort filters from red to blue
    flist = ["yzirg".find(x) for x in table["filter"]]
    table = table[np.argsort(flist)]

    if len(table) > 3:
        # pick only 3 filters
        table = table[[0, len(table) // 2, len(table) - 1]]

    for i, param in enumerate(["red", "green", "blue"]):
        url = url + f"&{param}={table['filename'][i]}"

    return url


def get_PS1_RGB_image(outfile, ra, dec, size=3, filters="grizy"):
    """Downloads an RGB image from the PS1 server.

    Parameters
    ----------
    outfile: str
        URL of the image to be downloaded
    ra: float
        Right Ascension in degrees.
    dec: float
        Declination in degrees.
    size: float or ~astropy.units.Quantity, default ``3``
        Image size. If a float is given, the units are assumed to be arcmin.
    filters: str
        Filters to include.

    Returns
    -------
    url: str
        The image's URL for a colour image.
    """
    data_format = os.path.splitext(outfile)[-1][1:]
    remote_url = get_PS1_url(ra, dec, size, filters, data_format)

    data = requests.get(remote_url)
    with open(outfile, "wb") as file:
        file.write(data.content)


def get_SDSS_RGB_image(outfile, ra, dec, size=3):
    """Downloads an RGB image from the SDSS server.

    Parameters
    ----------
    remote_url: str
        URL of the image to be downloaded
    ra: float
        Right Ascension in degrees.
    dec: float
        Declination in degrees.
    size: float or ~astropy.units.Quantity, default ``3``
        Image size. If a float is given, the units are assumed to be arcmin.

    Returns
    -------
    url: str
        The image's URL for a colour image.
    """
    pixel_scale = survey_pixel_scale("SDSS")
    if isinstance(size, (float, int)):
        size_arcsec = (size * u.arcmin).to(u.arcsec).value
    else:
        size_arcsec = size.to(u.arcsec).value

    size_pixels = int(size_arcsec / pixel_scale)

    remote_url = (
        "http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/"
        f"getjpeg?ra={ra}&dec={dec}&scale={pixel_scale}"
        f"&width={size_pixels}&height={size_pixels}"
    )

    data = requests.get(remote_url)
    with open(outfile, "wb") as file:
        file.write(data.content)
