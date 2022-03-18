from photutils.utils import calc_total_error
from astropy.stats import sigma_clipped_stats

def calc_sky_unc(image, exptime):
    """Calculates the uncertainty of the image from the
    sky standard deviation, sigma-clipped STD.

    Parameters
    ----------
    image: 2D array
        Image in a 2D numpy array.
    exptime: float
        Exposure time of the image.

    Returns
    -------
    error: float
        Estimated error of the image.
    """

    avg, sky, sky_std = sigma_clipped_stats(image[(image!= 0)],
                                                        sigma=3.0)
    error = calc_total_error(image, sky_std, exptime)

    return error
