import os
import sys
from pathlib import Path
from contextlib import contextmanager
import matplotlib.image as img
import matplotlib.pyplot as plt

import aplpy
from astropy.io import fits
from hostphot._constants import font_family

import warnings
from astropy.utils.exceptions import AstropyWarning

def check_work_dir(wokrdir: str | Path) -> None:
    """Checks if the working directory exists. If it
    does not, one is created.

    Parameters
    ----------
    wokrdir: str
        Working directory path.
    """
    work_path = Path(wokrdir)
    if work_path.is_dir() is False:
        work_path.mkdir(parents=True)


@contextmanager
def suppress_stdout():
    """Suppresses annoying outputs.

    Useful with astroquery and aplpy packages.
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
        
            
def plot_fits(fits_file: str | Path | list[fits.ImageHDU], ext: int = 0) -> None:
    """Plots a FITS file.

    Parameters
    ----------
    fits_file: FITS file.
    ext: Extension index.
    """
    if isinstance(fits_file, str) or isinstance(fits_file, Path):
        title = Path(fits_file).name.split()[0]
        fits_file = str(fits_file)
    else:
        title = None

    figure = plt.figure(figsize=(10, 10))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        fig = aplpy.FITSFigure(fits_file, hdu=ext, figure=figure)
    with suppress_stdout():
        fig.show_grayscale(stretch="arcsinh")

    # ticks
    fig.tick_labels.set_font(**{"family": font_family, "size": 18})
    fig.tick_labels.set_xformat("dd.dd")
    fig.tick_labels.set_yformat("dd.dd")
    fig.ticks.set_length(6)
    # ToDo: solve this deprecation warning (Aplpy should do it?)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        fig.axis_labels.set_font(**{"family": font_family, "size": 18})
    # title + theme
    fig.set_title(title, **{"family": font_family, "size": 24})
    fig.set_theme("publication")
    plt.show()


def plot_image(image_file: str | Path) -> None:
    """Plots an image file.

    E.g. '.png' and '.jpg' files.

    Parameters
    ----------
    image_file: Path to an image file.
    """
    _, ax = plt.subplots(figsize=(16, 8))
    im = img.imread(image_file)
    ax.imshow(im)
    plt.axis("off")
    plt.show()
