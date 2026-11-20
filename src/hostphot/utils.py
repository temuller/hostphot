import os
import time
import sys
import threading
import csv
import requests
import numpy as np
import pandas as pd
from io import BytesIO
from pathlib import Path
from typing import Optional
import matplotlib.image as img
import matplotlib.pyplot as plt
from contextlib import contextmanager
from datetime import datetime, timezone
from numpy.lib import recfunctions as rfn

import aplpy
from astropy.io import fits
from hostphot._constants import font_family

import warnings
from astropy.utils.exceptions import AstropyWarning

# maximum waiting time (seconds) for an archive query or a small download (cutouts); large downloads (whole maps
# or observations, e.g. Herschel, HST and JWST) get LARGE_DOWNLOAD_FACTOR times longer
NETWORK_TIMEOUT = 300.0
LARGE_DOWNLOAD_FACTOR = 4


def set_network_timeout(seconds: float) -> None:
    """Sets the maximum waiting time (in seconds) for the archive queries and downloads (300 s by default;
    large downloads of whole maps or observations get 4 times longer)."""
    global NETWORK_TIMEOUT
    NETWORK_TIMEOUT = float(seconds)


def network_timeout(large: bool = False) -> float:
    """Returns the current network timeout in seconds (longer for large downloads, see :data:`NETWORK_TIMEOUT`)."""
    return NETWORK_TIMEOUT * (LARGE_DOWNLOAD_FACTOR if large else 1)


def call_with_timeout(func, *args, timeout: Optional[float] = None, attempts: int = 2, label: str = "", **kwargs):
    """Calls a function that accesses an archive, with a timeout and retries.

    Some services (and the libraries used to access them, e.g. astroquery or pyvo) can wait indefinitely when a
    server stops responding. The function is run in a separate (daemon) thread, and if it does not finish within
    the timeout, a :class:`TimeoutError` is raised (the thread is left behind). Other errors are retried
    (``attempts`` times in total, with increasing waits), as servers can fail temporarily; a timeout is not
    retried (the server is unlikely to respond on a second attempt).

    Parameters
    ----------
    func: Function to call.
    args, kwargs: Arguments of the function.
    timeout: Timeout in seconds (``NETWORK_TIMEOUT`` if ``None``).
    attempts: Total number of attempts.
    label: Description of the call, used in the error message (e.g. "SDSS frame query").

    Returns
    -------
    result: Output of the function.
    """
    timeout = NETWORK_TIMEOUT if timeout is None else timeout
    error = None
    for attempt in range(attempts):
        result, exc = {}, {}

        def target():
            try:
                result["value"] = func(*args, **kwargs)
            except BaseException as e:  # (re-raised in the calling thread)
                exc["value"] = e

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            name = label or getattr(func, "__name__", "archive call")
            raise TimeoutError(
                f"{name}: no response within {timeout:.0f} s (the archive might be down; the timeout can be "
                "changed with hostphot.utils.set_network_timeout())."
            )
        if "value" not in exc:
            return result.get("value")
        error = exc["value"]
        if attempt < attempts - 1:
            time.sleep(2 * (attempt + 1))
    raise error


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


def open_fits_from_url(url: str) -> fits.hdu:
    """Opens a FITS file from a URL.

    Parameters
    ----------
    url: Link to the file.

    Returns
    -------
    hdu: FITS image.
    """
    import gzip
    r = requests.get(url, timeout=NETWORK_TIMEOUT)
    r.raise_for_status()

    content = r.content
    if content.startswith(b'\x1f\x8b'):
        content = gzip.decompress(content)

    hdu = fits.open(BytesIO(content))
    return hdu

def image_has_data(data: np.ndarray) -> bool:
    """Checks whether an image contains data, i.e. it is not constant, (almost) all zeros, or just the
    uniform noise left by image compression (as returned by some services where there is no data).

    For uniformly distributed values, the range between the 0.1 and 99.9 percentiles is ~2.7 times
    the robust standard deviation (from the median absolute deviation), while it is >~6 times for
    (Gaussian) sky noise, and larger with sources.

    Parameters
    ----------
    data: Image data.

    Returns
    -------
    has_data: Whether the image contains data.
    """
    values = np.asarray(data, dtype=float)
    # exact zeros are considered as pixels without data (e.g. outside the footprint)
    values = values[np.isfinite(values) & (values != 0)]
    if values.size < 0.05 * np.size(data):
        return False  # (almost) no pixels with data
    if np.std(values) == 0:
        return False  # constant image
    robust_std = 1.4826 * np.median(np.abs(values - np.median(values)))
    if robust_std == 0:
        return True  # mostly identical values with real ones (e.g. counts)
    low, high = np.percentile(values, [0.1, 99.9])
    return bool((high - low) / robust_std > 4)


# radius (in arcsec) around the position of the target within which an image must have data to be kept
TARGET_DATA_RADIUS = 3.0


def target_has_data(header: fits.Header, data: np.ndarray, ra: float, dec: float,
                    radius_arcsec: float = TARGET_DATA_RADIUS) -> bool:
    """Checks whether an image has data (finite, non-zero pixels) at the position of the target, i.e. within
    ``radius_arcsec`` of it (or the nearest pixels, for large pixels), e.g. not outside the image, nor in an
    area without coverage at the edge of the footprint (filled with zeros or NaN by some services).

    Parameters
    ----------
    header: Header of the image (with its WCS).
    data: Image data.
    ra, dec: Position of the target in degrees.
    radius_arcsec: Radius in arcsec.

    Returns
    -------
    has_data: Whether the image has data at the position of the target.
    """
    from astropy.wcs import WCS

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img_wcs = WCS(header, naxis=2)
            x, y = img_wcs.world_to_pixel_values(ra, dec)
            pixel_scale = np.sqrt(np.abs(np.linalg.det(img_wcs.pixel_scale_matrix))) * 3600
    except Exception:
        return True  # (not checked without a valid WCS)
    if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(pixel_scale) and pixel_scale > 0):
        return True
    values = np.asarray(data)
    if values.ndim != 2:
        return True
    ny, nx = values.shape
    r = max(radius_arcsec / pixel_scale, 1.5)
    yy, xx = np.ogrid[:ny, :nx]
    within = (xx - x) ** 2 + (yy - y) ** 2 <= r**2
    if not within.any():
        return False  # target outside the image
    stamp = values[within]
    return bool(np.any(np.isfinite(stamp) & (stamp != 0)))


def url_band(url: str) -> Optional[str]:
    """Returns the value of the ``band`` parameter of an image URL (e.g. of the WFAU archives,
    ``...&band=H&...``), or ``None`` if there is none.
    """
    from urllib.parse import parse_qs, urlparse

    values = parse_qs(urlparse(url).query).get("band")
    return values[0] if values else None


FULL_COVERAGE = 0.99  # the whole cutout is covered (allowing for rounding at the edges)


def _cutout_box(img_wcs, ra: float, dec: float, size_arcsec: float) -> tuple[int, int, int, int] | None:
    """Pixel limits (x0, y0, x1, y1) of a square cutout centred on the given position in an image
    (``None`` if the position cannot be converted into pixels)."""
    try:
        x, y = img_wcs.world_to_pixel_values(ra, dec)
        pixel_scale = np.sqrt(np.abs(np.linalg.det(img_wcs.pixel_scale_matrix))) * 3600
    except Exception:
        return None
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    npix = max(int(round(size_arcsec / pixel_scale)), 1)  # cutout size in pixels
    x0 = int(round(x - npix / 2))
    y0 = int(round(y - npix / 2))
    x1, y1 = x0 + npix, y0 + npix
    if npix > 2:
        # the outermost pixels are excluded, as the cutouts of the image services can be shifted by one pixel
        x0, y0, x1, y1 = x0 + 1, y0 + 1, x1 - 1, y1 - 1
    return x0, y0, x1, y1


def image_coverage(data: np.ndarray, img_wcs, ra: float, dec: float, size_arcsec: float) -> float:
    """Fraction of a square cutout (centred on the given position) covered by an image with valid
    data (finite, non-zero pixels).

    Parameters
    ----------
    data: Image data.
    img_wcs: WCS of the image.
    ra, dec: Position of the cutout centre in degrees.
    size_arcsec: Size of the cutout in arcsec.

    Returns
    -------
    coverage: Fraction of the cutout with valid data (0 if the position is outside the image).
    """
    box = _cutout_box(img_wcs, ra, dec, size_arcsec)
    if box is None:
        return 0.0
    x0, y0, x1, y1 = box
    ny, nx = data.shape
    total = (x1 - x0) * (y1 - y0)
    stamp = data[max(y0, 0) : max(min(y1, ny), 0), max(x0, 0) : max(min(x1, nx), 0)]
    valid = np.isfinite(stamp) & (stamp != 0)
    return float(valid.sum() / total)


def header_coverage(header: fits.Header, ra: float, dec: float, size_arcsec: float) -> float:
    """Fraction of a square cutout (centred on the given position) within the limits of an image, from
    its header alone (WCS and image size; pixels without data are not taken into account, unlike in
    :func:`image_coverage`).

    Parameters
    ----------
    header: Header of the image.
    ra, dec: Position of the cutout centre in degrees.
    size_arcsec: Size of the cutout in arcsec.

    Returns
    -------
    coverage: Fraction of the cutout within the image (0 if the position is outside the image).
    """
    from astropy.wcs import WCS

    try:
        nx, ny = int(header["NAXIS1"]), int(header["NAXIS2"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img_wcs = WCS(header, naxis=2)
    except Exception:
        return 0.0
    box = _cutout_box(img_wcs, ra, dec, size_arcsec)
    if box is None:
        return 0.0
    x0, y0, x1, y1 = box
    overlap_x = max(min(x1, nx) - max(x0, 0), 0)
    overlap_y = max(min(y1, ny) - max(y0, 0), 0)
    return float(overlap_x * overlap_y / ((x1 - x0) * (y1 - y0)))


def read_fits_header(url: str, max_blocks: int = 400) -> fits.Header | None:
    """Reads the primary header of a remote FITS file, downloading only the beginning of the file
    (the response is streamed until the end of the header).

    Parameters
    ----------
    url: Link to the file (plain or gzip-compressed FITS).
    max_blocks: Maximum number of 2880-byte FITS blocks read.

    Returns
    -------
    header: Primary header (``None`` if it could not be read).
    """
    import zlib

    try:
        with requests.get(url, stream=True, timeout=NETWORK_TIMEOUT) as response:
            response.raise_for_status()
            content = b""
            decompressor = None
            for chunk in response.iter_content(chunk_size=2880 * 4):
                if decompressor is None:
                    # gzip-compressed files are decompressed as they are read
                    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS) if chunk[:2] == b"\x1f\x8b" else False
                content += decompressor.decompress(chunk) if decompressor else chunk
                # the header ends with the END card (at the start of an 80-character card)
                n_cards = len(content) // 80
                cards = (content[80 * i : 80 * i + 3] for i in range(n_cards))
                end = next((i for i, card in enumerate(cards) if card == b"END"), None)
                if end is not None:
                    return fits.Header.fromstring(content[: 80 * (end + 1)].decode("ascii", errors="replace"))
                if len(content) > max_blocks * 2880:
                    return None
    except Exception:
        return None
    return None


def select_image(
    candidates: list,
    ra: float,
    dec: float,
    size_arcsec: float,
    label: str = "",
) -> Optional[int]:
    """Selects the best image among several candidates covering a position (e.g. overlapping tiles).

    Among the images covering the whole requested cutout, the deepest one is selected; if none
    covers the whole cutout, the one with the largest coverage is selected (with a warning).

    Parameters
    ----------
    candidates: List of (data, wcs, depth) tuples, where ``depth`` is a measure of the depth of
        the image (e.g. exposure time; larger is deeper).
    ra, dec: Position of the cutout centre in degrees.
    size_arcsec: Size of the cutout in arcsec.
    label: Description used in the warnings (e.g. survey and filter).

    Returns
    -------
    index: Index of the selected candidate, or ``None`` if no image covers the position.
    """
    if len(candidates) == 0:
        return None
    depths = np.array([depth if depth is not None and np.isfinite(depth) else -np.inf for _, _, depth in candidates])
    # check the images from the deepest to the shallowest (keeping the order if equally deep),
    # stopping at the first one covering the whole cutout
    coverages = np.zeros(len(candidates))
    for i in np.argsort(-depths, kind="stable"):
        data, img_wcs, _ = candidates[i]
        coverages[i] = image_coverage(data, img_wcs, ra, dec, size_arcsec)
        if coverages[i] >= FULL_COVERAGE:
            return int(i)
    if coverages.max() <= 0:
        return None
    best = int(np.argmax(coverages))
    warnings.warn(
        f"{label}: none of the available images covers the whole cutout; using the one with the largest "
        f"coverage ({100 * coverages[best]:.0f}% of the requested area)."
    )
    return best


def open_fits_from_urls(
    urls: list[str | None], max_workers: Optional[int] = None, attempts: int = 2, wait: float = 2.0
) -> list[fits.HDUList | None]:
    """Opens multiple FITS files from URLs in parallel.

    Parameters
    ----------
    urls: List of links to the files.
    max_workers: Maximum number of parallel downloads (default of ``ThreadPoolExecutor`` if ``None``).
    attempts: Number of attempts for each download (with increasing waits between them).
    wait: Wait before the second attempt in seconds (``wait * n`` before the ``n+1``-th attempt).

    Returns
    -------
    hdus: List of FITS images.
    """
    from concurrent.futures import ThreadPoolExecutor

    def safe_open(url: str | None) -> fits.HDUList | None:
        if url is None:
            return None
        # retries, as servers can fail temporarily (e.g. with parallel requests)
        for attempt in range(attempts):
            try:
                return open_fits_from_url(url)
            except Exception as exc:
                error = exc
                if attempt < attempts - 1:
                    time.sleep(wait * (attempt + 1))
        warnings.warn(f"The image could not be downloaded from {url}: {error!r}")
        return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        hdus = list(executor.map(safe_open, urls))
    return hdus

def add_fields(arr: np.ndarray, names: str | list, dtypes: str | list, data: np.ndarray) -> np.ndarray:
    """Add new fields to a NumPy structured array.

    Parameters
    ----------
    arr: Original structured array.
    names: Name(s) of the new fields.
    dtype: Dtype(s) of the new fields (e.g. 'f8', 'i4').
    data: Data for the new fields.

    Returns
    -------
    new_arr: New structured array with added fields.
    """
    if isinstance(names, str):
        names = [names]
    if isinstance(dtypes, (str, np.dtype)):
        dtypes = [dtypes]
    if not isinstance(data, (list, tuple)):
        data = [data]
        
    new_arr = rfn.append_fields(arr, names, data, dtypes=dtypes, usemask=False)

    return new_arr

_stdout_lock = threading.Lock()
_stdout_state = {"depth": 0, "saved": None, "devnull": None}


def orient_axis_labels(fig: aplpy.FITSFigure) -> bool:
    """Labels the axes of an aplpy figure with the coordinate that runs along each of them, for images rotated by
    more than 45 degrees (e.g. UHS), where RA runs along the y axis and Dec along the x axis. Afterwards, the
    ``x``/``y`` methods of ``fig.tick_labels`` and ``fig.axis_labels`` (e.g. ``hide_y()``) refer to the
    coordinate along each axis. The figures of other images are unchanged.

    Parameters
    ----------
    fig: Figure (before setting the labels).

    Returns
    -------
    swapped: Whether the axes of the image are rotated (labels swapped).
    """
    matrix = fig.ax.wcs.pixel_scale_matrix  # rows: world coordinates; columns: pixel axes
    if abs(matrix[0, 1]) <= abs(matrix[0, 0]):
        return False
    x_text, y_text = fig.axis_labels._x_text, fig.axis_labels._y_text
    for labels in (fig.tick_labels, fig.axis_labels):
        labels.x, labels.y = labels.y, labels.x
    fig.axis_labels.set_xtext(y_text)
    fig.axis_labels.set_ytext(x_text)
    fig.axis_labels.set_xposition("bottom")
    fig.axis_labels.set_yposition("left")
    fig.tick_labels.set_xposition("bottom")
    fig.tick_labels.set_yposition("left")
    return True


def save_tight_figure(figure: plt.Figure, outfile: str | Path) -> None:
    """Saves a figure as ``figure.savefig(outfile, bbox_inches="tight")`` does (identical output),
    with a cheaper layout pass: the images, clipped to their axes, are not resampled to find the
    bounding box.

    Parameters
    ----------
    figure: Figure to save.
    outfile: Path of the output file.
    """
    if plt.rcParams["savefig.dpi"] != "figure":
        # the bounding box below is found at the figure's dpi
        figure.savefig(outfile, bbox_inches="tight")
        return
    images = [image for ax in figure.axes for image in ax.images if image.get_visible()]
    for image in images:
        image.set_visible(False)
    try:
        figure.draw_without_rendering()
        bbox = figure.get_tightbbox()
    finally:
        for image in images:
            image.set_visible(True)
    figure.savefig(outfile, bbox_inches=bbox.padded(plt.rcParams["savefig.pad_inches"]))


@contextmanager
def suppress_stdout():
    """Suppresses annoying outputs.

    Useful with astroquery and aplpy packages. It can be used from several threads at the same time
    (e.g. parallel downloads): the output is restored when the last one finishes.
    """
    with _stdout_lock:
        if _stdout_state["depth"] == 0:
            _stdout_state["saved"] = sys.stdout
            _stdout_state["devnull"] = open(os.devnull, "w")
            sys.stdout = _stdout_state["devnull"]
        _stdout_state["depth"] += 1
    try:
        yield
    finally:
        with _stdout_lock:
            _stdout_state["depth"] -= 1
            if _stdout_state["depth"] == 0:
                sys.stdout = _stdout_state["saved"]
                _stdout_state["devnull"].close()
            
def add_provenance(
    header: fits.Header,
    survey: str,
    version: Optional[str] = None,
    size: Optional[float] = None,
    source: Optional[str] = None,
) -> None:
    """Adds provenance keywords to the header of an image saved by HostPhot: the HostPhot version (``HPVERS``),
    the time it was saved (``HPDATE``, UTC), the survey (``HPSURVEY``) and, if given, the version of the survey
    (``HPDATAV``, e.g. a data release; the version written by the download function, i.e. the one actually used,
    is kept), the requested size (``HPSIZE``, in arcmin) and the source file
    (``HPSOURCE``, e.g. for images given by the user).

    Parameters
    ----------
    header: Header of the image (updated in place).
    survey: Survey name.
    version: Version of the survey (the ``version`` parameter of the downloads).
    size: Requested image size in arcmin.
    source: Source of the image (e.g. a file name).
    """
    from hostphot._version import __version__

    header["HPVERS"] = (__version__, "HostPhot version")
    header["HPDATE"] = (datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"), "Date saved by HostPhot (UTC)")
    header["HPSURVEY"] = (survey, "Survey (HostPhot)")
    if version is not None and "HPDATAV" not in header:
        # the version actually used, if written by the download function (e.g. the default data release,
        # or the VISTA survey or GALEX project chosen for the position), is kept
        set_data_version(header, version)
    if size is not None:
        header["HPSIZE"] = (round(float(size), 4), "Requested image size in arcmin (HostPhot)")
    if source is not None:
        header["HPSOURCE"] = (str(source)[-68:], "Source of the image (HostPhot)")


def set_data_version(header: fits.Header, version: str) -> None:
    """Writes the version of the survey used for an image (``HPDATAV``, e.g. a data release) in its header.

    Parameters
    ----------
    header: Header of the image (updated in place).
    version: Version of the survey.
    """
    header["HPDATAV"] = (str(version), "Survey version/data release (HostPhot)")


# header keywords identifying the source image of each survey (see :func:`image_provenance`)
IMAGE_ID_KEYWORDS = {
    "PanSTARRS": ["SKYCELL"],
    "DES": ["TILENAME"],
    "SDSS": ["RUN", "CAMCOL", "FRAME"],
    "GALEX": ["TILENAME"],
    "2MASS": ["ORDATE", "SCANNO"],
    "UKIDSS": ["PROJECT", "OBJECT"],
    "UHS": ["PROJECT", "OBJECT"],
    "VISTA": ["PROJECT", "DATE-OBS"],
    "KiDS": ["ORIGFILE"],
    "SkyMapper": ["OBJECT", "DATAREL"],
    "SPLUS": ["FILENAME"],
    "Euclid": ["TILEID"],
    "HST": ["HPSOURCE", "FILENAME"],
    "JWST": ["HPSOURCE", "FILENAME"],
    "Herschel": ["OBSID", "PRODLVL"],
}
# labels of the keywords in the identifiers (e.g. "run 752 camcol 2 frame 483")
_IMAGE_ID_LABELS = {"RUN": "run", "CAMCOL": "camcol", "FRAME": "frame", "SCANNO": "scan", "TILEID": "tile", "OBSID": "obs"}


def image_provenance(header: fits.Header, survey: str) -> tuple[str, str]:
    """Identifier of the source image (from survey-specific header keywords, plus the survey version if given
    when downloading it, ``HPDATAV``) and the date the image was saved by HostPhot (``HPDATE``).

    Parameters
    ----------
    header: Header of the image.
    survey: Survey name.

    Returns
    -------
    image_id: Identifier of the image (empty if not available).
    download_date: Date the image was saved (UTC; empty for images saved by previous versions).
    """
    parts = []
    if survey == "Spitzer" and header.get("ARCHIVE") == "SHA":
        parts.append(f"SHA AOR {header.get('AORKEY')}")
    keywords = IMAGE_ID_KEYWORDS.get(survey, [])
    if survey in ["HST", "JWST"]:
        keywords = [key for key in keywords if key in header][:1]  # the first available file name
    for key in keywords:
        value = header.get(key)
        if value is None or str(value).strip() in ["", "NA", "nan"]:
            continue
        label = _IMAGE_ID_LABELS.get(key)
        parts.append(f"{label} {value}" if label else str(value).strip())
    if len(parts) == 0 and header.get("HPSOURCE") is not None:
        # e.g. the images used for WISE, unWISE and Spitzer (SEIP)
        parts.append(str(header["HPSOURCE"]))
    if header.get("HPDATAV") is not None:
        parts.append(str(header["HPDATAV"]))
    return " ".join(parts), str(header.get("HPDATE", ""))


def store_input(input_params: dict, inputs_file: Path) -> None:
    """Stores the input parameters of a function.

    Parameters
    ----------
    inputs_params: the parameters of a function.
    inputs_file: where to store the parameters.
    """
    from hostphot._version import __version__

    # save input parameters
    now = datetime.now(timezone.utc).isoformat()
    input_params.update({"timestamp": now, "hostphot_version": __version__})
    inputs_df = pd.DataFrame([input_params])
    if not inputs_file.exists():
        inputs_df.to_csv(inputs_file, index=False)
        return

    with open(inputs_file, newline="") as file:
        rows = list(csv.reader(file))
    header = rows[0] if len(rows) > 0 else []
    if header == list(inputs_df.columns):
        inputs_df.to_csv(inputs_file, index=False, mode="a", header=False)
        return

    # the columns have changed (e.g. new parameters in a newer version): merge the old and new
    # columns and rewrite the file, so that it can be read again (missing values are left empty)
    columns = header + [col for col in inputs_df.columns if col not in header]
    old_rows = []
    for row in rows[1:]:
        if len(row) > len(header):
            # rows appended with more values than the header
            if len(row) == len(header) + 1 and "timestamp" not in header:
                # timestamps were added in newer versions (always as the last value)
                extra_cols = ["timestamp"]
            else:
                extra_cols = [col for col in inputs_df.columns if col not in header]
            row_dict = dict(zip(header + extra_cols, row))
        else:
            row_dict = dict(zip(header, row))
        old_rows.append(row_dict)
    old_df = pd.DataFrame(old_rows, columns=columns, dtype=object)
    merged_df = pd.concat([old_df, inputs_df.astype(object)], ignore_index=True)[columns]
    merged_df.to_csv(inputs_file, index=False)
        
            
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
    orient_axis_labels(fig)  # rotated images
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
