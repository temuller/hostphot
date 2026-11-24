"""Quality flags of the photometry.

Some issues do not prevent the photometry from being measured, but make it less reliable. They are reported
with a warning and with a boolean flag in the photometry outputs (e.g. ``{filt}_coverage_flag``), so that the
users can decide whether to use each measurement (survey, image or filter).

The flags set during the photometry are found from the categories of the warnings (subclasses of
:class:`FlagWarning`), recorded with :func:`collect_flags`. The flags set when downloading or masking the images
are stored in their headers (see :data:`HEADER_FLAGS`), so they are only available for images downloaded or
masked with HostPhot v4.0.0 or later.

Flags:

- ``bkg``: the subtracted background is unreliable in the aperture (e.g. the galaxy fills most of the image);
  see :func:`hostphot.processing.background.background_flag`.
- ``coverage``: part of the aperture has no data (e.g. image edge or chip gap). Local apertures: the flux is
  measured from the remaining pixels and might be underestimated (by up to 5%). Global apertures: the estimated
  missing flux (up to 10%, assuming a symmetric galaxy) is added to the flux, and half of it to the uncertainty.
- ``psf``: the aperture radius is smaller than the PSF FWHM (local photometry only).
- ``err``: the blank-aperture noise floor could not be estimated, or was extrapolated from smaller apertures,
  so the uncertainty might be underestimated; or it was estimated from blank apertures far from the source,
  as those near it were contaminated (e.g. by faint galaxy light).
- ``aperture``: the global aperture might not include all the flux of the galaxy (global photometry only): the
  optimised Kron aperture might not fit in the image; or the profile of the curve-of-growth aperture was still
  significant at the edge of the image, or its Sérsic tail could not be estimated (see :func:`hostphot.photometry.global_photometry.cog_aperture`).
- ``calib``: less reliable calibration: VISTA image with a zeropoint uncertainty above the threshold
  (``MAGZRR``), KiDS image without the DR5 zeropoint correction (``DMAG``), or HST filter without a tabulated
  zeropoint uncertainty (2% assumed).
- ``fallback_image``: Spitzer post-BCD mosaic of the Spitzer Heritage Archive used instead of the SEIP mosaic.
- ``mask_fallback``: the host model could not be fitted when masking (``method="host_model"``), so the sources
  were masked with their ellipses.
- ``mask_nogaia``: the Gaia DR3 query failed when masking (``crossmatch=True``), so the stars could not be
  identified.
- ``psf_match``: the PSF matching of the image (local photometry with ``psf_match``) is less reliable: the PSF of
  the image is already broader than the target, so it was not matched; the FWHM of the image or of the target could
  not be measured from stars (typical value of the filter used); or the image has a PSF with strong wings (Spitzer,
  WISE, unWISE, GALEX), not modelled by the matching kernel.
"""

import warnings
from contextlib import contextmanager
from typing import Iterator, Optional

from astropy.io import fits


class FlagWarning(UserWarning):
    """Warning of an issue that sets a quality flag (given by the ``flag`` attribute)."""

    flag: str = ""


class BackgroundWarning(FlagWarning):
    flag = "bkg"


class CoverageWarning(FlagWarning):
    flag = "coverage"


class PSFWarning(FlagWarning):
    flag = "psf"


class ErrorFloorWarning(FlagWarning):
    flag = "err"


class ApertureWarning(FlagWarning):
    flag = "aperture"


class CalibrationWarning(FlagWarning):
    flag = "calib"


class FallbackImageWarning(FlagWarning):
    flag = "fallback_image"


class MaskFallbackWarning(FlagWarning):
    flag = "mask_fallback"


class GaiaWarning(FlagWarning):
    flag = "mask_nogaia"


class PSFMatchWarning(FlagWarning):
    flag = "psf_match"


# flags of the measurement in each aperture (local photometry)
APERTURE_FLAGS = ("bkg", "coverage", "psf", "err")
# flags of the image (the same for all the apertures)
IMAGE_FLAGS = ("calib", "fallback_image", "mask_fallback", "mask_nogaia")
# image flags of the local photometry only
LOCAL_IMAGE_FLAGS = IMAGE_FLAGS + ("psf_match",)
LOCAL_FLAGS = APERTURE_FLAGS + LOCAL_IMAGE_FLAGS
GLOBAL_FLAGS = ("bkg", "coverage", "err", "aperture") + IMAGE_FLAGS

# header keywords of the flags set when downloading or masking the images
HEADER_FLAGS = {
    "calib": "HPFCALIB",
    "fallback_image": "HPFFALLB",
    "mask_fallback": "HPFMSKFB",
    "mask_nogaia": "HPFNOGAI",
}
HEADER_FLAG_COMMENTS = {
    "calib": "HostPhot flag: less reliable calibration",
    "fallback_image": "HostPhot flag: fallback image used",
    "mask_fallback": "HostPhot flag: host model failed (ellipses)",
    "mask_nogaia": "HostPhot flag: Gaia query failed when masking",
}


def set_header_flag(header: fits.Header, flag: str) -> None:
    """Sets a flag (see :data:`HEADER_FLAGS`) in the header of an image."""
    header[HEADER_FLAGS[flag]] = (True, HEADER_FLAG_COMMENTS[flag])


def header_flags(header: Optional[fits.Header]) -> set[str]:
    """Flags set in the header of an image (see :data:`HEADER_FLAGS`)."""
    if header is None:
        return set()
    return {flag for flag, key in HEADER_FLAGS.items() if bool(header.get(key, False))}


@contextmanager
def collect_flags() -> Iterator[set[str]]:
    """Records the flags of the warnings raised within the context (see :class:`FlagWarning`).

    The warnings are raised again when leaving the context, so they are shown (or filtered) as usual. The
    yielded set is filled with the flags when leaving the context.

    Examples
    --------
    >>> with collect_flags() as flags:
    ...     warnings.warn("...", CoverageWarning)
    >>> flags
    {'coverage'}
    """
    flags: set[str] = set()
    caught: list = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            yield flags
    finally:
        # (also if an exception is raised, so that the warnings are not lost)
        for warning in caught:
            if issubclass(warning.category, FlagWarning):
                flags.add(warning.category.flag)
        for warning in caught:
            warnings.warn_explicit(warning.message, warning.category, warning.filename, warning.lineno)
