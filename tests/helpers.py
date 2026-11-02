"""Helpers for the tests: offline test images."""
import os
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits

from hostphot._constants import workdir

DATA_DIR = Path(__file__).parent / "data"


def install_test_images(name: str = "2002fk", survey: str = "PanSTARRS", filters: str = "grizy") -> None:
    """Copies the (compressed) test images into the working directory, as if downloaded with
    ``download_images`` (the PanSTARRS images of the host of SN 2002fk; 3 arcmin cutouts)."""
    survey_dir = Path(workdir, name, survey)
    survey_dir.mkdir(parents=True, exist_ok=True)
    for filt in filters:
        with fits.open(DATA_DIR / f"{survey}_2002fk_{filt}.fits.fz") as hdul:
            image = hdul[1]
            hdu = fits.PrimaryHDU(data=np.asarray(image.data, dtype=np.float32), header=image.header)
            # written to a temporary file and renamed (atomic), as tests running in parallel use the same images
            fd, tmp_file = tempfile.mkstemp(suffix=".fits", dir=survey_dir)
            os.close(fd)
            hdu.writeto(tmp_file, overwrite=True)
            os.replace(tmp_file, survey_dir / f"{survey}_{filt}.fits")
