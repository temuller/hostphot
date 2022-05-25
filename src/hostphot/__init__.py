# -*- coding: utf-8 -*-

from . import utils
from . import cutouts
from . import rgb_images

from . import objects_detect
from . import image_masking
from . import image_cleaning
from . import objects_detect
from . import coadd
from . import dust

from . import local_photometry
from . import global_photometry
from . import interactive_aperture
from ._constants import  __workdir__
from ._version import __version__
from . import _constants

def choose_workdir(workdir):
    """Updates the work directory across
    all the modules.

    Parameters
    ----------
    workdir: str
        Path to the work directory.
    """
    global __workdir__
    __workdir__ = workdir
    _constants._choose_workdir(workdir)
    cutouts._choose_workdir(workdir)
    image_masking._choose_workdir(workdir)
    coadd._choose_workdir(workdir)
    global_photometry._choose_workdir(workdir)
    local_photometry._choose_workdir(workdir)
    interactive_aperture._choose_workdir(workdir)
