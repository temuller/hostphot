.. _information_adding_surveys:

Adding New Surveys
==================

HostPhot is modular, meaning that new surveys con easily be added. However, there are a few things one needs before a new survey can be added: where to download the images from (e.g., using `astroquery`), zero-points to convert the images's flux/counts values into magnitudes, the magnitude system (e.g. AB, Vega), the pixel scaling of the images (in units of arcsec/pixel), the filters transmission functions and any other piece of information necessary to properly estimate magnitudes, errors, etc.


Cutouts
~~~~~~~

Usually, many surveys have their own databases where one can download the image cutouts from (in FITS format). Otherwise, `astroquery.skyview <https://astroquery.readthedocs.io/en/latest/skyview/skyview.html>`_ allows you to download images from many different surveys. Keep in mind that the headers of the images downloaded with the latter option do not contain all the same information that the original surveys' databases do. This can be troublesome in some cases as some essential information (e.g. zero-points) might not be present. For information about the surveys already implemented in HostPhot, check :ref:`Further Information: Cutouts<information_cutouts>`.


Zero-points
~~~~~~~~~~~

The pixel values of the images usually come in units of `counts` or `data number` (DN) instead of physical units (e.g. :math:`\text{erg} \text{s}^{-1} \text{cm}^{-2} \text{Å}^{-1}`). The purpose of this is to make things easier as a single zero-point (ZP) for the entire survey is needed to convert the pixel values to magnitudes. Sometimes, each filter might can a different ZP, and some other times, each image might have their own ZP, found in the header. The ZP is essential as HostPhot's outputs are given in magnitudes. Note that sometimes the calibration is more complex and other steps are needed to obtain correct magnitudes. For information about the surveys already implemented in HostPhot, check :ref:`Further Information: Photometry<information_photometry>`.


Magnitude System
~~~~~~~~~~~~~~~~

Most modern surveys use the AB mangitude system, but some might be in Vega or other systems. This information is always provided by the surveys. Keep in mind this information as it is the user's responsibility to not mix different magnitude systems.


Error Budget
~~~~~~~~~~~~

A proper error propagation is as important as the magnitude calibration. Different surveys have different sources of uncertainties, e.g. from the ZPs, calibration, etc.. This information is ideally provided by the surveys. Usually, the simplest way of estimating the error budget is by using parameters such as the gain, exposure time and readnoise of the images, which are in many cases included in the headers.


Pixel Scaling
~~~~~~~~~~~~~

The pixel scaling of the images is mainly used by HostPhot to download image cutouts with a given size (e.g., in units of arcmin). The scaling also helps the user understand the difference in resolution between different surveys.


Filters Transmission Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are many times provided by the surveys, but they can also be obtained from the `Spanish Virtual Observatory <http://svo2.cab.inta-csic.es/theory/fps/index.php?mode=browse>`_. Just make sure that these are the appropiate ones.
