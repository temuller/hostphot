.. _information_cutouts:

Cutouts
=======

The images are downloaded from different sources depending on what was the easiest at the moment of adding the survey.


PS1
~~~

These images are downloaded from the PS1 server directly. The code found at the end of the `PS1 Image Cutout Service <https://outerspace.stsci.edu/display/PANSTARRS/PS1+Image+Cutout+Service#PS1ImageCutoutService-ImportantFITSimageformat,WCS,andflux-scalingnotes>`_ was adapted for HostPhot. The pixel scaling of the images is 0.25 arcsec/pixel, as described in the `PS1 website <https://outerspace.stsci.edu/display/PANSTARRS/PS1+Image+Cutout+Service#PS1ImageCutoutService-ImportantFITSimageformat,WCS,andflux-scalingnotes>`_. The available filters are: :math:`grizy`.


DES
~~~

DES images are downloaded from the `NOAO DataLab <https://datalab.noirlab.edu/sia.php>`_ using the Simple Image Access (SIA) service. Here is an `example notebook <https://github.com/astro-datalab/notebooks-latest/blob/master/04_HowTos/SiaService/How_to_use_the_Simple_Image_Access_service.ipynb>`_. The pixel scaling of the images is 0.263 arcsec/pixel, as described in the `DES website <https://des.ncsa.illinois.edu/releases/dr1/dr1-docs/acquisition>`_. The available filters are: :math:`grizY`.


SDSS
~~~~

These images are downloaded using `astroquery.skyview <https://astroquery.readthedocs.io/en/latest/skyview/skyview.html>`_. The `astroquery.sdss <https://astroquery.readthedocs.io/en/latest/sdss/sdss.html>`_ module is not used as it downloads much larger images (taking longer to download) and thus need to be trimmed. The pixel scaling of the images is 0.396 arcsec/pixel, as described in the `SDSS website <https://www.sdss.org/dr12/imaging/images/>`_. **Note:** it is recommended to use PS1 instead of SDSS as PS1 covers the same patch of the sky as SDSS and more (unless :math:`u`-band is needed). Also, PS1 is better calibrated and goes deeper. The available filters are: :math:`ugriz`.


GALEX
~~~~~

These are downloaded with `astroquery.skyview`. Some header information missing in these images is retrieved from the images obtained with `astroquery.mast.Observations <https://astroquery.readthedocs.io/en/latest/mast/mast.html>`_ (e.g. exposure time). The pixel scaling of the images is 1.5 arcsec/pixel, as described in the `GALEX website <https://asd.gsfc.nasa.gov/archive/galex/FAQ/counts_background.html>`_. The available filters are: :math:`FUV` (far-UV) and :math:`NUV` (near-UV).



2MASS
~~~~~

These images are downloaded from the 2MASS database following `https://irsa.ipac.caltech.edu/ibe/docs/twomass/allsky/allsky/#main <https://irsa.ipac.caltech.edu/ibe/docs/twomass/allsky/allsky/#main>`_. Information such as the `ordate` is first obtained from the last rows of the header of the images downloaded with `PyVo <https://pyvo.readthedocs.io/en/latest/>`_. However, the latter are not used as they are much larger and the 2MASS zero-points are missing in their headers. The pixel scaling of the images is 1.0 arcsec/pixel, as described in the `2MASS website <https://irsa.ipac.caltech.edu/Missions/2MASS/docs/sixdeg/>`_. The available filters are: :math:`J`, :math:`H` and :math:`K_{S}`.


WISE
~~~~

These images are downloaded from the WISE database following `https://irsa.ipac.caltech.edu/ibe/docs/wise/allwise/p3am_cdd/#sample_code <https://irsa.ipac.caltech.edu/ibe/docs/wise/allwise/p3am_cdd/#sample_code>`_. Information such as the `coadd_id` is first obtained from the last rows of the header of the images downloaded with `astroquery.skyview`. However, the latter are not used as the 2MASS zero-points are missing in their headers. The pixel scaling of the images is 1.375 arcsec/pixel, as described in the `WISE website <https://wise2.ipac.caltech.edu/docs/release/prelim/>`_. The available filters are: :math:`W1` (:math:`3.4 \mu \text{m}`), :math:`W2` (:math:`4.6 \mu \text{m}`), :math:`W3` (:math:`12 \mu \text{m}`) and :math:`W4` (:math:`22 \mu \text{m}`).


unWISE
~~~~~~

unWISE is an unblurred coadd of the WISE imaging (i.e. better resolution). These images are directly obtained from the `unWISE image service <http://unwise.me/imgsearch/>`_. The pixel scaling of the images is 2.75 arcsec/pixel, as described in the website. The available filters are :math:`W1`, :math:`W2`, :math:`W3` and :math:`W4` for the AllWISE version, and only :math:`W1` and :math:`W2` for the NeoWISE-R1 and NeoWISE-R2 versions.
