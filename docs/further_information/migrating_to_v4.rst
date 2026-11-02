.. _migrating_to_v4:

Migrating from v3 to v4
=======================

HostPhot v4 fixes many errors in the photometry and uncertainties of v3, changes some defaults and adds new
surveys. This page summarises what changes for existing users. The details of each change (with the size of
the effects and the validation) are in the `changelog <https://github.com/temuller/hostphot/blob/main/CHANGELOG.md>`_.

**In short:** re-download your images, re-run the photometry and do not mix v3 and v4 results.


Changes that can break existing code
------------------------------------

* The ``dlr`` column of the aperture parameters, which contained the separation between the transient and the
  galaxy (in arcsec), is renamed ``sep_arcsec``. ``d_dlr`` (directional light-radius distance) is unchanged.
* ``extract_legacy_aperture_flux`` and ``extract_legacy_kron_flux`` are renamed ``extract_invvar_aperture_flux``
  and ``extract_invvar_kron_flux``, as they are used by several surveys.
* Some cases that previously failed silently (or gave wrong results) now raise errors:

  * ``common_aperture=True`` without ``ref_filt`` (previously, independent apertures were used for each filter);
  * using the aperture parameters of a reference filter or survey (``ref_filt``/``ref_survey``) before they have
    been saved (run the global photometry of the reference with ``save_aperture_params=True`` first);
  * HST and JWST downloads without filters;
  * a host galaxy that is not detected: the nearest detected object is only used if the galaxy coordinates are
    within its extent (previously, any object at any distance could be used, e.g. a nearby star). The error
    suggests a lower ``threshold``; with ``raise_exception=False``, NaN is returned.

* The VISTA ``version`` parameter (``VHS``, ``VIDEO`` or ``VIKING``) is now used (it was ignored before). If it
  is not given, the deepest survey covering the position is used, as before.


Changed defaults
----------------

The same call can give different results because some defaults changed:

* **Milky-Way extinction:** no longer corrected by default (``correct_extinction=False``). Use
  ``correct_extinction=True`` to correct it (the flux uncertainties are now also corrected).
* **Kron apertures (global photometry):** the standard Kron scale of 2.5 is used by default
  (``optimize_kronrad=False``), as for SExtractor's ``MAG_AUTO``. The optimisation of the scale is optional
  (``optimize_kronrad=True``).
* **Background subtraction:** now applied by default to Spitzer, JWST and Herschel images (besides 2MASS, WISE,
  VISTA, SkyMapper, UKIDSS and UHS), with a new estimation (sources and host galaxy masked, 30 arcsec mesh).
  GALEX images are downloaded already background subtracted (``intbgsub``).
* **Default filters:** HSC uses only the broadband filters (:math:`grizY`).


Corrected results
-----------------

Many magnitudes and uncertainties of v3 were wrong. The main changes are (see the changelog for the rest):

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Survey
     - Change in v4
   * - All
     - Uncertainties: blank-aperture noise floor (correlated noise) and calibration uncertainties. Background
       subtraction no longer removes part of the galaxy light (e.g. magnitudes ~0.1-0.3 mag brighter for large
       apertures and Kron magnitudes). Apertures partially without data are handled consistently (NaN, with a
       warning, if more than 5% is missing). Non-detections give NaN magnitudes (the fluxes are kept).
       Transferred apertures (common apertures and masks) are now correctly oriented
   * - PanSTARRS, DES, SkyMapper, S-PLUS
     - New error model (errors underestimated by up to ~2x for PanSTARRS and overestimated by ~2-3x for DES and
       S-PLUS)
   * - SDSS
     - Errors were overestimated by ~3-10x for faint sources
   * - 2MASS
     - Errors were overestimated by ~2-2.5x (local photometry) and underestimated for the global photometry
   * - WISE / unWISE
     - Errors were overestimated (e.g. 0.05-1.9 mag, now ~0.01-0.03 mag for a bright host). unWISE uses the
       inverse-variance maps
   * - Spitzer
     - Errors were overestimated by ~10-1000x. Background subtracted by default (previously, e.g. fluxes
       ~2-80x too large for stars in IRAC 2-4). IRAC extended-source correction (galaxies ~0.05-0.3 mag fainter)
   * - GALEX
     - Background-subtracted images (previously, up to several magnitudes too bright in NUV)
   * - UKIDSS
     - Pixel scale of micro-stepped images (apertures were half the requested radius); stacks with ``NEXP > 1``
       were 2.5 log10(NEXP) mag too faint; deep stacks are now used; calibration uncertainty
   * - VISTA
     - Tile selection by calibration and coverage; VIDEO uses the latest mosaics; calibration uncertainty
   * - HSC
     - Images without the local sky subtraction (``coadd/bg``), which removed the extended light of galaxies
       (e.g. ~2 mag for the Kron magnitude of NGC 799)
   * - HST
     - No point-source aperture correction (global magnitudes ~0.04 mag brighter); NICMOS errors were
       underestimated by ~2x
   * - JWST
     - Zeropoint uncertainty (was ~0.27 mag, now ~0.011 mag); Poisson noise from the pipeline variance
   * - Herschel
     - Background subtracted by default (e.g. SPIRE fluxes ~2x too large for NGC 628); maps cut to the requested
       size; deepest map covering the position; local photometry now supported


Re-download your images
-----------------------

We recommend re-downloading all the images (``overwrite=True``), as the image selection changed for several
surveys (e.g. SDSS, VISTA, UKIDSS, GALEX, Herschel and Spitzer, whose cutouts were also larger than requested).
For some surveys, images downloaded with v3 cannot be used correctly by v4:

* **unWISE:** the inverse-variance maps (used for the uncertainties) are now downloaded with the images;
* **HSC:** the images are now the ``coadd/bg`` coadds, with the mask plane;
* **JWST:** the effective gain (``EFFGAIN`` keyword) is calculated when downloading or setting the images
  (``set_JWST_image``);
* **GALEX:** the background-subtracted images (``intbgsub``) are now downloaded.


New in v4
---------

* New surveys: HSC, Euclid (Q1), UHS and KiDS (DR5).
* Spitzer: images from the Spitzer Heritage Archive (e.g. warm mission) when the SEIP mosaics do not cover the
  position.
* HST: ACS/WFC and NICMOS support.
* Herschel: local photometry.
* Clearer warnings: failed downloads, images without data, apertures without data, galaxies not detected and
  non-detections.

See the :ref:`cutouts <information_cutouts>` and :ref:`photometry <information_photometry>` pages for the details of each survey.
