---
title: 'HostPhot: global and local photometry of galaxies hosting supernovae or other transients'
tags:
  - Python
  - astronomy
  - supernova
  - galaxy
authors:
  - name: Tomás E. Müller-Bravo
    orcid: 0000-0003-3939-7167
    affiliation: "1, 2"
  - name: Lluís Galbany 
    orcid: 0000-0002-1296-6887
    affiliation: "1, 2"
affiliations:
 - name: Institute of Space Sciences (ICE, CSIC), Campus UAB, Carrer de Can Magrans, s/n, E-08193 Barcelona, Spain
   index: 1
 - name: Institut d’Estudis Espacials de Catalunya (IEEC), E-08034 Barcelona, Spain
   index: 2
date: 31 May 2022
bibliography: paper.bib
---

# Summary

Type Ia supernovae (SNe Ia) have assumed a fundamental role as cosmological distance indicators since the discovery of the accelerating expansion rate of the universe [@Riess1998; @Perlmutter1999]. Correlations between their optical peak luminosity, the decline rate of their light curves and their optical colours allow them to be standardised, reducing their observed r.m.s scatter [e.g. @Phillips1993; @Tripp1998]. Over a decade ago, the optical peak luminosity of SNe Ia was found to correlate with host galaxy stellar mass, further improving their standardisation [@Kelly2010; @Lampeitl2010; @Sullivan2010]. Since then, host galaxy properties have been used in cosmological analyses of SNe Ia [@Betoule2014; @Scolnic2018; @Brout2019] and tremendous effort has gone into findig the property, such as star formation rate [@Rigault2013], that fundamentally drives the correlation between SNe Ia and their host galaxies. Furthermore, it has been noted that the local environment in which the progenitors of SNe Ia evolve is much better at reducing the scatter in estimated distances than the global environment, i.e., the whole galaxy [@Roman2018; @Kelsey2021]. Therefore, the study of the effect of environment on SNe Ia is an active field of research and key in future cosmological analyses.

# Statement of need

`HostPhot` is an open-source Python package for measuring galaxy photmetry, both locally and globally. Galaxy photometry is fundamental as it is commmonly used to estimate the galaxy parameters, such as stellar mass and star formation rate. However, the codes used to calculate photometry by different groups can vary and there is no dedicated package for this. The API for `HostPhot` allows the user to extract public image cutouts of surveys, such as the Panoramic Survey Telescope and Rapid Response System (Pan-STARRS) Data Release 1 (PS1), Dark Energy Survey (DES) and Sloan Digital Sky Survey (SDSS). Different sets of filters are available depending on the chosen survey: *grizy* for PS1, *grizY* for DES and *ugriz* for SDSS. All photometry is corrected for Milky Way dust extinction. Furthermore, `HostPhot` also works with private data obtained by the user and can even be easily modified to include other surveys.

The major novelty of `HostPhot` is dealing with low-redshift galaxies (z $<$ 0.1) as obtaining photometry of these is not as simple as those at higher redshift. Foreground stars can be in the line of sight of nearby galaxies, making the extraction of the photometry a complex procedure. In addition, low-redshift galaxies have visible structures, while at high redshift they just look like simple ellipses. `HostPhot` is able to detect sources in the images, cross-match them with catalogs of stars (e.g., Gaia [@gaia]) and remove them by applying a convolution with a 2D Gaussian kernel. This process ensures that only stars (and in some cases other galaxies that are not of interest) are removed, keeping the structure of the galaxy intact.

`HostPhot` can calculate the photometry of an entire galaxy (global) or in a given circular aperture (local) and it heavily relies on the `Astropy` [@astropy; @astropy2] and `Photutils` [@photutils] packages for this. Local photometry can be calculated for different circular apertures in physical units (e.g., 4 kpc) at the redshift of the given object. In addition, as the physical size depends on the assumed cosmology, the cosmological model can be changed by the user, suiting their needs. On the other hand, for the global photometry, the user can choose between using a different aperture for each filter/image or a common aperture for all the filters/images. For the latter, `HostPhot` coadds images in the desired filters, as selected by the user (e.g., *riz*), and estimates the common aperture parameters from the coadd image. The aperture used for the global photometry can also be optimised, by increasing the size until the change in flux is negligible, encompassing the entire galaxy. In a few cases, nearby galaxies can have very complex structures. `HostPhot` offers the option of interactively setting the aperture via an intuitive GUI. This option also allows the user to test how the change in aperture shape can affect the calculated photometry.

`HostPhot` is user-friendly and well documented[^1], which allows the community to easily contribute to this package. `HostPhot` is already being used by different groups, such as HostFlows[^2] and DES, and will allow the supernova community to find exciting new scientific discoveries with future cosmological analyses. Finally, although `HostPhot` is mainly aimed at supernova science, it can be used in other fields in astronomy as well.

[^1]: https://hostphot.readthedocs.io/en/latest/
[^2]: https://hostflows.github.io/

Apart from `Astropy` and `Photutils` [@photutils], `HostPhot` also relies on `sep` [@sep] for global photometry, `Astroquery` [@astroquery] for image downloading and cross-matching with catalogs, reproject[^3] for the coadds, `extinction` [@extinction] and sfdmap[^4] for extinction correction. Finally, `HostPhot` makes use of the following packages as well: numpy [@numpy], matplotlib [@matplotlib], pandas [@pandas], pyvo [@pyvo], ipywidgets[^5] and ipympl[^6].

[^3]: https://pypi.org/project/reproject/
[^4]: https://github.com/kbarbary/sfdmap
[^5]: https://github.com/jupyter-widgets/ipywidgets
[^6]: https://github.com/matplotlib/ipympl

# Acknowledgements

TEMB and LG acknowledge financial support from the Spanish Ministerio de Ciencia e Innovación (MCIN), the Agencia Estatal de Investigación (AEI) 10.13039/501100011033 under the PID2020-115253GA-I00 HOSTFLOWS project, and from Centro Superior de Investigaciones Científicas (CSIC) under the PIE project 20215AT016, and the I-LINK 2021 LINKA20409. 
TEMB and LG are also partially supported by the program Unidad de Excelencia Maríia de Maeztu CEX2020-001058-M.
LG also acknowledges MCIN, AEI and the European Social Fund (ESF) "Investing in your future" under the 2019 Ramón y Cajal program RYC2019-027683-I.

# References
