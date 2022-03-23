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
    affiliation: 1
  - name: Lluís Galbany 
    orcid: 0000-0002-1296-6887
    affiliation: "1, 2"
affiliations:
 - name: Institute of Space Sciences (ICE, CSIC), Campus UAB, Carrer de Can Magrans, s/n, E-08193 Barcelona, Spain
   index: 1
 - name: Institut d’Estudis Espacials de Catalunya (IEEC), E-08034 Barcelona, Spain
   index: 2
date: 22 March 2022
bibliography: paper.bib
---

# Summary

Type Ia supernovae (SNe Ia) have assumed a fundamental role as cosmological distance indicators since the discovery of the accelerating expansion rate of the universe [@Riess1998; @Perlmutter1999].
Correlations between their optical peak luminosity, the decline rate of their light curves and their optical colours allow them to be standardised, reducing their observed r.m.s scatter [e.g. @Phillips1993; @Tripp1998].
Over a decade ago, the optical peak luminosity of SNe Ia was found to correlate with host galaxy stellar mass, further improving their standardisation [@Kelly2010; @Lampeitl2010; @Sullivan2010]. Since then, host galaxy properties have been used in cosmological analyses of SNe Ia [@Betoule2014; @Scolnic2018; @Brout2019] and tremendous effort has gone into findig the property, such as star formation rate [@Rigault2013], that fundamentally drives the correlation between SNe Ia and their host galaxies. Furthermore, it has been noted that the local environment, in which the progenitors of SNe Ia evolve, is much better at reducing the scatter in estimated distances than the global environment, i.e. the whole galaxy [@Roman2018; @Kelsey2021]. Therefore, the study of the effect of environment on SNe Ia is an active field of research and key in future cosmological analyses.

# Statement of need

`HostPhot` is an open-source Python package for measuring host galaxy photmetry, both locally and globally. Host galaxy photometry is fundamental as it is commmonly used to estimate the galaxy parameters, such as stellar mass and star formation rate. However, the codes used by different groups can vary and there is no dedicated package for this. The API for `HostPhot` allows the user to extract image cutouts of surveys, such as the Panoramic Survey Telescope and Rapid Response System (Pan-STARRS) Data Release 1 (PS1), Dark Energy Survey (DES) and Sloan Digital Sky Survey (SDSS). Different sets of filters are available depending on the chosen survey: `grizy` for PS1, `grizY` for DES and `ugriz` for SDSS. All photometry can be corrected for Milky Way dust extinction. 

Local photometry can be calculated with the desired circular aperture in physical units (e.g., 4 kpc) at the redshift of the given object. On the other hand, for the global photometry, the user can choose between using a common aperture for all the images, from a coadd image, and masking foreground stars for close-by galaxies. `HostPhot` is fast at calculating photometry (up to a few seconds per object) and user-friendly, which allows the community to easily contribute to this package. `HostPhot` will allow the supernova community to find exciting new scientific discoveries with future cosmological analyses. Finally, although `HostPhot` is mainly aimed at SNe Ia science, it can be used in other fields as well.

`HostPhot` heavily relies on the `Astropy` package [@astropy; @astropy2] for most functions, but also relies on `Photutils` [@photutils] for general photometry, `sep` [@sep] for global photometry and `PISCOLA` [@piscola] for extinction correction.

# Acknowledgements

TEMB and LG acknowledge financial support from the Spanish Ministerio de Ciencia e Innovación (MCIN), the Agencia Estatal de Investigación (AEI) 10.13039/501100011033 under the PID2020-115253GA-I00 HOSTFLOWS project, and from Centro Superior de Investigaciones Científicas (CSIC) under the PIE project 20215AT016, and the I-LINK 2021 LINKA20409. 
TEMB and LG are also partially supported by the program Unidad de Excelencia Maríia de Maeztu CEX2020-001058-M.
LG also acknowledges MCIN, AEI and the European Social Fund (ESF) "Investing in your future" under the 2019 Ramón y Cajal program RYC2019-027683-I.

# References
