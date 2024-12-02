.. _information_confic_file:

Configuration File
==================

The configureration file (``config.txt``) contains most of the necessary information described in :ref:`Further Information: Adding New Surveys<information_adding_surveys>` and it looks like this:


.. list-table:: config.txt
   :widths: 15 20 20 10 10 25
   :header-rows: 1
     
   * - survey
     - filters
     - zp
     - pixel_scale
     - mag_sys
     - mag_sys_conv
   * - DES
     - grizY
     - 30
     - 0.263
     - AB
     - 0.0
   * - PS1
     - grizy
     - 25
     - 0.25
     - AB
     - 0.0
   * - SDSS
     - ugriz
     - 22.5
     - 0.396
     - AB
     - 0.0
   * - GALEX
     - FUV,NUV
     - 18.82,20.08
     - 1.5
     - AB
     - 0.0
   * - 2MASS
     - J,H,Ks
     - header
     - 1.0
     - Vega
     - 0.91,1.39,1.85      
   * - WISE
     - W1,W2,W3,W4
     - header
     - 1.375
     - Vega
     - 2.699,3.339,5.174,6.620  
   * - unWISE
     - W1,W2,W3,W4
     - 22.5
     - 2.75
     - Vega
     - 2.699,3.339,5.174,6.620  
   * - LegacySurvey
     - grz
     - 22.5
     - 0.262
     - AB
     - 0.0
   * - Spitzer
     - IRAC.1,IRAC.2,IRAC.3,IRAC.4,MIPS.1
     - header
     - 0.6,2.45
     - AB
     - 0.0
   * - Vista
     - Z,Y,J,H,Ks
     - header
     - 0.339
     - Vega
     - 0.502,0.600,0.916,1.366,1.827
   * - HST
     - WFC3/UVIS,WFC3/IR
     - header
     - 0.04,0.13
     - AB
     - 0.0
   * - SkyMapper
     - uvgriz
     - header
     - 0.5
     - AB
     - 0.0
   * - SPLUS
     - u,F378,F395,F410,F430,g,F515,r,F660,i,F861,z
     - header
     - 0.55
     - AB
     - 0.0
   * - UKIDSS
     - ZYJHK
     - header
     - 0.4
     - Vega
     - 0.528,0.634,0.938,1.379,1.900
   * - JWST       
     - NIRCam_F090W,NIRCam_F150W,NIRCam_F277W	
     - header
     - 0.031,0.031,0.063
     - AB
     - 0.0 

where **survey** is the internal name of the survey used by HostPhot, **filters** are the available filters (or instrument for HST) either as a single string (for single character filters, e.g. grizy) or separated by commas (no spaces), **zp** are the zero-points (ZPs) separated by commas if multiple ZPs are used or the string `header` in which case HostPhot will look for the keyword ``MAGZP`` (sometimes added by HostPhot), **pixel_scale** is the pixel scaling of the images in units of arcsec/pixel, **mag_sys** is the magnitude system, and **mag_sys_conv** is the magnitude system convertion factor such that :math:`m_{\text{AB}} = m_{\text{Vega}} + \text{mag_sys_conv}`. The last twos columns are not used by HostPhot (at the moment at least). They are mainly for completeness in case a user needs the photometry in different units.

