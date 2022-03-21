# Check the following urls for more info about Pan-STARRS:
#
#     https://outerspace.stsci.edu/display/PANSTARRS/PS1+Image+Cutout+Service#PS1ImageCutoutService-ImportantFITSimageformat,WCS,andflux-scalingnotes
#     https://outerspace.stsci.edu/display/PANSTARRS/PS1+Stack+images#PS1Stackimages-Photometriccalibration
#
# For DES:
#
#     https://des.ncsa.illinois.edu/releases/dr1/dr1-docs/processing
#
# For SDSS:
#
#     https://www.sdss.org/dr12/algorithms/fluxcal/#SDSStoAB
#     https://data.sdss.org/datamodel/files/BOSS_PHOTOOBJ/frames/RERUN/RUN/CAMCOL/frame.html
#
# Some parts of this notebook are based on https://github.com/djones1040/PS1_surface_brightness/blob/master/Surface%20Brightness%20Tutorial.ipynb and codes from Lluís Galbany

import os
import glob
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import sep
from astropy.io import fits
from astropy.modeling import models, fitting
from astropy import coordinates as coords, units as u, wcs

from photutils import (CircularAnnulus,
                       CircularAperture,
                       aperture_photometry)
from photutils.detection import DAOStarFinder

import piscola
from piscola.extinction_correction import extinction_filter

from .utils import (get_survey_filters, extract_filters,
                check_survey_validity, check_filters_validity)

sep.set_sub_object_limit(1e4)

# Masking Stars
#-------------------------------
def create_circular_mask(h, w, centre, radius):
    """Creates a circular mask of an image.

    Parameters
    ----------
    h: int
        Image height.
    w: int
        Image width.
    centre: tuple-like
        Centre of the circular mask.
    radius: float
        Radius of the circular mask.

    Returns
    -------
    mask: 2D bool-array
        Circular mask (inside the circle = `True`).
    """

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - centre[0])**2 + (Y-centre[1])**2)
    mask = dist_from_center <= radius

    return mask

def inside_galaxy(star_center, gal_center, gal_r):
    """Checks whether a star is inside a galaxy.

    Parameters
    ==========
    star_center: tuple-like
        Centre of the star.
    gal_center: tuple-like
        Centre of the galaxy.
    gal_r: float
        Radius to define the galaxy size.

    Returns
    =======
    condition: bool
        `True` if the star is inside the galaxy,
        `False` otherwise.
    """

    dist_from_center = np.sqrt((star_center[0] - gal_center[0])**2 +
                               (star_center[1] - gal_center[1])**2)
    condition = dist_from_center <= gal_r

    return condition

def fit_2dgauss(star_data, x0=None, y0=None, plot_fit=False):
    """Fits a 2D gaussian to a star.

    Parameters
    ----------
    star_data: 2D array
        Image data.
    x0: int, default `None`
        Star's x-axis centre.
    y0: int, default `None`
        Star's y-axis centre.
    plot_fit: bool, default `False`
        If `True`, the model fit is plotted with `r_in`
        and `r_out`.

    Returns
    -------
    model_sigma: float
        2D gaussian sigma parameter. The largest between
        sigma_x and sigma_y.
    """

    # initial guess
    sigma = 0.5
    amp = np.max(star_data)
    if (x0 is None) | (y0 is None):
        y0, x0 = np.unravel_index(np.argmax(star_data),
                                  star_data.shape)

    fitter = fitting.LevMarLSQFitter()
    gaussian_model = models.Gaussian2D(amp, x0, y0, sigma, sigma)
    gaussian_model.fixed['x_mean'] = True
    gaussian_model.fixed['y_mean'] = True
    gaussian_model.bounds['x_stddev'] = (0, 10)
    gaussian_model.bounds['y_stddev'] = (0, 10)

    yi, xi = np.indices(star_data.shape)
    model_result = fitter(gaussian_model, xi, yi, star_data)

    model_sigma = max([model_result.x_stddev.value,
                       model_result.y_stddev.value])

    if plot_fit:
        model_data = model_result(xi, yi)

        fig, ax = plt.subplots()
        ax.imshow(star_data, interpolation='nearest', cmap='gray',
                       vmin=m-s, vmax=m+s, origin='lower')

        x_mean = model_result.x_mean.value
        y_mean = model_result.y_mean.value

        circle_in = plt.Circle((x_mean, y_mean), Rin_scale*model_sigma,
                               facecolor ='none', edgecolor = 'red',
                               linewidth = 2)
        circle_out = plt.Circle((x_mean, y_mean), Rout_scale*model_sigma,
                                facecolor ='none', edgecolor = 'red',
                                linewidth = 2)
        ax.add_patch(circle_in)
        ax.add_patch(circle_out)

        plt.show()

    return model_sigma

def mask_image(data, apertures, bkg, gal_center=None, gal_r=None, plot=False):
    """Creates a mask of the stars with the mean value of the background
    around them.

    **Note**: the galaxy coordinates are (y, x).

    Parameters
    ----------
    data: 2D array
        Image data.
    apertures: `photutils.aperture.circle.CircularAperture`
        Circular apertures of the stars.
    bkg: float
        Background level used to limit the aperture size to
        mask the stars. In other words, increase the aperture size
        of the mask until the mask value is <= 3*bkg to properly mask
        bright stars.
    gal_center: tuple-like, default `None`
        Centre of the galaxy (y, x) in pixels.
    gal_r: float, default `None`
        Radius to define the galaxy size.
    plot: bool, default `False`
        If `True`, the image with the masked stars are ploted.
    """
    h, w = data.shape[:2]
    masked_data = data.copy()

    model_sigmas = []
    skip_indeces = []
    Rin_scalings = []
    Rout_scalings = []
    for i, aperture in enumerate(apertures):
        star_y, star_x = aperture.positions
        size = 10
        xmin = max(int(star_x-2*size), 0)
        xmax = min(int(star_x+2*size), w)
        ymin = max(int(star_y-2*size), 0)
        ymax = min(int(star_y+2*size), h)

        # some stars close to the edges of the image fail,
        # but we can skip those anyway
        if (xmin<star_x<xmax) & (ymin<star_y<ymax):
            star_data = masked_data[xmin:xmax, ymin:ymax]
        else:
            skip_indeces.append(i)
            model_sigmas.append(np.nan)
            Rin_scalings.append(np.nan)
            Rout_scalings.append(np.nan)
            continue  # skip this star

        # fit a gaussian to the star and get sigma
        x0, y0 = star_x-xmin, star_y-ymin
        model_sigma = fit_2dgauss(star_data)
        model_sigmas.append(model_sigma)
        if model_sigma==10:
            # 10 is the limit I putted in `fit_2dgauss` --> fit failed
            skip_indeces.append(i)
            Rin_scalings.append(np.nan)
            Rout_scalings.append(np.nan)
            continue  # skip this star

        # check if the star is inside the galaxy aperture
        star_center = aperture.positions
        if (gal_center is None) or (gal_r is None):
            star_inside_galaxy = False
        else:
            star_inside_galaxy = inside_galaxy(star_center,
                                               gal_center, gal_r)
        if star_inside_galaxy:
            # obtain the "optimal" mask size
            sigma_scalings = np.arange(2, 14.5, 0.1)
            for Rin_scale in sigma_scalings:
                Rout_scale = Rin_scale+3
                r_in, r_out = Rin_scale*model_sigma, Rout_scale*model_sigma
                ann = CircularAnnulus(aperture.positions,
                                      r_in=r_in, r_out=r_out)
                ann_mean = aperture_photometry(
                                data, ann)['aperture_sum'][0] / ann.area
                if ann_mean<=3*bkg:
                    break
                if ann_mean>bkg*3 and Rin_scale==sigma_scalings.max():
                    # if the loop ends not successfully...
                    Rin_scale =7.0  # some "average"(abitrary) value
                    Rout_scale = Rin_scale+3
                    r_in, r_out = Rin_scale*model_sigma, Rout_scale*model_sigma

            Rin_scalings.append(Rin_scale)
            Rout_scalings.append(Rout_scale)

            mask = create_circular_mask(h, w, star_center, r_in)
            # basically remove the failed fits and avoid large objects
            not_big_object = r_in<(gal_r/3)
            # do not mask the galaxy
            dist2gal = np.sum(np.abs(apertures.positions - gal_center), axis=1)
            gal_id = np.argmin(dist2gal)

            if not np.isnan(ann_mean) and not_big_object and i!=gal_id:
                masked_data[mask] = ann_mean
            else:
                skip_indeces.append(i)
        else:
            Rin_scalings.append(5.0)
            Rout_scalings.append(8.0)

    if plot:
        m, s = np.nanmean(data), np.nanstd(data)

        fig, ax = plt.subplots(1, 3, figsize=(15, 10))
        # reference image
        ax[0].imshow(data, interpolation='nearest', cmap='gray',
                       vmin=m-s, vmax=m+s, origin='lower')
        ax[0].set_title('reference image')

        # apertures image
        ax[1].imshow(data, interpolation='nearest', cmap='gray',
                       vmin=m-s, vmax=m+s, origin='lower')
        ax[1].set_title('apertures image')
        if (gal_center is not None) or (gal_r is not None):
            gal_circle = plt.Circle(gal_center, gal_r,
                                            ec='b', fill=False)
            ax[1].add_patch(gal_circle)

        # masked image
        ax[2].imshow(masked_data, interpolation='nearest', cmap='gray',
                       vmin=m-s, vmax=m+s, origin='lower')
        ax[2].set_title('masked image')

        for i, (aperture, model_sigma,
                Rin_scale, Rout_scale) in enumerate(zip(apertures,
                                                        model_sigmas,
                                                        Rin_scalings,
                                                        Rout_scalings)):
            if i not in skip_indeces:
                aperture.r = Rin_scale*model_sigma
                aperture.plot(ax[1], color='red', lw=1.5, alpha=0.5)
                aperture.r = Rout_scale*model_sigma
                aperture.plot(ax[1], color='red', lw=1.5, alpha=0.5)

        plt.tight_layout()
        plt.show()

    return masked_data, model_sigmas

# Coadding images
#-------------------------------
def coadd_images(sn_name, filters='riz', resample=True, verbose=False,
                                                work_dir='', survey='PS1'):
    """Coadds images with SWarp for the choosen filters for
    common-aperture photometry.

    Parameters
    ----------
    sn_name: str
        SN name to be used for finding the images locally.
    filters: str, default `riz`
        Filters to use for the coadd image.
    resample: bool, default `True`
        If `True`, the images are resampled to a common frame
        before coadding them.
    verbose: bool, default `False`
        If `True`, the steps taken by this function are printed.
    work_dir: str
        Working directory where to find the objects'
        directories with the images.
    survey: str, default `PS1`
        Survey to use as prefix for the images.

    Returns
    -------
    A fits file with the coadded images is created with the filters
    used as the name of the file at the SN directory.
    """

    init_dir = os.path.abspath('.')
    sn_dir = os.path.join(work_dir, sn_name)
    fits_files = [os.path.join(sn_dir,
                               f'{survey}_{filt}.fits') for filt in filters]

    # change to SN directory
    os.chdir(sn_dir)
    if verbose:
        print(f'moving to {os.path.abspath(".")}')

    try:
        if resample:
            img = fits.open(os.path.basename(fits_files[0]))
            # Note that this assumes that all the images have the same size
            header = img[0].header
            stamp_sizex, stamp_sizey = header['NAXIS1'], header['NAXIS2']

            sci_frame_str = ''.join("%s "%''.join(os.path.basename(fits_file))
                                                    for fits_file in fits_files)
            # make a stamp as a det image
            resamp_cmd = ['swarp',
                          '-COMBINE', 'N',
                          '-RESAMPLE', 'Y',
                          '-IMAGE_SIZE', f'{stamp_sizex}, {stamp_sizey}',
                          '-BACK_SIZE', '512',
                          sci_frame_str]

            p = subprocess.Popen(resamp_cmd,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
            outs, errs = p.communicate()
            if verbose:
                print(errs.decode("utf-8"))

            # Now resample the image
            combine_frames = glob.glob(f'*resamp*.fits')
            combine_frame_str = ''.join("%s "%''.join(os.path.basename(combine_file))
                                                    for combine_file in combine_frames)
        else:
            combine_frame_str = ''.join("%s "%''.join(os.path.basename(fits_file))
                                                    for fits_file in fits_files)

        if verbose:
            print(f'Creating a {filters} image')

        combine_cmd = ['swarp',
                       '-COMBINE','Y',
                       '-RESAMPLE','N',
                       '-BACK_SIZE','512',
                       '-IMAGEOUT_NAME', 'riz.fits',
                       combine_frame_str]

        p = subprocess.Popen(combine_cmd,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        outs, errs = p.communicate()
        if verbose:
            print(errs.decode("utf-8"))

        # remove temporary files
        for file in glob.glob('*.fits'):
            if '.resamp.' in file:
                os.remove(file)
            if 'coadd' in file:
                os.remove(file)

        # move back to initial directory
        os.chdir(init_dir)
        if verbose:
            print(f'moving to {os.path.abspath(".")}')
    except Exception as message:
        os.chdir(init_dir)
        print(message)

# Photometry
#-------------------------------
def extract_aperture_params(fits_file, host_ra, host_dec, threshold, bkg_sub=True):
    """Extracts aperture parameters of a galaxy.

    **Note:** the galaxy must be ideally centred in the image.

    Parameters
    ==========
    fits_file: str
        Path to the fits file.
    host_ra: float
        Host-galaxy Right ascension of the galaxy in degrees.
    host_dec: float
        Host-galaxy Declination of the galaxy in degrees.
    threshold: float
        Threshold used by `sep.extract()` to extract objects.
    bkg_sub: bool, default `True`
        If `True`, the image gets background subtracted.

    Returns
    =======
    gal_object: numpy array
        Galaxy object extracted with `sep.extract()`.
    objects: numpy array
        All objects extracted with `sep.extract()`.
    """

    img = fits.open(fits_file)

    header = img[0].header
    data = img[0].data
    img_wcs = wcs.WCS(header, naxis=2)

    data = data.astype(np.float64)
    bkg = sep.Background(data)
    bkg_rms = bkg.globalrms
    if bkg_sub:
        data_sub = np.copy(data - bkg)
    else:
        data_sub = np.copy(data)

    # extract objects with Source Extractor
    objects = sep.extract(data_sub, threshold, err=bkg_rms)

    # obtain the galaxy data (hopefully centred in the image)
    gal_coords = coords.SkyCoord(ra=host_ra*u.degree,
                                 dec=host_dec*u.degree)
    gal_x, gal_y = img_wcs.world_to_pixel(gal_coords)

    x_diff = np.abs(objects['x']-gal_x)
    y_diff = np.abs(objects['y']-gal_y)
    gal_id = np.argmin(x_diff+y_diff)
    gal_object = objects[gal_id:gal_id+1]

    return gal_object, objects

def extract_global_photometry(fits_file, host_ra, host_dec, gal_object=None,
                              mask_stars=True, threshold=3, bkg_sub=True,
                              survey='PS1', show_plots=False):
    """Extracts PanSTARRS's global photometry of a galaxy. The use
    of `gal_object` is intended for common-aperture photometry.

    **Note:** the galaxy must be ideally centred in the image.

    Parameters
    ==========
    fits_file: str
        Path to the fits file.
    host_ra: float
        Host-galaxy Right ascension of the galaxy in degrees.
    host_dec: float
        Host-galaxy Declination of the galaxy in degrees.
    gal_object: numpy array, default `None`
        Galaxy object extracted with `extract_aperture_params()`.
        Use this for common-aperture photometry only.
    mask_stars: bool, default `True`
        If `True`, the stars identified inside the common aperture
        are masked with the mean value of the background around them.
    threshold: float, default `3`
        Threshold used by `sep.extract()` to extract objects.
    bkg_sub: bool, default `True`
        If `True`, the image gets background subtracted.
    survey: str, default `PS1`
        Survey to use for the zero-points.
    show_plots: bool, default `False`
        If `True`, diagnosis plot are shown.

    Returns
    =======
    mag: float
        Aperture magnitude.
    mag_err: float
        Error on the aperture magnitude.
    """
    check_survey_validity(survey)

    img = fits.open(fits_file)

    header = img[0].header
    data = img[0].data
    exptime = float(header['EXPTIME'])

    data = data.astype(np.float64)
    bkg = sep.Background(data)
    bkg_rms = bkg.globalrms
    if bkg_sub:
        data_sub = np.copy(data - bkg)
    else:
        data_sub = np.copy(data)

    if gal_object is None:
        # no common aperture
        gal_object, objects = extract_aperture_params(fits_file,
                                                      host_ra,
                                                      host_dec,
                                                      threshold,
                                                      bkg_sub)
    else:
        _, objects = extract_aperture_params(fits_file,
                                             host_ra,
                                             host_dec,
                                             threshold,
                                             bkg_sub)

    if mask_stars:
        # identify bright source.....
        #mean, median, std = sigma_clipped_stats(data_sub, sigma=3.0)
        #daofind = DAOStarFinder(fwhm=3.0, threshold=7.*std)  # avoids bogus sources
        #sources = daofind(data_sub)
        #positions = np.transpose((sources['xcentroid'], sources['ycentroid']))

        # ... or create apertures for the sources obtained with sep
        positions = np.transpose([objects['x'], objects['y']])

        apertures = CircularAperture(positions, r=4)  # the value of r is irrelevant

        # mask image
        gal_center = (gal_object['x'][0], gal_object['y'][0])
        gal_r = (6/2)*gal_object['a'][0]  # using the major-axis as radius
        masked_data, model_sigmas = mask_image(data_sub, apertures, bkg_rms,
                                                gal_center, gal_r, show_plots)
        data_sub = masked_data.copy()

    # aperture photometry
    # This uses what would be the default SExtractor parameters.
    # See https://sep.readthedocs.io/en/v1.1.x/apertures.html
    # NaNs are converted to mean values to avoid issues with the photometry.
    # The value used, slightly affects the results (~ 0.0x mag).
    masked_data = np.nan_to_num(data_sub, nan=np.nanmean(data_sub))
    kronrad, krflag = sep.kron_radius(masked_data,
                                      gal_object['x'],
                                      gal_object['y'],
                                      gal_object['a'],
                                      gal_object['b'],
                                      gal_object['theta'],
                                      6.0)

    r_min = 1.75  # minimum diameter = 3.5
    if kronrad*np.sqrt(gal_object['a']*gal_object['b']) < r_min:
        print(f'Warning: using circular photometry on {fits_file}')
        flux, flux_err, flag = sep.sum_circle(masked_data,
                                              gal_object['x'],
                                              gal_object['y'],
                                              r_min,
                                              err=bkg.globalrms,
                                              subpix=1)
    else:
        flux, flux_err, flag = sep.sum_ellipse(masked_data,
                                              gal_object['x'],
                                              gal_object['y'],
                                              gal_object['a'],
                                              gal_object['b'],
                                              gal_object['theta'],
                                              2.5*kronrad,
                                              err=bkg.globalrms,
                                              subpix=1)

    zp_dict = {'PS1':25 + 2.5*np.log10(exptime),
               'DES':30,
               'SDSS':22.5}
    zp = zp_dict[survey]

    mag = -2.5*np.log10(flux) + zp
    mag_err = 2.5/np.log(10)*flux_err/flux

    if show_plots:
        fig, ax = plt.subplots()
        m, s = np.nanmean(data_sub), np.nanstd(data_sub)
        im = ax.imshow(data_sub, interpolation='nearest',
                       cmap='gray',
                       vmin=m-s, vmax=m+s,
                       origin='lower')

        e = Ellipse(xy=(gal_object['x'][0], gal_object['y'][0]),
                    width=6*gal_object['a'][0],
                    height=6*gal_object['b'][0],
                    angle=gal_object['theta'][0]*180./np.pi)

        e.set_facecolor('none')
        e.set_edgecolor('red')
        ax.add_artist(e)
        plt.show()

    return mag[0], mag_err[0]

def multi_global_photometry(name_list, host_ra_list, host_dec_list, work_dir,
                            filters=None, coadd=True, coadd_filters='riz',
                            resample=True, mask_stars=True, threshold=3,
                            bkg_sub=True, survey="PS1", correct_extinction=True,
                            show_plots=False):
    """Extract global photometry for multiple SNe.

    Parameters
    ==========
    name_list: list-like
        List of SN names.
    host_ra_list: list-like
        List of host-galaxy right ascensions in degrees.
    host_dec_list: list-like
        List of host-galaxy declinations in degrees.
    work_dir: str
        Working directory where to find the objects'
        directories with the images.
    filters: str, defaul `None`
        Filters used to extract photometry. If `None`, use all
        the available filters for the given survey.
    coadd: bool, default `True`
        If `True`, a coadd image is created for common aperture.
    coadd_filters: str, default `riz`
        Filters to use for the coadd image.
    resample: bool, default `True`
        If `True`, the images are resampled before coadding them.
    mask_stars: bool, default `True`
        If `True`, the stars identified inside the common aperture
        are masked with the mean value of the background around them.
    threshold: float, default `3`
        Threshold used by `sep.extract()` to extract objects.
    bkg_sub: bool, default `True`
        If `True`, the image gets background subtracted.
    survey: str, default `PS1`
        Survey to use for the zero-points.
    correct_extinction: bool, default `True`
        If `True`, the magnitudes are corrected for extinction.
    show_plots: bool, default `False`
        If `True`, a diagnosis plot is shown.

    Returns
    =======
    global_phot_df: DataFrame
        Dataframe with the photometry, errors and SN name.
    """
    check_survey_validity(survey)
    check_filters_validity(filters, survey)
    if filters is None:
        filters = get_survey_filters(survey)

    # dictionary to save results
    mag_dict = {filt:[] for filt in filters}
    mag_err_dict = {filt+'_err':[] for filt in filters}
    mag_dict.update(mag_err_dict)

    results_dict = {'name':[], 'host_ra':[], 'host_dec':[]}
    results_dict.update(mag_dict)

    # filter funcstions for extinction correction
    filters_dict = extract_filters(filters, survey)

    for name, host_ra, host_dec in zip(name_list, host_ra_list,
                                                      host_dec_list):

        sn_dir = os.path.join(work_dir, name)
        image_files = [os.path.join(sn_dir, f'{survey}_{filt}.fits')
                                                    for filt in filters]

        if coadd:
            verbose = False
            coadd_images(name, coadd_filters, resample,
                                 verbose, work_dir, survey)
            coadd_file = os.path.join(sn_dir, f'{coadd_filters}.fits')
            gal_object, _ = extract_aperture_params(coadd_file,
                                                    host_ra, host_dec,
                                                    threshold, bkg_sub)
        else:
            gal_object = None

        for image_file, filt in zip(image_files, filters):
            try:
                mag, mag_err = extract_global_photometry(image_file,
                                                         host_ra,
                                                         host_dec,
                                                         gal_object,
                                                         mask_stars,
                                                         threshold,
                                                         bkg_sub,
                                                         survey,
                                                         show_plots)
                if correct_extinction:
                    wave = filters_dict[filt]['wave']
                    transmission = filters_dict[filt]['transmission']
                    A_ext = extinction_filter(wave, transmission,
                                                host_ra, host_dec)
                    mag -= A_ext
                results_dict[filt].append(mag)
                results_dict[filt+'_err'].append(mag_err)
            except Exception as message:
                results_dict[filt].append(np.nan)
                results_dict[filt+'_err'].append(np.nan)
                print(f'{name} failed with {filt} band: {message}')
        results_dict['name'].append(name)
        results_dict['host_ra'].append(host_ra)
        results_dict['host_dec'].append(host_dec)

    global_phot_df = pd.DataFrame(results_dict)

    return global_phot_df
