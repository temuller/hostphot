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
from astropy.stats import sigma_clipped_stats
from astropy.modeling import models, fitting
from astropy import coordinates as coords, units as u, wcs

from photutils import (CircularAnnulus,
                       CircularAperture,
                       aperture_photometry)
from photutils.detection import DAOStarFinder

import piscola
from piscola.extinction_correction import extinction_filter

from .phot_utils import calc_sky_unc
from .utils import (get_survey_filters, check_survey_validity,
                                        check_filters_validity)

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
        `True` if the a star is inside the galaxy.
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
def coadd_images(sn_name, filters='riz', resample=True, verbose=False, work_dir=''):
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

    Returns
    -------
    A fits file with the coadded images is created with the filters
    used as the name of the file at the SN directory.
    """

    init_dir = os.path.abspath('.')
    sn_dir = os.path.join(work_dir, sn_name)
    fits_files = [os.path.join(sn_dr,
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
def multi_global_photometry(name_list, filters = "grizy",
                             coadd=True, mask_stars=True, coadd_filters='riz',
                                threshold=3, bkg_sub=False, show_plots=False):
    """Extract global photometry for multiple SNe.

    Parameters
    ==========
    name_list: list-like
        List of SN names.
    filters: str, defaul `grizy`
        Filters used to extract photometry.
    coadd: bool, default `True`
        If `True`, a coadd image is created for common aperture.
    mask_stars: bool, default `True`
        If `True`, the stars identified inside the common aperture
        are masked with the mean value of the background around them.
    coadd_filters: str, default `riz`
        Filters to use for the coadd image.
    threshold: float, default `3`
        Threshold used by `sep.extract()` to extract objects.
    bkg_sub: bool, default `True`
        If `True`, the image gets background subtracted.
    show_plots: bool, default `False`
        If `True`, a diagnosis plot is shown.

    Returns
    =======
    global_phot_df: DataFrame
        Dataframe with the photometry, errors and SN name.
    """
    # SNe without PS1 data
    skip_sne = ['SN2006bh', 'SN2008bq']

    # dictionary to save results
    mag_dict = {filt:[] for filt in filters}
    mag_err_dict = {filt+'_err':[] for filt in filters}
    mag_dict.update(mag_err_dict)

    results_dict = {'name':[], 'host_name':[],
                   'host_ra':[], 'host_dec':[]}
    results_dict.update(mag_dict)

    # get host galaxy info
    osc_df = pd.read_csv('osc_results.csv')
    for name in name_list:
        if name in skip_sne:
            continue
        host_info = osc_df[osc_df.SN_Name==name]
        results_dict['host_name'].append(host_info.Host_Name.values[0])
        host_ra = host_info.Host_RA.values[0]
        host_dec = host_info.Host_DEC.values[0]
        results_dict['host_ra'].append(host_ra)
        results_dict['host_dec'].append(host_dec)

        sn_dir = f'fits_files/{name}'
        host_files = [os.path.join(sn_dir,
                                   f'host_{filt}.fits')
                                        for filt in filters]

        if coadd:
            resample = False  # shouldn't be necessary for PS1 images
            coadd_images(name, coadd_filters, resample)
            #coadd_images_linear(name, coadd_filters)
            coadd_file = os.path.join(sn_dir, f'host_{coadd_filters}.fits')
            gal_object, _ = extract_aperture_params(coadd_file,
                                                    host_ra, host_dec,
                                                    threshold, bkg_sub)
        else:
            gal_object = None

        for host_file, filt in zip(host_files, filters):
            try:
                mag, mag_err = extract_global_photometry(host_file,
                                                         host_ra,
                                                         host_dec,
                                                         gal_object,
                                                         mask_stars,
                                                         ZP,
                                                         threshold,
                                                         bkg_sub,
                                                         show_plots)
                results_dict[filt].append(mag)
                results_dict[filt+'_err'].append(mag_err)
            except Exception as message:
                results_dict[filt].append(np.nan)
                results_dict[filt+'_err'].append(np.nan)
                print(f'{name} failed with {filt} band: {message}')

        results_dict['name'].append(name)
    global_phot_df = pd.DataFrame(results_dict)

    return global_phot_df
