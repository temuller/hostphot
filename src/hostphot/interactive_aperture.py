import ipywidgets as widgets
from IPython.display import display
output = widgets.Output()

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import sep
from astropy.io import fits


from hostphot.utils import (get_image_gain, get_image_readnoise, survey_zp,
                            get_survey_filters, check_filters_validity)
from hostphot.image_cleaning import remove_nan
from hostphot._constants import __workdir__
#----------------------------------------
def _choose_workdir(workdir):
    """Updates the work directory.

    Parameters
    ----------
    workdir: str
        Path to the work directory.
    """
    global __workdir__
    __workdir__ = workdir

#----------------------------------------
class InteractiveAperture:
    """Class to interactively set the aperture to calculate photometry.

    Parameters
    ----------
    name: str
        Name of the object to find the path of the fits file.
    filt: str, default, ``g``
        Filter used to initiate the plot.
    filters: str, default, ``None``
        Filters to use to load the fits files. If `None` use all
        the filters of the given survey.
    survey: str, default ``PS1``
        Survey to use for the zero-points and pixel scale.
    masked: bool, default ``True``
        If ``True``, uses masked images.
    """
    def __init__(self, name, filt='g', survey='PS1', filters=None, masked=True):
        self.name = name
        self.filt = filt
        self.survey = survey
        if not filters:
            self.filters = get_survey_filters(survey)
        else:
            check_filters_validity(filters, survey)
            self.filters = filters

        if masked:
            self.masked = 'masked_'
        else:
            self.masked = ''

        base_file = os.path.join(f'{self.masked}{survey}_{filt}.fits')
        self.fits_file = os.path.join(__workdir__, name, base_file)

        self.ellipse_parameters = ['x', 'y', 'width',
                                  'height', 'angle']

        # this is where the photometry is saved
        self.flux_phot = {f:np.nan for f in self.filters}
        self.flux_phot.update({f'{f}_err':np.nan for f in self.filters})
        self.mag_phot = {f:np.nan for f in self.filters}
        self.mag_phot.update({f'{f}_err':np.nan for f in self.filters})

        # initiate plots + widgets
        self._initiate_plots()
        self._initiate_widgets()

    # -------------- Plots ------------------
    # ------
    # Images
    def _draw_image(self):
        """Plots the image."""
        m, s = np.nanmean(self.data), np.nanstd(self.data)
        self.im = self.ax.imshow(self.data, interpolation='nearest',
                           cmap='gray',
                           vmin=m-s, vmax=m+s,
                           origin='lower')

    def _change_image(self, filt):
        """Changes the plotted image."""
        base_file = os.path.join(f'{self.masked}{self.survey}_{filt}.fits')
        self.fits_file = os.path.join(__workdir__, self.name, base_file)
        img = fits.open(self.fits_file)

        self.header = img[0].header
        self.data = img[0].data
        self.data = self.data.astype(np.float64)

        self.bkg = sep.Background(self.data)
        self.err = self.bkg.globalrms

        self.exptime = float(self.header['EXPTIME'])
        self.gain = get_image_gain(self.header, self.survey)
        self.readnoise  = get_image_readnoise(self.header, self.survey)

        title = f'{self.name} (${self.filt}$-band)'
        self.title = title
        self.ax.set_title(title)
        self._draw_image()

    def _button_update_image(self, button):
        self.filt = button.description
        self._change_image(self.filt)

    # -------
    # Ellipse
    def _slider_update_ellipse(self, change):
        """Updates the parameters of the plotted ellipse"""
        slider = change.owner
        if slider.description=='x':
            self.e.center = (change.new, self.e.center[1])
        elif slider.description=='y':
            self.e.center = (self.e.center[0], change.new)
        else:
            self.e.update({slider.description:change.new})

        # removes previous ellipse
        self.ax.clear()
        self.ax.add_patch(self.e)
        self.ax.set_title(self.title)
        self.im = self.draw_image()

    def _get_init_eparams(self):
        """Gets initial ellipse parameters."""
        ydim, xdim = self.data.shape
        max_dim = max(xdim, ydim)
        self.eparams = {'x':{'value':xdim/2, 'min':0, 'max':xdim},
                 'y':{'value':ydim/2, 'min':0, 'max':ydim},
                 'width':{'value':max_dim/2, 'min':0, 'max':max_dim},
                 'height':{'value':max_dim/2, 'min':0, 'max':max_dim},
                 'angle':{'value':0, 'min':-90, 'max':90},
                }

    def _get_eparams(self):
        """Gets the ellipse parameters."""
        self.eparams = {}
        for key, slider in self.sliders.items():
            self.eparams[key] = slider.value

    def _onclick(self, event):
        """Updates the ellipse's center with a mouse click."""
        ix, iy = event.xdata, event.ydata
        self.sliders['x'].value = ix
        self.sliders['y'].value = iy

    # ---------------
    # Master Function
    def _initiate_plots(self):
        """Initiates the plot with the image and
        aperture (ellipse)."""
        self.fig, self.ax = plt.subplots(figsize=(6, 6))

        self.title = f'{self.name} (${self.filt}$-band)'
        self.ax.set_title(self.title)

        self._change_image(self.filt)

        # ellipse
        self._get_init_eparams()
        xy = (self.eparams['x']['value'],
              self.eparams['y']['value'])
        width = self.eparams['width']['value']
        height = self.eparams['height']['value']
        angle = self.eparams['angle']['value']

        self.e = Ellipse(xy=xy,
                        width=width,
                        height=height,
                        angle=angle)

        self.e.set_facecolor('none')
        self.e.set_edgecolor('red')
        self.ax.add_patch(self.e)

    # ---------- Widgets -------------
    # -------
    # widgets
    def _create_sliders(self):
        """Creates dictionary with sliders."""
        self.sliders = {}
        for key in self.ellipse_parameters:
            if self.eparams:
                value = self.eparams[key]['value']
                min = self.eparams[key]['min']
                max = self.eparams[key]['max']
            else:
                value, min, max = 1, 0, 180
            self.sliders[key] = widgets.IntSlider(value, min,
                                              max, step=1,
                                              description=key)

    def _create_textboxes(self):
        """Creates dictionary with text boxes."""
        keys = self.ellipse_parameters
        self.textboxes = {key:widgets.Combobox(description=key)
                                          for key in keys}

    def _create_buttons(self):
        """Creates dictionary with buttons."""
        self.buttons = {key:widgets.Button(description=key)
                              for key in self.filters}

    # ---------------
    # Master Function
    def _initiate_widgets(self):
        """Initiates the widgets and displays them."""
        # Sliders
        self._create_sliders()
        for slider in self.sliders.values():
            slider.observe(self._slider_update_ellipse, 'value')
        self.slider_VBox = widgets.VBox(list(self.sliders.values()))

        # Text Boxes
        self._create_textboxes()
        self.texbox_VBox = widgets.VBox(list(self.textboxes.values()))

        # Links: sliders with text boxes
        self.links = {}
        for key, slider in self.sliders.items():
            textbox = self.textboxes[key]
            self.links[key] = widgets.jslink((slider, 'value'),
                                             (textbox, 'value'))

        # Buttons
        self._create_buttons()
        for button in self.buttons.values():
            button.on_click(self._button_update_image)

        # Photometry Button
        self.phot_button = widgets.Button(description='Calc. Photometry')
        self.phot_button.on_click(self._calculate_phot)

        self.phot_text = widgets.Label(value='Magnitude:')

        # display widgets
        display(clear=True)
        display(widgets.HBox([self.slider_VBox, self.texbox_VBox]))
        display(widgets.HBox(list(self.buttons.values())))
        display(widgets.HBox([self.phot_button, self.phot_text]),
                                                            output)

        self.cid = self.fig.canvas.mpl_connect('button_press_event',
                                               self._onclick)

    # ------------ Photometry ----------
    # Photometry
    def _calculate_flux(self):
        """Calculates the flux within the aperture."""
        x, y = [self.eparams['x']], [self.eparams['y']]
        a, b = [self.eparams['width']], [self.eparams['height']]
        theta = [self.eparams['angle']*(np.pi/180)]  # in radians

        scale = 1  # fixed
        flux, flux_err, flag = sep.sum_ellipse(self.data, x, y,
                                               a, b, theta,
                                               scale, self.err,
                                               subpix=5,
                                               gain=self.gain)

        self.flux_phot[self.filt] =  flux[0]
        self.flux_phot[f'{self.filt}_err'] =  flux_err[0]

    def _calculate_mag(self):
        """Calculates the magnitude within the aperture."""
        flux = self.flux_phot[self.filt]
        flux_err = self.flux_phot[f'{self.filt}_err']
        zp = survey_zp(self.survey) + 2.5*np.log10(self.exptime)

        mag = -2.5*np.log10(flux) + zp
        mag_err = 2.5/np.log(10)*flux_err/flux

        # ellipse area = pi*a*b
        ap_area = np.pi*self.eparams['width']*self.eparams['height']
        extra_err = 1.0857*np.sqrt(ap_area*(self.readnoise**2)+flux/self.gain)/flux
        mag_err = np.sqrt(mag_err**2 + extra_err**2)

        self.mag_phot[self.filt] =  mag
        self.mag_phot[f'{self.filt}_err'] =  mag_err

    # ---------------
    # Master Function
    def _calculate_phot(self, button):
        """Calculates the photometry (flux and magnitude)
        within the aperture."""
        self._get_eparams()
        self._calculate_flux()
        self._calculate_mag()

        mag = self.mag_phot[self.filt]
        mag_err = self.mag_phot[f'{self.filt}_err']
        self.phot_text.value = f'Magnitude (${self.filt}$-band): {mag:.5f} +/- {mag_err:.5f}'

    # Export Photometry
    def export_photometry(self, outfile=None):
        """Exports the photometry (magnitudes) into a csv file.

        Parameters
        ----------
        outfile: str, default ``None``
            Name of the output file. If ``None``, use ``f'{self.name}_phot.csv'``
        """
        if not outfile:
            outfile = f'{self.name}_phot.csv'

        mag_phot = self.mag_phot.copy()
        for key in mag_phot.keys():
            mag_phot[key] = [mag_phot[key]]

        mag_phot['name'] = self.name
        self.phot_df = pd.DataFrame(mag_phot)
        self.phot_df.to_csv(outfile, index=False)
