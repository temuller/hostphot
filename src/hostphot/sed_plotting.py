import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import hostphot
from hostphot._constants import workdir, font_family
from hostphot.utils import get_survey_filters

path = hostphot.__path__[0]
config_file = os.path.join(path, 'filters', 'config.txt')
config_df = pd.read_csv(config_file, delim_whitespace=True)

colours = {'GALEX':'purple', 'PS1':'green', 'SDSS':'blue', 'DES':'lightblue', 
           'SkyMapper':'slateblue', 'SPLUS':'lime',
           '2MASS':'red', 'unWISE':'brown', 'WISE':'black',
           'VISTA':'coral', 'UKIDSS':'darkgoldenrod'}


def get_eff_wave(filt, survey):
    """Obtains the effective wavelength of a filter.
    
    Parameters
    ----------
    filt: str
        Filter name.
    survey: str
        Survey name.
        
    Returns
    -------
    eff_wave: float
        Effective wavelength in angstroms.
    """
    path = hostphot.__path__[0]
    if survey=='unWISE':
        survey = 'WISE'
    survey_files = glob.glob(os.path.join(path, 'filters', survey, '*'))
    
    filt_file = [file for file in survey_files if filt in os.path.basename(file)][0]
    wave, trans = np.loadtxt(filt_file).T
    eff_wave = np.sum(wave*trans)/np.sum(trans)
    
    return eff_wave


def plot_sed(name, phot_type='global', z=None, radius=None, include=None, exclude=None):
    """Plots the SED of an object.

    The SED will depend on the available photometry.

    Parameters
    ----------
    name : str
        Name of the object.
    phot_type : str, optional
        Type of photometry: ``global`` or ``local``. By default 'global'.
    z : float, optional
        Redshift of the object, by default ``None``. If given, corrects
        for time dilation.
    radius : int, float or str, optional
        Radius for the local photometry, by default ``None``.
    include: list, default ``None``
        List of surveys to include in the plot. Cannot be given together 
        with '``exclude``.
    exclude: list, default ``None``
        List of surveys to exclude from the plot. Cannot be given together 
        with '``include``.

    Raises
    ------
    ValueError
        The photometry type should be either ``global`` or ``local``.
    """
    if phot_type == 'local':
        assert radius is not None, "radius must be given with local photometry"
        
    global colours
    obj_path = os.path.join(workdir, name, '*')
    phot_files = [file for file in glob.glob(obj_path) 
                  if file.endswith(f'_{phot_type}.csv')]
    
    if include is not None and exclude is not None:
        raise ValueError("'inlcude' cannot be given together with 'exclude'!")
    
    if include is not None:
        include_files = []
        for file in phot_files:
            for pattern in include:
                if pattern in file:
                    include_files.append(file)
                    break
        phot_files = include_files

    if exclude is not None:
        include_files = []
        for file in phot_files:
            skip = False
            for pattern in exclude:
                if pattern in file:
                    skip = True
            if skip is False:
                include_files.append(file)
        phot_files = include_files
    
    if len(phot_files)==0:
        print(f'There is no photometry for {name}!')
        return None
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.invert_yaxis()  # for magnitude plot

    for file in phot_files:
        survey = os.path.basename(file).split('_')[0]
        filters = get_survey_filters(survey)

        # Vega to AB
        global config_df
        survey_df = config_df[config_df.survey==survey]
        mag_sys_conv = survey_df.mag_sys_conv.values[0]
        if ',' in mag_sys_conv:
            mag_sys_conv = {filt:float(conv) for filt, conv 
                            in zip(filters, mag_sys_conv.split(','))}
        else:
            mag_sys_conv = {filt:0.0 for filt in filters}

        # adapt name for local photometry
        if phot_type=='local':
            title = f'{name} - {phot_type} SED (r $= {radius}$ kpc)'
            ext = f'_{radius}'
        elif phot_type=='global':
            title = f'{name} - {phot_type} SED'
            ext = ''
        else:
            raise ValueError(f'Invalid photometry type: {phot_type}')


        phot_df = pd.read_csv(file)
        waves, phot, phot_err = [], [], []
        for filt in filters:
            filt_str = filt + ext
            filt_err_str = filt + ext + '_err'

            # get photometry
            wave = get_eff_wave(filt, survey)
            mag = phot_df[filt_str].values[0] + mag_sys_conv[filt]
            mag_err = phot_df[filt_err_str].values[0]
            waves.append(wave)
            phot.append(mag)
            phot_err.append(mag_err)

        if z is not None:
            # correct for time dilation
            waves = np.array(waves)/(1+z)
            phot = np.array(phot) - 2.5*np.log10((1+z))
            xlabel = r'Rest Wavelength ($\AA$)'
            if ')' in title:
                title = title.replace(')', f' @ $z={z}$)')
            else:
                title = title + f' ($z={z}$)'
        else:
            waves = np.array(waves)
            phot = np.array(phot)
            xlabel = r'Observed Wavelength ($\AA$)'
        phot_err = np.array(phot_err)

        lims = phot/phot_err < 3  # upper limits;inverted for magnitude plots
        phot_err[lims] = 1  # for visualization
        ax.errorbar(waves, phot, yerr=phot_err, marker='o', ms=10, ls='dashed', lw=3,
                    c=colours[survey], label=survey, lolims=lims)

    if len(phot_files)>3:
        ncol = 2
    else:
        ncol=1

    ax.set_xlabel(xlabel, fontsize=24, font=font_family)
    ax.set_ylabel('Magnitude (AB)', fontsize=24, font=font_family)
    ax.set_title(title, fontsize=28, font=font_family)

    for label in ax.get_xticklabels():
        label.set_fontproperties(font_family)
    for label in ax.get_yticklabels():
        label.set_fontproperties(font_family)
    ax.tick_params(labelsize=20)
    
    ax.legend(ncol=ncol, fancybox=True, framealpha=1, prop={"size": 18, "family": font_family})
    ax.set_xscale('log')

    plt.show()