import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import ticker
import matplotlib.pyplot as plt
from typing import Optional

import hostphot
from hostphot._constants import workdir, font_family
from hostphot.surveys_utils import get_survey_filters, filters_file, load_yml, extract_filter

path = Path(hostphot.__path__[0])
filters_config = load_yml(filters_file)
#config_file = path.joinpath("filters", "config.txt")
#config_df = pd.read_csv(config_file, sep="\\s+")


colours = {
    "GALEX": "purple",
    "PanSTARRS": "green",
    "SDSS": "blue",
    "DES": "lightblue",
    "SkyMapper": "slateblue",
    "SPLUS": "lime",
    "LegacySurvey": "gold",
    "2MASS": "red",
    "unWISE": "brown",
    "WISE": "black",
    "VISTA": "coral",
    "UKIDSS": "darkgoldenrod",
}


def get_eff_wave(filt: str, survey: str) -> float:
    """Obtains the effective wavelength of a filter.

    Parameters
    ----------
    filt: Filter name.
    survey: Survey name.

    Returns
    -------
    eff_wave: Effective wavelength in angstroms.
    """
    wave, trans = extract_filter(filt, survey)
    eff_wave = np.sum(wave * trans) / np.sum(trans)

    return eff_wave


def plot_sed(
    name: str,
    phot_type: str = "global",
    z: Optional[float] = None,
    radius: str | int | float = None,
    include: list = None,
    exclude: list = None,
    save_plot: bool = True,
    outfile: str | Path = None,
) -> None:
    """Plots the SED of an object.

    The SED will depend on the available photometry.

    Parameters
    ----------
    name: Object name.
    phot_type: Type of photometry: ``global`` or ``local``. 
    z: Redshift of the object, by default ``None``. If given, corrects
        for time dilation.
    radius : Radius for the local photometry, by default ``None``.
    include: List of surveys to include in the plot. Cannot be given together
        with '``exclude``.
    exclude: List of surveys to exclude from the plot. Cannot be given together
        with '``include``.
    save_plot: Whether to save the SED plot.
    outfile: If give, the plot is saved with this name instead of the default ones.

    Raises
    ------
    ValueError
        The photometry type should be either ``global`` or ``local``.
    """
    if phot_type == "local":
        assert radius is not None, "radius must be given with local photometry"

    global colours
    obj_path = Path(workdir, name)
    phot_files = [file for file in obj_path.rglob(f"*{phot_type}_photometry.csv")]

    if include is not None and exclude is not None:
        raise ValueError("'inlcude' cannot be given together with 'exclude'!")

    # include or exclude some surveys
    if include is not None:
        include_files = []
        for file in phot_files:
            for pattern in include:
                if pattern in str(file):
                    include_files.append(file)
                    break
        phot_files = include_files

    if exclude is not None:
        include_files = []
        for file in phot_files:
            skip = False
            for pattern in exclude:
                if pattern in str(file):
                    skip = True
            if skip is False:
                include_files.append(file)
        phot_files = include_files

    if len(phot_files) == 0:
        print(f"There is no photometry for {name}!")
        return None

    # start plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.invert_yaxis()  # for magnitude plot

    for file in phot_files:
        survey = file.parts[-2]
        filters = get_survey_filters(survey)

        # Vega to AB
        mag_sys_conv = {filt: filters_config[survey][filt]["Vega_to_AB"] 
                        for filt in filters}

        # adapt name for local photometry
        if phot_type == "local":
            title = fr"{name} - {phot_type} SED (r $= {radius}$ kpc)"
            ext = f"_{radius}"
        elif phot_type == "global":
            title = f"{name} - {phot_type} SED"
            ext = ""
        else:
            raise ValueError(f"Invalid photometry type: {phot_type}")

        phot_df = pd.read_csv(file)
        waves, phot, phot_err = [], [], []
        valid_filters = []
        for filt in filters:
            filt_str = filt + ext
            filt_err_str = filt + ext + "_err"
            if filt_str not in phot_df.columns:
                continue

            # get photometry
            wave = get_eff_wave(filt, survey)
            mag = phot_df[filt_str].values[0] + mag_sys_conv[filt]
            mag_err = phot_df[filt_err_str].values[0]
            waves.append(wave)
            phot.append(mag)
            phot_err.append(mag_err)
            valid_filters.append(filt)

        if z is not None:
            # correct for time dilation
            waves = np.array(waves) / (1 + z)
            phot = np.array(phot) - 2.5 * np.log10((1 + z))
            xlabel = r"Rest Wavelength ($\AA$)"
            if ")" in title:
                title = title.replace(")", fr" @ $z={z}$)")
            else:
                title = title + fr" ($z={z}$)"
        else:
            waves = np.array(waves)
            phot = np.array(phot)
            xlabel = r"Observed Wavelength ($\AA$)"
        phot_err = np.array(phot_err)

        # NaN mask
        print(file)
        print(phot, phot_err)
        mask = ~np.isnan(phot) * phot_err > 0
        waves = waves[mask]
        phot = phot[mask]
        phot_err = phot_err[mask]
        valid_filters = np.array(valid_filters)[mask]

        lims = phot / phot_err < 3  # upper limits;inverted for magnitude plots
        phot_err[lims] = 1  # for visualization
        label = f'{survey} ({", ".join(filt for filt in valid_filters)})'
        ax.errorbar(
            waves,
            phot,
            yerr=phot_err,
            marker="o",
            ms=10,
            ls="dashed",
            lw=3,
            c=colours[survey],
            mec="k",
            capsize=6,
            label=label,
            lolims=lims,
        )

    if len(phot_files) > 4:
        ncol = 2
    else:
        ncol = 1

    ax.set_xlabel(xlabel, fontsize=24, font=font_family)
    ax.set_ylabel("Magnitude (AB)", fontsize=24, font=font_family)
    ax.set_title(title, fontsize=28#, font=font_family
                 )

    for label in ax.get_xticklabels():
        label.set_fontproperties(font_family)
    for label in ax.get_yticklabels():
        label.set_fontproperties(font_family)
    ax.tick_params(labelsize=20)

    ax.legend(
        ncol=ncol, fancybox=True, framealpha=1, prop={"size": 16, "family": font_family}
    )
    ax.set_xscale("log")
    # format ticks
    ticks = np.array([2e3, 4e3, 9e3, 2e4, 4e4, 9e4, 2e5, 4e5, 9e5])
    start, end = ax.get_xlim()
    mask = (ticks >= start) & (ticks <= end)
    ax.set_xticks(ticks[mask])
    formatter = ticker.ScalarFormatter(useMathText=True)
    ax.get_xaxis().set_major_formatter(formatter)

    if save_plot is True:
        if outfile is None:
            obj_dir = Path(workdir, name)
            basename = f"sed_{phot_type}.jpg"
            outfile = obj_dir / basename
        plt.savefig(outfile)
    plt.show()
