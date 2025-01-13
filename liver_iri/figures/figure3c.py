"""Plots Figure 3c -- CTF 3: Eotaxin Interactions"""
import numpy as np
from scipy.stats import pearsonr
import xarray as xr

from .common import getSetup, plot_scatter
from ..dataimport import build_coupled_tensors

COLORS = ["tab:blue", "tab:orange"]
GRANULOCYTE_CYTOKINES = ["IL-17A", "IL-12P40", "IL-7", "VEGF"]


def makeFigure():
    ############################################################################
    # Data imports
    ############################################################################

    raw_data = build_coupled_tensors(
        lft_scaling=1,
        pv_scaling=1,
        no_missing=True,
        normalize=False,
        transform="log"
    )
    raw_val = build_coupled_tensors(
        lft_scaling=1,
        pv_scaling=1,
        no_missing=False,
        normalize=False,
        transform="log"
    )
    raw_data = xr.merge([raw_data, raw_val])
    cytokine_measurements = raw_data["Cytokine Measurements"]

    ############################################################################
    # Figure setup
    ############################################################################

    axs, fig = getSetup(
        (9, 3),
        {"nrows": 1, "ncols": 3}
    )

    ax = axs[0]

    cytokine_measurements = cytokine_measurements.loc[
        {
            "Cytokine": ["IL-1RA", "IL-15", "Eotaxin"]
        }
    ]
    for index, cytokine_tp in enumerate(
            cytokine_measurements["Cytokine Timepoint"].values
    ):
        cytokines = cytokine_measurements.loc[
            {
                "Cytokine Timepoint": cytokine_tp
            }
        ].squeeze().to_pandas()
        il_1ra = cytokines.loc[:, ["IL-1RA", "Eotaxin"]].dropna(axis=0)
        il_15 = cytokines.loc[:, ["IL-15", "Eotaxin"]].dropna(axis=0)
        il_1ra_corr = pearsonr(
            il_1ra.loc[:, "IL-1RA"],
            il_1ra.loc[:, "Eotaxin"]
        )
        il_15_corr = pearsonr(
            il_15.loc[:, "IL-15"],
            il_15.loc[:, "Eotaxin"]
        )

        ax.bar(
            index * 3,
            il_1ra_corr.statistic,
            width=1,
            color="tab:blue"
        )
        ax.bar(
            index * 3 + 1,
            il_15_corr.statistic,
            width=1,
            color="tab:orange"
        )

    x_lims = ax.get_xlim()
    ax.plot([-100, 100], [0, 0], linestyle="--", color="k")
    ax.set_xlim(x_lims)

    ax.set_xticks(
        np.arange(
            0.5,
            len(cytokine_measurements["Cytokine Timepoint"].values) * 3,
            3
        )
    )

    ax.set_ylabel("Pearson Correlation")
    ax.legend()

    ax = axs[1]

    eotaxin_il15 = cytokine_measurements.loc[
        {
            "Cytokine": ["IL-15", "Eotaxin"],
            "Cytokine Timepoint": "LF"
        }
    ].squeeze().to_pandas().dropna()
    eotaxin_il15.columns = "LF: " + eotaxin_il15.columns

    plot_scatter(
        eotaxin_il15,
        ax
    )

    ax = axs[2]

    eotaxin_il1ra = cytokine_measurements.loc[
        {
            "Cytokine": ["IL-1RA", "Eotaxin"],
            "Cytokine Timepoint": "LF"
        }
    ].squeeze().to_pandas().dropna()
    eotaxin_il15.columns = "LF: " + eotaxin_il15.columns

    plot_scatter(
        eotaxin_il1ra,
        ax
    )

    return fig
