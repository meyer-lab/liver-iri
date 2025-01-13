"""Plots Figure 3a -- CTF 1: Th2 Returns"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import xarray as xr

from .common import getSetup, plot_scatter
from ..dataimport import build_coupled_tensors


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

    ############################################################################
    # IL-9 scatters
    ############################################################################

    d1 = cytokine_measurements.loc[{
        "Cytokine": "IL-9",
        "Cytokine Timepoint": ["PO", "D1"]
    }].squeeze().to_pandas()
    m1 = cytokine_measurements.loc[{
        "Cytokine": "IL-9",
        "Cytokine Timepoint": ["PO", "M1"]
    }].squeeze().to_pandas()

    plot_scatter(
        d1,
        axs[0],
    )
    x_lims = axs[0].get_xlim()
    axs[0].set_xlim([0, x_lims[1]])

    plot_scatter(
        m1,
        axs[1],
    )
    x_lims = axs[1].get_xlim()
    axs[1].set_xlim([0, x_lims[1]])

    ############################################################################
    # Th2 Pre-Op / Post-Op Correlations
    ############################################################################

    ax = axs[2]

    th2_cytokines = cytokine_measurements.loc[
        {
            "Cytokine": ["IL-1a", "IL-4", "IL-5", "IL-9", "IL-13"],
            "Cytokine Timepoint": ["PO", "D1", "W1", "M1"]
        }
    ]

    correlations = pd.DataFrame(
        index=th2_cytokines["Cytokine"].values,
        columns=th2_cytokines["Cytokine Timepoint"].values[1:]
    )
    for cytokine in th2_cytokines["Cytokine"].values:
        df = th2_cytokines.loc[{"Cytokine": cytokine}].squeeze().to_pandas()
        df = df.dropna(axis=0)
        for tp in correlations.columns:
            correlations.loc[cytokine, tp] = pearsonr(
                df.loc[:, "PO"],
                df.loc[:, tp]
            ).statistic

    correlations = correlations.sort_values(by="W1", ascending=False)

    for index, tp in enumerate(correlations.columns):
        ax.bar(
            np.arange(index, 4 * correlations.shape[0], 4),
            correlations.loc[:, tp],
            width=1
        )

    ax.set_xticks(np.arange(1, 4 * correlations.shape[0], 4))
    ax.set_xticklabels(correlations.index)
    ax.set_ylabel("Pearson Correlation")

    return fig
