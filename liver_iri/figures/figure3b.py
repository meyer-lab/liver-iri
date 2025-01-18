"""Plots Figure 3b -- CTF 2: IFNg & VEGF"""

import numpy as np
import xarray as xr
from scipy.stats import pearsonr

from ..dataimport import build_coupled_tensors
from .common import getSetup, plot_scatter


def makeFigure():
    ############################################################################
    # Data imports
    ############################################################################

    raw_data = build_coupled_tensors(
        lft_scaling=1,
        pv_scaling=1,
        no_missing=True,
        normalize=False,
        transform="log",
    )
    raw_val = build_coupled_tensors(
        lft_scaling=1,
        pv_scaling=1,
        no_missing=False,
        normalize=False,
        transform="log",
    )
    raw_data = xr.merge([raw_data, raw_val])
    cytokine_measurements = raw_data["Cytokine Measurements"]

    ############################################################################
    # Figure setup
    ############################################################################

    axs, fig = getSetup((6, 3), {"nrows": 1, "ncols": 2})

    ############################################################################
    # IFN-gamma & VEGF correlations
    ############################################################################

    ax = axs[0]
    correlations = []
    for tp in cytokine_measurements["Cytokine Timepoint"].values:
        ifng = (
            cytokine_measurements.loc[
                {"Cytokine": "IFNg", "Cytokine Timepoint": tp}
            ]
            .squeeze()
            .to_pandas()
            .dropna()
        )
        vegf = (
            cytokine_measurements.loc[
                {"Cytokine": ["VEGF"], "Cytokine Timepoint": tp}
            ]
            .squeeze()
            .to_pandas()
            .dropna()
        )
        result = pearsonr(ifng, vegf)
        correlations.append(result.statistic)

    ax.bar(np.arange(len(correlations)), correlations)

    ############################################################################
    # IFN-gamma & VEGF at Day 1
    ############################################################################

    ax = axs[1]

    ifng_vegf = (
        cytokine_measurements.loc[
            {"Cytokine": ["IFNg", "VEGF"], "Cytokine Timepoint": "D1"}
        ]
        .squeeze()
        .to_pandas()
        .dropna()
    )
    ifng_vegf.columns = "D1: " + ifng_vegf.columns

    plot_scatter(ifng_vegf, ax)

    return fig
