"""Plots Figure 6b -- tPLS 2: IL-4 & EGF/IL-6"""

import xarray as xr

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
    # IL-4 & regenerative factors M1 comparisons
    ############################################################################

    ax = axs[0]

    egf_il4 = (
        cytokine_measurements.loc[
            {"Cytokine": ["IL-4", "EGF"], "Cytokine Timepoint": "M1"}
        ]
        .squeeze()
        .to_pandas()
    )
    egf_il4.columns = "M1: " + egf_il4.columns
    plot_scatter(egf_il4, ax)

    ax = axs[1]

    il6_il4 = (
        cytokine_measurements.loc[
            {"Cytokine": ["IL-4", "IL-6"], "Cytokine Timepoint": "M1"}
        ]
        .squeeze()
        .to_pandas()
    )
    il6_il4.columns = "M1: " + il6_il4.columns
    plot_scatter(il6_il4, ax)

    return fig
