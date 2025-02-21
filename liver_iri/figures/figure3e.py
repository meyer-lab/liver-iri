"""Plots Figure 3e -- CTF 1: IL-9 Plotting"""

import numpy as np
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

    axs, fig = getSetup((12, 3), {"nrows": 1, "ncols": 4})

    ############################################################################
    # IL-9 Errorbars
    ############################################################################

    il_9 = cytokine_measurements.loc[{"Cytokine": "IL-9"}].squeeze().to_pandas()

    il_9 = il_9.sort_values("PO", ascending=False)
    high_il_9 = il_9.loc[il_9.loc[:, "PO"] >= 3.5, :]
    low_il_9 = il_9.loc[il_9.loc[:, "PO"] <= 2, :]
    mid_il_9 = il_9.drop(high_il_9.index).drop(low_il_9.index)

    ax = axs[0]
    colors = ["tab:green", "black", "tab:red"]
    for index, (df, color) in enumerate(
        zip([low_il_9, mid_il_9, high_il_9], colors, strict=False)
    ):
        ax.errorbar(
            np.arange(df.shape[1]) + (index - 1) * 0.1,
            df.mean(axis=0),
            yerr=df.std(axis=0),
            capsize=1,
            color=color,
        )

    ############################################################################
    # IL-9 Correlations
    ############################################################################

    for df, title, ax in zip(
        [low_il_9, mid_il_9, high_il_9],
        ["Low", "Middle", "High"],
        axs[1:],
        strict=False,
    ):
        df = df.loc[:, ["PO", "D1"]]
        plot_scatter(df, ax)
        ax.set_title(title)

    return fig
