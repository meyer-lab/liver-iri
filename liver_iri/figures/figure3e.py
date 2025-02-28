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

    axs, fig = getSetup((15, 3), {"nrows": 1, "ncols": 5})

    ############################################################################
    # IL-9 Errorbars
    ############################################################################

    il_9 = cytokine_measurements.loc[{"Cytokine": "IL-9"}].squeeze().to_pandas()

    il_9 = il_9.sort_values("PO", ascending=False)
    high_il_9 = il_9.loc[il_9.loc[:, "PO"] >= 3.5, :]
    low_il_9 = il_9.loc[il_9.loc[:, "PO"] <= 2, :]
    mid_il_9 = il_9.drop(high_il_9.index).drop(low_il_9.index)

    colors = ["tab:green", "black", "tab:red"]
    labels = ["Low Pre-Op IL-9", "Medium Pre-Op IL-9", "High Pre-Op IL-9"]
    for index, (df, label, color) in enumerate(zip(
        [low_il_9, mid_il_9, high_il_9],
        labels,
        colors,
        strict=False
    )):
        axs[0].plot(
            np.arange(df.shape[1]),
            df.T,
            color=color,
            alpha=0.25,
        )
        axs[1].errorbar(
            np.arange(df.shape[1]) + (index - 1) * 0.1,
            df.mean(axis=0),
            yerr=df.std(axis=0),
            capsize=1,
            color=color,
        )

    for ax in axs[:2]:
        ax.legend(labels)
        ax.set_ylabel("IL-9")
        ax.set_xticks(np.arange(il_9.shape[1]))
        ax.set_xticklabels(il_9.columns)

    ############################################################################
    # IL-9 Correlations
    ############################################################################

    for df, title, ax in zip(
        [low_il_9, mid_il_9, high_il_9],
        labels,
        axs[2:],
        strict=False,
    ):
        df = df.loc[:, ["PO", "D1"]]
        df.columns = df.columns + ": IL-9"
        plot_scatter(df, ax)
        ax.set_title(f"{title} (n={str(df.shape[0])})")

    return fig
