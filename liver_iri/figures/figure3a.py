"""Plots Figure 3a -- Resolving NK responses"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import xarray as xr

from .common import getSetup, plot_scatter
from ..dataimport import build_coupled_tensors
from ..tensor import run_coupled

COLORS = ["tab:blue", "tab:orange"]
NK_AXIS = ["IL-2", "IL-15"]


def makeFigure():
    ############################################################################
    # Factorization
    ############################################################################

    data = build_coupled_tensors(
        pv_scaling=1,
        lft_scaling=1,
        no_missing=True
    )

    _, cp = run_coupled(data, rank=4)
    patient_factor = cp.x["_Patient"].to_pandas()

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
    lft_measurements = raw_data["LFT Measurements"]

    ############################################################################
    # Figure setup
    ############################################################################

    axs, fig = getSetup(
        (12, 3),
        {"nrows": 1, "ncols": 4}
    )

    ############################################################################
    # Component histograms
    ############################################################################

    ax = axs[0]

    ax.hist(
        patient_factor.loc[:, 2],
        range=(-1, 1),
        bins=20,
        alpha=0.75,
        color=COLORS[0]
    )
    ax.hist(
        patient_factor.loc[:, 3],
        range=(-1, 1),
        bins=20,
        alpha=0.75,
        color=COLORS[1]
    )

    ax.set_xlim([-1, 1])
    ax.set_xlabel("Component Association")
    ax.set_ylabel("Frequency")

    ############################################################################
    # IL-2/IL-15 Boxplots
    ############################################################################

    ax = axs[1]

    for bar_index, (color, cytokine) in enumerate(zip(COLORS, NK_AXIS)):
        cytokine_df = cytokine_measurements.loc[
            {
                "Cytokine": cytokine
            }
        ].squeeze().to_pandas()
        for index, tp in enumerate(cytokine_df.columns):
            patch = ax.boxplot(
                cytokine_df.loc[:, tp].dropna(),
                patch_artist=True,
                positions=[index * 3 + bar_index],
                notch=True,
                vert=True,
                widths=0.9,
                flierprops={
                    "markersize": 6,
                    "markerfacecolor": color
                }
            )
            patch["boxes"][0].set_facecolor(color)

    ax.set_xticks(np.arange(0.5, 6 * 3, 3))
    ax.set_xticklabels(cytokine_df.columns)

    ax.set_xlim([-1, 3 * 6 - 1])
    ax.set_ylabel("Cytokine Expression")

    ############################################################################
    # IL-15/Eotaxin Interaction
    ############################################################################

    ax = axs[2]

    il_15 = cytokine_measurements.loc[
        {
            "Cytokine": "IL-15"
        }
    ].squeeze().to_pandas()
    eotaxin = cytokine_measurements.loc[
        {
            "Cytokine": "Eotaxin"
        }
    ].squeeze().to_pandas()

    df = pd.concat(
        [
            il_15.loc[:, "LF"],
            eotaxin.loc[:, "LF"]
        ],
        axis=1
    )
    df.columns = ["LF: IL-15", "LF: Eotaxin"]
    plot_scatter(df, ax)

    ############################################################################
    # NK/LFT Associations
    ############################################################################

    ax = axs[3]

    il_15 = cytokine_measurements.loc[
        {
            "Cytokine": "IL-15",
            "Cytokine Timepoint": "PV"
        }
    ].squeeze().to_pandas()

    for lft_index, lft_score in enumerate(lft_measurements["LFT Score"].values):
        lfts = lft_measurements.loc[
            {
                "LFT Score": lft_score,
                "LFT Timepoint": ["Opening"] + [str(i) for i in range(1, 8)]
            }
        ].squeeze().to_pandas()
        correlations = []
        for lft_tp in lfts.columns:
            df = pd.concat([il_15, lfts.loc[:, lft_tp]], axis=1)
            df = df.dropna(axis=0)
            result = pearsonr(
                df.iloc[:, 0],
                df.iloc[:, 1]
            )
            correlations.append(result.statistic)

        ax.bar(
            np.arange(lft_index, len(correlations) * 4, 4),
            correlations,
            width=1,
            label=lft_score
        )

    x_lims = ax.get_xlim()
    ax.plot([-100, 100], [0, 0], linestyle="--", color="k")
    ax.set_xlim(x_lims)

    ax.set_xticks(np.arange(1, lfts.shape[1] * 4, 4))
    tick_labels = list(lfts.columns)
    tick_labels[1:] = [f"Day {i}" for i in tick_labels[1:]]
    ax.set_xticklabels(
        tick_labels,
        ha="right",
        ma="right",
        va="top",
        rotation=45
    )

    ax.set_ylabel("Pearson Correlation")
    ax.legend()

    return fig
