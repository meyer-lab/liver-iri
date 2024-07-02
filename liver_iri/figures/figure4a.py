"""Plots Figure 4a -- Long-term Histogram"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from .common import getSetup, plot_scatter
from ..dataimport import build_coupled_tensors
from ..tensor import run_coupled

COLORS = ["tab:blue", "tab:orange"]
TH2_CYTOKINES = ["IL-4", "IL-5", "IL-9", "IL-13"]
THRESHOLDS = {
    "IL-4": 6,
    "IL-5": 2,
    "IL-9": 2,
    "IL-13": 3
}


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
    cytokine_measurements = raw_data["Cytokine Measurements"]
    lft_measurements = raw_data["LFT Measurements"]

    ############################################################################
    # Figure setup
    ############################################################################

    axs, fig = getSetup(
        (9, 6),
        {"nrows": 2, "ncols": 3}
    )

    ############################################################################
    # Component Histogram
    ############################################################################

    ax = axs[0]

    ax.hist(
        patient_factor.loc[:, 1],
        range=(-1, 1),
        bins=20,
        alpha=0.75,
        color="tab:orange"
    )
    ax.hist(
        patient_factor.loc[:, 4],
        range=(-1, 1),
        bins=20,
        alpha=0.75,
        color="tab:blue"
    )

    ax.set_xlim([-1, 1])
    ax.set_xlabel("Component Association")
    ax.set_ylabel("Frequency")

    ############################################################################
    # IL-4 scatters
    ############################################################################

    df = cytokine_measurements.loc[{
        "Cytokine": "IL-4",
        "Cytokine Timepoint": ["PO", "W1"]
    }].squeeze().to_pandas()
    comp = patient_factor.loc[:, 1]

    plot_scatter(
        df,
        axs[1],
        cmap=comp
    )
    x_lims = axs[1].get_xlim()
    axs[1].set_xlim([0, x_lims[1]])

    trimmed = df.loc[
        df.iloc[:, 0].between(6, 10),
        :
    ]
    trimmed = trimmed.loc[
        trimmed.iloc[:, 1] > 2,
        :
    ]

    plot_scatter(
        trimmed,
        axs[2],
        cmap=comp
    )
    x_lims = axs[2].get_xlim()
    y_lims = axs[2].get_ylim()
    axs[2].set_xlim([6, x_lims[1]])
    axs[2].set_ylim([2, y_lims[1]])

    ############################################################################
    # Th2 Pre-Op / Week 1 Correlations
    ############################################################################

    ax = axs[3]

    th2_cytokines = cytokine_measurements.loc[
        {
            "Cytokine": ["IL-4", "IL-5", "IL-9", "IL-13"],
            "Cytokine Timepoint": ["PO", "W1"]
        }
    ]
    correlations = pd.Series(index=th2_cytokines["Cytokine"].values)
    for cytokine in th2_cytokines["Cytokine"].values:
        df = th2_cytokines.loc[{"Cytokine": cytokine}].squeeze().to_pandas()
        df = df.dropna(axis=0)
        df = df.loc[
            df.iloc[:, 0] > THRESHOLDS[cytokine],
            :
        ]
        result = pearsonr(
            df.iloc[:, 0],
            df.iloc[:, 1]
        )
        correlations.loc[cytokine] = result.statistic

    correlations = correlations.sort_values(ascending=False)
    ax.bar(
        np.arange(len(correlations)),
        correlations,
        width=0.9
    )

    ax.set_xticks(np.arange(len(correlations)))
    ax.set_xticklabels(correlations.index)
    ax.set_ylabel("Pearson Correlation")

    ############################################################################
    # IL-1a / IL-4 / LFT Associations
    ############################################################################

    il_1a = cytokine_measurements.loc[
        {
            "Cytokine": "IL-1a",
            "Cytokine Timepoint": "W1"
        }
    ].squeeze().to_pandas()
    il_4 = cytokine_measurements.loc[
        {
            "Cytokine": "IL-4",
            "Cytokine Timepoint": "W1"
        }
    ].squeeze().to_pandas()
    lft = lft_measurements.loc[
        {
            "LFT Score": "alt",
            "LFT Timepoint": "7"
        }
    ].squeeze().to_pandas()

    df_1 = pd.concat([il_4, il_1a], axis=1)
    df_2 = pd.concat([il_1a, lft], axis=1)
    df_1.columns = ["W1: IL-4", "W1: IL-1a"]
    df_2.columns = ["W1: IL-1a", "W1: ALT"]
    for ax, df in zip(axs[4:6], [df_1, df_2]):
        plot_scatter(df, ax)

    return fig
