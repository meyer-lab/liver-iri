"""Plots Figure 3b -- Resolving Granulocyte Responses"""
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, pearsonr
import xarray as xr

from .common import getSetup, plot_scatter
from ..dataimport import build_coupled_tensors
from ..tensor import run_coupled

COLORS = ["tab:blue", "tab:orange"]
GRANULOCYTE_CYTOKINES = ["IL-17A", "IL-12P40", "IL-7", "VEGF"]


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
    patient_factor = patient_factor.sort_values(by=2, ascending=False)
    high_comp = patient_factor.index[:30]
    low_comp = patient_factor.index[-30:]

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
        (12, 6),
        {"nrows": 2, "ncols": 4}
    )

    ############################################################################
    # Granulocyte component boxplots
    ############################################################################

    for cytokine, ax in zip(GRANULOCYTE_CYTOKINES, axs[:4]):
        cytokine_df = cytokine_measurements.loc[
            {
                "Cytokine": cytokine
            }
        ].squeeze().to_pandas()
        timepoints = list(cytokine_df.columns)

        for index, tp in enumerate(cytokine_df.columns):
            low_patch = ax.boxplot(
                cytokine_df.loc[low_comp, tp].dropna(),
                patch_artist=True,
                positions=[index * 3],
                notch=True,
                vert=True,
                widths=0.9,
                flierprops={
                    "markersize": 6,
                    "markerfacecolor": COLORS[0]
                }
            )
            high_patch = ax.boxplot(
                cytokine_df.loc[high_comp, tp].dropna(),
                patch_artist=True,
                positions=[index * 3 + 1],
                notch=True,
                vert=True,
                widths=0.9,
                flierprops={
                    "markersize": 6,
                    "markerfacecolor": COLORS[1]
                }
            )
            low_patch["boxes"][0].set_facecolor(COLORS[0])
            high_patch["boxes"][0].set_facecolor(COLORS[1])
            result = ttest_ind(
                cytokine_df.loc[high_comp, tp].dropna(),
                cytokine_df.loc[low_comp, tp].dropna()
            )
            if result.pvalue < 0.05:
                timepoints[index] = timepoints[index] + "*"

        ax.set_xticks(np.arange(0.5, 6 * 3, 3))
        ax.set_xticklabels(timepoints)

        ax.set_xlim([-1, 3 * 6 - 1])
        ax.set_ylabel("Cytokine Expression")

        ax.set_title(cytokine)

    ############################################################################
    # IFN-gamma interactions
    ############################################################################

    ifng_vegf = cytokine_measurements.loc[
        {
            "Cytokine": ["IFNg", "VEGF"],
            "Cytokine Timepoint": "D1"
        }
    ].squeeze().to_pandas()
    ifng_il7 = cytokine_measurements.loc[
        {
            "Cytokine": ["IFNg", "IL-7"],
            "Cytokine Timepoint": "PV"
        }
    ].squeeze().to_pandas()

    ax = axs[4]
    ifng_vegf.columns = ["D1: IFNg", "D1: VEGF"]
    plot_scatter(ifng_vegf, ax)

    ax = axs[5]
    ifng_il7.columns = ["PV: IFNg", "PV: IL-7"]
    plot_scatter(ifng_il7, ax)

    ############################################################################
    # IL-17A LFT associations
    ############################################################################

    ax = axs[6]

    il_17a = cytokine_measurements.loc[
        {
            "Cytokine": "IL-17A",
            "Cytokine Timepoint": "PO"
        }
    ].squeeze().to_pandas()
    il_17a = il_17a.loc[
        il_17a > 2
    ]

    for lft_index, lft_score in enumerate(lft_measurements["LFT Score"].values):
        lfts = lft_measurements.loc[
            {
                "LFT Score": lft_score,
                "LFT Timepoint": ["Opening"] + [str(i) for i in range(1, 8)]
            }
        ].squeeze().to_pandas()
        correlations = []
        for lft_tp in lfts.columns:
            df = pd.concat([il_17a, lfts.loc[:, lft_tp]], axis=1)
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
