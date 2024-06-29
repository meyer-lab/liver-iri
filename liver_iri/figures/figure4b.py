"""Plots Figure 4c -- Flt-3L & GRO"""
import warnings

from scipy.stats import pearsonr, ttest_ind
import numpy as np
import xarray as xr

from ..dataimport import build_coupled_tensors
from ..tensor import run_coupled
from .common import getSetup

M_EFFECTORS = ["Flt-3L", "GRO"]

warnings.filterwarnings("ignore")


def makeFigure():
    ############################################################################
    # Factorization
    ############################################################################

    data = build_coupled_tensors(
        pv_scaling=1,
        lft_scaling=1,
        no_missing=True
    )
    val_data = build_coupled_tensors(
        lft_scaling=1,
        pv_scaling=1,
        no_missing=False
    )
    non_missing = val_data["Cytokine Measurements"].isnull().any(
        "Cytokine Timepoint"
    ).all("Cytokine")
    val_data = val_data.loc[{"Patient": non_missing}]
    data = xr.merge([data, val_data])

    _, cp = run_coupled(data, rank=4)
    patient_factor = cp.x["_Patient"].to_pandas()
    patient_factor = patient_factor.sort_values(4, ascending=False)
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
        (9, 6),
        {"nrows": 2, "ncols": 3}
    )

    ############################################################################
    # Component Boxplots
    ############################################################################

    for ax_index, cytokine in enumerate(M_EFFECTORS):
        cytokine_df = cytokine_measurements.loc[
            {
                "Cytokine": cytokine
            }
        ].squeeze().to_pandas()
        timepoints = list(cytokine_df.columns)
        for index, tp in enumerate(cytokine_df.columns):
            expression_ax = axs[ax_index]
            comparison_ax = axs[ax_index + 2]
            expression_ax.boxplot(
                cytokine_df.loc[:, tp].dropna(),
                positions=[index * 3],
                notch=True,
                vert=True,
                widths=0.9,
                flierprops={
                    "markersize": 6,
                    "markerfacecolor": "tab:blue"
                }
            )
            low_patch = comparison_ax.boxplot(
                cytokine_df.loc[low_comp, tp].dropna(),
                patch_artist=True,
                positions=[index * 3],
                notch=True,
                vert=True,
                widths=0.9,
                flierprops={
                    "markersize": 6,
                    "markerfacecolor": "tab:blue"
                }
            )
            high_patch = comparison_ax.boxplot(
                cytokine_df.loc[high_comp, tp].dropna(),
                patch_artist=True,
                positions=[index * 3 + 1],
                notch=True,
                vert=True,
                widths=0.9,
                flierprops={
                    "markersize": 6,
                    "markerfacecolor": "tab:orange"
                }
            )
            low_patch["boxes"][0].set_facecolor("tab:blue")
            high_patch["boxes"][0].set_facecolor("tab:orange")
            result = ttest_ind(
                cytokine_df.loc[high_comp, tp].dropna(),
                cytokine_df.loc[low_comp, tp].dropna()
            )
            if result.pvalue < 0.05:
                timepoints[index] = timepoints[index] + "*"

        for boxplot_ax in [expression_ax, comparison_ax]:
            boxplot_ax.set_xticks(np.arange(0.5, 6 * 3, 3))
            boxplot_ax.set_xticklabels(timepoints)
            boxplot_ax.set_xlim([-1, 3 * 6 - 1])
            boxplot_ax.set_ylabel("Cytokine Expression")
            boxplot_ax.set_title(cytokine)

    ############################################################################
    # GRO / TBIL Associations
    ############################################################################

    ax = axs[4]

    tbil = lft_measurements.loc[{"LFT Score": "tbil"}].squeeze().to_pandas()
    timepoints = list(tbil.columns)
    for index, tp in enumerate(tbil.columns):
        low_patch = ax.boxplot(
            tbil.loc[low_comp, tp].dropna(),
            patch_artist=True,
            positions=[index * 3],
            notch=True,
            vert=True,
            widths=0.9,
            flierprops={
                "markersize": 6,
                "markerfacecolor": "tab:blue"
            }
        )
        high_patch = ax.boxplot(
            tbil.loc[high_comp, tp].dropna(),
            patch_artist=True,
            positions=[index * 3 + 1],
            notch=True,
            vert=True,
            widths=0.9,
            flierprops={
                "markersize": 6,
                "markerfacecolor": "tab:orange"
            }
        )
        low_patch["boxes"][0].set_facecolor("tab:blue")
        high_patch["boxes"][0].set_facecolor("tab:orange")
        result = ttest_ind(
            tbil.loc[high_comp, tp].dropna(),
            tbil.loc[low_comp, tp].dropna()
        )
        if result.pvalue < 0.05:
            timepoints[index] = timepoints[index] + "*"

    ax.set_xticks(np.arange(0.5, 8 * 3, 3))
    ax.set_xticklabels(timepoints)

    ax.set_xlim([-1, 3 * 8 - 1])
    ax.set_ylabel("Cytokine Expression")


    ############################################################################
    # FLT-3L / TBIL Correlations
    ############################################################################

    ax = axs[5]

    flt3l = cytokine_measurements.loc[{
        "Cytokine": "Flt-3L",
        "Cytokine Timepoint": "PO"
    }].squeeze().to_pandas().dropna()

    timepoints = list(tbil.columns)
    correlations = []
    for index, tp in enumerate(tbil.columns):
        tbil_col = tbil.loc[flt3l.index, tp].dropna()
        _flt3l = flt3l.loc[tbil_col.index]
        result = pearsonr(tbil_col, _flt3l)
        correlations.append(result.correlation)
        if result.pvalue < 0.01:
            timepoints[index] = timepoints[index] + "**"
        elif result.pvalue < 0.05:
            timepoints[index] = timepoints[index] + "*"

    ax.bar(
        np.arange(len(correlations)),
        correlations,
        width=1
    )

    ax.set_xticks(np.arange(len(correlations)))
    ax.set_xticklabels(timepoints)
    ax.set_ylabel("Pearson Correlation")

    return fig
