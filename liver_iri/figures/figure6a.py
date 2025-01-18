"""Plots Figure 6a -- tPLS Boxplots"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import xarray as xr

from ..dataimport import build_coupled_tensors, import_meta
from ..predict import oversample, run_coupled_tpls_classification
from ..tensor import convert_to_numpy
from .common import getSetup

warnings.filterwarnings("ignore")

CYTOKINES = {
    1: ["IL-1RA", "IL-2", "IL-17A", "IFNg"],
    2: ["IL-4", "EGF"]
}


def makeFigure():
    ############################################################################
    # Data imports
    ############################################################################

    meta = import_meta()
    labels = meta.loc[:, "graft_death"]
    labels = labels.dropna()

    val_meta = import_meta(no_missing=False)
    val_labels = val_meta.loc[:, "graft_death"]
    val_labels = val_labels.dropna()

    data = build_coupled_tensors()
    val_data = build_coupled_tensors(no_missing=False)

    all_data = xr.merge([data, val_data])
    all_labels = pd.Series(pd.concat([labels, val_labels]))
    all_tensors, all_labels = convert_to_numpy(all_data, all_labels)

    raw_data = build_coupled_tensors(
        lft_scaling=1,
        pv_scaling=1,
        transform="log",
        normalize=False
    )
    raw_val = build_coupled_tensors(
        no_missing=False,
        lft_scaling=1,
        pv_scaling=1,
        transform="log",
        normalize=False
    )
    raw_data = xr.merge([raw_data, raw_val])

    cytokine_measurements = raw_data["Cytokine Measurements"]

    ############################################################################
    # Factorization
    ############################################################################

    tensors, labels = convert_to_numpy(data, labels)
    oversampled_tensors, oversampled_labels = oversample(tensors, labels)

    (tpls, lr_model), tpls_acc, tpls_proba = run_coupled_tpls_classification(
        tensors, labels, return_proba=True
    )
    tpls.fit(oversampled_tensors, oversampled_labels.values)

    ############################################################################
    # tPLS patients
    ############################################################################

    factor = tpls.transform(all_tensors)
    patient_factors = pd.DataFrame(
        factor,
        index=all_labels.index,
        columns=np.arange(1, factor.shape[1] + 1),
    )
    patient_factors = patient_factors.loc[all_labels.index, :]
    patient_factors /= abs(patient_factors).max(axis=0)

    n_axs = len(CYTOKINES[1]) + len(CYTOKINES[2])
    axs, fig = getSetup(
        (n_axs * 3, 3),
        {"ncols": n_axs, "nrows": 1}
    )
    ax_index = 0

    ############################################################################
    # Component boxplots
    ############################################################################

    for component, cytokines in CYTOKINES.items():
        patient_factors = patient_factors.sort_values(
            component, 
            ascending=False
        )
        high_comp, low_comp = (patient_factors.index[:30],
                               patient_factors.index[-30:])
        for cytokine in cytokines:
            ax = axs[ax_index]
            cytokine_df = cytokine_measurements.loc[
                {
                    "Cytokine": cytokine,
                }
            ].to_pandas()
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
                        "markerfacecolor": "tab:blue"
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
                        "markerfacecolor": "tab:orange"
                    }
                )
                low_patch["boxes"][0].set_facecolor("tab:blue")
                high_patch["boxes"][0].set_facecolor("tab:orange")
    
                result = ttest_ind(
                    cytokine_df.loc[high_comp, tp].dropna(),
                    cytokine_df.loc[low_comp, tp].dropna()
                )
                if result.pvalue < 0.01:
                    timepoints[index] = timepoints[index] + "**"
                elif result.pvalue < 0.05:
                    timepoints[index] = timepoints[index] + "*"
    
            ax.set_xticks(np.arange(0.5, 6 * 3, 3))
            ax.set_xticklabels(timepoints)
            ax.set_xlim([-1, 3 * 6 - 1])
            ax.set_ylabel("Cytokine Expression")
            ax.set_title(f"tPLS {component}: {cytokine}")

            ax_index += 1

    return fig
