"""Plots Figure 6 -- Model Comparisons"""

import warnings

import numpy as np
import pandas as pd
import xarray as xr
from lifelines import KaplanMeierFitter
from sklearn.metrics import roc_curve
from sklearn.preprocessing import scale

from ..dataimport import build_coupled_tensors, import_meta
from ..predict import (oversample, predict_categorical,
                       run_coupled_tpls_classification, run_survival,
                       run_tpls_survival)
from ..tensor import convert_to_numpy
from .common import getSetup

warnings.filterwarnings("ignore")

METHODS = ["LFTs", "PV Cytokines", "Peripheral Cytokines", "tPLS"]


def makeFigure():
    ############################################################################
    # Figure setup
    ############################################################################

    fig_size = (6.5, 4)
    layout = {"nrows": 2, "ncols": 4}
    axs, fig = getSetup(fig_size, layout)

    ############################################################################
    # Data imports
    ############################################################################

    meta = import_meta()
    val_meta = import_meta(no_missing=False, long_survival=False)
    labels = meta.loc[:, ["graft_death", "survival_time"]]
    val_labels = val_meta.loc[:, ["graft_death", "survival_time"]]

    data = build_coupled_tensors()
    val_data = build_coupled_tensors(no_missing=False)
    tensors, labels = convert_to_numpy(data, labels)
    val_tensors, val_labels = convert_to_numpy(val_data, val_labels)

    cytokine_data = (
        data["Cytokine Measurements"]
        .stack(Flattened=["Cytokine", "Cytokine Timepoint"])
        .to_pandas()
    )
    lft_data = (
        data["LFT Measurements"]
        .stack(Flattened=["LFT Score", "LFT Timepoint"])
        .to_pandas()
    )

    cytokine_data.columns = np.arange(cytokine_data.shape[1])
    lft_data.columns = np.arange(lft_data.shape[1])

    pv_timepoints = []
    for i in np.arange(0, cytokine_data.shape[1], 6):
        pv_timepoints.extend([i + 1, i + 2])
    pv_cytokines = cytokine_data.iloc[:, pv_timepoints]
    peripheral_cytokines = cytokine_data.drop(pv_cytokines.columns, axis=1)

    (tpls, _), tpls_acc, tpls_proba = run_coupled_tpls_classification(
        tensors, labels.loc[:, "graft_death"], return_proba=True
    )

    lft_data = lft_data.dropna(axis=0)
    pv_cytokines = pv_cytokines.dropna(axis=0)
    peripheral_cytokines = peripheral_cytokines.dropna(axis=0)

    ############################################################################
    # Figure 6A-C: Raw Data Comparisons
    ############################################################################

    pv_acc, pv_model, pv_proba = predict_categorical(
        pv_cytokines,
        labels.loc[pv_cytokines.index, "graft_death"],
        return_proba=True,
    )
    peripheral_acc, peripheral_model, peripheral_proba = predict_categorical(
        peripheral_cytokines,
        labels.loc[peripheral_cytokines.index, "graft_death"],
        return_proba=True,
    )
    lft_acc, lft_model, lft_proba = predict_categorical(
        lft_data, labels.loc[lft_data.index, "graft_death"], return_proba=True
    )
    pv_fpr, pv_tpr, _ = roc_curve(
        labels.loc[pv_cytokines.index, "graft_death"], pv_proba
    )
    peripheral_fpr, peripheral_tpr, _ = roc_curve(
        labels.loc[peripheral_cytokines.index, "graft_death"], peripheral_proba
    )
    lft_fpr, lft_tpr, _ = roc_curve(
        labels.loc[lft_data.index, "graft_death"], lft_proba
    )
    tpls_fpr, tpls_tpr, _ = roc_curve(labels.loc[:, "graft_death"], tpls_proba)

    ax = axs[0]
    ax.bar(
        np.arange(4),
        [lft_acc, pv_acc, peripheral_acc, tpls_acc],
        width=1,
        color=["tab:blue", "tab:orange", "tab:green", "tab:red"],
        label=METHODS,
    )
    ax.set_xticklabels(METHODS, ha="center", ma="right", va="top", rotation=45)
    ax.legend()
    ax.set_ylim([0, 1])

    ax = axs[1]
    curves = [
        (lft_fpr, lft_tpr),
        (pv_fpr, pv_tpr),
        (peripheral_fpr, peripheral_tpr),
        (tpls_fpr, tpls_tpr),
    ]
    ax.plot([0, 1], [0, 1], color="k", linestyle="--")

    for curve, method in zip(curves, METHODS, strict=False):
        ax.plot(curve[0], curve[1], label=method)

    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    ############################################################################
    # Figure 6D-E: Cox-PH
    ############################################################################

    (tpls, cox_ph), c_index, cph_expected = run_tpls_survival(tensors, labels)
    oversampled_tensors, oversampled_labels = oversample(
        tensors, labels, column="graft_death"
    )
    tpls.fit(
        oversampled_tensors, oversampled_labels.loc[:, "graft_death"].values
    )

    _, pv_c_index, _ = run_survival(pv_cytokines, labels)
    _, peripheral_c_index, _ = run_survival(peripheral_cytokines, labels)
    _, lft_c_index, _ = run_survival(lft_data, labels)
    _, liri_c_index, _ = run_survival(meta.loc[:, "liri"].to_frame(), labels)

    ax = axs[3]

    ax.bar(
        np.arange(5),
        [lft_c_index, pv_c_index, peripheral_c_index, liri_c_index, c_index],
        width=1,
        color=["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red"],
        label=[
            "LFTs",
            "PV Cytokines",
            "Peripheral Cytokines",
            "Pathology Score",
            "tPLS",
        ],
    )
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(
        [
            "LFTs",
            "PV Cytokines",
            "Peripheral\nCytokines",
            "Pathology\nScore",
            "tPLS",
        ],
        ha="right",
        ma="right",
        va="top",
        rotation=45,
    )

    ax.legend()
    ax.set_ylim([0, 1])

    ax.set_xlabel("Method")
    ax.set_ylabel("C-Index")

    ax = axs[4]

    ax.plot([0, 0], [-1, 10], linestyle="--", color="k")
    ax.errorbar(
        cox_ph.params_,
        np.arange(len(cox_ph.params_)),
        linestyle="",
        marker="o",
        capsize=5,
        xerr=cox_ph.standard_errors_ * 1.96,
    )

    ax.set_yticks([0, 1])
    ax.set_ylim([-0.5, 1.5])

    ax.set_xlabel("Hazard Ratio")
    ax.set_ylabel("tPLS Component")

    ############################################################################
    # Figures 6F-H: Kaplan-Meier Curves
    ############################################################################

    merged_data = xr.merge([data, val_data])
    merged_labels = pd.concat([labels, val_labels])

    merged_tensors, merged_labels = convert_to_numpy(merged_data, merged_labels)
    components = pd.DataFrame(
        tpls.transform(merged_tensors),
        index=merged_labels.index,
        columns=np.arange(tpls.n_components) + 1,
    )
    components.loc[:, "Sum"] = scale(components).sum(axis=1)
    threshold = int(components.shape[0] / 10)
    kmf = KaplanMeierFitter()

    for ax, column in zip(axs[5:], components.columns, strict=False):
        components = components.sort_values(by=column, ascending=False)
        high_index = components.index[:threshold]
        low_index = components.index[threshold:]
        kmf.fit(
            merged_labels.loc[high_index, "survival_time"],
            merged_labels.loc[high_index, "graft_death"],
        )
        ax.plot(
            kmf.survival_function_.index,
            kmf.survival_function_.iloc[:, 0],
            label=f"High {column}",
            color="tab:blue",
        )
        max_index = kmf.survival_function_.index[-1]
        kmf.fit(
            merged_labels.loc[low_index, "survival_time"],
            merged_labels.loc[low_index, "graft_death"],
        )
        ax.plot(
            kmf.survival_function_.index,
            kmf.survival_function_.iloc[:, 0],
            label=f"Low {column}",
            color="tab:orange",
        )
        if kmf.survival_function_.index[-1] > max_index:
            max_index = kmf.survival_function_.index[-1]

        ax.set_ylim([0, 1])
        ax.set_xlim([0, max_index])

        ax.set_ylabel("Probability of\nNon-Rejection")
        ax.set_xlabel("Time")
        ax.legend()

    return fig
