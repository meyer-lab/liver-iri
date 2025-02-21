"""Plots Figure 4 -- tPLS Model Accuracy"""

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.preprocessing import LabelEncoder

from ..dataimport import (
    build_coupled_tensors,
    cytokine_data,
    import_meta,
    lft_data,
)
from ..predict import (
    predict_categorical,
    predict_continuous,
    run_coupled_tpls_classification,
)
from ..tensor import convert_to_numpy
from .common import getSetup

warnings.filterwarnings("ignore")

CATEGORICAL = ["dsx", "rsx", "rrc", "abox", "iri"]
CYTO_TIMEPOINTS = ["PO", "D1", "W1", "M1", "LF", "PV"]
LFT_TIMEPOINTS = ["Opening", "1", "2", "3", "4", "5", "6", "7"]
METHODS = ["LFTs", "PV Cytokines", "Peripheral Cytokines", "tPLS"]
REGRESSION = ["dage", "rag", "dtbili", "dalt", "cit", "wit", "txmeld"]
TRANSLATIONS = {
    "dsx": "Donor Sex",
    "dage": "Donor Age",
    "dtbili": "Donor TBIL",
    "dalt": "Donor ALT",
    "abox": "ABO\nCompatibility",
    "cit": "Cold\nIschemia Time",
    "wit": "Warm\nIschemia Time",
    "txmeld": "Transplant\nMELD Score",
    "rrc": "Race",
    "iri": "LIRI > 1",
    "graft_death": "Graft Death",
    "rsx": "Recipient Sex",
    "rag": "Recipient Age",
    "liri": "LIRI",
}


def get_clinical_accuracies(factors, meta):
    encoder = LabelEncoder()
    accuracies = pd.Series()
    q2ys = pd.Series()

    for target in CATEGORICAL:
        labels = meta.loc[:, target]
        labels = labels.dropna()
        labels[:] = encoder.fit_transform(labels)
        labels = labels.astype(int)

        data = factors.loc[labels.index, :]
        score, _, _ = predict_categorical(data, labels)
        accuracies.loc[target] = score

    for target in REGRESSION:
        labels = meta.loc[:, target]
        labels = labels.dropna()

        data = factors.loc[labels.index, :]
        score, _ = predict_continuous(data, labels)
        q2ys.loc[target] = score

    return accuracies, q2ys


def makeFigure():
    ############################################################################
    # Figure setup
    ############################################################################

    fig_size = (9, 6)
    layout = {"nrows": 2, "ncols": 3}
    axs, fig = getSetup(fig_size, layout)

    ############################################################################
    # Data imports
    ############################################################################

    meta = import_meta()
    graft = meta.loc[:, "graft_death"]
    cyto = cytokine_data()
    lfts = lft_data()

    ############################################################################
    # Figure 4A: Rank v. Accuracy
    ############################################################################

    data = build_coupled_tensors()
    tensors, labels = convert_to_numpy(data, graft)
    ranks = np.arange(1, 6)
    accuracies = pd.Series(0, index=ranks)

    for rank in ranks:
        (_, _), _acc, _ = run_coupled_tpls_classification(
            tensors, labels, return_proba=True, rank=rank
        )
        accuracies.loc[rank] = _acc

    ax = axs[0]
    ax.plot(accuracies.index, accuracies)
    ax.set_ylim([0.5, 0.75])
    ax.set_xlabel("tPLS Components")
    ax.set_ylabel("Prediction Accuracy")

    ############################################################################
    # Figure 4B: Scaling heatmap
    ############################################################################

    # scalings = [1 / 8, 1 / 6, 1 / 4, 1 / 2, 1, 2, 4, 6, 8]
    # accuracies = pd.DataFrame(index=scalings, columns=scalings, dtype=float)
    # for pv_scaling in scalings:
    #     for lft_scaling in scalings:
    #         _data = build_coupled_tensors(
    #             peripheral_scaling=1,
    #             pv_scaling=pv_scaling,
    #             lft_scaling=lft_scaling,
    #         )
    #         _tensors, _labels = convert_to_numpy(_data, graft)
    #         (_, _), _acc, _ = run_coupled_tpls_classification(_tensors, _labels)
    #         accuracies.loc[pv_scaling, lft_scaling] = _acc
    #
    # ax = axs[1]
    # sns.heatmap(
    #     accuracies, cmap="rocket", vmin=0.5, annot=True, fmt=".2f", ax=ax
    # )
    # ax.set_xticklabels([round(i, 3) for i in accuracies.index])
    # ax.set_yticklabels([round(i, 3) for i in accuracies.columns])
    # ax.set_xlabel("LFT Scaling")
    # ax.set_ylabel("PV Scaling")

    ############################################################################
    # Figures 4C/4D: Bar and ROC plots
    ############################################################################

    (tpls, _), tpls_acc, tpls_proba = run_coupled_tpls_classification(
        tensors, labels, return_proba=True
    )

    pv_matrices = []
    peripheral_matrices = []
    for tp in CYTO_TIMEPOINTS:
        if tp in ["PV", "LF"]:
            pv_matrices.append(
                cyto.sel({"Cytokine Timepoint": tp})
                .to_array()
                .squeeze()
                .to_pandas()
                .loc[meta.index, :]
            )
        else:
            peripheral_matrices.append(
                cyto.sel({"Cytokine Timepoint": tp})
                .to_array()
                .squeeze()
                .to_pandas()
                .loc[meta.index, :]
            )
    pv_matrix = pd.concat(pv_matrices, axis=1).dropna(axis=0)
    peripheral_matrix = pd.concat(peripheral_matrices, axis=1).dropna(axis=0)

    matrices = []
    for tp in LFT_TIMEPOINTS:
        matrices.append(
            lfts.sel({"LFT Timepoint": tp})
            .to_array()
            .squeeze()
            .to_pandas()
            .loc[meta.index, :]
        )
    lft_matrix = pd.concat(matrices, axis=1).dropna(axis=0)

    pv_acc, pv_model, pv_proba = predict_categorical(
        pv_matrix, graft.loc[pv_matrix.index], return_proba=True
    )
    peripheral_acc, peripheral_model, peripheral_proba = predict_categorical(
        peripheral_matrix, graft.loc[peripheral_matrix.index], return_proba=True
    )
    lft_acc, lft_model, lft_proba = predict_categorical(
        lft_matrix, graft.loc[lft_matrix.index], return_proba=True
    )

    pv_fpr, pv_tpr, _ = roc_curve(graft.loc[pv_matrix.index], pv_proba)
    peripheral_fpr, peripheral_tpr, _ = roc_curve(
        graft.loc[peripheral_matrix.index], peripheral_proba
    )
    lft_fpr, lft_tpr, _ = roc_curve(graft.loc[lft_matrix.index], lft_proba)
    tpls_fpr, tpls_tpr, _ = roc_curve(labels, tpls_proba)

    ax = axs[2]
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

    ax = axs[3]
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
    # Figures 4E/4F: Clinical Predictions
    ############################################################################

    patient_factors = pd.DataFrame(
        tpls.Xs_factors[0][0],
        index=labels.index,
        columns=np.arange(tpls.n_components) + 1,
    )
    accuracies, q2ys = get_clinical_accuracies(patient_factors, meta)

    accuracies = accuracies.sort_values(ascending=True)
    q2ys = q2ys.sort_values(ascending=True)

    ax = axs[4]
    ax.bar(np.arange(len(accuracies)), accuracies)

    ticks = [TRANSLATIONS[metric] for metric in accuracies.index]
    ax.set_xticks(range(len(accuracies)))
    ax.set_xticklabels(ticks, rotation=45, va="top", ha="right")

    ax.set_ylim([0, 1])
    ax.set_xlim([-0.5, len(accuracies) - 0.5])

    ax = axs[5]
    ax.bar(np.arange(len(q2ys)), q2ys)

    ticks = [TRANSLATIONS[metric] for metric in q2ys.index]
    ax.set_xticks(range(len(q2ys)))
    ax.set_xticklabels(ticks, rotation=45, va="top", ha="right")

    ax.set_ylim([0, 1])
    ax.set_xlim([-0.5, len(q2ys) - 0.5])

    return fig
