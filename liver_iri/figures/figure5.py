"""Plots Figure 5 -- Clinical Correlations"""
import warnings

import numpy as np
import pandas as pd
from matplotlib.patches import Patch
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

TO_CORRELATE = [
    "abox",
    "aih",
    "cit",
    "dage",
    "dalt",
    "dbmi",
    "drace",
    "dsx",
    "dtbili",
    "etoh",
    "hbv",
    "hcv",
    "liri",
    "pbc",
    "postchol",
    "postinf",
    "postnec",
    "poststeat",
    "psc",
    "rag",
    "rrc",
    "rsx",
    "txmeld",
    "wit",
]
COLORS = pd.Series(
    {
        "Compatibility": "c",
        "Autoimmune Hepatitis": "g",
        "Cold\nIschemia Time": "c",
        "Donor Age": "b",
        "Donor ALT": "b",
        "Donor BMI": "b",
        "Donor Race": "b",
        "Donor Sex": "b",
        "Donor TBIL": "b",
        "Donor Weight": "b",
        "Alcoholic Hepatitis": "g",
        "Hepatitis B": "g",
        "Hepatitis C": "g",
        "LIRI Score": "c",
        "Primary Biliary Cholangitis": "g",
        "Cholesterol": "c",
        "Inflammation": "c",
        "Necrotization": "c",
        "Steatosis": "c",
        "Primary Sclerosing Cholangitis": "g",
        "Recipient Age": "r",
        "Race": "r",
        "Recipient Sex": "r",
        "Transplant\nMELD Score": "c",
        "Warm\nIschemia Time": "c",
    }
)
CONVERSIONS = {
    "abox": "Compatibility",
    "aih": "Autoimmune Hepatitis",
    "cit": "Cold\nIschemia Time",
    "dage": "Donor Age",
    "dalt": "Donor ALT",
    "dbmi": "Donor BMI",
    "drace": "Donor Race",
    "dsx": "Donor Sex",
    "dtbili": "Donor TBIL",
    "etoh": "Alcoholic Hepatitis",
    "hbv": "Hepatitis B",
    "hcv": "Hepatitis C",
    "liri": "LIRI Score",
    "pbc": "Primary Biliary Cholangitis",
    "postchol": "Cholesterol",
    "postinf": "Inflammation",
    "postnec": "Necrotization",
    "poststeat": "Steatosis",
    "psc": "Primary Sclerosing Cholangitis",
    "rag": "Recipient Age",
    "rrc": "Race",
    "rsx": "Recipient Sex",
    "txmeld": "Transplant\nMELD Score",
    "wit": "Warm\nIschemia Time",
}
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


def get_clinical_accuracies(graft, meta):
    le = LabelEncoder()
    p_values = pd.Series(0, index=TO_CORRELATE)

    for name in TO_CORRELATE:
        var = meta.loc[:, name]
        var = var.dropna()
        var[:] = le.fit_transform(var)

        p_values.loc[name], _, _ = predict_categorical(
            var, graft.loc[var.index]
        )

    return p_values.sort_values(ascending=False)


def plot_accuracies(p_values, ax, x_label=None, y_label=None):
    ax.bar(range(len(p_values)), p_values, color=COLORS.loc[p_values.index])
    for index, p_value in enumerate(p_values):
        ax.text(
            index,
            p_value + 0.01,
            s=round(p_value, 2),
            ha="center",
            va="bottom",
            fontsize=6,
        )

    ax.legend(
        [
            Patch(facecolor="b"),
            Patch(facecolor="r"),
            Patch(facecolor="g"),
            Patch(facecolor="c"),
        ],
        [
            "Donor Characteristics",
            "Recipient Characteristics",
            "Etiologies",
            "Clinical Measurements",
        ],
        loc="upper left",
    )
    ax.set_xticks(range(len(p_values)))
    ax.set_xticklabels(
        p_values.index,
        ha="right",
        ma="right",
        va="top",
        rotation=45,
    )
    ax.set_xlim([-0.5, len(p_values) - 0.5])
    ax.set_ylim([0, 1.1])

    if x_label is not None:
        ax.set_xlabel(x_label)

    if y_label is not None:
        ax.set_ylabel(y_label)


def makeFigure():
    ############################################################################
    # Figure setup
    ############################################################################

    fig_size = (10, 3)
    layout = {"nrows": 1, "ncols": 5}
    axs, fig = getSetup(fig_size, layout)

    ############################################################################
    # Data imports
    ############################################################################

    meta = import_meta()
    graft = meta.loc[:, "graft_death"]
    cyto = cytokine_data()
    lfts = lft_data()

    data = build_coupled_tensors()
    tensors, labels = convert_to_numpy(data, graft)

    ############################################################################
    # Figure 5F: Clinical correlations
    ############################################################################

    p_values = get_clinical_accuracies(graft, meta)
    p_values.index = [CONVERSIONS.get(i, i) for i in p_values.index]
    plot_accuracies(p_values, axs[4], y_label="Correlation p-value")

    ############################################################################
    # Figures 5A/5B: Bar and ROC plots
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

    for curve, method in zip(curves, METHODS):
        ax.plot(curve[0], curve[1], label=method)

    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    ############################################################################
    # Figures 5C/D: Clinical Predictions
    ############################################################################

    patient_factors = pd.DataFrame(
        tpls.Xs_factors[0][0],
        index=labels.index,
        columns=np.arange(tpls.n_components) + 1,
    )
    accuracies, q2ys = get_clinical_accuracies(patient_factors, meta)

    accuracies = accuracies.sort_values(ascending=True)
    q2ys = q2ys.sort_values(ascending=True)

    ax = axs[2]
    ax.bar(np.arange(len(accuracies)), accuracies)

    ticks = [TRANSLATIONS[metric] for metric in accuracies.index]
    ax.set_xticks(range(len(accuracies)))
    ax.set_xticklabels(ticks, rotation=45, va="top", ha="right")

    ax.set_ylim([0, 1])
    ax.set_xlim([-0.5, len(accuracies) - 0.5])

    ax = axs[3]
    ax.bar(np.arange(len(q2ys)), q2ys)

    ticks = [TRANSLATIONS[metric] for metric in q2ys.index]
    ax.set_xticks(range(len(q2ys)))
    ax.set_xticklabels(ticks, rotation=45, va="top", ha="right")

    ax.set_ylim([0, 1])
    ax.set_xlim([-0.5, len(q2ys) - 0.5])

    return fig
