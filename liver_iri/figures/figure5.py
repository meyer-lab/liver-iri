"""Plots Figure 5 -- tPLS Model Performance"""
import warnings

from lifelines import KaplanMeierFitter
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import LabelEncoder, scale
import xarray as xr

from ..dataimport import build_coupled_tensors, import_meta
from ..predict import (oversample, predict_categorical, predict_clinical,
                       predict_continuous, run_coupled_tpls_classification,
                       run_survival, run_tpls_survival)
from ..tensor import convert_to_numpy
from .common import getSetup

warnings.filterwarnings("ignore")

TO_PREDICT = [
    "abox",
    "cit",
    "dage",
    "dalt",
    "dbmi",
    "drace",
    "dsx",
    "dtbili",
    "liri",
    "postchol",
    "postinf",
    "postnec",
    "poststeat",
    "rag",
    "rrc",
    "rsx",
    "txmeld",
    "wit",
    "etiology"
]
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
    "etiology": "Etiology"
}
METHODS = ["LFTs", "PV Cytokines", "Peripheral Cytokines", "tPLS"]


def clinical_predict(meta: pd.DataFrame, labels: pd.Series):
    """
    Evaluates predictive ability of clinical variables.

    Args:
        meta (pd.DataFrame): patient meta-data
        labels (pd.Series): patient outcomes

    Returns:
        accuracies (pd.Series): clinical variable accuracies
    """
    encoder = LabelEncoder()
    accuracies = pd.Series(index=TO_PREDICT)
    c_indices = accuracies.copy(deep=True)
    labels = labels.loc[meta.index, :]

    for variable_name in TO_PREDICT:
        if variable_name == "etiology":
            data = meta.loc[
                :,
                ["aih", "etoh", "hbv", "hcv", "pbc", "psc"]
            ]
            accuracies.loc[variable_name], _, _ = predict_categorical(
                data,
                labels.loc[:, "graft_death"]
            )
        else:
            variable = meta.loc[:, variable_name]
            variable = variable.dropna()
            _labels = labels.loc[variable.index, :]

            if variable.dtype != float:
                variable[:] = encoder.fit_transform(variable)

            accuracies.loc[variable_name] = predict_clinical(
                variable,
                _labels.loc[:, "graft_death"]
            )
            _, c_indices.loc[variable_name], _ = run_survival(
                variable,
                _labels
            )

    return accuracies.sort_values(), c_indices.sort_values()


def makeFigure():
    ############################################################################
    # Figure setup
    ############################################################################

    axs, fig = getSetup((12, 9), {"ncols": 4, "nrows": 3})

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
    all_labels = pd.concat([labels, val_labels])

    cytokine_measurements = data["Cytokine Measurements"]
    lft_measurements = data["LFT Measurements"]

    tensors, labels = convert_to_numpy(data, labels)
    oversampled_tensors, oversampled_labels = oversample(tensors, labels)
    all_tensors, all_labels = convert_to_numpy(all_data, all_labels)
    survival_labels = meta.loc[:, ["graft_death", "survival_time"]]

    clinical_accuracies, c_indices = clinical_predict(meta, survival_labels)
    (tpls, lr_model), tpls_acc, tpls_proba = run_coupled_tpls_classification(
        tensors, labels, return_proba=True
    )
    tpls.fit(oversampled_tensors, oversampled_labels.values)

    ############################################################################
    # Patient factors
    ############################################################################

    ax = axs[0]
    factor = tpls.transform(all_tensors)
    patient_factors = pd.DataFrame(
        factor,
        index=all_labels.index,
        columns=np.arange(1, factor.shape[1] + 1),
    )
    patient_factors = patient_factors.loc[all_labels.index, :]

    xx, yy = np.meshgrid(
        np.linspace(
            patient_factors.loc[:, 1].min() * 1.5,
            patient_factors.loc[:, 1].max() * 1.5,
            100,
        ),
        np.linspace(
            patient_factors.loc[:, 2].min() * 1.5,
            patient_factors.loc[:, 2].max() * 1.5,
            100,
        ),
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = lr_model.predict_proba(grid)[:, 0].reshape(xx.shape)

    xx /= abs(patient_factors.loc[:, 1]).max()
    yy /= abs(patient_factors.loc[:, 2]).max()
    patient_factors /= patient_factors.max(axis=0)

    res = pearsonr(patient_factors.iloc[:, 0], patient_factors.iloc[:, 1])
    score, model = predict_continuous(
        patient_factors.iloc[:, 0],
        patient_factors.iloc[:, 1]
    )

    cs = ax.contourf(xx, yy, probs, 11, cmap="RdBu", alpha=0.75)
    fig.colorbar(cs)
    ax.plot(
        [
            -1.1,
            1.1,
        ],
        [0, 0],
        linestyle="--",
        color="k",
    )
    ax.plot(
        [0, 0],
        [
            -1.1,
            1.1,
        ],
        linestyle="--",
        color="k",
    )
    ax.scatter(
        patient_factors.loc[all_labels == 0, 1],
        patient_factors.loc[all_labels == 0, 2],
        c="blue",
        edgecolor="black",
        alpha=0.75,
        label="No Transplant Rejection",
    )
    ax.scatter(
        patient_factors.loc[all_labels == 1, 1],
        patient_factors.loc[all_labels == 1, 2],
        c="red",
        edgecolor="black",
        alpha=0.75,
        label="Transplant Rejection",
    )
    ax.text(
        1,
        -1,
        s=f"Pearson: {round(res.statistic, 2)}\np-value: {res.pvalue}",
        ha="right",
        ma="right",
        va="bottom"
    )

    xs = [-1.1, 1.1]
    ys = [
        model.params.iloc[0] + model.params.iloc[1] * xs[0],
        model.params.iloc[0] + model.params.iloc[1] * xs[1]
    ]

    ax.plot(xs, ys, color="k", linestyle="--", zorder=3)

    ax.legend()
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title("Patient Factors")

    ############################################################################
    # Cytokine factors
    ############################################################################

    ax = axs[1]

    cyto_factors = pd.DataFrame(
        tpls.Xs_factors[0][2], index=data.Cytokine.values, columns=[1, 2]
    )
    cyto_factors /= abs(cyto_factors).max(axis=0)
    ax.set_title("Cytokine Factors")

    ax.plot([-2, 2], [0, 0], linestyle="--", color="k")
    ax.plot([0, 0], [-2, 2], linestyle="--", color="k")
    colors = []
    for cyto in cyto_factors.index:
        if cyto_factors.loc[cyto, 1] > 0.75 > cyto_factors.loc[cyto, 2]:
            colors.append("#1f77b4")
        elif cyto_factors.loc[cyto, 2] > 0.75:
            colors.append("#ff7f0e")
        else:
            colors.append("grey")

    ax.scatter(
        cyto_factors.loc[:, 1], cyto_factors.loc[:, 2], c=colors, edgecolor="k"
    )

    xticks = [round(i, 1) for i in np.arange(-1, 1.1, 0.5)]
    yticks = [round(i, 1) for i in np.arange(-1, 1.1, 0.5)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)

    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])

    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    for cyto in cyto_factors.index:
        ax.text(
            cyto_factors.loc[cyto, 1],
            cyto_factors.loc[cyto, 2],
            s=cyto,
            ha="center",
            va="center",
        )

    ax = axs[2]

    time_factors = pd.DataFrame(
        tpls.Xs_factors[0][1],
        index=data["Cytokine Timepoint"].values,
        columns=[1, 2],
    )
    time_factors /= abs(time_factors).max(axis=0)
    time_factors.loc[["PV", "LF"], :] /= abs(time_factors.loc[["PV", "LF"], :]).max()

    ax.plot(
        [-0.1, time_factors.shape[0] - 0.9], [0, 0], linestyle="--", color="k"
    )
    ax.plot(
        range(time_factors.shape[0]),
        time_factors.loc[:, 1],
        label="Component 1",
    )
    ax.plot(
        range(time_factors.shape[0]),
        time_factors.loc[:, 2],
        label="Component 2",
    )

    yticks = [round(i, 1) for i in np.arange(-1, 1.1, 0.5)]
    ax.set_xticks(range(time_factors.shape[0]))
    ax.set_xticklabels(data["Cytokine Timepoint"].values)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)

    ax.set_xlim([-0.1, time_factors.shape[0] - 0.9])
    ax.set_ylim([-1.1, 1.1])

    ax.legend()
    ax.set_ylabel("Component Assocation")
    ax.set_title("Cytokine Time Factors")

    ############################################################################
    # LFT factors
    ############################################################################

    ax = axs[3]

    lft_factors = pd.DataFrame(
        tpls.Xs_factors[1][2], index=data["LFT Score"].values, columns=[1, 2]
    )
    lft_factors /= abs(lft_factors).max(axis=0)

    ax.plot([-2, 2], [0, 0], linestyle="--", color="k")
    ax.plot([0, 0], [-2, 2], linestyle="--", color="k")

    ax.scatter(
        lft_factors.loc[:, 1],
        lft_factors.loc[:, 2],
        c=["tab:blue", "tab:orange", "tab:green"],
        edgecolor="k"
    )

    xticks = [round(i, 1) for i in np.arange(-1, 1.1, 0.5)]
    yticks = [round(i, 1) for i in np.arange(-1, 1.1, 0.5)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)

    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])

    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    ax.set_title("LFT Factors")

    ax = axs[4]

    time_factors = pd.DataFrame(
        tpls.Xs_factors[1][1],
        index=data["LFT Timepoint"].values,
        columns=[1, 2],
    )
    time_factors /= abs(time_factors).max(axis=0)
    index = list(time_factors.index)
    index[1:] = ["Day " + i for i in index[1:]]
    time_factors.index = index

    ax.plot([-0.1, 7.1], [0, 0], color="k", linestyle="--")
    for col in time_factors.columns:
        ax.plot(
            range(time_factors.shape[0]),
            time_factors.loc[:, col],
            label=f"Component {col}",
        )

    ax.legend()
    ax.set_xticks(range(time_factors.shape[0]))
    ax.set_xticklabels(
        time_factors.index, rotation=45, ha="right", va="top", ma="right"
    )
    yticks = [round(i, 2) for i in np.arange(0, 1.05, 0.2)]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)
    ax.set_ylabel("Component Association")

    ax.set_ylim([-1.1, 1.1])
    ax.set_xlim([-0.2, 7.2])
    ax.set_title("LFT Factor Timepoints")

    ############################################################################
    # Bar and ROC plots
    ############################################################################

    pv_matrices = []
    peripheral_matrices = []
    for tp in data["Cytokine Timepoint"].values:
        if tp in ["PV", "LF"]:
            pv_matrices.append(
                cytokine_measurements.sel({"Cytokine Timepoint": tp})
                .squeeze()
                .to_pandas()
                .loc[meta.index, :]
            )
        else:
            peripheral_matrices.append(
                cytokine_measurements.sel({"Cytokine Timepoint": tp})
                .squeeze()
                .to_pandas()
                .loc[meta.index, :]
            )

    pv_matrix = pd.concat(pv_matrices, axis=1).dropna(axis=0)
    peripheral_matrix = pd.concat(peripheral_matrices, axis=1).dropna(axis=0)

    matrices = []
    for tp in lft_measurements["LFT Timepoint"].values:
        matrices.append(
            lft_measurements.sel({"LFT Timepoint": tp})
            .squeeze()
            .to_pandas()
            .loc[meta.index, :]
        )
    lft_matrix = pd.concat(matrices, axis=1).dropna(axis=0)

    pv_acc, pv_model, pv_proba = predict_categorical(
        pv_matrix, all_labels.loc[pv_matrix.index], return_proba=True
    )
    peripheral_acc, peripheral_model, peripheral_proba = predict_categorical(
        peripheral_matrix, all_labels.loc[peripheral_matrix.index], return_proba=True
    )
    lft_acc, lft_model, lft_proba = predict_categorical(
        lft_matrix, all_labels.loc[lft_matrix.index], return_proba=True
    )
    pv_fpr, pv_tpr, _ = roc_curve(all_labels.loc[pv_matrix.index], pv_proba)
    peripheral_fpr, peripheral_tpr, _ = roc_curve(
        all_labels.loc[peripheral_matrix.index], peripheral_proba
    )
    lft_fpr, lft_tpr, _ = roc_curve(all_labels.loc[lft_matrix.index], lft_proba)
    tpls_fpr, tpls_tpr, _ = roc_curve(labels, tpls_proba)

    gs = axs[6].get_gridspec()
    axs[6].remove()
    axs[7].remove()

    axs = axs[:-1]
    axs[6] = fig.add_subplot(gs[1, -2:])

    ax = axs[6]

    molecular_accuracies = pd.Series(
        [tpls_acc, peripheral_acc, pv_acc, lft_acc],
        index=["tPLS", "Peripheral Cytokines", "PV Cytokines", "LFTs"]
    )
    accuracies = pd.concat([molecular_accuracies, clinical_accuracies])
    accuracies = accuracies.sort_values(ascending=False)

    colors = pd.Series("tab:blue", index=accuracies.index)
    colors.loc[molecular_accuracies.index] = "tab:orange"

    ax.bar(
        np.arange(len(accuracies)),
        accuracies,
        width=1,
        color=colors
    )
    ax.set_xticks(np.arange(len(accuracies)))
    ax.set_xticklabels(
        accuracies.index.to_series().replace(CONVERSIONS),
        ha="right",
        ma="right",
        va="top",
        rotation=45
    )
    ax.set_xlim([-1, len(accuracies)])
    ax.set_ylim([0, 1])

    ax = axs[5]
    curves = [
        (lft_fpr, lft_tpr),
        (pv_fpr, pv_tpr),
        (peripheral_fpr, peripheral_tpr),
        (tpls_fpr, tpls_tpr),
    ]
    ax.plot([0, 1], [0, 1], color="k", linestyle="--")

    for curve, method in zip(curves, METHODS):
        ax.plot(
            curve[0],
            curve[1],
            label=f"{method} (AUC: {round(auc(curve[0], curve[1]), 2)})"
        )

    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    ############################################################################
    # Cox-PH
    ############################################################################

    survival_labels = meta.loc[:, ["graft_death", "survival_time"]]
    (tpls, cox_ph), c_indices.loc["tPLS"], cph_expected = run_tpls_survival(
        tensors, survival_labels
    )
    oversampled_tensors, oversampled_labels = oversample(
        tensors, survival_labels, column="graft_death"
    )
    tpls.fit(
        oversampled_tensors,
        oversampled_labels.loc[:, "graft_death"].values
    )
    survival_factors = pd.DataFrame(
        tpls.transform(tensors),
        index=survival_labels.index,
        columns=np.arange(1, tpls.n_components + 1),
    )

    _, c_indices.loc["PV Cytokines"], _ = run_survival(
        pv_matrix,
        survival_labels
    )
    _, c_indices.loc["Peripheral Cytokines"], _ = run_survival(
        peripheral_matrix,
        survival_labels
    )
    _, c_indices.loc["LFTs"], _ = run_survival(
        lft_matrix,
        survival_labels
    )

    gs = axs[9].get_gridspec()
    axs[9].remove()
    axs[10].remove()

    axs = axs[:-1]
    axs[9] = fig.add_subplot(gs[2, -2:])

    ax = axs[9]

    c_indices = c_indices.dropna().sort_values(ascending=False)
    ax.bar(
        np.arange(len(c_indices)),
        c_indices,
        width=1
    )
    ax.set_xticks(np.arange(len(c_indices)))
    ax.set_xticklabels(
        c_indices.index.to_series().replace(CONVERSIONS),
        ha="right",
        ma="right",
        va="top",
        rotation=45
    )
    ax.set_xlim([-1, len(c_indices)])
    ax.set_ylim([0, 1])

    ax.legend()
    ax.set_ylim([0, 1])

    ax.set_xlabel("Method")
    ax.set_ylabel("C-Index")

    ############################################################################
    # Kaplan-Meier curves
    ############################################################################

    ax = axs[8]

    survival_factors[:] = scale(survival_factors)
    component_sum = survival_factors.sum(axis=1)
    component_sum = component_sum.sort_values(ascending=False)
    kmf = KaplanMeierFitter()

    high_index = component_sum.index[:30]
    low_index = component_sum.index[30:]
    kmf.fit(
        survival_labels.loc[high_index, "survival_time"],
        survival_labels.loc[high_index, "graft_death"]
    )
    ax.plot(
        kmf.survival_function_.index,
        kmf.survival_function_.iloc[:, 0],
        label=f"High tPLS 1 + 2",
        color="tab:blue"
    )
    max_index = kmf.survival_function_.index[-1]
    kmf.fit(
        survival_labels.loc[low_index, "survival_time"],
        survival_labels.loc[low_index, "graft_death"]
    )
    ax.plot(
        kmf.survival_function_.index,
        kmf.survival_function_.iloc[:, 0],
        label=f"Low tPLS 1 + 2",
        color="tab:orange"
    )
    if kmf.survival_function_.index[-1] > max_index:
        max_index = kmf.survival_function_.index[-1]

    ax.set_ylim([0, 1])
    ax.set_xlim([0, max_index])

    ax.set_ylabel("Probability of\nNon-Rejection")
    ax.set_xlabel("Time")
    ax.legend()

    ############################################################################
    # Rank v. Accuracy
    ############################################################################

    ranks = np.arange(1, 6)
    accuracies = pd.Series(0, index=ranks)

    for rank in ranks:
        (_, _), _acc, _ = run_coupled_tpls_classification(
            tensors, labels, return_proba=True, rank=rank
        )
        accuracies.loc[rank] = _acc

    ax = axs[7]
    ax.plot(accuracies.index, accuracies)
    ax.set_ylim([0.5, 0.75])
    ax.set_xlabel("tPLS Components")
    ax.set_ylabel("Prediction Accuracy")

    return fig
