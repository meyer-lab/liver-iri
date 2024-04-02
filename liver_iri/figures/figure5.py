"""Plots Figure 5 -- Cytokine Associations"""
import warnings

import numpy as np
import pandas as pd
import seaborn as sns

from ..dataimport import cytokine_data, import_meta
from ..predict import predict_categorical
from ..utils import reorder_table
from .common import getSetup

warnings.filterwarnings("ignore")

LFT_TIMEPOINTS = ["Opening", "1", "2", "3", "4", "5", "6", "7"]
LFT_NAMES = ["ALT", "AST", "INR", "TBIL"]
LFT_CONVERSIONS = {i: f"Day {i}" for i in LFT_TIMEPOINTS[1:]}
LFT_CONVERSIONS["Opening"] = "Pre-Op"
TIMEPOINTS = ["PO", "D1", "W1", "M1"]
TP_CONVERSIONS = {
    "PO": "Pre-Op",
    "D1": "1 Day Post-Op",
    "W1": "1 Week Post-Op",
    "M1": "1 Month Post-Op",
    "PV": "Pre-Op PV",
    "LF": "Post-Op PV",
}


def get_coefs(matrices, graft, names):
    coefs = pd.DataFrame(0, index=names, columns=matrices[0].columns)
    for matrix, tp in zip(matrices, coefs.index):
        data = matrix.dropna(axis=0)
        labels = graft.loc[data.index]
        _, _, coef = predict_categorical(data, labels, return_coef=True)
        coefs.loc[tp, :] = coef

    return coefs


def plot_coefficients(coefs, ax):
    coefs = coefs.T / abs(coefs).max(axis=1)
    coefs = reorder_table(coefs)
    sns.heatmap(
        coefs, cmap="vlag", ax=ax, cbar_kws={"label": "Graft Death Association"}
    )

    ax.set_xticks(np.arange(0.5, coefs.shape[1]))
    ax.set_xticklabels(
        coefs.columns, rotation=45, va="top", ha="right", ma="right"
    )
    ax.set_yticks(np.arange(0.5, coefs.shape[0]))
    ax.set_yticklabels(coefs.index)

    ax.set_ylabel("Cytokine")
    ax.set_xlabel("Time Point")


def makeFigure():
    cyto = cytokine_data()
    meta = import_meta()
    cyto = cyto.sel(Patient=meta.index)
    graft = meta.loc[:, "graft_death"]

    matrices = []
    for tp in TIMEPOINTS:
        matrices.append(
            cyto.sel({"Cytokine Timepoint": tp})
            .to_array()
            .squeeze()
            .to_pandas()
        )

    cyto_names = [TP_CONVERSIONS.get(i, i) for i in TIMEPOINTS]
    coefs = get_coefs(matrices, graft, cyto_names)

    fig_size = (6, 4)
    layout = {"nrows": 1, "ncols": 1}
    axs, fig = getSetup(fig_size, layout)

    plot_coefficients(coefs, axs[0])

    return fig
