import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from .common import getSetup
from ..dataimport import cytokine_data, import_meta, import_lfts
from ..predict import predict_categorical, get_probabilities

warnings.filterwarnings('ignore')

LFT_TIMEPOINTS = [
    'Opening', '1', '2', '3', '4', '5', '6', '7'
]
LFT_NAMES = [
    'ALT', 'AST', 'INR', 'TBIL'
]
LFT_CONVERSIONS = {
    i: f'Day {i}' for i  in LFT_TIMEPOINTS[1:]
}
LFT_CONVERSIONS['Opening'] = 'Pre-Op'
TIMEPOINTS = ['PO', 'D1', 'W1', 'M1']
TP_CONVERSIONS = {
    'PO': 'Pre-Op',
    'D1': '1 Day\nPost-Op',
    'W1': '1 Week\nPost-Op',
    'M1': '1 Month\nPost-Op',
    'PV': 'Pre-Op\nPV',
    'LF': 'Post-Op\nPV'
}


def get_accuracies(matrices, graft, names):
    accuracies = pd.Series(
        0,
        index=names
    )
    curves = []
    for matrix, tp in zip(matrices, accuracies.index):
        data = matrix.dropna(axis=0)
        labels = graft.loc[data.index]
        acc, model = predict_categorical(
            data,
            labels
        )
        accuracies.loc[tp] = acc

        proba = get_probabilities(model, data, labels)
        fpr, tpr, _ = roc_curve(
            labels,
            proba
        )
        curves.append((fpr, tpr))

    return accuracies, curves


def plot_curves(curves, ax, names):
    for curve, name in zip(curves, names):
        ax.plot(
            curve[0],
            curve[1],
            label=TP_CONVERSIONS.get(name, name).replace('\n', ' ')
        )

    ax.legend()

    ax.plot([0, 1], [0, 1], color='k', linestyle='--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')


def plot_accuracies(accuracies, ax, x_label=None, y_label=None):
    ax.bar(
        range(len(accuracies)),
        accuracies
    )
    for index, accuracy in enumerate(accuracies):
        ax.text(
            index,
            accuracy + 0.01,
            s=round(accuracy, 2),
            ha='center',
            va='bottom'
        )

    ax.set_xticks(range(len(accuracies)))
    labels = [TP_CONVERSIONS.get(index, index) for index in accuracies.index]
    ax.set_xticklabels(
        labels,
        ha='center',
        ma='center',
        va='top'
    )
    ax.set_ylim([0, 1])
    ax.set_xlim([-0.5, len(accuracies) - 0.5])

    if x_label is not None:
        ax.set_xlabel(x_label)

    if y_label is not None:
        ax.set_ylabel(y_label)


def makeFigure():
    meta = import_meta(balanced=True)
    graft = meta.loc[:, 'graft_death']
    cyto = cytokine_data(
        column=None,
        uniform_lod=False,
        transform='log',
        mean_center=False,
        drop_pv=False
    )
    lfts = import_lfts()
    lfts = lfts.loc[meta.index, :]

    matrices = []
    for tp in TIMEPOINTS:
        matrices.append(
            cyto.sel(
                {'Cytokine Timepoint': tp}
            ).to_array().squeeze().to_pandas().loc[meta.index, :]
        )
    matrices.append(
        pd.concat(
            matrices,
            axis=1
        ).dropna(axis=0)
    )

    lft_matrices = []
    for tp in LFT_TIMEPOINTS:
        matrix = lfts.loc[:, lfts.columns.str.contains(tp.lower())]
        matrix = matrix.loc[np.isfinite(matrix).all(axis=1), :]
        lft_matrices.append(
            matrix
        )
    lft_matrices.append(
        pd.concat(
            lft_matrices,
            axis=1
        )
    )

    cyto_names = [TP_CONVERSIONS.get(i, i) for i in TIMEPOINTS] + \
                 ['Flattened\nCytokines']
    lft_names = [LFT_CONVERSIONS.get(i, i) for i in LFT_TIMEPOINTS] + \
                ['Flattened\nLFTs']
    cyto_accuracies, cyto_curves = get_accuracies(
        matrices,
        graft,
        cyto_names
    )
    lft_accuracies, lft_curves = get_accuracies(
        lft_matrices,
        graft,
        lft_names
    )

    fig_size = (8, 6)
    layout = {'nrows': 2, 'ncols': 2}
    axs, fig = getSetup(
        fig_size,
        layout
    )

    plot_accuracies(
        cyto_accuracies,
        axs[0],
        x_label='Cytokine Measurements',
        y_label='Classification Accuracy'
    )
    plot_curves(cyto_curves, axs[1], cyto_names)

    plot_accuracies(
        lft_accuracies,
        axs[2],
        x_label='LFT Measurements',
        y_label='Classification Accuracy'
    )
    plot_curves(lft_curves, axs[3], lft_names)

    return fig
