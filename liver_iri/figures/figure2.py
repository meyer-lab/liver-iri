"""Plots Figure 2 -- Accuracy vs. Rank"""
import numpy as np
import pandas as pd

from .common import getSetup
from ..dataimport import build_coupled_tensors, import_meta
from ..predict import run_coupled_tpls_classification


def makeFigure():
    factor_count = np.arange(1, 21)
    accuracies = pd.Series(index=factor_count, dtype=float)

    meta = import_meta()
    labels = meta.loc[:, 'graft_death']
    labels = labels.dropna()

    data = build_coupled_tensors(
        cytokine_params={
            'coupled_scaling': 1,
            'pv_scaling': 1
        },
        rna_params=False,
        lft_params={
            'coupled_scaling': 1
        }
    )
    for n_factors in factor_count:
        (_, _), acc, _ = run_coupled_tpls_classification(
            data,
            labels,
            rank=n_factors
        )
        accuracies.loc[n_factors] = acc

    fig_size = (6, 3)
    layout = {'nrows': 1, 'ncols': 1}
    axs, fig = getSetup(
        fig_size,
        layout
    )
    ax = axs[0]

    ax.plot(factor_count, accuracies)

    ax.set_xticks(factor_count)
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Transplant Outcome Prediction Accuracy')

    return fig
