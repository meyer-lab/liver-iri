"""Plots Figure 2 -- Accuracy vs. Rank"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .common import getSetup
from ..dataimport import build_coupled_tensors, import_meta
from ..predict import predict_categorical_svc
from ..tensor import run_coupled


def makeFigure():
    factor_count = np.arange(1, 11)
    accuracies = pd.Series(index=factor_count, dtype=float)

    meta = import_meta()
    labels = meta.loc[:, 'graft_death']
    labels = labels.dropna()

    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels)

    data = build_coupled_tensors(
        cytokine_params={},
        lft_params={},
        pv_params={}
    )
    for n_factors in factor_count:
        factors, _ = run_coupled(
            data,
            rank=n_factors
        )

        factors = factors.loc[labels.index, :]
        score, _ = predict_categorical_svc(factors, encoded)
        accuracies.loc[n_factors] = score

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
