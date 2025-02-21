"""Plots Figure S8 -- tPLS Parameter Tuning"""

import warnings

import numpy as np
import pandas as pd

from ..dataimport import build_coupled_tensors, import_meta
from ..predict import run_coupled_tpls_classification
from ..tensor import convert_to_numpy
from .common import getSetup

warnings.filterwarnings("ignore")


def makeFigure():
    ############################################################################
    # Data imports
    ############################################################################

    meta = import_meta()
    labels = meta.loc[:, "graft_death"]
    labels = labels.dropna()
    tpls_data = build_coupled_tensors()
    tensors, labels = convert_to_numpy(tpls_data, labels)

    ############################################################################
    # Figure setup
    ############################################################################

    axs, fig = getSetup((6, 6), {"ncols": 2, "nrows": 2})

    ############################################################################
    # Rank evaluation
    ############################################################################

    ax = axs[0]
    ranks = np.arange(1, 6)
    accuracies = pd.Series(0, index=ranks, dtype=float)
    for rank in ranks:
        _, tpls_acc, _ = run_coupled_tpls_classification(
            tensors, labels, rank=rank
        )
        accuracies.loc[rank] = tpls_acc

    ax.plot(ranks, accuracies)
    ax.set_ylim([0, 1])
    ax.set_xlabel("Rank")
    ax.set_ylabel("Balanced Accuracy")

    ############################################################################
    # Scaling comparisons
    ############################################################################

    scalings = np.logspace(-4, 4, 9)
    default_scalings = {
        "pv_scaling": 1,
        "peripheral_scaling": 1,
        "lft_scaling": 1,
        "transform": "power",
        "no_missing": True,
        "normalize": True,
    }
    labels = meta.loc[:, "graft_death"]
    labels = labels.dropna()

    for ax, dataset in zip(
        axs[1:],
        ["pv_scaling", "lft_scaling", "peripheral_scaling"],
        strict=False,
    ):
        accuracies = pd.Series(0, index=scalings, dtype=float)
        for scaling in scalings:
            _scalings = default_scalings.copy()
            _scalings[dataset] = scaling
            tpls_data = build_coupled_tensors(**_scalings)
            tensors, _labels = convert_to_numpy(tpls_data, labels)
            _, tpls_acc, _ = run_coupled_tpls_classification(
                tensors, labels, return_proba=True
            )
            accuracies.loc[scaling] = tpls_acc

        ax.semilogx(scalings, accuracies)
        ax.set_xticks(scalings)
        ax.set_ylim([0, 1])
        ax.set_xlabel(f"{dataset} Scaling")
        ax.set_ylabel("Balanced Accuracy")

    return fig
