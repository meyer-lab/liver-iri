"""Plots Figure 2c -- CP Component Correlations"""
from decimal import Decimal

import numpy as np

from .common import getSetup
from ..dataimport import build_coupled_tensors
from ..predict import predict_continuous
from ..tensor import run_coupled


def plot_scatter(df, ax):
    df = df.dropna(axis=0)

    ax.scatter(
        df.iloc[:, 0],
        df.iloc[:, 1],
        s=6
    )

    score, model = predict_continuous(
        df.iloc[:, 0],
        df.iloc[:, 1]
    )

    ax.set_xlabel(df.columns[0])
    ax.set_ylabel(df.columns[1])

    xs = [-1.1, 1.1]
    ys = [
        model.params.iloc[0] + model.params.iloc[1] * xs[0],
        model.params.iloc[0] + model.params.iloc[1] * xs[1]
    ]
    ax.plot(xs, ys, color="k", linestyle="--")
    ax.plot([-1.1, 1.1], [0, 0], color="grey", linestyle="--")
    ax.plot([0, 0], [-1.1, 1.1], color="grey", linestyle="--")

    ax.text(
        0.98,
        0.02,
        s=f"R2: {round(score, 3)}\np-value: {Decimal(model.pvalues[1]):.2E}",
        ha="right",
        ma="right",
        va="bottom",
        transform=ax.transAxes
    )

    ax.set_xticks(np.arange(-1, 1.1, 0.5))
    ax.set_yticks(np.arange(-1, 1.1, 0.5))

    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])


def makeFigure():
    ############################################################################
    # Data imports
    ############################################################################

    data = build_coupled_tensors(
        pv_scaling=1,
        lft_scaling=1,
        no_missing=True
    )

    ############################################################################
    # Factorization
    ############################################################################

    _, cp = run_coupled(data, rank=4)
    patient_factor = cp.x["_Patient"].to_pandas()
    patient_factor /= abs(patient_factor).max(axis=0)

    axs, fig = getSetup(
        (3 * 3, 2 * 3), {"nrows": 2, "ncols": 3}
    )
    ax_index = 0

    for i in np.arange(cp.rank):
        for j in np.arange(i + 1, cp.rank):
            ax = axs[ax_index]
            df = patient_factor.iloc[:, [i, j]]
            df.columns = [f"Component {i + 1}", f"Component {j + 1}"]
            plot_scatter(df, ax)
            ax_index += 1

    return fig
