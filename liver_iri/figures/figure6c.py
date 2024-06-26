"""Plots Figure 2e -- Resolving Cytokine Response"""
from decimal import Decimal
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from ..dataimport import build_coupled_tensors, import_meta
from ..predict import predict_continuous
from ..tensor import convert_to_numpy
from ..utils import reorder_table
from .common import getSetup

GRANULOCYTE_CYTOKINES = ["IL-17A", "IFNg", "IL-12P70"]

warnings.filterwarnings("ignore")


def plot_scatter(cytokine_measurements, cytokine, timepoints, ax):
    df = cytokine_measurements.loc[{
        "Cytokine": cytokine,
        "Cytokine Timepoint": timepoints
    }].to_pandas()
    df = df.loc[
        df.iloc[:, 0] >= 2,
        :
    ]
    df = df.dropna(axis=0)
    score, model = predict_continuous(
        df.iloc[:, 0],
        df.iloc[:, 1]
    )

    xs = [2, df.iloc[:, 0].max() * 1.05]
    ys = [
        model.params.iloc[0] + model.params.iloc[1] * xs[0],
        model.params.iloc[0] + model.params.iloc[1] * xs[1]
    ]
    ax.plot(xs, ys, color="k", linestyle="--")

    ax.scatter(
        df.iloc[:, 0],
        df.iloc[:, 1],
        s=6
    )
    ax.set_xlabel(df.columns[0])
    ax.set_ylabel(df.columns[1])

    ax.text(
        0.98,
        0.02,
        s=f"R2: {round(score, 3)}\np-value: {Decimal(model.pvalues[1]):.2E}",
        ha="right",
        ma="right",
        va="bottom",
        transform=ax.transAxes
    )
    ax.set_xlim([2, xs[1]])
    ax.set_ylim([0, ax.get_ylim()[-1]])
    ax.set_title(cytokine)


def makeFigure():
    ############################################################################
    # Data imports
    ############################################################################

    raw_data = build_coupled_tensors(
        lft_scaling=1,
        pv_scaling=1,
        no_missing=True,
        normalize=False,
        transform="log"
    )
    raw_val = build_coupled_tensors(
        lft_scaling=1,
        pv_scaling=1,
        no_missing=False,
        normalize=False,
        transform="log"
    )
    raw_data = xr.merge([raw_data, raw_val])
    cytokine_measurements = raw_data["Cytokine Measurements"]

    axs, fig = getSetup(
        (9, 3),
        {"nrows": 1, "ncols": 3}
    )

    for ax, cytokine in zip(axs, GRANULOCYTE_CYTOKINES):
        plot_scatter(cytokine_measurements, cytokine, ["PV", "W1"], ax)

    return fig
