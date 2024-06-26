"""Plots Figure 3b -- NK / LFT Association"""
from decimal import Decimal
import warnings

import numpy as np
from scipy.stats import pearsonr
import pandas as pd
import xarray as xr

from ..dataimport import build_coupled_tensors, import_meta
from ..predict import predict_continuous
from .common import getSetup

warnings.filterwarnings("ignore")


def plot_scatter(df, meta, ax):
    df = df.dropna(axis=0)
    meta = meta.loc[df.index, :]

    ax.scatter(
        df.iloc[:, 0],
        df.iloc[:, 1],
        s=6,
        c=meta.loc[:, "graft_death"].replace(
            {
                1: "red",
                0: "green"
            }
        )
    )

    score, model = predict_continuous(
        df.iloc[:, 0],
        df.iloc[:, 1]
    )

    x_lims = ax.get_xlim()
    y_lims = ax.get_ylim()

    ax.set_xlabel(df.columns[0])
    ax.set_ylabel(df.columns[1])

    xs = [0, df.iloc[:, 0].max() * 1.05]
    ys = [
        model.params.iloc[0] + model.params.iloc[1] * xs[0],
        model.params.iloc[0] + model.params.iloc[1] * xs[1]
    ]
    ax.plot(xs, ys, color="k", linestyle="--")

    ax.text(
        0.98,
        0.02,
        s=f"R2: {round(score, 3)}\np-value: {Decimal(model.pvalues[1]):.2E}",
        ha="right",
        ma="right",
        va="bottom",
        transform=ax.transAxes
    )
    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)


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
    lft_measurements = raw_data["LFT Measurements"]

    meta = import_meta(long_survival=False, no_missing=True)
    val_meta = import_meta(long_survival=False, no_missing=False)
    meta = pd.concat([meta, val_meta])

    axs, fig = getSetup(
        (3, 3),
        {"nrows": 1, "ncols": 1}
    )
    ax = axs[0]

    il_15 = cytokine_measurements.loc[
        {
            "Cytokine": "IL-15",
            "Cytokine Timepoint": "PV"
        }
    ].squeeze().to_pandas()

    for lft_index, lft_score in enumerate(lft_measurements["LFT Score"].values):
        lfts = lft_measurements.loc[
            {
                "LFT Score": lft_score,
                "LFT Timepoint": ["Opening"] + [str(i) for i in range(1, 8)]
            }
        ].squeeze().to_pandas()
        correlations = []
        for lft_tp in lfts.columns:
            df = pd.concat([il_15, lfts.loc[:, lft_tp]], axis=1)
            df = df.dropna(axis=0)
            result = pearsonr(
                df.iloc[:, 0],
                df.iloc[:, 1]
            )
            correlations.append(result.pvalue)

        correlations = -np.log(correlations)
        ax.bar(
            np.arange(lft_index, len(correlations) * 4, 4),
            correlations,
            width=1,
            label=lft_score
        )

    x_lims = ax.get_xlim()
    ax.plot(
        [-100, 100],
        [-np.log(0.05)] * 2,
        linestyle="--",
        color="k",
        zorder=-3
    )

    ax.set_xlim(x_lims)

    ax.set_xticks(np.arange(1, lfts.shape[1] * 4, 4))
    tick_labels = list(lfts.columns)
    tick_labels[1:] = [f"Day {i}" for i in tick_labels[1:]]
    ax.set_xticklabels(
        tick_labels,
        ha="right",
        ma="right",
        va="top",
        rotation=45
    )

    ax.set_ylabel("-log(p-value)")
    ax.legend()

    return fig
