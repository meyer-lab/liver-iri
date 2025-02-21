"""Plots Figure 3d -- Granulocyte / LFT Association"""

import warnings

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import pearsonr

from ..dataimport import build_coupled_tensors
from .common import getSetup

warnings.filterwarnings("ignore")


def makeFigure():
    ############################################################################
    # Data imports
    ############################################################################

    raw_data = build_coupled_tensors(
        lft_scaling=1,
        pv_scaling=1,
        no_missing=True,
        normalize=False,
        transform="log",
    )
    raw_val = build_coupled_tensors(
        lft_scaling=1,
        pv_scaling=1,
        no_missing=False,
        normalize=False,
        transform="log",
    )
    raw_data = xr.merge([raw_data, raw_val])
    cytokine_measurements = raw_data["Cytokine Measurements"]
    lft_measurements = raw_data["LFT Measurements"]

    axs, fig = getSetup((3, 3), {"nrows": 1, "ncols": 1})
    ax = axs[0]

    il_17a = (
        cytokine_measurements.loc[
            {"Cytokine": "IL-17A", "Cytokine Timepoint": "PO"}
        ]
        .squeeze()
        .to_pandas()
    )
    il_17a = il_17a.loc[il_17a > 2]

    for lft_index, lft_score in enumerate(lft_measurements["LFT Score"].values):
        lfts = (
            lft_measurements.loc[
                {
                    "LFT Score": lft_score,
                    "LFT Timepoint": ["Opening"]
                    + [str(i) for i in range(1, 8)],
                }
            ]
            .squeeze()
            .to_pandas()
        )
        correlations = []
        for lft_tp in lfts.columns:
            df = pd.concat([il_17a, lfts.loc[:, lft_tp]], axis=1)
            df = df.dropna(axis=0)
            result = pearsonr(df.iloc[:, 0], df.iloc[:, 1])
            correlations.append(result.statistic)

        ax.bar(
            np.arange(lft_index, len(correlations) * 4, 4),
            correlations,
            width=1,
            label=lft_score,
        )

    x_lims = ax.get_xlim()
    ax.plot([-100, 100], [0, 0], linestyle="--", color="k")
    ax.set_xlim(x_lims)

    ax.set_xticks(np.arange(1, lfts.shape[1] * 4, 4))
    tick_labels = list(lfts.columns)
    tick_labels[1:] = [f"Day {i}" for i in tick_labels[1:]]
    ax.set_xticklabels(
        tick_labels, ha="right", ma="right", va="top", rotation=45
    )

    ax.set_ylabel("Pearson Correlation")
    ax.legend()

    return fig
