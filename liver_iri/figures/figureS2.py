"""Plots Figure S2 -- LFT Timecourses"""
import numpy as np
import pandas as pd
from scipy.signal import argrelmax
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import xarray as xr

from ..dataimport import build_coupled_tensors, import_meta
from ..utils import reorder_table
from .common import getSetup


def makeFigure():
    ############################################################################
    # Data imports
    ############################################################################

    data = build_coupled_tensors(
        peripheral_scaling=1, pv_scaling=1, lft_scaling=1, normalize=False,
        transform="log"
    )
    val_data = build_coupled_tensors(
        peripheral_scaling=1,
        pv_scaling=1,
        lft_scaling=1,
        normalize=False,
        no_missing=False,
        transform="log"
    )
    data = xr.merge([data, val_data])
    data = data.drop_sel({"Patient": [34]})

    ############################################################################
    # Figure setup
    ############################################################################

    axs, fig = getSetup(
        (9, 3),
        {"ncols": 3, "nrows": 1}
    )

    ############################################################################
    # Threshold LFTs
    ############################################################################

    lfts = data["LFT Measurements"]
    thresholds = pd.Series(
        [35, 45, 3.5],
        index=["ast", "alt", "tbil"],
        dtype=float
    )
    thresholds.loc[:] = np.log(thresholds)
    types = pd.DataFrame(
        index=lfts.Patient.values,
        columns=lfts["LFT Score"].values
    )

    ############################################################################
    # Plot LFTs by timecourse
    ############################################################################

    for score, ax in zip(lfts["LFT Score"].values, axs):
        score_array = lfts.sel(
            {
                "LFT Score": score
            }
        ).squeeze().to_pandas()
        high_end = score_array.loc[
            score_array.iloc[:, -1] >= thresholds[score],
            :
        ]
        low_end = score_array.loc[
            score_array.iloc[:, -1] < thresholds[score],
            :
        ]
        recurring = high_end.loc[
            high_end.iloc[:, -3:].max(axis=1) > high_end.iloc[:, -4],
            :
        ]
        resolving = high_end.loc[
            high_end.iloc[:, -3:].max(axis=1) <= high_end.iloc[:, -4],
            :
        ]

        types.loc[low_end.index, score] = "Resolving"
        types.loc[recurring.index, score] = "Recurring"
        types.loc[resolving.index, score] = "High"

        ax.errorbar(
            np.arange(score_array.shape[1]) - 0.2,
            low_end.mean(axis=0),
            yerr=low_end.std(axis=0),
            label=f"Resolving (n={low_end.shape[0]})",
            capsize=2,
            color="tab:green"
        )
        ax.errorbar(
            np.arange(score_array.shape[1]),
            recurring.mean(axis=0),
            yerr=recurring.std(axis=0),
            label=f"Recurring (n={recurring.shape[0]})",
            capsize=2,
            color="tab:orange"
        )
        ax.errorbar(
            np.arange(score_array.shape[1]) + 0.2,
            resolving.mean(axis=0),
            yerr=resolving.std(axis=0),
            label=f"High (n={resolving.shape[0]})",
            capsize=2,
            color="tab:red"
        )

        ax.set_title(score)
        ax.plot(
            [-1, 10],
            [thresholds[score], thresholds[score]],
            linestyle="--",
            color="k"
        )
        ax.legend()
        ax.set_xlim([-0.5, 7.5])

    return fig
