"""Plots Figure 1j -- All Cytokine Errorbars"""
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, pearsonr
from sklearn.preprocessing import LabelEncoder
import xarray as xr

from .common import getSetup
from ..dataimport import build_coupled_tensors, import_meta



def makeFigure():
    ############################################################################
    # Factorization
    ############################################################################

    meta = import_meta(long_survival=False)
    val_meta = import_meta(no_missing=False, long_survival=False)
    meta = pd.concat([meta, val_meta])

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
    cytokines = data["Cytokine Measurements"]

    axs, fig = getSetup(
        (3, 3 * len(cytokines["Cytokine"].values)),
        {
            "nrows": len(cytokines["Cytokine"].values),
            "ncols": 1
        }
    )

    ############################################################################
    # Correlation
    ############################################################################

    ax_index = 0
    le = LabelEncoder()
    for cytokine, ax in zip(cytokines["Cytokine"].values, axs):
        df = cytokines.sel(
            {
                "Cytokine": cytokine
            }
        ).squeeze().to_pandas()
        ax.errorbar(
            np.arange(df.shape[1]),
            df.mean(axis=0),
            yerr=df.std(axis=0),
            capsize=2,
            color="black"
        )

        ticks = list(df.columns)

        ax.legend()
        ax.set_title(cytokine)
        ax.set_xticks(np.arange(df.shape[1]))
        ax.set_xticklabels(ticks)
        ax_index += 1

    return fig
