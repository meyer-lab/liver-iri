"""Plots Figure 1c -- Cytokine Binned Clinical Comparisons"""
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.preprocessing import LabelEncoder
import xarray as xr

from .common import getSetup
from ..dataimport import build_coupled_tensors, import_meta


DIFFERENCES = [
    "bnscores", "postinf", "postnec", "poststeat", "postcong", "postbal"
]


def makeFigure():
    ############################################################################
    # Data imports
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

    ############################################################################
    # Figure setup
    ############################################################################

    axs, fig = getSetup(
        (3 * len(DIFFERENCES), 3 * len(cytokines["Cytokine"].values)),
        {
            "nrows": len(cytokines["Cytokine"].values),
            "ncols": len(DIFFERENCES)
        }
    )

    ############################################################################
    # Binned variable cytokine clinical comparisons
    ############################################################################

    ax_index = 0
    le = LabelEncoder()
    for cytokine in cytokines["Cytokine"].values:
        df = cytokines.sel(
            {
                "Cytokine": cytokine
            }
        ).squeeze().to_pandas()
        df = df.loc[meta.index, :]
        for to_diff in DIFFERENCES:
            ax = axs[ax_index]
            meta_col = meta.loc[:, to_diff].dropna()
            meta_col = meta_col > 1
            meta_col[:] = le.fit_transform(meta_col)
            _df = df.loc[meta_col.index, :]

            if le.classes_ is None:
                continue

            ax.errorbar(
                np.arange(_df.shape[1]) + 0.1,
                _df.loc[meta_col == 0, :].mean(axis=0),
                yerr=_df.loc[meta_col == 0, :].std(axis=0),
                capsize=2,
                label=f"{le.classes_[0]}: "
                      f"n={_df.loc[meta_col == 0, :].shape[0]}"
            )
            ax.errorbar(
                np.arange(_df.shape[1]) - 0.1,
                _df.loc[meta_col == 1, :].mean(axis=0),
                yerr=_df.loc[meta_col == 1, :].std(axis=0),
                capsize=2,
                label=f"{le.classes_[1]}: "
                      f"n={_df.loc[meta_col == 1, :].shape[0]}"
            )

            ticks = list(_df.columns)
            for index, col in enumerate(_df.columns):
                result = ttest_ind(
                    _df.loc[meta_col == 0, col].dropna(),
                    _df.loc[meta_col == 1, col].dropna()
                )
                if result.pvalue < 0.01:
                    ticks[index] = ticks[index] + "**"
                elif result.pvalue < 0.05:
                    ticks[index] = ticks[index] + "*"

            ax.legend()
            ax.set_title(f"{cytokine} v. {to_diff}")

            ax.set_xticks(np.arange(_df.shape[1]))
            ax.set_xticklabels(ticks)
            ax_index += 1

    return fig
