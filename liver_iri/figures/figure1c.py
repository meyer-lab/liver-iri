"""Plots Figure 1c -- Raw Data Heatmaps"""
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import xarray as xr

from ..dataimport import build_coupled_tensors, import_meta
from ..utils import reorder_table
from .common import getSetup


def makeFigure():
    # Figure setup
    axs, fig = getSetup(
        (5, 2), {"ncols": 4, "nrows": 1, "width_ratios": [1, 10, 10, 10]}
    )

    # Data imports
    data = build_coupled_tensors(
        peripheral_scaling=1, pv_scaling=1, lft_scaling=1, normalize=False
    )
    val_data = build_coupled_tensors(
        peripheral_scaling=1,
        pv_scaling=1,
        lft_scaling=1,
        normalize=False,
        no_missing=False,
    )
    data = xr.merge([data, val_data])

    meta = import_meta(long_survival=False)
    val_meta = import_meta(long_survival=False, no_missing=False)
    meta = pd.concat([meta, val_meta])
    meta = meta.loc[:, ["etiol", "iri", "graft_death"]]

    le = LabelEncoder()
    for column in meta.columns:
        meta.loc[:, column] = le.fit_transform(meta.loc[:, column])

    cytokines = (
        data["Cytokine Measurements"]
        .stack(Flattened=["Cytokine Timepoint", "Cytokine"])
        .to_pandas()
    )
    lfts = (
        data["LFT Measurements"]
        .stack(Flattened=["LFT Timepoint", "LFT Score"])
        .to_pandas()
    )

    merged = pd.concat([cytokines, lfts], axis=1)
    missing = np.isnan(merged)
    merged = reorder_table(merged.fillna(-1), plot_ax=axs[1])

    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].set_xlabel("")
    axs[1].set_ylabel("")

    # Transplant subset colorbars
    sns.heatmap(
        meta.loc[merged.index, :].astype(float),
        cmap="tab10",
        ax=axs[0],
        cbar=False
    )

    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_xlabel("")
    axs[0].set_ylabel("")

    # Cytokines
    heatmap = sns.heatmap(
        merged.iloc[:, :cytokines.shape[1]],
        cmap="coolwarm",
        ax=axs[2],
        cbar=False,
        mask=missing.iloc[:, : cytokines.shape[1]].values,
    )
    heatmap.set_facecolor("lightgrey")

    axs[2].set_xticks([])
    axs[2].set_yticks([])
    axs[2].set_xlabel("")

    # LFTs
    heatmap = sns.heatmap(
        merged.iloc[:, cytokines.shape[1]:],
        cmap="coolwarm",
        ax=axs[3],
        mask=missing.iloc[:, cytokines.shape[1]:].values,
    )
    heatmap.set_facecolor("lightgrey")

    axs[3].set_xticks([])
    axs[3].set_yticks([])
    axs[3].set_xlabel("")
    axs[3].set_ylabel("")

    return fig
