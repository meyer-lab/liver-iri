"""Plots Figure S1 -- Missingness Quantification"""

import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

from ..dataimport import build_coupled_tensors
from .common import getSetup


def makeFigure():
    ############################################################################
    # Data imports
    ############################################################################

    data = build_coupled_tensors(
        pv_scaling=1, lft_scaling=1, no_missing=True, normalize=False
    )
    val_data = build_coupled_tensors(
        pv_scaling=1, lft_scaling=1, no_missing=False, normalize=False
    )
    data = xr.merge([data, val_data])

    ############################################################################
    # Figure setup
    ############################################################################

    axs, fig = getSetup((8, 4), {"ncols": 1, "nrows": 2})

    ############################################################################
    # IL-4 heatmap
    ############################################################################

    for ax, d_type in zip(axs, data.data_vars, strict=False):
        data_set = data[d_type]
        missingness = pd.DataFrame(
            0,
            columns=data.Patient.values,
            index=data[data_set.dims[1]].values,
            dtype=int,
        )
        tensor = np.isnan(data_set.to_numpy())
        for tp in np.arange(tensor.shape[1]):
            missingness.iloc[tp, :] = tensor[:, tp, :].any(axis=1).astype(int)

        sns.heatmap(
            missingness,
            cmap="rocket",
            vmin=0,
            vmax=2,
            ax=ax,
            linewidths=0.1,
            cbar=False,
        )
        ax.set_xticks([])
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_ylabel(d_type)

    return fig
