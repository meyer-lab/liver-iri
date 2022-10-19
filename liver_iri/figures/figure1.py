from ..dataimport import cytokine_data
import pandas as pd
import xarray as xr
import numpy as np
from tensorpack import perform_CP, reorient_factors
from tensorly.cp_tensor import cp_flip_sign
import seaborn as sns

from .common import getSetup
from ..utils import reorder_table


def makeFigure():
    data = cytokine_data(None, log_scaling=True, uniform_lod=True)
    return makeComponentPlot(data, 8, ["Patient", "Cytokine"])


def makeComponentPlot(data:xr.DataArray, rank: int, reorder=[]):
    cp = perform_CP(data.to_numpy(), rank)
    cp = cp_flip_sign(cp)
    cp = reorient_factors(cp)

    ddims = len(data.coords)
    axes_names = list(data.coords)

    factors = [pd.DataFrame(cp[1][rr], columns=[f"Cmp. {i}" for i in np.arange(1, rank + 1)],
                            index=data.coords[axes_names[rr]]) for rr in range(ddims)]

    for r_ax in reorder:
        if isinstance(r_ax, int):
            assert r_ax < ddims
            factors[r_ax] = reorder_table(factors[r_ax])
        elif isinstance(r_ax, str):
            assert r_ax in axes_names
            rr = axes_names.index(r_ax)
            factors[rr] = reorder_table(factors[rr])

    fig_size = (5 * ddims, 6)
    layout = {'nrows': 1, 'ncols': ddims, 'wspace': 0.1}
    axes, fig = getSetup(
        fig_size,
        layout
    )
    comp_labels = [str(ii + 1) for ii in range(rank)]

    for rr in range(ddims):
        sns.heatmap(factors[rr], cmap="PiYG", center=0, xticklabels=comp_labels, yticklabels=factors[rr].index,
                    cbar=True, vmin=-1.0, vmax=1.0, ax=axes[rr])
        axes[rr].set_xlabel("Components")
        axes[rr].set_title(axes_names[rr])

    return fig
