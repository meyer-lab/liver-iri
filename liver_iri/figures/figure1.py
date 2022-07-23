from ..dataimport import cytokine_data
import pandas as pd
import xarray as xr
import numpy as np
from tensorpack import perform_CP
from matplotlib import gridspec, pyplot as plt
import seaborn as sns

def makeFigure():
    data = cytokine_data()
    return makeComponentPlot(data, 8)


def makeComponentPlot(data:xr.DataArray, rank: int):
    cp = perform_CP(data.to_numpy(), rank)
    ddims = len(data.coords)
    axes_names = list(data.coords)

    factors = [pd.DataFrame(cp[1][rr], columns=[f"Cmp. {i}" for i in np.arange(1, rank + 1)],
                            index=data.coords[axes_names[rr]]) for rr in range(ddims)]

    f = plt.figure(figsize=(5*ddims, 6))
    gs = gridspec.GridSpec(1, ddims, wspace=0.5)
    axes = [plt.subplot(gs[rr]) for rr in range(ddims)]
    comp_labels = [str(ii + 1) for ii in range(rank)]

    for rr in range(ddims):
        sns.heatmap(factors[rr], cmap="PiYG", center=0, xticklabels=comp_labels, yticklabels=factors[rr].index,
                    cbar=True, vmin=-1.0, vmax=1.0, ax=axes[rr])
        axes[rr].set_xlabel("Components")
        axes[rr].set_title(axes_names[rr])

    return f