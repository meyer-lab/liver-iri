"""Plots Figure 2b -- CTF Factorization Timepoints"""

import numpy as np

from ..dataimport import build_coupled_tensors
from ..tensor import run_coupled
from .common import getSetup


def makeFigure():
    ############################################################################
    # Data imports
    ############################################################################

    data = build_coupled_tensors(
        peripheral_scaling=1, pv_scaling=1, lft_scaling=1, no_missing=True
    )

    ############################################################################
    # Factorization
    ############################################################################

    _, cp = run_coupled(data, rank=4)
    factors = {}
    for mode in cp.modes:
        if "Timepoint" in str(mode):
            factors[mode] = cp.x[f"_{mode}"].to_pandas()

    ############################################################################
    # Figure setup
    ############################################################################

    axs, fig = getSetup(
        (len(factors) * 3, 3), {"nrows": 1, "ncols": len(factors)}
    )

    ############################################################################
    # Plot timepoint associations
    ############################################################################

    for ax, (name, df) in zip(axs, factors.items()):
        for component in df.columns:
            ax.plot(
                np.arange(df.shape[0]),
                df.loc[:, component],
                label=f"Component {component}",
            )

        ax.legend()
        ax.plot([-1, df.shape[0]], [0, 0], linestyle="--", color="k")

        ax.set_xlim([-0.5, df.shape[0] - 0.5])
        ax.set_ylim([-1.1, 1.1])
        ax.set_xticks(np.arange(df.shape[0]))
        ax.set_yticks(np.arange(-1, 1.1, 0.5))
        ax.set_xticklabels(df.index)
        ax.set_title(name)
        ax.set_xlabel("Timepoint")
        ax.set_ylabel("Component Association")
        ax.grid(True)

    return fig
