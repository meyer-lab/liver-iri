"""Plots Figure 2 -- CP Factorization R2Xs"""
import numpy as np
import pandas as pd
from tqdm import tqdm
import xarray as xr

from .common import getSetup
from ..dataimport import build_coupled_tensors
from ..tensor import calc_r2x, run_coupled


def makeFigure():
    ############################################################################
    # Figure setup
    ############################################################################

    axs, fig = getSetup(
        (8, 2),
        {
            "ncols": 4,
            "nrows": 1,
        },
    )

    ############################################################################
    # Data import & factorization
    ############################################################################

    data = build_coupled_tensors(
        pv_scaling=2,
        lft_scaling=4,
    )
    val_data = build_coupled_tensors(
        pv_scaling=2, lft_scaling=4, no_missing=False
    )
    data = xr.merge([data, val_data])

    ############################################################################
    # Figure 2A: R2X vs. Rank
    ############################################################################

    ranks = np.arange(8) + 1
    r2xs = pd.Series(0, index=ranks)
    r2x_averaged = r2xs.copy(deep=True)
    for rank in tqdm(ranks):
        _, cp = run_coupled(data, rank=rank)
        r2xs.loc[rank] = cp.R2X()
        cyto_reconstructed = cp.to_CPTensor("Cytokine Measurements").to_tensor()
        cyto_actual = cp.data["Cytokine Measurements"].to_numpy()
        lft_reconstructed = cp.to_CPTensor("LFT Measurements").to_tensor()
        lft_actual = cp.data["LFT Measurements"].to_numpy()
        r2x_averaged.loc[rank] = np.mean(
            [
                calc_r2x(
                    cyto_actual[:, [1, 2], :], cyto_reconstructed[:, [1, 2], :]
                ),
                calc_r2x(
                    cyto_actual[:, [0, 3, 4, 5], :],
                    cyto_reconstructed[:, [0, 3, 4, 5], :],
                ),
                calc_r2x(lft_actual, lft_reconstructed),
            ]
        )

    ax = axs[0]
    ax.plot(r2xs.index, r2xs, label="R2X")
    ax.plot(r2x_averaged.index, r2x_averaged, label="Averaged")
    ax.set_ylim([0, 1])
    ax.set_ylabel("R2X")
    ax.set_xlabel("Rank")

    ############################################################################
    # Figure 2B-D: R2X vs. Scaling
    ############################################################################

    scalings = np.logspace(-5, 5, base=2, num=11)
    r2xs = pd.DataFrame(
        index=scalings,
        columns=[
            "Total",
            "Peripheral Cytokines",
            "PV Cytokines",
            "LFT Measurements",
        ],
    )
    ds_scalings = ["peripheral_scaling", "pv_scaling", "lft_scaling"]
    for dataset, ax in zip(ds_scalings, axs[1:]):
        for scaling in scalings:
            scalings_values = {ds: 1 for ds in ds_scalings}
            scalings_values[dataset] = scaling
            data = build_coupled_tensors(
                **scalings_values,
            )
            val_data = build_coupled_tensors(
                **scalings_values, no_missing=False
            )
            data = xr.merge([data, val_data])

            _, cp = run_coupled(data)
            r2xs.loc[scaling, "Total"] = cp.R2X()

            cyto_reconstructed = cp.to_CPTensor(
                "Cytokine Measurements"
            ).to_tensor()
            cyto_actual = cp.data["Cytokine Measurements"].to_numpy()
            r2xs.loc[scaling, "PV Cytokines"] = calc_r2x(
                cyto_actual[:, [1, 2], :], cyto_reconstructed[:, [1, 2], :]
            )
            r2xs.loc[scaling, "Peripheral Cytokines"] = calc_r2x(
                cyto_actual[:, [0, 3, 4, 5], :],
                cyto_reconstructed[:, [0, 3, 4, 5], :],
            )

            lft_reconstructed = cp.to_CPTensor("LFT Measurements").to_tensor()
            lft_actual = cp.data["LFT Measurements"].to_numpy()
            r2xs.loc[scaling, "LFT Measurements"] = calc_r2x(
                lft_actual, lft_reconstructed
            )

        for column in r2xs.columns:
            ax.plot(r2xs.index, r2xs.loc[:, column], label=column)

        ax.set_xticks(r2xs.index)
        ax.set_ylim([0, 1])

        ax.legend()
        ax.set_xscale("log")
        ax.set_xlabel(dataset)
        ax.set_ylabel("R2X")

    return fig
