"""Plots Figure 2 -- CP Factorization R2Xs"""
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from .common import getSetup
from ..dataimport import build_coupled_tensors
from ..tensor import calc_r2x, run_coupled


def makeFigure():
    ############################################################################
    # Figure setup
    ############################################################################

    axs, fig = getSetup(
        (6, 4),
        {
            'ncols': 3,
            'nrows': 2,
            # 'width_ratios': [1, 1, 1.1]
        }
    )

    ############################################################################
    # Data import & factorization
    ############################################################################

    data = build_coupled_tensors(
        pv_scaling=2,
        lft_scaling=4
    )

    ############################################################################
    # Figure 2A: R2X vs. Rank
    ############################################################################

    ranks = np.arange(1, 11)
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
                    cyto_actual[:, [1, 2], :],
                    cyto_reconstructed[:, [1, 2], :]
                ),
                calc_r2x(
                    cyto_actual[:, [0, 3, 4, 5], :],
                    cyto_reconstructed[:, [0, 3, 4, 5], :]
                ),
                calc_r2x(
                    lft_actual,
                    lft_reconstructed
                )
            ]
        )

    ax = axs[0]
    ax.plot(
        r2xs.index,
        r2xs,
        label="R2X"
    )
    ax.plot(
        r2x_averaged.index,
        r2x_averaged,
        label="Averaged"
    )
    ax.set_ylabel('R2X')
    ax.set_xlabel('Rank')

    ############################################################################
    # Figure 2B-F: R2X vs. Scaling
    ############################################################################

    scalings = np.logspace(-5, 5, base=2, num=11)
    r2x_total = pd.DataFrame(index=scalings, columns=scalings, dtype=float)
    r2x_pv = r2x_total.copy(deep=True)
    r2x_peripheral = r2x_total.copy(deep=True)
    r2x_lfts = r2x_total.copy(deep=True)
    r2x_averaged = r2x_total.copy(deep=True)
    for pv_scaling in tqdm(scalings):
        for lft_scaling in scalings:
            data = build_coupled_tensors(
                pv_scaling=pv_scaling,
                lft_scaling=lft_scaling
            )
            _, cp = run_coupled(data)
            r2x_total.loc[lft_scaling, pv_scaling] = cp.R2X()

            cyto_reconstructed = cp.to_CPTensor(
                "Cytokine Measurements"
            ).to_tensor()
            cyto_actual = cp.data["Cytokine Measurements"].to_numpy()
            r2x_pv.loc[lft_scaling, pv_scaling] = calc_r2x(
                cyto_actual[:, [1, 2], :],
                cyto_reconstructed[:, [1, 2], :]
            )
            r2x_peripheral.loc[lft_scaling, pv_scaling] = calc_r2x(
                cyto_actual[:, [0, 3, 4, 5], :],
                cyto_reconstructed[:, [0, 3, 4, 5], :]
            )

            lft_reconstructed = cp.to_CPTensor("LFT Measurements").to_tensor()
            lft_actual = cp.data["LFT Measurements"].to_numpy()
            r2x_lfts.loc[lft_scaling, pv_scaling] = calc_r2x(
                lft_actual,
                lft_reconstructed
            )
            r2x_averaged.loc[lft_scaling, pv_scaling] = np.mean(
                [
                    r2x_peripheral.loc[lft_scaling, pv_scaling],
                    r2x_pv.loc[lft_scaling, pv_scaling],
                    r2x_lfts.loc[lft_scaling, pv_scaling]
                ]
            )

    heatmaps = [
        r2x_total,
        r2x_averaged,
        r2x_peripheral,
        r2x_pv,
        r2x_lfts
    ]
    names = [
        "Total",
        "Averaged",
        "Peripheral",
        "PV",
        "LFTs"
    ]

    r2x_total.to_csv("total.csv")
    r2x_averaged.to_csv("averaged.csv")
    r2x_peripheral.to_csv("peripheral.csv")
    r2x_pv.to_csv("pv.csv")
    r2x_lfts.to_csv("lfts.csv")

    for ax, name, heatmap in zip(axs[1:], names, heatmaps):
        sns.heatmap(
            heatmap,
            cmap='Greens',
            vmin=0,
            vmax=1,
            ax=ax,
            cbar=ax == axs[2]
        )
        ax.set_xticks(np.arange(0.5, heatmap.shape[0]))
        ax.set_yticks(np.arange(0.5, heatmap.shape[1]))
        ax.set_xticklabels(np.arange(-5, 6, 1))
        ax.set_yticklabels(np.arange(-5, 6, 1), rotation=0)
        ax.invert_yaxis()
        ax.set_ylabel('LFT Scaling')
        ax.set_xlabel('PV Scaling')
        ax.set_title(name)

    return fig