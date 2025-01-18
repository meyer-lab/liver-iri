"""Plots Figure S3 -- FMS Tuning"""

import numpy as np
import pandas as pd
import xarray as xr
from statsmodels.multivariate.pca import PCA
from tlviz.factor_tools import factor_match_score as fms
from tqdm import tqdm

from ..dataimport import build_coupled_tensors
from ..tensor import run_coupled
from .common import getSetup

RANKS = 5
N_TRIALS = 10
RNG = np.random.default_rng(215)


def resample(data: xr.Dataset):
    patients = data["Patient"].values
    sampled_patients = RNG.choice(patients, size=len(patients))

    return data.loc[{"Patient": sampled_patients}]


def makeFigure():
    ############################################################################
    # Data imports
    ############################################################################

    data = build_coupled_tensors(pv_scaling=1, lft_scaling=1, no_missing=True)

    ############################################################################
    # Figure setup
    ############################################################################

    axs, fig = getSetup((8, 4), {"nrows": 1, "ncols": 3})

    ############################################################################
    # Base factors
    ############################################################################

    lft_scores = pd.DataFrame(
        index=np.arange(N_TRIALS) + 1, columns=np.arange(RANKS) + 1, dtype=float
    )
    cyto_scores = lft_scores.copy(deep=True)
    r2xs = pd.DataFrame(
        index=np.array(["CTF"]), columns=np.arange(RANKS) + 1, dtype=float
    )

    ############################################################################
    # Generate resampled factors
    ############################################################################

    for rank in tqdm(np.arange(RANKS) + 1):
        _, cp = run_coupled(data, rank=rank)
        lft_cp = cp.to_CPTensor(dvar="LFT Measurements")
        cyto_cp = cp.to_CPTensor(dvar="Cytokine Measurements")

        r2xs.loc["CTF", rank] = cp.R2X()

        for trial in np.arange(N_TRIALS) + 1:
            resampled_data = resample(data)
            _, resampled_cp = run_coupled(resampled_data, rank=rank)
            lft_resampled = resampled_cp.to_CPTensor(dvar="LFT Measurements")
            cyto_resampled = resampled_cp.to_CPTensor(
                dvar="Cytokine Measurements"
            )
            lft_score = fms(
                lft_cp, lft_resampled, consider_weights=False, skip_mode=0
            )
            cyto_score = fms(
                cyto_cp, cyto_resampled, consider_weights=False, skip_mode=0
            )

            lft_scores.loc[trial, rank] = lft_score
            cyto_scores.loc[trial, rank] = cyto_score

    ############################################################################
    # Plot FMS
    ############################################################################

    for ax, score_df in zip(axs, [lft_scores, cyto_scores]):
        mean = score_df.mean(axis=0)
        dev = score_df.std(axis=0)
        ax.fill_between(dev.index, mean - dev, mean + dev, alpha=0.25)
        ax.plot(mean.index, mean)
        ax.set_xticks(np.arange(1, 6, 1))
        ax.set_yticks(np.arange(0, 1.25, 0.25))
        ax.set_ylim([0, 1])
        ax.set_xlabel("Rank")
        ax.set_ylabel("FMS")

    axs[0].set_title("LFTs")
    axs[1].set_title("Cytokines")

    ############################################################################
    # R2X plots
    ############################################################################

    flattened = (
        data["Cytokine Measurements"]
        .stack(merged=("Cytokine", "Cytokine Timepoint"))
        .to_pandas()
    )
    lfts = (
        data["LFT Measurements"]
        .stack(merged=("LFT Score", "LFT Timepoint"))
        .to_pandas()
    )
    flattened = pd.concat([flattened, lfts], axis=1)  # type: ignore

    pca = PCA(flattened, missing="fill-em", ncomp=5)
    assert pca.rsquare is not None
    r2xs.loc["PCA", :] = pca.rsquare.iloc[1:].values
    ax = axs[2]
    for line in r2xs.index:
        ax.plot(np.arange(r2xs.shape[1]) + 1, r2xs.loc[line, :], label=line)

    ax.set_xticks(np.arange(1, 6, 1))
    ax.set_yticks(np.arange(0, 0.6, 0.1))
    ax.set_xlabel("Rank")
    ax.set_ylabel("R2X")
    ax.legend()

    return fig
