"""Plots Figure 3S -- FMS"""
import numpy as np
import pandas as pd
from tlviz.factor_tools import factor_match_score as fms
from tqdm import tqdm
import xarray as xr

from .common import getSetup
from ..dataimport import build_coupled_tensors
from ..tensor import run_coupled

RANKS = 5
N_TRIALS = 10
RNG = np.random.default_rng(215)


def resample(data: xr.Dataset):
    patients = data["Patient"].values
    sampled_patients = RNG.choice(patients, size=len(patients))

    return data.loc[
        {
            "Patient": sampled_patients
        }
    ]


def makeFigure():
    ############################################################################
    # Data imports
    ############################################################################

    data = build_coupled_tensors(
        pv_scaling=1,
        lft_scaling=1,
        no_missing=True
    )

    ############################################################################
    # Factorization
    ############################################################################

    axs, fig = getSetup(
        (8, 4),
        {"nrows": 1, "ncols": 2}
    )

    lft_scores = pd.DataFrame(
        index=np.arange(N_TRIALS) + 1,
        columns=np.arange(RANKS) + 1,
        dtype=float
    )
    cyto_scores = lft_scores.copy(deep=True)
    for rank in tqdm(np.arange(RANKS) + 1):
        _, cp = run_coupled(data, rank=rank)
        lft_cp = cp.to_CPTensor(dvar="LFT Measurements")
        cyto_cp = cp.to_CPTensor(dvar="Cytokine Measurements")
        for trial in np.arange(N_TRIALS) + 1:
            resampled_data = resample(data)
            _, resampled_cp = run_coupled(resampled_data, rank=rank)
            lft_resampled = resampled_cp.to_CPTensor(dvar="LFT Measurements")
            cyto_resampled = resampled_cp.to_CPTensor(dvar="Cytokine Measurements")
            lft_score = fms(lft_cp, lft_resampled, consider_weights=False)
            cyto_score = fms(cyto_cp, cyto_resampled, consider_weights=False)

            lft_scores.loc[trial, rank] = lft_score
            cyto_scores.loc[trial, rank] = cyto_score

    for ax, score_df in zip(axs, [lft_scores, cyto_scores]):
        mean = score_df.mean(axis=0)
        dev = score_df.std(axis=0)
        ax.fill_between(
            dev.index,
            mean - dev,
            mean + dev,
            alpha=0.25
        )
        ax.plot(
            mean.index,
            mean
        )
        ax.set_xlabel("Rank")
        ax.set_ylabel("FMS")

    axs[0].set_title("LFTs")
    axs[1].set_title("Cytokines")

    lft_scores.to_csv("lft_fms.csv")
    cyto_scores.to_csv("cyto_fms.csv")

    return fig
