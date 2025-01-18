"""Plots Figure S9 -- tPLS 2 v. CTF 1"""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from ..dataimport import build_coupled_tensors, import_meta
from ..predict import oversample, run_coupled_tpls_classification
from ..tensor import convert_to_numpy, run_coupled
from .common import getSetup

warnings.filterwarnings("ignore")


def makeFigure():
    ############################################################################
    # Data imports
    ############################################################################

    meta = import_meta()
    labels = meta.loc[:, "graft_death"]
    labels = labels.dropna()

    all_meta = import_meta(no_missing=True, long_survival=False)
    all_labels = all_meta.loc[:, "graft_death"]
    all_labels = all_labels.dropna()

    tpls_data = build_coupled_tensors()
    cp_data = build_coupled_tensors(
        peripheral_scaling=1, pv_scaling=1, lft_scaling=1, no_missing=True
    )
    raw_data = build_coupled_tensors(
        pv_scaling=1,
        lft_scaling=1,
        no_missing=True,
        normalize=False,
        transform="log",
    )

    ############################################################################
    # Factorization
    ############################################################################

    _, cp = run_coupled(cp_data, rank=4)
    tensors, labels = convert_to_numpy(tpls_data, labels)
    cp_tensors, all_labels = convert_to_numpy(tpls_data, all_labels)
    oversampled_tensors, oversampled_labels = oversample(tensors, labels)

    (tpls, lr_model), tpls_acc, tpls_proba = run_coupled_tpls_classification(
        tensors, labels, return_proba=True
    )
    tpls.fit(oversampled_tensors, oversampled_labels.values)

    ctf_factors = {
        "Patient": cp.x["_Patient"].to_pandas().iloc[:, 0],
        "Cytokine": cp.x["_Cytokine"].to_pandas().iloc[:, 0],
        "Cytokine Timepoint": cp.x["_Cytokine Timepoint"]
        .to_pandas()
        .iloc[:, 0],
        "LFT": cp.x["_LFT Score"].to_pandas().iloc[:, 0],
        "LFT Timepoint": cp.x["_LFT Timepoint"].to_pandas().iloc[:, 0],
    }
    tpls_factors = {
        "Patient": pd.Series(
            tpls.transform(cp_tensors)[:, 1],  # type: ignore
            index=all_labels.index,
        ),
        "Cytokine": tpls.Xs_factors[0][2][:, 1],
        "Cytokine Timepoint": tpls.Xs_factors[0][1][:, 1],
        "LFT": tpls.Xs_factors[1][2][:, 1],
        "LFT Timepoint": tpls.Xs_factors[1][1][:, 1],
    }

    ############################################################################
    # Figure setup
    ############################################################################

    axs, fig = getSetup((9, 9), {"nrows": 3, "ncols": 3})

    ticks = np.arange(-1, 1.1, 0.5)

    ############################################################################
    # Patient factor
    ############################################################################

    tpls_factor = tpls_factors["Patient"]
    ctf_factor = ctf_factors["Patient"]

    graft_death = all_labels.replace({0: "tab:green", 1: "tab:red"})
    component_association = pd.Series("grey", index=all_labels.index, dtype=str)
    component_association.loc[
        tpls_factor.sort_values(ascending=False).index[:40]
    ] = "tab:cyan"
    component_association.loc[
        ctf_factor.sort_values(ascending=True).index[:40]
    ] = "tab:blue"
    component_association.loc[
        list(
            set(ctf_factor.sort_values(ascending=True).index[:40])
            & set(tpls_factor.sort_values(ascending=False).index[:40])
        )
    ] = "black"

    tpls_factor /= abs(tpls_factor).max()
    ctf_factor /= abs(ctf_factor).max()
    factor_diff = tpls_factor + ctf_factor
    factor_diff = factor_diff.sort_values(ascending=True)

    for ax, color in zip(axs[:2], [graft_death, component_association]):
        ax.scatter(tpls_factor, ctf_factor, c=color, edgecolors="black")
        ax.plot([-1.1, 1.1], [1.1, -1.1], linestyle="--", color="black")

        ax.set_xlabel("tPLS")
        ax.set_ylabel("CTF")

        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])

    ############################################################################
    # Cytokine / LFTs plotting
    ############################################################################

    for ax, dimension in zip(axs[2:4], ["Cytokine", "LFT"]):
        tpls_factor = tpls_factors[dimension]
        ctf_factor = ctf_factors[dimension]
        tpls_factor /= abs(tpls_factor).max()
        ctf_factor /= abs(ctf_factor).max()

        ax.scatter(tpls_factor, ctf_factor)
        ax.plot([-1.1, 1.1], [-1.1, 1.1], linestyle="--", color="black")
        for index, val in enumerate(ctf_factor.index):
            ax.text(
                tpls_factor[index],
                ctf_factor.loc[val],
                s=val,
                ha="center",
                ma="center",
                va="center",
            )

        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])

        ax.set_xlabel("tPLS")
        ax.set_ylabel("CTF")
        ax.set_title(dimension)

    ############################################################################
    # Timepoint plotting
    ############################################################################

    for ax, dimension in zip(axs[4:], ["Cytokine Timepoint", "LFT Timepoint"]):
        tpls_factor = tpls_factors[dimension]
        ctf_factor = ctf_factors[dimension]
        tpls_factor /= abs(tpls_factor).max()
        ctf_factor /= abs(ctf_factor).max()

        if dimension == "Cytokine Timepoint":
            tpls_factor[[1, 2]] /= abs(tpls_factor[[1, 2]]).max()

        ax.plot([-10, 10], [0, 0], linestyle="--", color="black")
        ax.plot(
            np.arange(len(tpls_factor)),
            tpls_factor,
            label="tPLS",
            color="tab:cyan",
        )
        ax.plot(
            np.arange(len(ctf_factor)),
            ctf_factor,
            label="CTF",
            color="tab:blue",
        )

        ax.set_xticks(np.arange(len(tpls_factor)))
        ax.set_yticks(ticks)
        ax.set_xlim([-0.1, len(tpls_factor) - 0.9])
        ax.set_ylim([-1.1, 1.1])

        ax.legend()
        ax.set_title(dimension)

    ############################################################################
    # Eotaxin, EGF, and TGFa plotting
    ############################################################################

    cytokine_measurements = raw_data["Cytokine Measurements"]
    high_tpls = factor_diff.index[-30:]
    high_ctf = factor_diff.index[:30]

    for cytokine, ax in zip(["TGFa", "EGF", "Eotaxin"], axs[6:]):
        cytokine_df = (
            cytokine_measurements.loc[{"Cytokine": cytokine}]
            .squeeze()
            .to_pandas()
        )
        timepoints = list(cytokine_df.columns)
        for index, tp in enumerate(cytokine_df.columns):
            low_patch = ax.boxplot(
                cytokine_df.loc[high_ctf, tp].dropna(),
                patch_artist=True,
                positions=[index * 3],
                notch=True,
                vert=True,
                widths=0.9,
                flierprops={"markersize": 6, "markerfacecolor": "tab:blue"},
            )
            high_patch = ax.boxplot(
                cytokine_df.loc[high_tpls, tp].dropna(),
                patch_artist=True,
                positions=[index * 3 + 1],
                notch=True,
                vert=True,
                widths=0.9,
                flierprops={"markersize": 6, "markerfacecolor": "tab:orange"},
            )
            low_patch["boxes"][0].set_facecolor("tab:blue")
            high_patch["boxes"][0].set_facecolor("tab:orange")
            result = ttest_ind(
                cytokine_df.loc[high_ctf, tp].dropna(),
                cytokine_df.loc[high_tpls, tp].dropna(),
            )
            if result.pvalue < 0.01:
                timepoints[index] = timepoints[index] + "**"
            elif result.pvalue < 0.05:
                timepoints[index] = timepoints[index] + "*"

        ax.set_xticks(np.arange(0.5, 6 * 3, 3))
        ax.set_xticklabels(timepoints)

        ax.set_xlim([-1, 3 * 6 - 1])
        ax.set_ylabel("Cytokine Expression")

        ax.set_title(cytokine)

    return fig
