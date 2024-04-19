"""Plots Figure 6 -- Survival Analyses"""
import warnings

from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
import xarray as xr

from ..dataimport import build_coupled_tensors, import_meta
from ..predict import oversample, run_survival, run_tpls_survival
from ..tensor import convert_to_numpy
from .common import getSetup

warnings.filterwarnings("ignore")


def makeFigure():
    ############################################################################
    # Figure setup
    ############################################################################

    fig_size = (6, 4)
    layout = {"nrows": 2, "ncols": 3}
    axs, fig = getSetup(fig_size, layout)

    ############################################################################
    # Data imports
    ############################################################################

    meta = import_meta()
    val_meta = import_meta(no_missing=False, long_survival=False)
    labels = meta.loc[:, ["graft_death", "survival_time"]]
    val_labels = val_meta.loc[:, ["graft_death", "survival_time"]]

    data = build_coupled_tensors()
    val_data = build_coupled_tensors(no_missing=False)
    tensors, labels = convert_to_numpy(data, labels)
    val_tensors, val_labels = convert_to_numpy(val_data, val_labels)

    ############################################################################
    # Figure 6A: Cox-PH
    ############################################################################

    (tpls, cox_ph), c_index, cph_expected = run_tpls_survival(
        tensors, labels
    )
    oversampled_tensors, oversampled_labels = oversample(
        tensors, labels, column="graft_death"
    )
    tpls.fit(
        oversampled_tensors,
        oversampled_labels.loc[:, "graft_death"].values
    )

    cytokine_data = data["Cytokine Measurements"].stack(
        Flattened=["Cytokine", "Cytokine Timepoint"]
    ).to_pandas()
    lft_data = data["LFT Measurements"].stack(
        Flattened=["LFT Score", "LFT Timepoint"]
    ).to_pandas()

    cytokine_data.columns = np.arange(cytokine_data.shape[1])
    lft_data.columns = np.arange(lft_data.shape[1])

    pv_timepoints = []
    for i in np.arange(0, cytokine_data.shape[1], 6):
        pv_timepoints.extend([i + 1, i + 2])
    pv_cytokines = cytokine_data.iloc[:, pv_timepoints]
    peripheral_cytokines = cytokine_data.drop(pv_cytokines.columns, axis=1)

    _, pv_c_index, _ = run_survival(pv_cytokines, labels)
    _, peripheral_c_index, _ = run_survival(peripheral_cytokines, labels)
    _, lft_c_index, _ = run_survival(lft_data, labels)
    _, liri_c_index, _ = run_survival(meta.loc[:, "liri"].to_frame(), labels)

    ax = axs[0]

    ax.bar(
        np.arange(5),
        [lft_c_index, pv_c_index, peripheral_c_index, liri_c_index, c_index],
        width=1,
        color=["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red"],
        label=[
            "LFTs",
            "PV Cytokines",
            "Peripheral Cytokines",
            "Pathology Score",
            "tPLS"
        ]
    )
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(
        [
            "LFTs",
            "PV Cytokines",
            "Peripheral\nCytokines",
            "Pathology\nScore",
            "tPLS"
        ],
        ha="right", ma="right", va="top", rotation=45
    )

    ax.legend()
    ax.set_ylim([0, 1])

    ax.set_xlabel("Method")
    ax.set_ylabel("C-Index")

    ax = axs[1]

    ax.plot([0, 0], [-1, 10], linestyle="--", color="k")
    ax.errorbar(
        cox_ph.params_,
        np.arange(len(cox_ph.params_)),
        linestyle="",
        marker="o",
        capsize=5,
        xerr=cox_ph.standard_errors_ * 1.96
    )

    ax.set_yticks([0, 1])
    ax.set_ylim([-0.5, 1.5])

    ax.set_xlabel("Hazard Ratio")
    ax.set_ylabel("tPLS Component")

    ############################################################################
    # Figures 6C-D: Kaplan-Meier Curves
    ############################################################################

    merged_data = xr.merge([data, val_data])
    merged_labels = pd.concat([labels, val_labels])

    merged_tensors, merged_labels = convert_to_numpy(merged_data, merged_labels)
    components = pd.DataFrame(
        tpls.transform(merged_tensors),
        index=merged_labels.index,
        columns=np.arange(tpls.n_components) + 1
    )
    components.loc[:, "Sum"] = scale(components).sum(axis=1)
    threshold = int(components.shape[0] / 10)
    kmf = KaplanMeierFitter()

    for ax, column in zip(axs[3:], components.columns):
        components = components.sort_values(by=column, ascending=False)
        high_index = components.index[:threshold]
        low_index = components.index[threshold:]
        kmf.fit(
            merged_labels.loc[high_index, "survival_time"],
            merged_labels.loc[high_index, "graft_death"]
        )
        ax.plot(
            kmf.survival_function_.index,
            kmf.survival_function_.iloc[:, 0],
            label=f"High {column}",
            color="tab:blue"
        )
        max_index = kmf.survival_function_.index[-1]
        kmf.fit(
            merged_labels.loc[low_index, "survival_time"],
            merged_labels.loc[low_index, "graft_death"]
        )
        ax.plot(
            kmf.survival_function_.index,
            kmf.survival_function_.iloc[:, 0],
            label=f"Low {column}",
            color="tab:orange"
        )
        if kmf.survival_function_.index[-1] > max_index:
            max_index = kmf.survival_function_.index[-1]

        ax.set_ylim([0, 1])
        ax.set_xlim([0, max_index])

        ax.set_ylabel("Probability of\nNon-Rejection")
        ax.set_xlabel("Time")
        ax.legend()

    return fig
