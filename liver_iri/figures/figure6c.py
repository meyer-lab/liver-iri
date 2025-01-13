"""Plots Figure 6bc-- tPLS 2 Heatmap"""
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

from ..dataimport import build_coupled_tensors, import_meta
from ..predict import oversample, run_coupled_tpls_classification
from ..tensor import convert_to_numpy
from ..utils import reorder_table
from .common import getSetup

warnings.filterwarnings("ignore")


def makeFigure():
    ############################################################################
    # Data imports
    ############################################################################

    meta = import_meta()
    labels = meta.loc[:, "graft_death"]
    labels = labels.dropna()

    val_meta = import_meta(no_missing=False)
    val_labels = val_meta.loc[:, "graft_death"]
    val_labels = val_labels.dropna()

    data = build_coupled_tensors()
    val_data = build_coupled_tensors(no_missing=False)

    all_data = xr.merge([data, val_data])
    all_labels = pd.concat([labels, val_labels])
    all_tensors, all_labels = convert_to_numpy(all_data, all_labels)

    raw_data = build_coupled_tensors(
        lft_scaling=1,
        pv_scaling=1,
        transform="log",
        normalize=False
    )
    raw_val = build_coupled_tensors(
        no_missing=False,
        lft_scaling=1,
        pv_scaling=1,
        transform="log",
        normalize=False
    )
    raw_data = xr.merge([raw_data, raw_val])

    cytokine_measurements = raw_data["Cytokine Measurements"]

    tensors, labels = convert_to_numpy(data, labels)
    oversampled_tensors, oversampled_labels = oversample(tensors, labels)

    (tpls, lr_model), tpls_acc, tpls_proba = run_coupled_tpls_classification(
        tensors, labels, return_proba=True
    )
    tpls.fit(oversampled_tensors, oversampled_labels.values)

    ############################################################################
    # tPLS Patients
    ############################################################################

    factor = tpls.transform(all_tensors)
    patient_factors = pd.DataFrame(
        factor,
        index=all_labels.index,
        columns=np.arange(1, factor.shape[1] + 1),
    )
    patient_factors = patient_factors.loc[all_labels.index, :]
    patient_factors /= abs(patient_factors).max(axis=0)

    patient_factors = patient_factors.sort_values(2, ascending=False)
    high_2, low_2 = patient_factors.index[:30], patient_factors.index[-30:]

    axs, fig = getSetup(
        (6, 3),
        {"ncols": 1, "nrows": 1}
    )
    ax = axs[0]

    high_mean = cytokine_measurements.sel(
        {
            "Patient": high_2,
        }
    ).mean("Patient").to_pandas()
    low_mean = cytokine_measurements.sel(
        {
            "Patient": low_2,
        }
    ).mean("Patient").to_pandas()
    diff = high_mean - low_mean
    diff = reorder_table(diff.T).T

    sns.heatmap(
        diff,
        center=0,
        ax=ax,
        cmap="coolwarm",
        square=True,
        linewidths=0.1
    )
    ax.set_yticklabels(diff.index)
    ax.set_xticks(np.arange(0.5, diff.shape[1]))
    ax.set_xticklabels(diff.columns, rotation=90)

    return fig
