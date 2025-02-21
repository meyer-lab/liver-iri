"""Plots Figure 6d -- tPLS Correlates"""

import numpy as np
import pandas as pd
import xarray as xr

from ..dataimport import build_coupled_tensors, import_meta
from ..predict import oversample, run_coupled_tpls_classification
from ..tensor import convert_to_numpy
from .common import getSetup, plot_scatter

CORRELATES = {
    "wit": {"name": "Warm Ischemia Time", "component": 1},
    "dage": {"name": "Donor Age", "component": 2},
    "dri": {"name": "Donor Risk Index", "component": 2},
}


def makeFigure():
    ############################################################################
    # Data imports
    ############################################################################

    meta = import_meta()
    labels = meta.loc[:, "graft_death"]

    val_meta = import_meta(no_missing=False, long_survival=False)
    val_labels = val_meta.loc[:, "graft_death"]

    data = build_coupled_tensors()
    val_data = build_coupled_tensors(no_missing=False)

    all_data = xr.merge([data, val_data])
    all_labels = pd.Series(pd.concat([labels, val_labels]))
    all_tensors, all_labels = convert_to_numpy(all_data, all_labels)

    ############################################################################
    # Figure setup
    ############################################################################

    axs, fig = getSetup((9, 3), {"nrows": 1, "ncols": 3})

    ############################################################################
    # Factorization
    ############################################################################

    tensors, labels = convert_to_numpy(data, labels)
    oversampled_tensors, oversampled_labels = oversample(tensors, labels)

    (tpls, lr_model), tpls_acc, tpls_proba = run_coupled_tpls_classification(
        tensors, labels, return_proba=True
    )
    tpls.fit(oversampled_tensors, oversampled_labels.values)

    ############################################################################
    # tPLS patients
    ############################################################################

    factor = tpls.transform(all_tensors)
    patient_factor = pd.DataFrame(
        factor,
        index=all_labels.index,
        columns=np.arange(1, factor.shape[1] + 1),
    )
    patient_factor = patient_factor.loc[all_labels.index, :]
    patient_factor /= abs(patient_factor).max(axis=0)

    meta = pd.concat([meta, val_meta])
    meta = meta.loc[patient_factor.index, :]

    ############################################################################
    # Correlation plots
    ############################################################################

    for ax_index, (abbr, sub_dict) in enumerate(CORRELATES.items()):
        ax = axs[ax_index]
        meta_name = sub_dict["name"]
        comp = sub_dict["component"]

        meta_col = meta.loc[:, abbr]
        meta_col = meta_col.dropna()
        df = patient_factor.loc[meta_col.index, comp].to_frame()
        df.loc[:, meta_name] = meta_col
        df.columns = [f"Component {comp}", meta_name]

        if abbr == "cit":
            df = df.loc[df.loc[:, meta_name] < 1000, :]

        plot_scatter(df, ax)

    return fig
