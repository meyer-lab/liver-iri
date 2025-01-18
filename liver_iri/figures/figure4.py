"""Plots Figure 4: Clinical Correlates"""

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.preprocessing import LabelEncoder

from ..dataimport import build_coupled_tensors, import_meta
from ..tensor import run_coupled
from .common import getSetup, plot_scatter

CTF_CORRELATES = {
    "Correlation": {
        1: ["dage", "dri"],
        2: ["dalt", "cit"],
    },
    "T-Test": {
        2: ["psens"],
        3: ["etoh", "hbv", "postinf", "postcong", "precong"],
        4: ["preinf", "preiri", "postcong", "precong"],
    },
}


def makeFigure():
    ############################################################################
    # Data import
    ############################################################################

    data = build_coupled_tensors(
        peripheral_scaling=1, pv_scaling=1, lft_scaling=1, no_missing=True
    )

    ############################################################################
    # Factorization
    ############################################################################

    _, cp = run_coupled(data)
    patient_factor = cp.x["_Patient"].to_pandas()
    meta = import_meta(no_missing=True, long_survival=False)
    meta = meta.loc[patient_factor.index, :]

    ############################################################################
    # Figure setup
    ############################################################################

    n_axes = 0
    for assoc_dict in CTF_CORRELATES.values():
        for variables in assoc_dict.values():
            n_axes += len(variables)

    ax_index = 0
    n_cols = int(np.ceil(n_axes / 3))
    axs, fig = getSetup((n_cols * 3, 9), {"nrows": 3, "ncols": n_cols})

    ############################################################################
    # Correlation
    ############################################################################

    for component, correlates in CTF_CORRELATES["Correlation"].items():
        for correlate in correlates:
            ax = axs[ax_index]
            concatenated = pd.concat(
                [
                    patient_factor.loc[:, component],
                    np.log(meta.loc[:, correlate]),
                ],
                axis=1,
            )
            concatenated = concatenated.dropna(axis=0)
            concatenated.columns = [f"Component {component}", correlate]

            plot_scatter(concatenated, ax)
            ax_index += 1

    ############################################################################
    # Differences
    ############################################################################

    le = LabelEncoder()
    for component, variables in CTF_CORRELATES["T-Test"].items():
        for variable in variables:
            ax = axs[ax_index]
            meta_col = meta.loc[:, variable]
            meta_col = meta_col.dropna()

            if meta_col.max() > 1:
                meta_col = meta_col > 1

            meta_col[:] = le.fit_transform(meta_col)

            _patient_factor = patient_factor.loc[meta_col.index, :]
            _patient_factor.loc[:, "group"] = meta_col

            result = ttest_ind(
                _patient_factor.loc[meta_col == 0, component],
                _patient_factor.loc[meta_col == 1, component],
            )

            ax.errorbar(
                0,
                _patient_factor.loc[meta_col == 0, component].mean(),
                yerr=_patient_factor.loc[meta_col == 0, component].std(),
                capsize=10,
                markersize=20,
                linewidth=3,
                markeredgewidth=3,
                marker="_",
                color="k",
                zorder=10,
            )
            ax.errorbar(
                1,
                _patient_factor.loc[meta_col == 1, component].mean(),
                yerr=_patient_factor.loc[meta_col == 1, component].std(),
                capsize=10,
                markersize=20,
                linewidth=3,
                markeredgewidth=3,
                marker="_",
                color="k",
                zorder=10,
            )
            sns.swarmplot(_patient_factor, x="group", y=component, ax=ax)
            ax.text(
                0.99,
                0.01,
                s=f"p-value: {str(round(result.pvalue, 4))}",
                ha="right",
                ma="right",
                va="bottom",
                transform=ax.transAxes,
            )

            ax.set_ylabel(f"Component {component}")
            ax.set_xlabel(variable)
            ax.set_xticklabels(["-", "+"])
            ax_index += 1

    return fig
