"""Plots Figure S4/5 -- Cytokine-Clinical Correlates"""

import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from scipy.stats import pearsonr, ttest_ind
from sklearn.preprocessing import LabelEncoder

from ..dataimport import build_coupled_tensors, import_meta
from .common import getSetup

COLORS = ["tab:blue", "tab:orange"]
CONTINUOUS = np.array(
    [
        "dage",
        "rag",
        "dbmi",
        "dtbili",
        "dalt",
        "cit",
        "wit",
        "listmeld",
        "txmeld",
        "liri",
        "dri",
    ]
)
CATEGORICAL = np.array(
    [
        "dsx",
        "rsx",
        "etoh",
        "hcv",
        "nash",
        "hcc",
        "psens",
        "esens",
        "lsens",
        "iri",
    ]
)
BINNED = np.array(
    [
        "clscores",
        "bnscores",
        "postinf",
        "postnec",
        "poststeat",
        "postcong",
        "postbal",
    ]
)


def makeFigure():
    ############################################################################
    # Data imports
    ############################################################################

    data = build_coupled_tensors(
        pv_scaling=1, lft_scaling=1, no_missing=True, normalize=False
    )
    val_data = build_coupled_tensors(
        pv_scaling=1, lft_scaling=1, no_missing=False, normalize=False
    )
    data = xr.merge([data, val_data])
    cytokines = data["Cytokine Measurements"]
    cytokines = cytokines.stack(Flattened=["Cytokine", "Cytokine Timepoint"])
    cytokines = cytokines.to_pandas()

    meta = import_meta(long_survival=False)
    val_meta = import_meta(no_missing=False, long_survival=False)
    meta = pd.concat([meta, val_meta])
    meta = meta.loc[cytokines.index]

    correlations = pd.DataFrame(
        index=CONTINUOUS, columns=cytokines.columns, dtype=float
    )
    corr_p = correlations.copy(deep=True)
    t_tests = pd.DataFrame(columns=cytokines.columns, dtype=float)
    t_p = t_tests.copy(deep=True)

    ############################################################################
    # Figure setup
    ############################################################################

    axs, fig = getSetup((8, 8), {"nrows": 4, "ncols": 3})

    ############################################################################
    # Pearson correlations
    ############################################################################

    for cytokine in cytokines.columns:
        for meta_var in CONTINUOUS:
            concatenated = pd.concat(
                [cytokines.loc[:, cytokine], meta.loc[:, meta_var]], axis=1
            )
            concatenated = concatenated.dropna(axis=0)
            result = pearsonr(concatenated.iloc[:, 0], concatenated.iloc[:, 1])
            correlations.loc[meta_var, cytokine] = result.statistic
            corr_p.loc[meta_var, cytokine] = result.pvalue

    ############################################################################
    # T-tests
    ############################################################################

    meta.loc[:, BINNED] = (meta.loc[:, BINNED] > 1).astype(int)
    CATEGORICAL.extend(BINNED)
    le = LabelEncoder()
    for meta_var in CATEGORICAL:
        meta_col = meta.loc[:, meta_var].dropna()
        meta_col.loc[:] = le.fit_transform(meta_col)
        meta_col = meta_col.astype(int)
        _cytokines = cytokines.loc[meta_col.index]

        result = ttest_ind(
            _cytokines.loc[meta_col == 1, :],
            _cytokines.loc[meta_col == 0, :],
            nan_policy="omit",
        )
        t_tests.loc[meta_var, :] = result.statistic
        t_p.loc[meta_var, :] = result.pvalue

    ############################################################################
    # Heatmap plotting
    ############################################################################

    for ax_row, (stat, label, p_values) in enumerate(
        zip(
            [t_tests, correlations],
            ["T-Test", "Pearson Correlation"],
            [t_p, corr_p],
        )
    ):
        v_max = abs(stat).max().max()
        for tp_index, tp in enumerate(t_tests.columns.unique(level=1)):
            tp_stat = stat.xs(tp, level=1, axis=1).T
            tp_p = p_values.xs(tp, level=1, axis=1).T
            ax = axs[ax_row * 6 + tp_index]

            annot = np.empty(tp_p.shape, dtype=np.dtype("U100"))
            annot[tp_p < 0.05] = "*"
            annot[tp_p < 0.01] = "**"

            sns.heatmap(
                tp_stat,
                annot=annot,
                cmap="coolwarm",
                vmin=-v_max,
                vmax=v_max,
                fmt="s",
                ax=ax,
                cbar=tp_index % 3 == 2,
                annot_kws={"ha": "center", "ma": "center", "va": "top"},
                cbar_kws={"label": label, "shrink": 8 / stat.shape[0]},
                xticklabels=tp_index > 2,
                yticklabels=tp_index % 3 == 0,
            )

            ax.set_ylabel("")

    return fig
