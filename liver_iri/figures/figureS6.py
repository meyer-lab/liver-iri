"""Plots Figure S6 -- Clinical Correlates"""
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import fisher_exact, pearsonr, ttest_ind
from sklearn.preprocessing import LabelEncoder
import xarray as xr

from .common import getSetup
from ..dataimport import build_coupled_tensors, import_meta

COLORS = ["tab:blue", "tab:orange"]
CONTINUOUS = np.array([
    "dage", "rag", "dbmi", "dtbili", "dalt", "cit", "wit", "listmeld",
    "txmeld", "liri", "dri"
])
CATEGORICAL = np.array([
    "dsx", "rsx", "etoh", "hcv", "nash", "hcc", "psens", "esens", "lsens"
])
BINNED = np.array([
    "clscores", "bnscores", "postinf", "postnec", "poststeat",
    "postcong", "postbal"
])


def makeFigure():
    ############################################################################
    # Data imports
    ############################################################################

    meta = import_meta(long_survival=False)
    val_meta = import_meta(no_missing=False, long_survival=False)
    meta = pd.concat([meta, val_meta])
    meta.loc[:, BINNED] = (meta.loc[:, BINNED] > 1).astype(int)
    CATEGORICAL.extend(BINNED)

    ############################################################################
    # Pearson correlations
    ############################################################################

    correlations = pd.DataFrame(
        index=CONTINUOUS,
        columns=CONTINUOUS,
        dtype=float
    )
    corr_p = correlations.copy(deep=True)
    for index_1, variable_1 in enumerate(CONTINUOUS):
        for index_2, variable_2 in enumerate(CONTINUOUS[index_1 + 1:]):
            concatenated = pd.concat(
                [
                    meta.loc[:, variable_1],
                    meta.loc[:, variable_2]
                ],
                axis=1
            )
            concatenated = concatenated.dropna(axis=0)
            result = pearsonr(
                concatenated.iloc[:, 0],
                concatenated.iloc[:, 1]
            )
            correlations.loc[variable_1, variable_2] = result.statistic
            corr_p.loc[variable_1, variable_2] = result.pvalue

    ############################################################################
    # T-tests
    ############################################################################

    t_tests = pd.DataFrame(
        index=CONTINUOUS,
        columns=CATEGORICAL,
        dtype=float
    )
    t_p = t_tests.copy(deep=True)
    le = LabelEncoder()
    for index_1, variable_1 in enumerate(CONTINUOUS):
        for index_2, variable_2 in enumerate(CATEGORICAL):
            concatenated = pd.concat(
                [
                    meta.loc[:, variable_1],
                    meta.loc[:, variable_2]
                ],
                axis=1
            )
            concatenated = concatenated.dropna(axis=0)
            concatenated.iloc[:, 1] = le.fit_transform(concatenated.iloc[:, 1])
            result = ttest_ind(
                concatenated.loc[concatenated.iloc[:, 1] != 0, variable_1],
                concatenated.loc[concatenated.iloc[:, 1] == 0, variable_1],
                nan_policy="omit"
            )
            t_tests.loc[variable_1, variable_2] = result.statistic
            t_p.loc[variable_1, variable_2] = result.pvalue

    ############################################################################
    # Fisher's exact tests
    ############################################################################

    fe_tests = pd.DataFrame(
        index=CATEGORICAL,
        columns=CATEGORICAL,
        dtype=float
    )
    fe_p = fe_tests.copy(deep=True)
    le = LabelEncoder()
    for index_1, variable_1 in enumerate(CATEGORICAL):
        for index_2, variable_2 in enumerate(CATEGORICAL[index_1 + 1:]):
            concatenated = pd.concat(
                [
                    meta.loc[:, variable_1],
                    meta.loc[:, variable_2]
                ],
                axis=1
            )
            concatenated = concatenated.dropna(axis=0)
            for col in concatenated.columns:
                concatenated.loc[:, col] = le.fit_transform(
                    concatenated.loc[:, col]
                )

            table = np.zeros((2, 2))
            for row in [0, 1]:
                for col in [0, 1]:
                    table[row, col] = concatenated.loc[
                        np.logical_and(
                            concatenated.loc[:, variable_1] == row,
                            concatenated.loc[:, variable_2] == col
                        )
                    ].shape[0]

            result = fisher_exact(table)
            fe_tests.loc[variable_1, variable_2] = result.statistic
            fe_p.loc[variable_1, variable_2] = result.pvalue

    ############################################################################
    # Heatmap plotting
    ############################################################################

    axs, fig = getSetup(
        (8, 4),
        {"nrows": 1, "ncols": 3}
    )
    for ax, stat, label, p_values in zip(
        axs,
        [t_tests, fe_tests, correlations],
        ["T-Test", "Fisher Exact", "Pearson Correlation"],
        [t_p, fe_p, corr_p]
    ):
        stat = stat.T
        p_values = p_values.T
        v_max = min([abs(stat).max().max(), 15])
        annot = np.empty(stat.shape, dtype=np.dtype('U100'))
        annot[p_values < 0.05] = "*"
        annot[p_values < 0.01] = "**"

        sns.heatmap(
            stat,
            annot=annot,
            cmap="coolwarm",
            vmin=-v_max,
            vmax=v_max,
            fmt="s",
            ax=ax,
            cbar=True,
            linewidths=0.1,
            annot_kws={
                "ha": "center",
                "ma": "center",
                "va": "top"
            },
            cbar_kws={
                "label": label,
                "shrink": 8 / stat.shape[0]
            },
            xticklabels=True,
            yticklabels=True
        )

        ax.set_ylabel("")

    return fig
