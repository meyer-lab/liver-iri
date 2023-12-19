"""Plots Figure 6 -- Clinical Correlations"""
import warnings

from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, spearmanr
from sklearn.preprocessing import LabelEncoder

from .common import getSetup
from ..dataimport import import_meta

warnings.filterwarnings("ignore")

TO_CORRELATE = [
    "abox",
    "aih",
    "cit",
    "dage",
    "dalt",
    "dbmi",
    "drace",
    "dsx",
    "dtbili",
    "etoh",
    "hbv",
    "hcv",
    "liri",
    "pbc",
    "postchol",
    "postinf",
    "postnec",
    "poststeat",
    "psc",
    "rag",
    "rrc",
    "rsx",
    "txmeld",
    "wit",
]
# r = rec, b = donor, g = etio, c = clinical
COLORS = pd.Series(
    {
        "Compatibility": "c",
        "Autoimmune Hepatitis": "g",
        "Cold\nIschemia Time": "c",
        "Donor Age": "b",
        "Donor ALT": "b",
        "Donor BMI": "b",
        "Donor Race": "b",
        "Donor Sex": "b",
        "Donor TBIL": "b",
        "Donor Weight": "b",
        "Alcoholic Hepatitis": "g",
        "Hepatitis B": "g",
        "Hepatitis C": "g",
        "LIRI Score": "c",
        "Primary Biliary Cholangitis": "g",
        "Cholesterol": "c",
        "Inflammation": "c",
        "Necrotization": "c",
        "Steatosis": "c",
        "Primary Sclerosing Cholangitis": "g",
        "Recipient Age": "r",
        "Race": "r",
        "Recipient Sex": "r",
        "Transplant\nMELD Score": "c",
        "Warm\nIschemia Time": "c",
    }
)
CONVERSIONS = {
    "abox": "Compatibility",
    "aih": "Autoimmune Hepatitis",
    "cit": "Cold\nIschemia Time",
    "dage": "Donor Age",
    "dalt": "Donor ALT",
    "dbmi": "Donor BMI",
    "drace": "Donor Race",
    "dsx": "Donor Sex",
    "dtbili": "Donor TBIL",
    "etoh": "Alcoholic Hepatitis",
    "hbv": "Hepatitis B",
    "hcv": "Hepatitis C",
    "liri": "LIRI Score",
    "pbc": "Primary Biliary Cholangitis",
    "postchol": "Cholesterol",
    "postinf": "Inflammation",
    "postnec": "Necrotization",
    "poststeat": "Steatosis",
    "psc": "Primary Sclerosing Cholangitis",
    "rag": "Recipient Age",
    "rrc": "Race",
    "rsx": "Recipient Sex",
    "txmeld": "Transplant\nMELD Score",
    "wit": "Warm\nIschemia Time",
}


def get_correlations(graft, meta):
    le = LabelEncoder()
    p_values = pd.Series(0, index=TO_CORRELATE)

    for name in TO_CORRELATE:
        var = meta.loc[:, name]
        var = var.dropna()
        if name == "poststeat":
            print()

        if (var.dtype != int) & (var.dtype != float):
            var[:] = le.fit_transform(var)
            labels = graft.loc[var.index]
            table = np.zeros((2, 2))
            table[0, 0] = (var.loc[labels == 0] == 0).sum()
            table[0, 1] = (var.loc[labels == 1] == 0).sum()
            table[1, 0] = (var.loc[labels == 0] == 1).sum()
            table[1, 1] = (var.loc[labels == 1] == 1).sum()
            _, p_val = fisher_exact(table)
        elif var.max() > 1:
            _, p_val = spearmanr(var, graft.loc[var.index])
        else:
            labels = graft.loc[var.index]
            table = np.zeros((2, 2))
            table[0, 0] = (var.loc[labels == 0] == 0).sum()
            table[0, 1] = (var.loc[labels == 1] == 0).sum()
            table[1, 0] = (var.loc[labels == 0] == 1).sum()
            table[1, 1] = (var.loc[labels == 1] == 1).sum()
            _, p_val = fisher_exact(table)

        p_values.loc[name] = p_val

    return p_values.sort_values(ascending=True)


def plot_correlations(p_values, ax, x_label=None, y_label=None):
    ax.plot([-10, 100], [0.05, 0.05], color="k", linestyle="--", alpha=0.5)
    ax.bar(range(len(p_values)), p_values, color=COLORS.loc[p_values.index])
    for index, p_value in enumerate(p_values):
        ax.text(
            index,
            p_value + 0.01,
            s=round(p_value, 2),
            ha="center",
            va="bottom",
            fontsize=6,
        )

    ax.legend(
        [
            Patch(facecolor="b"),
            Patch(facecolor="r"),
            Patch(facecolor="g"),
            Patch(facecolor="c")

        ],
        [
            "Donor Characteristics",
            "Recipient Characteristics",
            "Etiologies",
            "Clinical Measurements"
        ],
        loc="upper left",
    )
    ax.set_xticks(range(len(p_values)))
    ax.set_xticklabels(
        p_values.index,
        ha="right",
        ma="right",
        va="top",
        rotation=45,
    )
    ax.set_xlim([-0.5, len(p_values) - 0.5])
    ax.set_ylim([0, 1.1])

    if x_label is not None:
        ax.set_xlabel(x_label)

    if y_label is not None:
        ax.set_ylabel(y_label)


def makeFigure():
    meta = import_meta()
    graft = meta.loc[:, "graft_death"]

    p_values = get_correlations(graft, meta)
    p_values.index = [CONVERSIONS.get(i, i) for i in p_values.index]

    fig_size = (4.5, 3)
    layout = {"nrows": 1, "ncols": 1}
    axs, fig = getSetup(fig_size, layout)

    plot_correlations(p_values, axs[0], y_label="Correlation p-value")

    return fig
