import warnings

from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, spearmanr
from sklearn.preprocessing import LabelEncoder

from .common import getSetup
from ..dataimport import cytokine_data, import_meta
from ..predict import predict_categorical
from ..utils import reorder_table

warnings.filterwarnings('ignore')

TO_CORRELATE = [
    'age', 'abocompt', 'postrepiri', 'postinf', 'postnec', 'postchol',
    'poststeat', 'etoh', 'hbv', 'hcv', 'aih', 'pbc', 'psc', 'dage', 'dbmi',
    'dweight'
]
COLORS = pd.Series(
    {
        'Age': 'r',
        'LIRI Score': 'b',
        'Compatibility': 'r',
        'Inflammation': 'b',
        'Necrotization': 'b',
        'Cholesterol': 'b',
        'Steatosis': 'b',
        'Alcoholic Hepatitis': 'c',
        'Hepatitis B': 'c',
        'Hepatitis C': 'c',
        'Autoimmune Hepatitis': 'c',
        'Primary Biliary Cholangitis': 'c',
        'Primary Sclerosing Cholangitis': 'c',
        'Donor Age': 'g',
        'Donor BMI': 'g',
        'Donor Weight': 'g'
    }
)
CONVERSIONS = {
    'age': 'Age',
    'postrepiri': 'LIRI Score',
    'abocompt': 'Compatibility',
    'postinf': 'Inflammation',
    'postnec': 'Necrotization',
    'postchol': 'Cholesterol',
    'poststeat': 'Steatosis',
    'etoh': 'Alcoholic Hepatitis',
    'hbv': 'Hepatitis B',
    'hcv': 'Hepatitis C',
    'aih': 'Autoimmune Hepatitis',
    'pbc': 'Primary Biliary Cholangitis',
    'psc': 'Primary Sclerosing Cholangitis',
    'dage': 'Donor Age',
    'dbmi': 'Donor BMI',
    'dweight': 'Donor Weight'
}


def get_correlations(graft, meta):
    le = LabelEncoder()
    p_values = pd.Series(
        0,
        index=TO_CORRELATE
    )

    for name in TO_CORRELATE:
        var = meta.loc[:, name]
        var = var.dropna()
        if name == 'poststeat':
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
            _, p_val = spearmanr(
                var,
                graft.loc[var.index]
            )
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
    ax.plot([-10, 100], [0.05, 0.05], color='k', linestyle='--', alpha=0.5)
    ax.bar(
        range(len(p_values)),
        p_values,
        color=COLORS.loc[p_values.index]
    )
    for index, p_value in enumerate(p_values):
        ax.text(
            index,
            p_value + 0.01,
            s=round(p_value, 2),
            ha='center',
            va='bottom'
        )

    ax.legend(
        [
            Patch(facecolor='b'),
            Patch(facecolor='g'),
            Patch(facecolor='c'),
            Patch(facecolor='r')
        ],
        [
            'Post-Op Measurements',
            'Donor Characteristics',
            'Etiologies',
            'Patient Characteristics'
        ],
        loc='upper left'
    )
    ax.set_xticks(range(len(p_values)))
    ax.set_xticklabels(
        p_values.index,
        ha='right',
        ma='right',
        va='top',
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
    graft = meta.loc[:, 'graft_death']

    p_values = get_correlations(
        graft,
        meta
    )
    p_values.index = [CONVERSIONS.get(i, i) for i in p_values.index]

    fig_size = (4.5, 3)
    layout = {'nrows': 1, 'ncols': 1}
    axs, fig = getSetup(
        fig_size,
        layout
    )

    plot_correlations(
        p_values,
        axs[0],
        y_label='Correlation p-value'
    )
