"""Plots Figure 5 -- tPLS Model Interpretation"""
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from ..dataimport import build_coupled_tensors, import_meta
from ..predict import oversample, run_coupled_tpls_classification
from ..tensor import convert_to_numpy
from ..utils import reorder_table
from .common import getSetup

warnings.filterwarnings("ignore")

LFT_TIMEPOINTS = ["Opening", "1", "2", "3", "4", "5", "6", "7"]
LFT_NAMES = ["ALT", "AST", "INR", "TBIL"]
LFT_CONVERSIONS = {i: f"Day {i}" for i in LFT_TIMEPOINTS[1:]}
LFT_CONVERSIONS["Opening"] = "Pre-Op"
TIMEPOINTS = ["PO", "D1", "W1", "M1"]
TP_CONVERSIONS = {
    "PO": "Pre-Op",
    "D1": "1 Day Post-Op",
    "W1": "1 Week Post-Op",
    "M1": "1 Month Post-Op",
    "PV": "Pre-Op PV",
    "LF": "Post-Op PV",
}


def makeFigure():
    ############################################################################
    # Figure setup
    ############################################################################

    axs, fig = getSetup(
        (6, 9),
        {
            'ncols': 2,
            'nrows': 3
        }
    )

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

    tensors, labels = convert_to_numpy(data, labels)
    oversampled_tensors, oversampled_labels = oversample(tensors, labels)
    all_tensors, all_labels = convert_to_numpy(all_data, all_labels)

    (tpls, lr_model), acc, predicted = run_coupled_tpls_classification(
        tensors, labels
    )
    tpls.fit(oversampled_tensors, oversampled_labels.values)

    # Figure 5A: Patient factors

    ax = axs[0]
    factor = tpls.transform(all_tensors)
    patient_factors = pd.DataFrame(
        factor,
        index=all_labels.index,
        columns=np.arange(1, factor.shape[1] + 1)
    )
    patient_factors = patient_factors.loc[all_labels.index, :]

    xx, yy = np.meshgrid(
        np.linspace(
            patient_factors.loc[:, 1].min() - 1,
            patient_factors.loc[:, 1].max() + 1,
            100
        ),
        np.linspace(
            patient_factors.loc[:, 2].min() - 1,
            patient_factors.loc[:, 2].max() + 1,
            100
        )
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = lr_model.predict_proba(grid)[:, 0].reshape(xx.shape)

    cs = ax.contourf(xx, yy, probs, 11, cmap='RdBu', alpha=0.75)
    fig.colorbar(cs)
    ax.plot(
        [
            patient_factors.loc[:, 1].min() - 1,
            patient_factors.loc[:, 1].max() + 1
        ],
        [0, 0],
        linestyle='--',
        color='k'
    )
    ax.plot(
        [0, 0],
        [
            patient_factors.loc[:, 2].min() - 1,
            patient_factors.loc[:, 2].max() + 1
        ],
        linestyle='--',
        color='k'
    )
    ax.scatter(
        patient_factors.loc[all_labels == 0, 1],
        patient_factors.loc[all_labels == 0, 2],
        c='blue',
        edgecolor='black',
        alpha=0.75,
        label='No Transplant Rejection'
    )
    ax.scatter(
        patient_factors.loc[all_labels == 1, 1],
        patient_factors.loc[all_labels == 1, 2],
        c='red',
        edgecolor='black',
        alpha=0.75,
        label='Transplant Rejection'
    )

    ax.legend()
    ax.set_xticklabels([int(tick) for tick in ax.get_xticks()])
    ax.set_yticklabels([int(tick) for tick in ax.get_yticks()])
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_title('Patient Factors')

    ############################################################################
    # CYTOKINE FACTORS
    ############################################################################

    ax = axs[2]

    cyto_factors = pd.DataFrame(
        tpls.Xs_factors[0][2],
        index=data.Cytokine.values,
        columns=[1, 2]
    )
    cyto_factors /= abs(cyto_factors).max(axis=0)
    ax.set_title('Cytokine Factors')

    ax.plot([-2, 2], [0, 0], linestyle='--', color='k')
    ax.plot([0, 0], [-2, 2], linestyle='--', color='k')
    colors = []
    for cyto in cyto_factors.index:
        if cyto_factors.loc[cyto, 1] > 0.75 > cyto_factors.loc[cyto, 2]:
            colors.append('#1f77b4')
        elif cyto_factors.loc[cyto, 2] > 0.75:
            colors.append('#ff7f0e')
        else:
            colors.append('grey')

    ax.scatter(
        cyto_factors.loc[:, 1],
        cyto_factors.loc[:, 2],
        c=colors,
        edgecolor='k'
    )

    xticks = [round(i, 1) for i in np.arange(-1, 1.1, 0.5)]
    yticks = [round(i, 1) for i in np.arange(-1, 1.1, 0.5)]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)

    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])

    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')

    for cyto in cyto_factors.index:
        ax.text(
            cyto_factors.loc[cyto, 1],
            cyto_factors.loc[cyto, 2],
            s=cyto,
            ha='center',
            va='center'
        )

    ax = axs[3]

    time_factors = pd.DataFrame(
        tpls.Xs_factors[0][1],
        index=data['Cytokine Timepoint'].values,
        columns=[1, 2]
    )
    time_factors /= abs(time_factors).max(axis=0)

    ax.plot(
        [-0.1, time_factors.shape[0] - 0.9],
        [0, 0],
        linestyle='--',
        color='k'
    )
    ax.plot(
        range(time_factors.shape[0]),
        time_factors.loc[:, 1],
        label='Component 1'
    )
    ax.plot(
        range(time_factors.shape[0]),
        time_factors.loc[:, 2],
        label='Component 2'
    )

    yticks = [round(i, 1) for i in np.arange(-1, 1.1, 0.5)]
    ax.set_xticks(range(time_factors.shape[0]))
    ax.set_xticklabels(data['Cytokine Timepoint'].values)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)

    ax.set_xlim([-0.1, time_factors.shape[0] - 0.9])
    ax.set_ylim([-1.1, 1.1])

    ax.legend()
    ax.set_ylabel('Component Assocation')
    ax.set_title('Cytokine Time Factors')

    ############################################################################
    # LFT FACTORS
    ############################################################################

    ax = axs[4]

    lft_factors = pd.DataFrame(
        tpls.Xs_factors[1][2],
        index=data['LFT Score'].values,
        columns=[1, 2]
    )
    lft_factors /= abs(lft_factors).max(axis=0)

    ax.plot(
        [-0.6, 6.6],
        [0, 0],
        linestyle='--',
        color='k'
    )
    ax.bar(
        [0, 4],
        lft_factors.loc['tbil', :],
        width=1,
        label='TBIL'
    )
    ax.bar(
        [1, 5],
        lft_factors.loc['alt', :],
        width=1,
        label='ALT'
    )
    ax.bar(
        [2, 6],
        lft_factors.loc['ast', :],
        width=1,
        label='AST'
    )
    ax.legend()

    yticks = [round(i, 2) for i in np.arange(0, 1.05, 0.2)]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)
    ax.set_ylabel('Component Association')
    ax.set_xticks([1, 5])
    ax.set_xticklabels(['Component 1', 'Component 2'])

    ax.set_title('LFT Factors')
    ax.set_xlim([-0.6, 6.6])

    ax = axs[5]

    time_factors = pd.DataFrame(
        tpls.Xs_factors[1][1],
        index=data['LFT Timepoint'].values,
        columns=[1, 2]
    )
    time_factors /= abs(time_factors).max(axis=0)
    index = list(time_factors.index)
    index[1:] = ['Day ' + i for i in index[1:]]
    time_factors.index = index

    ax.plot([-0.1, 7.1], [0, 0], color='k', linestyle='--')
    for col in time_factors.columns:
        ax.plot(
            range(time_factors.shape[0]),
            time_factors.loc[:, col],
            label=f'Component {col}'
        )

    ax.legend()
    ax.set_xticks(range(time_factors.shape[0]))
    ax.set_xticklabels(
        time_factors.index,
        rotation=45,
        ha='right',
        va='top',
        ma='right'
    )
    yticks = [round(i, 2) for i in np.arange(0, 1.05, 0.2)]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)
    ax.set_ylabel('Component Association')

    ax.set_ylim([-0.1, 1.1])
    ax.set_xlim([-0.2, 7.2])
    ax.set_title('LFT Factor Timepoints')

    return fig
