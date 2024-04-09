"""Plots Figure 3 -- CP Factor Interpretation"""
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

from .common import getSetup
from ..dataimport import build_coupled_tensors, import_meta
from ..tensor import run_coupled
from ..utils import reorder_table


def makeFigure():
    ############################################################################
    # Data imports
    ############################################################################

    meta = import_meta()
    coupled = build_coupled_tensors(
    )
    coupled = coupled.sel(Patient=meta.index)

    ############################################################################
    # Factorization
    ############################################################################

    _, cp = run_coupled(coupled, rank=4)
    factors = {}
    for mode in cp.modes:
        if 'Timepoint' not in mode:
            factors[mode] = cp.x[f'_{mode}'].to_pandas()

    ############################################################################
    # Patient meta-data heatmap
    ############################################################################

    labels = meta.loc[
        :,
        [
            'liri', 'graft_death', 'etiol'
        ]
    ]
    labels.loc[:, 'liri'] = labels.loc[:, 'liri'].fillna(-1)
    labels = labels.loc[cp.x.Patient.values, :]

    le = LabelEncoder()
    labels.loc[:, 'etiology'] = le.fit_transform(labels.loc[:, 'etiol'])
    labels = labels.sort_values(by=['graft_death', 'liri', 'etiol'])

    liri_cmap = LinearSegmentedColormap.from_list(
        'Custom',
        ['grey', 'green', 'red'],
        3
    )
    colors = [
        'orange',
        'yellow',
        'greenyellow',
        'blue',
        'indigo',
        'purple',
        'violet',
        'hotpink',
        'pink',
        'lightcoral'
    ]
    etio_cmap = LinearSegmentedColormap.from_list(
        'Custom',
        colors
    )
    graft_cmap = LinearSegmentedColormap.from_list(
        'Custom',
        ['darkgreen', 'darkred']
    )

    legend_elements = [
        Patch(facecolor='darkred'),
        Patch(facecolor='darkgreen'),
        Patch(facecolor='white'),
        Patch(facecolor='grey'),
        Patch(facecolor='red'),
        Patch(facecolor='green'),
        Patch(facecolor='white')
    ]
    legend_names = [
        'Transplant\nRejection',
        'No Transplant\nRejection',
        '',
        'LIRI Unknown',
        'High LIRI',
        'Low LIRI',
        '',
    ]
    for i in range(len(le.classes_)):
        legend_elements.append(
            Patch(facecolor=colors[i])
        )
        legend_names.append(le.classes_[i])

    ############################################################################
    # Figure setup
    ############################################################################

    width_ratios = [
        cp.rank * 0.9,
        1,
        1,
        1,
        cp.rank,
        cp.rank * 0.6,
        cp.rank,
        cp.rank * 0.3,
        cp.rank,
        cp.rank * 0.25
    ]

    axs, fig = getSetup(
        (len(factors) * 3, 3),
        {
            'nrows': 1,
            'ncols': len(width_ratios),
            'width_ratios': width_ratios
        }
    )

    spacers = [0, 1, 2, 3] + list(range(5, len(width_ratios), 2))
    for spacer in sorted(spacers, reverse=True):
        axs[spacer].set_frame_on(False)
        axs[spacer].set_xticks([])
        axs[spacer].set_yticks([])

    ############################################################################
    # Meta-data heatmaps
    ############################################################################

    sns.heatmap(
        np.expand_dims(labels.loc[:, 'etiology'].values, 1),
        cmap=etio_cmap,
        cbar=False,
        ax=axs[1],
    )
    sns.heatmap(
        np.expand_dims(labels.loc[:, 'liri'].values, 1),
        cmap=liri_cmap,
        cbar=False,
        ax=axs[2],
    )
    sns.heatmap(
        np.expand_dims(labels.loc[:, 'graft_death'].values, 1),
        cmap=graft_cmap,
        cbar=False,
        ax=axs[3],
    )

    axs[0].legend(legend_elements, legend_names, loc='center left')
    axs[1].set_xticklabels(['Etiology'], va='top', ha='right',
                           rotation=45)
    axs[2].set_xticklabels(['LIRI Score'], va='top', ha='right',
                           rotation=45)
    axs[3].set_xticklabels(['Graft Rejection'], va='top', ha='right',
                           rotation=45)
    axs[1].set_yticks([])
    axs[1].set_ylabel('')
    axs[2].set_yticks([])
    axs[2].set_ylabel('')
    axs[3].set_yticks([])
    axs[3].set_ylabel('')

    ############################################################################
    # Factor heatmaps
    ############################################################################

    for ax, name in zip(axs[list(range(4, len(axs), 2))], factors.keys()):
        data = factors[name]

        if name not in ['Cytokine Timepoint', 'PV Timepoint']:
            data /= abs(data).max()
        else:
            data /= abs(factors['PV Timepoint']).max()

        data = data.fillna(0)

        if name == 'Patient':
            data = data.loc[labels.index, :]
        elif 'Timepoint' not in name:
            data = reorder_table(data)

        cbar = False
        if ax == axs[-2]:
            cbar = True

        sns.heatmap(
            data,
            ax=ax,
            cmap='vlag',
            cbar=cbar,
            vmin=-1,
            vmax=1,
            center=0
        )
        if name in ['Patient', 'Gene']:
            ax.set_yticks([])
            if name == 'Gene':
                ax.set_ylabel(name)
            else:
                ax.set_ylabel('')
        else:
            ax.set_yticks(
                np.arange(0.5, data.shape[0])
            )
            ax.set_yticklabels(data.index)
            ax.set_ylabel(name)

        ax.set_xticks(np.arange(0.5, data.shape[1]))
        ax.set_xlabel('Components')

    return fig
