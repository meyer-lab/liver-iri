"""Plots Figure 3 -- Component Associations to Clinical Characteristics"""
import warnings

import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder, scale

from .common import getSetup
from ..dataimport import import_meta
from ..tensor import run_coupled
from ..predict import predict_categorical, predict_continuous

warnings.filterwarnings('ignore')

CATEGORICAL = ['gender', 'rrc', 'abo', 'liri', 'death']
REGRESSION = ['age', 'postrepiri']
TRANSLATIONS = {
    'dsx': 'Donor Sex',
    'dag': 'Donor Age',
    'dtbili': 'Donor TBIL',
    'dalt': 'Donor ALT',
    'abox': 'ABO Compatibility',
    'cit': 'Cold Ischemia Time',
    'wit': 'Warm Ischemia Time',
    'txmeld': 'Transplant MELD Score',
    'rrc': 'Race',
    'iri': 'LIRI > 1',
    'graft_death': 'Graft Death',
    'rsx': 'Recipient Sex',
    'rag': 'Recipient Age',
    'liri': 'LIRI'
}
N_BOOTSTRAP = 30


def get_accuracies(factors, meta):
    encoder = LabelEncoder()
    accuracies = pd.Series()
    q2ys = pd.Series()
    models = {}

    for target in CATEGORICAL:
        labels = meta.loc[:, target]
        labels = labels.dropna()

        encoded = encoder.fit_transform(labels)
        data = factors.loc[labels.index, :]
        score, model = predict_categorical(data, encoded, oversample=False)
        accuracies.loc[target] = score
        models[target] = model

    for target in REGRESSION:
        labels = meta.loc[:, target]
        labels = labels.dropna()

        data = factors.loc[labels.index, :]
        score, model = predict_continuous(data, labels)
        q2ys.loc[target] = score
        models[target] = model

    return accuracies, q2ys, models


def plot_accuracies(accuracies, ax, y_label=None):
    ax.bar(
        range(len(accuracies)),
        accuracies
    )
    for index, accuracy in enumerate(accuracies):
        ax.text(
            index,
            accuracy + 0.01,
            s=round(accuracy, 4),
            ha='center',
            va='bottom'
        )

    ticks = [TRANSLATIONS[metric] for metric in accuracies.index]
    ax.set_xticks(range(len(accuracies)))
    ax.set_xticklabels(ticks, rotation=45, va='top', ha='right')

    ax.set_ylim([0, 1])
    ax.set_xlim([-0.5, len(accuracies) - 0.5])

    if y_label is not None:
        ax.set_ylabel(y_label)


def bootstrap_weights(factors, meta, models):
    feature_weights = pd.DataFrame(
        0,
        index=factors.columns,
        columns=['gender', 'liri']
    )
    feature_devs = feature_weights.copy()
    encoder = LabelEncoder()
    for target in feature_weights.columns:
        scores = np.zeros((N_BOOTSTRAP, factors.shape[1]))
        encoder.fit(meta.loc[:, target].dropna())
        for trial in range(N_BOOTSTRAP):
            labels = meta.loc[:, target]
            labels = labels.dropna()
            encoded = encoder.transform(labels)
            data = factors.loc[labels.index, :]

            _data, _labels = resample(data, encoded)
            models[target].fit(_data, _labels)
            scores[trial, :] = models[target].coef_

        scores = scale(scores, axis=1)
        feature_weights.loc[:, target] = scores.mean(axis=0)
        feature_devs.loc[:, target] = scores.std(axis=0)

    return feature_weights, feature_devs


def plot_errorbars(means, devs, ax):
    ax.errorbar(
        means.loc[:, 'liri'],
        np.arange(0, 6 * means.shape[0], 6),
        xerr=devs.loc[:, 'liri'],
        linestyle='',
        marker='.',
        capsize=4
    )
    ax.errorbar(
        means.loc[:, 'gender'],
        np.arange(1, 6 * means.shape[0], 6),
        xerr=devs.loc[:, 'gender'],
        linestyle='',
        marker='.',
        capsize=4
    )
    ax.legend(['LIRI', 'Gender'])

    ax.set_yticks(np.arange(0.5, 6 * means.shape[0], 6))
    ax.set_yticklabels(np.arange(1, means.shape[0] + 1))
    ax.set_ylabel('Component')
    ax.set_xlabel('Feature Weight')

    ax.plot([0, 0], [-1, 6 * means.shape[0] - 3], linestyle='--', color='k')
    ax.set_ylim([-2, 6 * means.shape[0] - 3])


def makeFigure():
    factors, _ = run_coupled()
    meta = import_meta()

    accuracies, q2ys, models = get_accuracies(factors, meta)
    weight_means, weight_devs = bootstrap_weights(factors, meta, models)

    fig_size = (9, 3)
    layout = {'nrows': 1, 'ncols': 3}
    axs, fig = getSetup(
        fig_size,
        layout
    )

    plot_accuracies(accuracies, axs[0], 'Balanced Accuracy')
    plot_accuracies(q2ys, axs[1], 'Q2Y')
    plot_errorbars(weight_means, weight_devs, axs[2])

    return fig
