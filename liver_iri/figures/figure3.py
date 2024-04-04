"""Plots Figure 3 -- Component Associations to Clinical Characteristics"""
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, scale
from sklearn.utils import resample

from ..dataimport import build_coupled_tensors, import_meta
from ..predict import (predict_categorical, predict_continuous,
                       run_coupled_tpls_classification)
from ..tensor import convert_to_numpy
from .common import getSetup

warnings.filterwarnings("ignore")

CATEGORICAL = ["dsx", "rsx", "rrc", "abox", "iri", "graft_death"]
REGRESSION = ["dage", "rag", "dtbili", "dalt", "cit", "wit", "txmeld"]
TRANSLATIONS = {
    "dsx": "Donor Sex",
    "dage": "Donor Age",
    "dtbili": "Donor TBIL",
    "dalt": "Donor ALT",
    "abox": "ABO\nCompatibility",
    "cit": "Cold\nIschemia Time",
    "wit": "Warm\nIschemia Time",
    "txmeld": "Transplant\nMELD Score",
    "rrc": "Race",
    "iri": "LIRI > 1",
    "graft_death": "Graft Death",
    "rsx": "Recipient Sex",
    "rag": "Recipient Age",
    "liri": "LIRI",
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
        labels[:] = encoder.fit_transform(labels)
        labels = labels.astype(int)

        data = factors.loc[labels.index, :]
        score, model, _ = predict_categorical(
            data, labels
        )
        models[target] = model

        if target != "graft_death":
            accuracies.loc[target] = score

    for target in REGRESSION:
        labels = meta.loc[:, target]
        labels = labels.dropna()

        data = factors.loc[labels.index, :]
        score, model = predict_continuous(data, labels)
        q2ys.loc[target] = score
        models[target] = model

    return accuracies, q2ys, models


def plot_accuracies(accuracies, ax, y_label=None):
    accuracies = accuracies.sort_values(ascending=True)
    ax.bar(range(len(accuracies)), accuracies)
    for index, accuracy in enumerate(accuracies):
        ax.text(
            index,
            accuracy + 0.01,
            s=round(accuracy, 3),
            ha="center",
            va="bottom",
        )

    ticks = [TRANSLATIONS[metric] for metric in accuracies.index]
    ax.set_xticks(range(len(accuracies)))
    ax.set_xticklabels(ticks, rotation=45, va="top", ha="right")

    ax.set_ylim([0, 1])
    ax.set_xlim([-0.5, len(accuracies) - 0.5])

    if y_label is not None:
        ax.set_ylabel(y_label)


def bootstrap_weights(factors, meta, models):
    feature_weights = pd.DataFrame(
        0, index=factors.columns, columns=["graft_death", "iri"]
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
        means.loc[:, "iri"],
        np.arange(0, 6 * means.shape[0], 6),
        xerr=devs.loc[:, "iri"],
        linestyle="",
        marker=".",
        capsize=4,
    )
    ax.errorbar(
        means.loc[:, "graft_death"],
        np.arange(1, 6 * means.shape[0], 6),
        xerr=devs.loc[:, "graft_death"],
        linestyle="",
        marker=".",
        capsize=4,
    )
    ax.legend(["LIRI", "Graft Death"])

    ax.set_yticks(np.arange(0.5, 6 * means.shape[0], 6))
    ax.set_yticklabels(np.arange(1, means.shape[0] + 1))
    ax.set_ylabel("Component")
    ax.set_xlabel("Feature Weight")

    ax.plot([0, 0], [-1, 6 * means.shape[0] - 3], linestyle="--", color="k")
    ax.set_ylim([-2, 6 * means.shape[0] - 3])


def makeFigure():
    meta = import_meta()
    labels = meta.loc[:, "graft_death"]
    labels = labels.dropna()

    data = build_coupled_tensors()
    tensors, labels = convert_to_numpy(data, labels)

    (tpls, _), _, _ = run_coupled_tpls_classification(tensors, labels)
    factors = pd.DataFrame(
        tpls.Xs_factors[0][0],
        index=labels.index,
        columns=np.arange(tpls.n_components) + 1,
    )

    accuracies, q2ys, models = get_accuracies(factors, meta)
    weight_means, weight_devs = bootstrap_weights(factors, meta, models)

    fig_size = (9, 3)
    layout = {"nrows": 1, "ncols": 3}
    axs, fig = getSetup(fig_size, layout)

    plot_accuracies(accuracies, axs[0], "Balanced Accuracy")
    plot_accuracies(q2ys, axs[1], "Q2Y")
    plot_errorbars(weight_means, weight_devs, axs[2])

    return fig
