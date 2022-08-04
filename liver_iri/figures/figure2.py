import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .common import getSetup
from ..dataimport import import_meta
from ..predict import predict_categorical
from ..tensor import get_factors


def makeFigure():
    factor_count = np.arange(1, 16)
    accuracies = pd.Series(index=factor_count, dtype=float)

    meta = import_meta()
    labels = meta.loc[:, 'liri']
    labels = labels.dropna()

    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels)

    for n_factors in factor_count:
        factors = get_factors(n_factors)

        data = factors.loc[labels.index, :]
        score, _ = predict_categorical(data, encoded)
        accuracies.loc[n_factors] = score

    fig_size = (6, 3)
    layout = {'nrows': 1, 'ncols': 1}
    axs, fig = getSetup(
        fig_size,
        layout
    )
    ax = axs[0]

    ax.plot(factor_count, accuracies)

    ax.set_xticks(factor_count)
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('LIRI Prediction Accuracy')

    return fig
