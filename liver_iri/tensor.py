import numpy as np
import pandas as pd
from tensorpack import perform_CP
from tensorpack.coupled import CoupledTensor

from .dataimport import cytokine_data

OPTIMAL_COMPONENTS = 9


def get_factors(rank=9, nonneg=False):
    if nonneg:
        data = cytokine_data(
            None,
            log_scaling=False,
            uniform_lod=True,
            mean_center=False
        )
        decomposer = CoupledTensor(
            data,
            rank
        )
        decomposer.initialize(method='nmf')
    else:
        data = cytokine_data(
            None,
            log_scaling=True,
            uniform_lod=True,
            mean_center=True
        )
        decomposer = CoupledTensor(
            data,
            rank
        )
        decomposer.initialize(method='svd')

    decomposer.fit(nonneg=nonneg)
    factors = decomposer.x._Patient.to_pandas()

    return factors
