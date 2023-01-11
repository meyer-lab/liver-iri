import numpy as np
from tensorpack.coupled import CoupledTensor

from liver_iri.dataimport import build_coupled_tensor

OPTIMAL_RANK = 6


def get_factors(data=None, rank=OPTIMAL_RANK, nonneg=False):
    """
    Runs coupled CP and returns factor matrices.

    Parameters:
        data (xr.Dataset, default: None): coupled tensors to factorize; if
            'None', builds default coupled tensor (see
            dataimport.build_coupled_tensor)
        rank (int, default: OPTIMAL_RANK): tensor factorization rank
        nonneg (bool, default: False): runs non-negative factorization

    Returns:
        factors (pd.DataFrame): patient factors for provided data
        decomposer (CoupledTensor): decomposition object
    """
    np.random.seed(215)
    rank = int(rank)

    if data is None:
        data = build_coupled_tensor()

    if nonneg:
        decomposer = CoupledTensor(
            data,
            rank
        )
        decomposer.initialize(method='svd')
    else:
        decomposer = CoupledTensor(
            data,
            rank
        )
        decomposer.initialize(method='randomized_svd')

    decomposer.fit(nonneg=nonneg, tol=1E-6, maxiter=2500)
    decomposer.normalize_factors()
    factors = decomposer.x._Patient.to_pandas()

    return factors, decomposer
