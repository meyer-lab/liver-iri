import numpy as np
from numpy.linalg import norm
import pandas as pd
import xarray as xr
from tensorpack import perform_CP
from tensorpack.coupled import CoupledTensor

from liver_iri.dataimport import build_coupled_tensors

OPTIMAL_RANK = 4


def calc_r2x(actual: np.ndarray, reconstructed: np.ndarray):
    """
    Calculates R2X of reconstruction.

    Parameters:
        actual (np.ndarray): actual tensor
        reconstructed (np.ndarray): reconstructed tensor

    Returns:
        R2X (float): reconstruction fidelity
    """
    mask = np.isfinite(actual)
    actual = np.nan_to_num(actual)
    top = norm(reconstructed * mask - actual) ** 2
    bottom = norm(actual) ** 2
    return 1 - top / bottom


def run_cp(data, rank=OPTIMAL_RANK):
    """
    Runs CP on provided data.

    Parameters:
        data (xr.Dataset, default: None): coupled tensors to factorize; if
            'None', builds default coupled tensor (see
            dataimport.build_coupled_tensors)
        rank (int, default: OPTIMAL_RANK): tensor factorization rank
        nonneg (bool, default: False): runs non-negative factorization

    Returns:
        factors (pd.DataFrame): patient factors for provided data
        decomposer (CoupledTensor): decomposition object
    """
    if isinstance(data, xr.Dataset):
        data = data.to_array().squeeze()
        data = data.to_numpy()
    elif isinstance(data, xr.DataArray):
        data = data.to_numpy()
    elif not isinstance(data, np.ndarray):
        raise TypeError("Unrecognized data format provided")

    cp = perform_CP(data, rank)
    return cp


def run_coupled(data=None, rank=OPTIMAL_RANK, nonneg=False):
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
        data = build_coupled_tensors()

    if nonneg:
        decomposer = CoupledTensor(data, rank)
        decomposer.initialize(method="svd")
    else:
        decomposer = CoupledTensor(data, rank)
        decomposer.initialize(method="randomized_svd")

    decomposer.fit(nonneg=nonneg, tol=1e-6, maxiter=2500, progress=False)
    decomposer.normalize_factors()
    factors = decomposer.x._Patient.to_pandas()

    return factors, decomposer


def cp_impute(data: xr.Dataset, labels: pd.Series, rank: int = 2):
    """Imputes missing values via coupled CP decomposition."""
    _, cp = run_coupled(data, rank)
    reconstructed = cp.reconstruct()
    tensors = []
    for var in data.data_vars:
        tensor = data[var].sel(Patient=labels.index).to_numpy()
        imputed = reconstructed[var].sel(Patient=labels.index).to_numpy()
        mask = np.isnan(tensor)
        tensor[mask] = imputed[mask]
        tensors.append(tensor)

    return tensors


def convert_to_numpy(
    data: xr.Dataset, labels: pd.Series, impute_method: str = None
):
    """Converts xr.Dataset to tPLS-compatible numpy arrays."""
    if impute_method not in ["cp", "drop", "zero", None]:
        raise ValueError('impute_method must be one of "cp", "drop", or "zero"')

    shared_patients = sorted(list(set(data.Patient.values) & set(labels.index)))
    data = data.sel(Patient=shared_patients)
    labels = labels.loc[shared_patients]

    if impute_method == "drop":
        tensors = [data[var].to_numpy() for var in data.data_vars]
        patients_all = np.array(
            [np.isfinite(tensor).any(axis=1).any(axis=1) for tensor in tensors]
        ).all(axis=0)
        data = data.sel(Patient=data.Patient.values[patients_all])
        labels = labels.loc[data.Patient.values]
        tensors = [
            data[var].sel(Patient=labels.index).to_numpy()
            for var in data.data_vars
        ]
    elif impute_method == "zero":
        tensors = [
            data[var].sel(Patient=labels.index).to_numpy()
            for var in data.data_vars
        ]
        for index, tensor in enumerate(tensors):
            all_missing = np.isnan(tensor).all(axis=1).all(axis=1)
            tensors[index][all_missing, :, :] = 0
    elif impute_method == "cp":
        tensors = cp_impute(data, labels)
    else:
        tensors = [
            data[var].sel(Patient=labels.index).to_numpy()
            for var in data.data_vars
        ]

    return tensors, labels
