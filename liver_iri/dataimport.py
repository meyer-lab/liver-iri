from os.path import abspath, dirname, join

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import zscore
from sklearn.preprocessing import power_transform

REPO_PATH = dirname(dirname(abspath(__file__)))
VISIT_TYPES = ["PO", "PV", "LF", "D1", "W1", "M1"]


def transform_data(data: pd.DataFrame, transform: str = "power"):
    """
    Applies transform to provided data.

    Args:
        data (pd.DataFrame): data to transform
        transform (str, default:'log'): transform to apply

    Returns:
        pd.DataFrame: transformed version of provided data
    """
    if (not isinstance(transform, str)) or (
        transform.lower() not in ["log", "power", "reciprocal"]
    ):
        raise ValueError(
            '"transform" parameter must be "log", "power", or "reciprocal"'
        )
    transform = transform.lower()

    if transform == "power":
        data[:] = power_transform(data)
    elif transform == "log":
        data[:] = np.log(data + 1)
    elif transform == "reciprocal":
        data[:] = np.reciprocal(data)

    return data


# noinspection PyArgumentList
def cytokine_data(
    transform: str = "power",
    peripheral_scaling: float = 1,
    pv_scaling: float = 1,
    no_missing: bool = True,
    normalize: bool = True,
):
    """
    Import cytokine data into tensor form.

    Parameters:
        transform (str, default:'power'): specifies transformation to use
        peripheral_scaling (float, default:1): scaling to apply to peripheral
            measurements
        pv_scaling (float, default:1): scaling to apply to PV measurements
        no_missing (bool, default:True): return only patients with all
            measurements at all time-points
        normalize (bool, default:True): z-score measurements

    Returns:
        xarray.Dataset: cytokine data in tensor form
    """
    if no_missing:
        df = pd.read_csv(
            join(REPO_PATH, "liver_iri", "data", "cytokines_no_missing.csv"),
            index_col=0,
        )
    else:
        df = pd.read_csv(
            join(REPO_PATH, "liver_iri", "data", "cytokines_validation.csv"),
            index_col=0,
        )

    df = df.drop(["IL-3", "MIP-1a"], axis=1)

    meta = df.loc[:, :"plate"]
    df = df.loc[:, "EGF":]

    data = xr.DataArray(
        coords={
            "Patient": meta["PID"].unique(),
            "Cytokine Timepoint": VISIT_TYPES,
            "Cytokine": df.columns,
        },
        dims=["Patient", "Cytokine Timepoint", "Cytokine"],
    )

    if transform is not None:
        df[:] = transform_data(df, transform)

    if normalize:
        df[:] = zscore(df, axis=1, nan_policy="omit")
        df.loc[meta.loc[:, "visit"].isin(["PV", "LF"]), :] = (
            zscore(
                df.loc[meta.loc[:, "visit"].isin(["PV", "LF"]), :],
                axis=0,
                nan_policy="omit",
            )
            * pv_scaling
        )
        df.loc[meta.loc[:, "visit"].isin(["PO", "D1", "W1", "M1"]), :] = (
            zscore(
                df.loc[meta.loc[:, "visit"].isin(["PO", "D1", "W1", "M1"]), :],
                axis=0,
                nan_policy="omit",
            )
            * peripheral_scaling
        )

    for index in meta.index:
        meta_row = meta.loc[index, :]
        data.loc[meta_row["PID"], meta_row["visit"], :] = df.loc[index, :]

    return data.to_dataset(name="Cytokine Measurements")


def lft_data(
    transform: str = "power", no_missing: bool = True, normalize: bool = True
):
    """
    Import LFT data into tensor form.

    Parameters:
        transform (str, default:'power'): specifies transformation to use
        no_missing (bool, default:True): return only patients with all
            measurements at all time-points
        normalize (bool, default:True): z-score measurements

    Returns:
        xarray.Dataset: RNA expression data in tensor form
    """
    if no_missing:
        lfts = pd.read_csv(
            join(REPO_PATH, "liver_iri", "data", "lft_scores_no_missing.csv"),
            index_col=0,
        )
    else:
        lfts = pd.read_csv(
            join(REPO_PATH, "liver_iri", "data", "lft_scores_validation.csv"),
            index_col=0,
        )

    lfts = lfts.loc[:, ~lfts.columns.str.contains("inr")]
    if transform is not None:
        lfts.loc[:, ~lfts.columns.str.contains("inr")] = transform_data(
            lfts.loc[:, ~lfts.columns.str.contains("inr")], transform
        )

    lfts.index = lfts.index.astype(int)
    scores = ["ast", "alt", "tbil"]

    patients = lfts.index.values
    if normalize:
        lfts[:] = zscore(lfts, nan_policy="omit", axis=1)
        lfts[:] = zscore(lfts, nan_policy="omit", axis=0)

    data = xr.DataArray(
        coords={
            "Patient": patients,
            "LFT Timepoint": ["Opening"] + [str(i) for i in range(1, 8)],
            "LFT Score": scores,
        },
        dims=["Patient", "LFT Timepoint", "LFT Score"],
    )

    for score in scores:
        data.loc[:, :, score] = lfts.loc[
            patients, lfts.columns.str.contains(score)
        ]

    return data.to_dataset(name="LFT Measurements")


def build_coupled_tensors(
    transform: str = "power",
    peripheral_scaling: float | int = 1e2,
    pv_scaling: float | int = 1,
    lft_scaling: float | int = 1,
    no_missing: bool = True,
    normalize: bool = True,
):
    """
    Builds datasets and couples across shared patient dimension.

    Parameters:
        transform (str, default:'power'): specifies transformation to use
        peripheral_scaling (float, default: 1): peripheral cytokine scaling
        pv_scaling (float, default: 1): PV cytokine scaling
        lft_scaling (float, default: 1): LFT scaling
        no_missing (bool, default:True): return only patients with all
            measurements at all time-points
        normalize (bool, default:True): z-score measurements

    Returns:
        xr.Dataset: coupled datasets merged into one object
    """
    tensors = [
        cytokine_data(
            peripheral_scaling=peripheral_scaling,
            pv_scaling=pv_scaling,
            no_missing=no_missing,
            normalize=normalize,
            transform=transform,
        ),
        lft_data(
            no_missing=no_missing, normalize=normalize, transform=transform
        )
        * lft_scaling,
    ]
    coupled = xr.merge(tensors)

    return coupled


def import_meta(long_survival: bool = True, no_missing: bool = True):
    """
    Imports patient meta-data.

    Parameters:
        long_survival (bool, default: True): removes recent patients with
            unknown outcomes
        no_missing (bool, default:True): return only patients with all
            measurements at all time-points

    Returns:
        pandas.DataFrame: patient meta-data
    """
    if no_missing:
        data = pd.read_csv(
            join(REPO_PATH, "liver_iri", "data", "patient_meta_no_missing.csv"),
            index_col=0,
        )
        data.index = data.index.astype(int)
    else:
        data = pd.read_csv(
            join(REPO_PATH, "liver_iri", "data", "patient_meta_validation.csv"),
            index_col=0,
        )
        data.index = data.index.astype(int)

    if long_survival:
        rejection = data.loc[data.loc[:, "graft_death"].astype(bool), :]
        survived = data.loc[~data.loc[:, "graft_death"].astype(bool), :]
        survived = survived.loc[survived.loc[:, "survival_time"] > 1e3, :]
        data = pd.concat([survived, rejection], axis=0)

    return data
