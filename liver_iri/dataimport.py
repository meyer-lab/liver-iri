from os.path import abspath, dirname, join

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.preprocessing import power_transform, scale

REPO_PATH = dirname(dirname(abspath(__file__)))


def transform_data(data, transform="log"):
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
    plate_scale: bool = True,
    transform: str = "log",
    normalize: bool = True,
    peripheral_scaling: float = 1,
    pv_scaling: float = 2,
):
    """
    Import cytokine data into tensor form.

    Parameters:
        plate_scale (bool, default:False): normalizes each plate independently
        transform (str, default:'log'): specifies transformation to use
        normalize (bool, default:False): sets zero-mean, variance one
        peripheral_scaling (float, default:1): scaling to apply to peripheral
            measurements
        pv_scaling (float, default:1): scaling to apply to PV measurements

    Returns:
        xarray.Dataset: cytokine data in tensor form
    """
    df = pd.read_csv(
        join(REPO_PATH, "liver_iri", "data", "cytokines.csv"), index_col=0
    )
    df = df.drop(["IL-3", "MIP-1a"], axis=1)
    visit_types = df.loc[:, "visit"].unique()

    meta = df.loc[:, :"plate"]
    df = df.loc[:, "EGF":]

    data = xr.DataArray(
        coords={
            "Patient": meta["PID"].unique(),
            "Cytokine Timepoint": visit_types,
            "Cytokine": df.columns,
        },
        dims=["Patient", "Cytokine Timepoint", "Cytokine"],
    )

    if plate_scale:
        for group in meta.loc[:, "plate"].unique():
            group_cytokines = df.loc[meta.loc[:, "plate"] == group, :]
            col_min = np.min(group_cytokines.where(group_cytokines > 0), axis=0)
            group_cytokines[:] = np.clip(
                group_cytokines, col_min, np.inf, axis=1
            )

            if transform is not None:
                group_cytokines[:] = transform_data(group_cytokines, transform)

            if normalize:
                group_cytokines[:] = scale(group_cytokines)

            df.loc[
                group_cytokines.index, group_cytokines.columns
            ] = group_cytokines
    else:
        if transform is not None:
            df[:] = transform_data(df, transform)

        if normalize:
            df[:] = scale(df)

    for index, meta_row in meta.iterrows():
        data.loc[meta_row["PID"], meta_row["visit"], :] = df.iloc[index, :]

    data.loc[{"Cytokine Timepoint": ["PV", "LF"]}] *= pv_scaling
    data.loc[
        {"Cytokine Timepoint": ["PO", "D1", "W1", "M1"]}
    ] *= peripheral_scaling

    return data.to_dataset(name="Cytokine Measurements")


def lft_data(transform="power", normalize=True, drop_inr=True):
    """
    Import LFT data into tensor form.

    Parameters:
        transform (str, default:None): specifies transformation to use
        normalize (bool, default:False): sets zero-mean, variance one
        drop_inr (bool, default:True): drops INR measurements

    Returns:
        xarray.Dataset: RNA expression data in tensor form
    """
    lfts = import_lfts(transform=transform)
    lfts.index = lfts.index.astype(int)

    if drop_inr is not None:
        lfts = lfts.loc[:, ~lfts.columns.str.contains("inr")]
        scores = ["ast", "alt", "tbil"]
    else:
        scores = ["ast", "alt", "inr", "tbil"]

    patients = lfts.index.values
    if normalize:
        lfts[:] = scale(lfts)

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
    peripheral_scaling: float = 1,
    pv_scaling: float = 2,
    lft_scaling: float = 0.5,
):
    """
    Builds datasets and couples across shared patient dimension.

    Parameters:
        peripheral_scaling (float, default: 1): peripheral cytokine scaling
        pv_scaling (float, default: 1): PV cytokine scaling
        lft_scaling (float, default: 1): LFT scaling

    Returns:
        xr.Dataset: coupled datasets merged into one object
    """
    tensors = [
        cytokine_data(
            peripheral_scaling=peripheral_scaling, pv_scaling=pv_scaling
        ),
        lft_data() * lft_scaling,
    ]

    return xr.merge(tensors)


def import_meta():
    """
    Imports patient meta-data.

    Returns:
        pandas.DataFrame: patient meta-data
    """
    data = pd.read_csv(
        join(REPO_PATH, "liver_iri", "data", "patient_meta.csv"),
        index_col=0,
    )

    data.index = data.index.astype(int)

    return data


def import_lfts(score=None, transform="power"):
    """
    Imports liver function test scores.

    Parameters:
        score (str, default:None): liver function test to return; if 'None' is
            passed, returns all liver function tests; if provided, must be one
            of 'alt', 'ast', 'inr', or 'tbil'
        transform (str, default:'log'): transform to apply

    Returns:
        pandas.DataFrame: requested liver function test scores
    """
    lft = pd.read_csv(
        join(REPO_PATH, "liver_iri", "data", "lft_scores.csv"), index_col=0
    )

    if transform is not None:
        lft.loc[:, ~lft.columns.str.contains("inr")] = transform_data(
            lft.loc[:, ~lft.columns.str.contains("inr")], transform
        )
        lft.index = lft.index.astype(str)

    if score is not None:
        score = score.lower()
        if score not in ["alt", "ast", "inr", "tbil"]:
            raise ValueError(
                'score must be one of "alt", "ast", "inr", or "tbil"'
            )
        lft = lft.loc[:, lft.columns.str.contains(score)]

    lft.index = lft.index.astype(int)

    return lft
