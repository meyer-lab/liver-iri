from os.path import abspath, dirname, join

import numpy as np
import pandas as pd
from sklearn.preprocessing import scale, power_transform
import xarray as xr

REPO_PATH = dirname(dirname(abspath(__file__)))


def transform_data(data, transform='log'):
    """
    Applies transform to provided data.

    Args:
        data (pd.DataFrame): data to transform
        transform (str, default:'log'): transform to apply

    Returns:
        pd.DataFrame: transformed version of provided data
    """
    if (not isinstance(transform, str)) or \
            (transform.lower() not in ['log', 'power', 'reciprocal']):
        raise ValueError(
            '"transform" parameter must be "log", "power", or "reciprocal"'
        )
    transform = transform.lower()

    if transform == 'power':
        data[:] = power_transform(data)
    elif transform == 'log':
        data[:] = np.log(data)
    elif transform == 'reciprocal':
        data[:] = np.reciprocal(data)

    return data


# noinspection PyArgumentList
def cytokine_data(column=None, uniform_lod=False, transform='log',
                  normalize=False, drop_unknown=True, drop_pv=False,
                  pv_scaling=1):
    """
    Import cytokine data into tensor form.

    Parameters:
        column (str, default:None): normalizes unique values in provided column
            independently
        uniform_lod (bool, default:False): enforces uniform limit of detection
        transform (str, default:'log'): specifies transformation to use
        normalize (bool, default:False): sets zero-mean, variance one
        drop_unknown (bool, default:True): drop patients without metadata
        drop_pv (bool, default:True): drop measurements taken from portal vein
        pv_scaling (float, default:1): scaling to apply to PV measurements

    Returns:
        xarray.Dataset: cytokine data in tensor form
    """
    df = pd.read_csv(
        join(
            REPO_PATH,
            'liver_iri',
            'data',
            'cytokines.csv'
        ),
        index_col=0
    )
    df = df.drop(['IL-3', 'MIP-1a'], axis=1)

    if drop_unknown:
        meta = import_meta()
        patients = set(meta.index.astype(int))
        keep_rows = [pid in patients for pid in df.loc[:, 'PID']]
        df = df.loc[keep_rows, :]

    if drop_pv:
        visit_types = ['PO', 'D1', 'W1', 'M1']
        df = df.loc[df.loc[:, 'visit'] != 'PV', :]
        df = df.loc[df.loc[:, 'visit'] != 'LF', :]
    else:
        visit_types = ['PO', 'PV', 'LF', 'D1', 'W1', 'M1']

    data = xr.DataArray(coords={
        "Patient": df["PID"].unique(),
        "Cytokine Timepoint": visit_types,
        "Cytokine": df.columns[3:],
    },
        dims=["Patient", "Cytokine Timepoint", "Cytokine"]
    )

    if uniform_lod:
        col_min = pd.read_csv(
            join(REPO_PATH, 'liver_iri', 'data', 'cytokine_minimums.csv'),
            index_col=0
        ).squeeze()
        df.iloc[:, 3:] = np.clip(
            df.iloc[:, 3:],
            col_min,
            np.inf,
            axis=1
        )

    if column is not None:
        for group in df.loc[:, column].unique():
            group_cytokines = df.loc[df.loc[:, column] == group]
            group_cytokines = group_cytokines.iloc[:, 3:]
            col_min = np.min(
                group_cytokines.where(group_cytokines > 0),
                axis=0
            )
            group_cytokines[:] = np.clip(
                group_cytokines,
                col_min,
                np.inf,
                axis=1
            )

            if transform is not None:
                group_cytokines[:] = transform_data(group_cytokines, transform)

            if normalize:
                group_cytokines -= np.mean(group_cytokines, axis=0)
                group_cytokines /= np.std(group_cytokines, axis=0)

            df.loc[group_cytokines.index, group_cytokines.columns] = \
                group_cytokines
    else:
        col_min = np.min(df.iloc[:, 3:].where(df.iloc[:, 3:] > 0), axis=0)
        df.iloc[:, 3:] = np.clip(df.iloc[:, 3:], col_min, np.inf, axis=1)

        if transform is not None:
            df.iloc[:, 3:] = transform_data(df.iloc[:, 3:], transform)

        if normalize:
            df.iloc[:, 3:] -= np.mean(df.iloc[:, 3:], axis=0)
            df.iloc[:, 3:] /= np.std(df.iloc[:, 3:], axis=0)

    for rrow in df.iterrows():
        data.loc[rrow[1]["PID"], rrow[1]["visit"], :] = rrow[1][3:]

    if not drop_pv:
        data.loc[{"Cytokine Timepoint": ['PV', 'LF']}] *= pv_scaling

    return data.to_dataset(name='Cytokine Measurements')


def rna_data(transform='power', normalize='full', drop_unknown=True,
             shuffle=None):
    """
    Import RNA data into tensor form.

    Parameters:
        transform (bool, default:True): log-transforms RNA expression
        normalize (str, default:'full'): specifies whether to z-score
            RNA measurements altogether or individually by time point
        shuffle (rng, default:None): shuffles rna data
        drop_unknown (bool, default:True): drop patients without metadata

    Returns:
        xarray.Dataset: RNA expression data in tensor form
    """
    if normalize is not None:
        if (not isinstance(normalize, str)) or \
                normalize.lower() not in ['full', 'box']:
            raise ValueError('normalize must be None, "full", or "box"')
        normalize = normalize.lower()

    df = pd.read_csv(
        join(
            REPO_PATH,
            'liver_iri',
            'data',
            'rna_tpm.txt'
        ),
        index_col=0
    )

    if drop_unknown:
        meta = import_meta()
        pids = set(meta.index)
        keep_rows = [pid[:-4] in pids for pid in df.columns]
        df = df.loc[:, keep_rows]
        patients = df.columns.str[:-4].unique()
    else:
        patients = df.columns.str[:-4].unique()

    data = xr.DataArray(
        coords={
            "Patient": patients,
            "Gene Timepoint": ['Pre-Op', 'Post-Op'],
            "Gene": df.index,
        },
        dims=["Patient", "Gene Timepoint", "Gene"]
    )

    if transform is not None:
        df[:] = transform_data(df, transform)

    if normalize == 'full':
        df[:] = scale(df, axis=1)

    if normalize == 'box':
        for box in ['Bx1', 'Bx2']:
            box_df = df.loc[:, df.columns.str.contains(box)]
            df.loc[:, df.columns.str.contains(box)] = scale(box_df, axis=1)

    if shuffle is not None:
        df[:] = df.sample(frac=1, random_state=shuffle, axis=0).values

    for patient in patients:
        data.loc[patient, :, :] = df.loc[:, df.columns.str.contains(patient)].T

    return data.to_dataset(name='RNA Measurements')


def lft_data(transform='power', normalize=False, drop_inr=True):
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
        lfts = lfts.loc[:, ~lfts.columns.str.contains('inr')]
        scores = ['ast', 'alt', 'tbil']
    else:
        scores = ['ast', 'alt', 'inr', 'tbil']

    patients = lfts.index.values
    if normalize:
        lfts[:] = scale(lfts)

    data = xr.DataArray(
        coords={
            "Patient": patients,
            "LFT Timepoint": ['Opening'] + [str(i) for i in range(1, 8)],
            "LFT Score": scores,
        },
        dims=["Patient", "LFT Timepoint", "LFT Score"]
    )

    for score in scores:
        data.loc[:, :, score] = lfts.loc[
            patients,
            lfts.columns.str.contains(score)
        ]

    return data.to_dataset(name='LFT Measurements')


def build_coupled_tensors(
        cytokine_params=None,
        rna_params=None,
        lft_params=None
    ):
    """
    Builds datasets and couples across shared patient dimension.

    Parameters:
        cytokine_params (dict, None, or False): cytokine constructor parameters
        rna_params (dict, None, or False): RNA-seq constructor parameters
        lft_params (dict, None, or False): LFT constructor parameters

    Returns:
        xr.Dataset: coupled datasets merged into one object

    Notes:
        Parameters for each data constructor should be formatted as a dict
        mapping arguments to desired values. If None is provided for a
        constructor's arguments, default parameters are used. If False is
        provided, the dataset is removed from the merged datasets.
    """
    tensors = []
    for params, func, name in zip(
            [cytokine_params, rna_params, lft_params],
            [cytokine_data, rna_data, lft_data],
            ['cytokine_params', 'rna_params', 'lft_params']
    ):
        if not isinstance(params, dict):
            if params is not None and params is not False:
                raise TypeError(f'{name} must be dict, None, or False')
        if params is not False:
            if params is None:
                params = {}
            if 'coupled_scaling' in params.keys():
                scaling = params.pop('coupled_scaling')
            else:
                scaling = 1
            tensors.append(func(**params) * scaling)

    return xr.merge(tensors)


def import_meta(balanced=False):
    """
    Imports patient meta-data.

    Returns:
        pandas.DataFrame: patient meta-data
    """
    if balanced:
        file_name = 'balanced_meta.csv'
    else:
        file_name = 'patient_meta.csv'

    data = pd.read_csv(
        join(
            REPO_PATH,
            'liver_iri',
            'data',
            file_name
        ),
        index_col=0,
    )

    data.index = data.index.astype(int)

    return data


def import_lfts(score=None, transform='power'):
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
        join(
            REPO_PATH,
            'liver_iri',
            'data',
            'lft_scores.csv'
        ),
        index_col=0
    )

    if transform is not None:
        lft.loc[:, ~lft.columns.str.contains('inr')] = transform_data(
            lft.loc[
                :,
                ~lft.columns.str.contains('inr')
            ],
            transform
        )
        lft.index = lft.index.astype(str)

    if score is not None:
        score = score.lower()
        if score not in ['alt', 'ast', 'inr', 'tbil']:
            raise ValueError(
                'score must be one of "alt", "ast", "inr", or "tbil"'
            )
        lft = lft.loc[:, lft.columns.str.contains(score)]

    lft.index = lft.index.astype(int)

    return lft
