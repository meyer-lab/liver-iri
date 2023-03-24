from os.path import abspath, dirname, join

import numpy as np
import pandas as pd
from sklearn.preprocessing import scale, power_transform
import xarray as xr

REPO_PATH = dirname(dirname(abspath(__file__)))
OPTIMAL_SCALING = 1


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


def pv_data(column=None, uniform_lod=True, scaling='log',
                  mean_center=True, drop_unknown=True):
    """
    Import portal vein cytokine data into tensor form.

    Parameters:
        column (str, default:None): normalizes unique values in provided column
            independently
        uniform_lod (bool, default:True): enforces uniform limit of detection
        scaling (str, default:None): specifies scaling transformation to use
        mean_center (bool, default:True): sets zero-mean, variance one
        drop_unknown (bool, default:True): drop patients without metadata
        drop_pv (bool, default:True): drop measurements taken from portal vein

    Returns:
        xarray.Dataset: cytokine data in tensor form
    """
    if scaling is not None:
        if scaling not in ['log', 'power', 'reciprocal']:
            raise AssertionError(
                '"scaling" parameter must be "log", "power", or "reciprocal"'
            )

    df = pd.read_csv(
        join(
            REPO_PATH,
            'liver_iri',
            'data',
            'cytokine_20201120.csv'
        )
    )
    df = df.drop(['IL-3', 'MIP-1a'], axis=1)

    if drop_unknown:
        meta = import_meta()
        patients = set(meta.index)
        keep_rows = [pid in patients for pid in df.loc[:, 'PID']]
        df = df.loc[keep_rows, :]

    visit_types = ['PV', 'LF']
    df = df.loc[df.loc[:, 'Visit Type'] != 'PO', :]
    df = df.loc[df.loc[:, 'Visit Type'] != 'D1', :]
    df = df.loc[df.loc[:, 'Visit Type'] != 'W1', :]
    df = df.loc[df.loc[:, 'Visit Type'] != 'M1', :]

    data = xr.DataArray(coords={
        "Patient": pd.unique(df["PID"]),
        "PV Timepoint": visit_types,
        "Cytokine": df.columns[6:],
        },
        dims=["Patient", "PV Timepoint", "Cytokine"]
    )

    if uniform_lod:
        col_min = pd.read_csv(
            join(REPO_PATH, 'liver_iri', 'data', 'cytokine_minimums.csv'),
            index_col=0
        ).squeeze()
        df.iloc[:, 6:] = np.clip(
            df.iloc[:, 6:],
            col_min,
            np.inf,
            axis=1
        )

    if column is not None:
        for group in df.loc[:, column].unique():
            group_cytokines = df.loc[df.loc[:, column] == group]
            group_cytokines = group_cytokines.iloc[:, 6:]
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

            if scaling is not None:
                if scaling == 'power':
                    group_cytokines[:] = power_transform(group_cytokines)
                elif scaling == 'log':
                    group_cytokines[:] = np.log(group_cytokines)
                elif scaling == 'reciprocal':
                    group_cytokines[:] = np.reciprocal(group_cytokines)

            if mean_center:
                group_cytokines -= np.mean(group_cytokines, axis=0)
                group_cytokines /= np.std(group_cytokines, axis=0)

            df.loc[group_cytokines.index, group_cytokines.columns] = \
                group_cytokines
    else:
        col_min = np.min(df.iloc[:, 6:].where(df.iloc[:, 6:]>0), axis=0)
        df.iloc[:, 6:] = np.clip(df.iloc[:, 6:], col_min, np.inf, axis=1)

        if scaling is not None:
            if scaling == 'power':
                df.iloc[:, 6:] = power_transform(df.iloc[:, 6:])
            elif scaling == 'log':
                df.iloc[:, 6:] = np.log(df.iloc[:, 6:])
            elif scaling == 'reciprocal':
                df.iloc[:, 6:] = np.reciprocal(df.iloc[:, 6:])

        if mean_center:
            df.iloc[:, 6:] -= np.mean(df.iloc[:, 6:], axis=0)
            df.iloc[:, 6:] /= np.std(df.iloc[:, 6:], axis=0)

    for rrow in df.iterrows():
        data.loc[rrow[1]["PID"], rrow[1]["Visit Type"], :] = rrow[1][6:]

    return data.to_dataset(name='PV Measurements')


# noinspection PyArgumentList
def cytokine_data(column=None, uniform_lod=True, transform='log',
                  mean_center=True, drop_unknown=True, drop_pv=True):
    """
    Import cytokine data into tensor form.

    Parameters:
        column (str, default:None): normalizes unique values in provided column
            independently
        uniform_lod (bool, default:True): enforces uniform limit of detection
        transform (str, default:None): specifies transformation to use
        mean_center (bool, default:True): sets zero-mean, variance one
        drop_unknown (bool, default:True): drop patients without metadata
        drop_pv (bool, default:True): drop measurements taken from portal vein

    Returns:
        xarray.Dataset: cytokine data in tensor form
    """
    df = pd.read_csv(
        join(
            REPO_PATH,
            'liver_iri',
            'data',
            'cytokine_20201120.csv'
        )
    )
    df = df.drop(['IL-3', 'MIP-1a'], axis=1)

    if drop_unknown:
        meta = import_meta()
        patients = set(meta.index)
        keep_rows = [pid in patients for pid in df.loc[:, 'PID']]
        df = df.loc[keep_rows, :]

    if drop_pv:
        visit_types = ['PO', 'D1', 'W1', 'M1']
        df = df.loc[df.loc[:, 'Visit Type'] != 'PV', :]
        df = df.loc[df.loc[:, 'Visit Type'] != 'LF', :]
    else:
        visit_types = ['PO', 'PV', 'LF', 'D1', 'W1', 'M1']

    data = xr.DataArray(coords={
        "Patient": pd.unique(df["PID"]),
        "Cytokine Timepoint": visit_types,
        "Cytokine": df.columns[6:],
        },
        dims=["Patient", "Cytokine Timepoint", "Cytokine"]
    )

    if uniform_lod:
        col_min = pd.read_csv(
            join(REPO_PATH, 'liver_iri', 'data', 'cytokine_minimums.csv'),
            index_col=0
        ).squeeze()
        df.iloc[:, 6:] = np.clip(
            df.iloc[:, 6:],
            col_min,
            np.inf,
            axis=1
        )

    if column is not None:
        for group in df.loc[:, column].unique():
            group_cytokines = df.loc[df.loc[:, column] == group]
            group_cytokines = group_cytokines.iloc[:, 6:]
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

            if mean_center:
                group_cytokines -= np.mean(group_cytokines, axis=0)
                group_cytokines /= np.std(group_cytokines, axis=0)

            df.loc[group_cytokines.index, group_cytokines.columns] = \
                group_cytokines
    else:
        col_min = np.min(df.iloc[:, 6:].where(df.iloc[:, 6:]>0), axis=0)
        df.iloc[:, 6:] = np.clip(df.iloc[:, 6:], col_min, np.inf, axis=1)

        if transform is not None:
            df.iloc[:, 6:] = transform_data(df.iloc[:, 6:], transform)

        if mean_center:
            df.iloc[:, 6:] -= np.mean(df.iloc[:, 6:], axis=0)
            df.iloc[:, 6:] /= np.std(df.iloc[:, 6:], axis=0)

    for rrow in df.iterrows():
        data.loc[rrow[1]["PID"], rrow[1]["Visit Type"], :] = rrow[1][6:]

    return data.to_dataset(name='Cytokine Measurements')


def rna_data(transform='power', mean_center='full', drop_unknown=True, shuffle=None):
    """
    Import RNA data into tensor form.

    Parameters:
        transform (bool, default:True): log-transforms RNA expression
        mean_center (str, default:'full'): specifies whether to z-score
            RNA measurements altogether or individually by time point
        shuffle (rng, default:None): shuffles rna data
        drop_unknown (bool, default:True): drop patients without metadata

    Returns:
        xarray.Dataset: RNA expression data in tensor form
    """
    if mean_center is not None:
        if (not isinstance(mean_center, str)) or \
                mean_center.lower() not in ['full', 'box']:
            raise ValueError('mean_center must be None, "full", or "box"')
        mean_center = mean_center.lower()

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

    if mean_center == 'full':
        df[:] = scale(df, axis=1)

    if mean_center == 'box':
        for box in ['Bx1', 'Bx2']:
            box_df = df.loc[:, df.columns.str.contains(box)]
            df.loc[:, df.columns.str.contains(box)] = scale(box_df, axis=1)

    if shuffle is not None:
        df[:] = df.sample(frac=1, random_state=shuffle, axis=0).values

    for patient in patients:
        data.loc[patient, :, :] = df.loc[:, df.columns.str.contains(patient)].T

    return data.to_dataset(name='RNA Measurements')


def lft_data(scaling='power', normalize=False, drop_inr=True):
    lfts = import_lfts()
    if drop_inr is not None:
        lfts = lfts.loc[:, ~lfts.columns.str.contains('inr')]
        scores = ['ast', 'alt', 'tbil']
    else:
        scores = ['ast', 'alt', 'inr', 'tbil']

    patients = lfts.index.values

    if scaling:
        if scaling == 'log':
            lfts += 1
            lfts[:] = np.log(lfts)
        elif scaling == 'power':
            lfts[:] = power_transform(lfts)
        else:
            raise ValueError(
                f'Unrecognized scaling parameter provided: {scaling}:',
                'should be one of "log" or "power"'
            )

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
        lft_params=None,
        pv_params=None
    ):
    if cytokine_params is not None and not isinstance(cytokine_params, dict):
        raise ValueError('cytokine_params must be a dict')

    if rna_params is not None and not isinstance(rna_params, dict):
        raise ValueError('rna_params must be a dict')

    if lft_params is not None and not isinstance(lft_params, dict):
        raise ValueError('lft_params must be a dict')

    if pv_params is not None and not isinstance(pv_params, dict):
        raise ValueError('pv_params must be a dict')

    tensors = []
    for params, func in zip(
        [cytokine_params, rna_params, lft_params, pv_params],
        [cytokine_data, rna_data, lft_data, pv_data]
    ):
        if params is not None:
            if 'coupled_scaling' in params.keys():
                scaling = params.pop('coupled_scaling')
            else:
                scaling = 1
            tensors.append(func(**params) * scaling)

    return xr.merge(tensors)


def build_triple_tensor(
        scaling=(1, 1, 1),
        cytokine_params=None,
        rna_params=None,
        lft_params=None,
    ):
    if cytokine_params is not None and not isinstance(cytokine_params, dict):
        raise ValueError('cytokine_params must be a dict')

    if rna_params is not None and not isinstance(rna_params, dict):
        raise ValueError('rna_params must be a dict')

    if lft_params is not None and not isinstance(lft_params, dict):
        raise ValueError('lft_params must be a dict')

    if cytokine_params is None:
        cytokine_params = {}

    if rna_params is None:
        rna_params = {}

    if lft_params is None:
        lft_params = {}

    rna = rna_data(**rna_params) * scaling[0]
    cytokine = cytokine_data(**cytokine_params) * scaling[1]
    lft = lft_data(**lft_params) * scaling[2]

    return xr.merge([rna, cytokine, lft])


def build_coupled_tensor(
        scaling=OPTIMAL_SCALING,
        cytokine_params=None,
        rna_params=None,
        drop_unknown=False
    ):
    """
    Builds coupled cytokine and RNA tensors.

    Parameters:
        scaling (float, default:OPTIMAL_SCALING): variance scaling between RNA
            and cytokine tensors; values > 1 increase RNA emphasis
        cytokine_params (dict, default:None): parameters to be passed to
            cytokine tensor creation; refer to cytokine_data
        rna_params (dict, default:None): parameters to be passed to RNA tensor
            creation; refer to rna_data
        drop_unknown (bool, default:True): drop patients without metadata

    Returns:
        xarray.Dataset: Coupled RNA and cytokine tensors
    """
    meta = import_meta()

    if cytokine_params is not None and not isinstance(cytokine_params, dict):
        raise ValueError('cytokine_params must be a dict')

    if rna_params is not None and not isinstance(rna_params, dict):
        raise ValueError('rna_params must be a dict')

    if cytokine_params is None:
        cytokine_params = {}

    if rna_params is None:
        rna_params = {}

    rna = rna_data(**rna_params) * scaling
    cytokine = cytokine_data(**cytokine_params)

    if drop_unknown:
        rna = rna.sel(
            Patient=sorted(list(set(meta.index) & set(rna.Patient.values)))
        )
        cytokine = cytokine.sel(
            Patient=sorted(list(set(meta.index) & set(cytokine.Patient.values)))
        )

    return xr.merge([rna, cytokine])


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
    data.index = data.index.astype(str)

    return data


def import_lfts(score=None, transform='log'):
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

    return lft
