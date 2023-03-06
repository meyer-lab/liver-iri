from os.path import abspath, dirname, join

import numpy as np
import pandas as pd
from sklearn.preprocessing import scale, power_transform
import xarray as xr

REPO_PATH = dirname(dirname(abspath(__file__)))
OPTIMAL_SCALING = 1


# noinspection PyArgumentList
def cytokine_data(column=None, uniform_lod=True, scaling='log',
                  mean_center=True, drop_unknown=True, drop_pv=True):
    """
    Import cytokine data into tensor form.

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
    # if uniform_lod:
    #     print('Uniform LOD enforced; "column" argument is ignored')
    #     column = None

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

    return data.to_dataset(name='Cytokine Measurements')


def rna_data(log_scaling=True, normalization='full', drop_unknown=True, shuffle=None):
    """
    Import RNA data into tensor form.

    Parameters:
        log_scaling (bool, default:True): log-transforms RNA expression
        normalization (str, default:'full'): specifies whether to z-score
            RNA measurements altogether or individually by time point
        shuffle (rng, default:None): shuffles rna data
        drop_unknown (bool, default:True): drop patients without metadata

    Returns:
        xarray.Dataset: RNA expression data in tensor form
    """
    if normalization:
        if normalization.lower() not in ['full', 'box']:
            raise ValueError('normalization must be "False", "full", or "box"')

    df = pd.read_csv(
        join(
            REPO_PATH,
            'liver_iri',
            'data',
            'tpm_higher_frequency.txt'
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

    if log_scaling:
        df[:] = power_transform(df)

    if normalization == 'full':
        df[:] = scale(df, axis=1)

    if normalization == 'Box':
        for box in ['Bx1', 'Bx2']:
            box_df = df.loc[:, df.columns.str.contains(box)]
            df.loc[:, df.columns.str.contains(box)] = scale(box_df, axis=1)

    if shuffle is not None:
        df[:] = df.sample(frac=1, random_state=shuffle, axis=0).values

    for patient in patients:
        data.loc[patient, :, :] = df.loc[:, df.columns.str.contains(patient)].T

    return data.to_dataset(name='RNA Measurements')


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


def import_meta():
    """
    Imports patient meta-data.

    Returns:
        pandas.DataFrame: patient meta-data
    """
    data = pd.read_csv(
        join(
            REPO_PATH,
            'liver_iri',
            'data',
            'patient_meta_v2.csv'
        ),
        index_col=0,
    )
    data.index = data.index.astype(str)

    return data
