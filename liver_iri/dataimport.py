from os.path import abspath, dirname, join

import numpy as np
import pandas as pd
import xarray as xr

REPO_PATH = dirname(dirname(abspath(__file__)))


# noinspection PyArgumentList
def cytokine_data(column=None, uniform_lod=False, log_scaling=True, mean_center=False):
    if uniform_lod:
        print('Uniform LOD enforced; "column" argument is ignored')
        column = None

    df = pd.read_csv(
        join(
            REPO_PATH,
            'liver_iri',
            'data',
            'cytokine_20201120.csv'
        )
    )
    data = xr.DataArray(coords={
        "Patient": pd.unique(df["PID"]),
        "Visit Type": pd.unique(df["Visit Type"]),
        "Cytokine": df.columns[6:],
        },
        dims=["Patient", "Visit Type", "Cytokine"]
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
            group_cytokines = np.clip(
                group_cytokines,
                col_min,
                np.inf,
                axis=1
            )

            if log_scaling:
                group_cytokines = np.log(group_cytokines)

            if mean_center:
                group_cytokines -= np.mean(group_cytokines, axis=0)
                group_cytokines /= np.std(group_cytokines, axis=0)

            df.loc[group_cytokines.index, group_cytokines.columns] = \
                group_cytokines
    else:
        # log standardize data
        col_min = np.min(df.iloc[:, 6:].where(df.iloc[:, 6:]>0), axis=0)
        df.iloc[:, 6:] = np.clip(df.iloc[:, 6:], col_min, np.inf, axis=1)

        if log_scaling:
            df.iloc[:, 6:] = np.log(df.iloc[:, 6:])

        if mean_center:
            df.iloc[:, 6:] -= np.mean(df.iloc[:, 6:], axis=0)
            df.iloc[:, 6:] /= np.std(df.iloc[:, 6:], axis=0)

    for rrow in df.iterrows():
        data.loc[rrow[1]["PID"], rrow[1]["Visit Type"], :] = rrow[1][6:]

    return data.to_dataset(name='Cytokine Measurements')


def import_meta():
    data = pd.read_csv(
        join(
            REPO_PATH,
            'liver_iri',
            'data',
            'patient_meta.csv'
        ),
        index_col=0,
    )
    data.index = data.index.astype(str)

    return data
