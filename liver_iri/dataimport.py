import pandas as pd
import xarray as xr
import numpy as np

def cytokine_data():
    df = pd.read_csv("./liver_iri/data/cytokine_20201120.csv")
    data = xr.DataArray(coords={
        "Patient": pd.unique(df["PID"]),
        "Visit Type": pd.unique(df["Visit Type"]),
        "Cytokine": df.columns[6:],
        },
        dims=["Patient", "Visit Type", "Cytokine"]
    )

    # log standardize data
    col_min = np.min(df.iloc[:, 6:].where(df.iloc[:, 6:]>0), axis=0)
    df.iloc[:, 6:] = np.clip(df.iloc[:, 6:], col_min, np.inf, axis=1)
    df.iloc[:, 6:] = np.log(df.iloc[:, 6:])
    df.iloc[:, 6:] -= np.mean(df.iloc[:, 6:], axis=0)
    df.iloc[:, 6:] /= np.std(df.iloc[:, 6:], axis=0)

    for rrow in df.iterrows():
        data.loc[rrow[1]["PID"], rrow[1]["Visit Type"], :] = rrow[1][6:]
    return data
