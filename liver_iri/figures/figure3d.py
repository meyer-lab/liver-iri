"""Plots Figure 3d -- CTF 4: Returning GRO, Flt-3L"""
import xarray as xr

from .common import getSetup, plot_scatter
from ..dataimport import build_coupled_tensors


def makeFigure():
    ############################################################################
    # Data imports
    ############################################################################

    raw_data = build_coupled_tensors(
        lft_scaling=1,
        pv_scaling=1,
        no_missing=True,
        normalize=False,
        transform="log"
    )
    raw_val = build_coupled_tensors(
        lft_scaling=1,
        pv_scaling=1,
        no_missing=False,
        normalize=False,
        transform="log"
    )
    raw_data = xr.merge([raw_data, raw_val])
    cytokine_measurements = raw_data["Cytokine Measurements"]

    ############################################################################
    # Figure setup
    ############################################################################

    axs, fig = getSetup(
        (12, 3),
        {"nrows": 1, "ncols": 4}
    )

    ############################################################################
    # Component Histogram
    ############################################################################

    ax = axs[0]

    gro = cytokine_measurements.loc[{"Cytokine": "GRO"}].squeeze().to_pandas()
    flt3l = cytokine_measurements.loc[{
        "Cytokine": "Flt-3L"
    }].squeeze().to_pandas()

    df = gro.loc[:, ["LF", "M1"]]
    df.columns = "GRO: " + df.columns
    plot_scatter(
        df,
        ax
    )

    ax = axs[1]

    df = flt3l.loc[:, ["LF", "M1"]]
    df.columns = "Flt-3L: " + df.columns
    plot_scatter(
        df,
        ax
    )

    for ax, df, name in zip(axs[2:], [flt3l, gro], ["Flt-3L", "GRO"]):
        for index, tp in enumerate(df.columns):
            ax.boxplot(
                df.loc[:, tp].dropna(),
                patch_artist=True,
                positions=[index * 2],
                notch=True,
                vert=True,
                widths=0.9,
                flierprops={
                    "markersize": 6,
                }
            )
        ax.set_title(name)
        ax.set_xlim([-1, 11])

    return fig
