"""
This file contains functions that are used in multiple figures.
"""

import logging
import sys
import time
from collections.abc import Iterable
from decimal import Decimal
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from ..predict import predict_continuous

matplotlib.use("AGG")

matplotlib.rcParams["axes.labelsize"] = 10
matplotlib.rcParams["axes.linewidth"] = 0.6
matplotlib.rcParams["axes.titlesize"] = 12
matplotlib.rcParams["font.family"] = ["sans-serif"]
matplotlib.rcParams["font.sans-serif"] = ["Arial"]
matplotlib.rcParams["font.size"] = 8
matplotlib.rcParams["grid.linestyle"] = "dotted"
matplotlib.rcParams["legend.borderpad"] = 0.35
matplotlib.rcParams["legend.fontsize"] = 7
matplotlib.rcParams["legend.framealpha"] = 0.5
matplotlib.rcParams["legend.handlelength"] = 0.5
matplotlib.rcParams["legend.handletextpad"] = 0.5
matplotlib.rcParams["legend.labelspacing"] = 0.2
matplotlib.rcParams["legend.markerscale"] = 0.7
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["xtick.labelsize"] = 8
matplotlib.rcParams["xtick.major.pad"] = 1.0
matplotlib.rcParams["xtick.minor.pad"] = 0.9
matplotlib.rcParams["ytick.labelsize"] = 8
matplotlib.rcParams["ytick.major.pad"] = 1.0
matplotlib.rcParams["ytick.minor.pad"] = 0.9


def getSetup(
    figsize: tuple[int | int],
    gridd: dict[str | Any],
    multz: dict[int | int] | None = None,
    empts: Iterable[int] | None = None,
    style: str = "whitegrid",
):
    """Establish figure set-up with subplots."""
    sns.set(
        style=style,  # noqa
        font_scale=0.7,
        color_codes=True,
        palette="colorblind",
        rc=plt.rcParams,
    )

    # create empty list if empts isn't specified
    if empts is None:
        empts = []

    if multz is None:
        multz = dict()

    # Setup plotting space and grid
    f = plt.figure(figsize=figsize, constrained_layout=True, dpi=200)
    gs = gridspec.GridSpec(**gridd, figure=f)

    # Get list of axis objects
    x = 0
    ax = list()
    while x < gridd["nrows"] * gridd["ncols"]:
        if (
            x not in empts and x not in multz
        ):  # If this is just a normal subplot
            ax.append(f.add_subplot(gs[x]))
        elif x in multz:  # If this is a subplot that spans grid elements
            ax.append(f.add_subplot(gs[x : x + multz[x] + 1]))
            x += multz[x]
        x += 1

    return np.array(ax), f


def genFigure():
    """Main figure generation function."""
    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
    fdir = "./output/"
    start = time.time()
    nameOut = "figure" + sys.argv[1]

    exec(
        "from liver_iri.figures." + nameOut + " import makeFigure",  # noqa
        globals(),
    )
    ff = makeFigure()  # noqa
    ff.savefig(
        fdir + nameOut + ".svg", dpi=300, bbox_inches="tight", pad_inches=0
    )

    logging.info(
        f"Figure {sys.argv[1]} is done after {time.time() - start} seconds."
    )


def plot_scatter(df: pd.DataFrame, ax: Axes, cmap: pd.Series | None = None):
    """
    Plots scatter with regression line.

    Args:
        df (pd.DataFrame): data to plot; each column corresponds to a variable
        ax (matplotlib.ax.Axes): ax to plot to
        cmap (pd.Series): colormap colors

    Returns:
        None. Modifies provided ax.
    """
    df = df.dropna(axis=0)
    score, model = predict_continuous(df.iloc[:, 0], df.iloc[:, 1])

    if cmap is None:
        ax.scatter(df.iloc[:, 0], df.iloc[:, 1], s=6)
    else:
        ax.scatter(
            df.iloc[:, 0],
            df.iloc[:, 1],
            c=cmap.loc[df.index],
            cmap="coolwarm",
            s=6,
        )

    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()

    xs = [0, df.iloc[:, 0].max() * 1.05]
    ys = [
        model.params.iloc[0] + model.params.iloc[1] * xs[0],
        model.params.iloc[0] + model.params.iloc[1] * xs[1],
    ]

    ax.plot(xs, ys, color="k", linestyle="--")
    ax.set_xlabel(df.columns[0])  # noqa
    ax.set_ylabel(df.columns[1])  # noqa

    ax.text(
        0.98,
        0.02,
        s=f"R2: {round(score, 3)}\np-value: {Decimal(model.pvalues[1]):.2E}",
        ha="right",
        ma="right",
        va="bottom",
        transform=ax.transAxes,
    )
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
