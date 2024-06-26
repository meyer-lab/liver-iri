"""
This file contains functions that are used in multiple figures.
"""
from decimal import Decimal

import logging
import sys
import time
from string import ascii_lowercase

from matplotlib.axes import Axes
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import svgutils.transform as st
from matplotlib import gridspec
from matplotlib import pyplot as plt

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


def getSetup(figsize, gridd, multz=None, empts=None, style="whitegrid"):
    """Establish figure set-up with subplots."""
    sns.set(
        style=style,
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
            x not in empts and x not in multz.keys()
        ):  # If this is just a normal subplot
            ax.append(f.add_subplot(gs[x]))
        elif x in multz.keys():  # If this is a subplot that spans grid elements
            ax.append(f.add_subplot(gs[x : x + multz[x] + 1]))
            x += multz[x]
        x += 1

    return np.array(ax), f


def subplotLabel(axs):
    """Place subplot labels on figure."""
    for ii, ax in enumerate(axs):
        ax.text(
            -0.2,
            1.2,
            ascii_lowercase[ii],
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            va="top",
        )


def genFigure():
    """Main figure generation function."""
    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
    fdir = "./output/"
    start = time.time()
    nameOut = "figure" + sys.argv[1]

    exec("from liver_iri.figures." + nameOut + " import makeFigure", globals())
    ff = makeFigure()
    ff.savefig(
        fdir + nameOut + ".svg", dpi=300, bbox_inches="tight", pad_inches=0
    )

    logging.info(
        f"Figure {sys.argv[1]} is done after {time.time() - start} seconds."
    )


def overlayCartoon(figFile, cartoonFile, x, y, scalee=1):
    """Add cartoon to a figure file."""

    # Overlay Figure cartoons
    template = st.fromfile(figFile)
    cartoon = st.fromfile(cartoonFile).getroot()

    cartoon.moveto(x, y, scale=scalee)

    template.append(cartoon)
    template.save(figFile)


def plot_scatter(df: pd.DataFrame, ax: Axes):
    """
    Plots scatter with regression line.

    Args:
        df (pd.DataFrame): data to plot; each column corresponds to a variable
        ax (matplotlib.ax.Axes): ax to plot to

    Returns:
        None. Modifies provided ax.
    """
    df = df.dropna(axis=0)
    score, model = predict_continuous(
        df.iloc[:, 0],
        df.iloc[:, 1]
    )

    ax.scatter(
        df.iloc[:, 0],
        df.iloc[:, 1],
        s=6
    )

    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()

    xs = [0, df.iloc[:, 0].max() * 1.05]
    ys = [
        model.params.iloc[0] + model.params.iloc[1] * xs[0],
        model.params.iloc[0] + model.params.iloc[1] * xs[1]
    ]
    ax.plot(xs, ys, color="k", linestyle="--")

    ax.set_xlabel(df.columns[0])
    ax.set_ylabel(df.columns[1])

    ax.text(
        0.98,
        0.02,
        s=f"R2: {round(score, 3)}\np-value: {Decimal(model.pvalues[1]):.2E}",
        ha="right",
        ma="right",
        va="bottom",
        transform=ax.transAxes
    )
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
