"""
This file contains functions that are used in multiple figures.
"""
import seaborn as sns
import matplotlib
from matplotlib import gridspec, pyplot as plt
import svgutils.transform as st

from string import ascii_lowercase
import sys
import logging
import time


OPTIMAL_COMPONENTS = 9

matplotlib.use('TkAgg')

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
    """ Establish figure set-up with subplots. """
    sns.set(
        style=style,
        font_scale=0.7,
        color_codes=True,
        palette="colorblind",
        rc=plt.rcParams
    )

    # create empty list if empts isn't specified
    if empts is None:
        empts = []

    if multz is None:
        multz = dict()

    # Setup plotting space and grid
    f = plt.figure(figsize=figsize, constrained_layout=True)
    gs = gridspec.GridSpec(**gridd, figure=f)

    # Get list of axis objects
    x = 0
    ax = list()
    while x < gridd['nrows'] * gridd['ncols']:
        if x not in empts and x not in multz.keys():  # If this is just a normal subplot
            ax.append(f.add_subplot(gs[x]))
        elif x in multz.keys():  # If this is a subplot that spans grid elements
            ax.append(f.add_subplot(gs[x: x + multz[x] + 1]))
            x += multz[x]
        x += 1

    return ax, f


def subplotLabel(axs):
    """ Place subplot labels on figure. """
    for ii, ax in enumerate(axs):
        ax.text(-0.2, 1.2, ascii_lowercase[ii], transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")


def genFigure():
    """ Main figure generation function. """
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    fdir = './output/'
    start = time.time()
    nameOut = 'figure' + sys.argv[1]

    exec('from tfac.figures.' + nameOut + ' import makeFigure', globals())
    ff = makeFigure()
    ff.savefig(fdir + nameOut + '.svg', dpi=300, bbox_inches='tight', pad_inches=0)

    logging.info(f'Figure {sys.argv[1]} is done after {time.time() - start} seconds.')


def overlayCartoon(figFile, cartoonFile, x, y, scalee=1):
    """ Add cartoon to a figure file. """

    # Overlay Figure cartoons
    template = st.fromfile(figFile)
    cartoon = st.fromfile(cartoonFile).getroot()

    cartoon.moveto(x, y, scale=scalee)

    template.append(cartoon)
    template.save(figFile)