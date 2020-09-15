"""
Import this in a Jupyter notebook to set up the plotting environment.
"""
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.animation as animation
from matplotlib.ticker import MaxNLocator, LogLocator
from mpl_toolkits import axes_grid1
import corner

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')

PARAMS = {
    'text.latex.preamble': ['\\usepackage{gensymb}'],
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'axes.grid': False,
    'savefig.dpi': 150,  # to adjust notebook inline plot size
    'axes.labelsize': 14, # fontsize for x and y labels (was 10)
    'axes.titlesize': 12,
    'font.size': 12, # was 10
    'legend.fontsize': 12, # was 10
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    #'text.usetex': True,
    'figure.figsize': [8, 6],
    'figure.dpi':110,
    'font.family': 'serif'
}

matplotlib.rcParams.update(PARAMS)

def Normalize(vmin, vmax):
    """
    Simplify the expression for normalizing plots.
    """
    return mcolors.Normalize(vmax=vmax, vmin=vmin)

def overplot_shell(ax, origin=(0, 0)):
    """
    Draw the MST shell overtop of a 2D plot.
    """
    minor_radius = 0.52
    thick = 0.03
    ax.add_patch(patches.Wedge(origin, minor_radius+thick, 0, 360, width=thick, color='gray'))
    
def text(x, y, string, ax, fontsize=12, **kwargs):
    ax.text(x, y, string, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes,
            fontsize=fontsize, **kwargs)

def color_axis(ax, color='red'):
    ax.tick_params(axis='x', colors=color)
    ax.tick_params(axis='y', colors=color)
    ax.yaxis.label.set_color(color)
    ax.xaxis.label.set_color(color)
    
def distplot(samples, labels=None, **kwargs):
    """
    Function to plot posteriors using corner.py and scipy's gaussian KDE function.
    """
    fig = corner.corner(samples, labels=labels, hist_kwargs={'density': True}, **kwargs)

    # plot KDE smoothed version of distributions
    n = samples.shape[1]
    diags = [n*i+i for i in range(0,n)]
    for axidx, samps in zip(diags, samples.T):
        kde = sp.stats.gaussian_kde(samps)
        xvals = fig.axes[axidx].get_xlim()
        xvals = np.linspace(xvals[0], xvals[1], 100)
        fig.axes[axidx].plot(xvals, kde(xvals), color='firebrick')

def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """
    Add a vertical color bar to an image plot.
    """
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes('right', size=width, pad=pad)
    plt.sca(current_ax)

    return im.axes.figure.colorbar(im, cax=cax, **kwargs)