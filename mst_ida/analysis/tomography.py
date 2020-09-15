"""

"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1
import matplotlib.animation as animation
from mst_ida.utilities.graphics import *
plt.rcParams["animation.html"] = "jshtml"
plt.rcParams['animation.embed_limit'] = 2**128

def plot_emiss(tomo, time_n, label=None, cmap=cm.plasma):
    """
    Make a goo-loking plot of an SXR tomographic inversion. The input tomo is the dictionary returned
        by mst_ida.data.sxt.invert_brightness, and time_n is the index specifying the desired time point.
    """
    vals = np.copy(tomo.emiss[:,:,time_n])
    xs = tomo.xs - tomo.major
    ys = tomo.ys
    
    # Remove anything outside of MST
    xx, yy = np.meshgrid(xs,ys)
    vals[(xx**2 + yy**2) > 0.53**2] = np.nan
    
    # Make the contour plot
    fig, ax = plt.subplots(1,1)
    im = plt.imshow(vals.T, extent=[tomo.xs[0],tomo.xs[-1],tomo.ys[0],tomo.ys[-1]],
        interpolation='None', vmin=0, cmap=cmap)
    
    # Make it look good
    overplot_shell(ax, origin=(tomo.major,0))
    cbar = add_colorbar(im, label=r'Emissivity (W m$^{-3}$ sr$^{-1}$)')
    ax.set_xlabel(r'$R$')
    ax.set_ylabel(r'$Z$')
    
    if label is not None:
        text(0.1,0.95,label,ax, fontsize=14)

    ax.set_title(r'SXR-ABCD, t = {0:4.1f} sec.'.format(tomo.time[time_n]))
    
    return fig, ax

def plot_emiss_2c(tomo1, tomo2, time_n, labels=None, cmap=cm.plasma):
    """
    Make a goo-loking plot of an SXR tomographic inversion. The input tomo is the dictionary returned
        by mst_ida.data.sxt.invert_brightness, and time_n is the index specifying the desired time point.
        This version plots to inversion side-by-side.
    """
    vals1 = np.copy(tomo1.emiss[:,:,time_n])
    vals2 = np.copy(tomo2.emiss[:,:,time_n])
    
    # Remove anything outside of MST
    xs = tomo1.xs - tomo1.major
    ys = tomo1.ys
    xx, yy = np.meshgrid(xs,ys)
    vals1[(xx**2 + yy**2) > 0.53**2] = np.nan

    xs = tomo2.xs - tomo2.major
    ys = tomo2.ys
    xx, yy = np.meshgrid(xs,ys)
    vals2[(xx**2 + yy**2) > 0.53**2] = np.nan
    
    # Make the contour plot
    fig, axs = plt.subplots(1, 2, figsize=(8,4))

    im1 = axs[0].imshow(vals1.T, extent=[tomo1.xs[0],tomo1.xs[-1],tomo1.ys[0],tomo1.ys[-1]],
        interpolation='None', vmin=0, cmap=cmap)
    im2 = axs[1].imshow(vals2.T, extent=[tomo2.xs[0],tomo2.xs[-1],tomo2.ys[0],tomo2.ys[-1]],
        interpolation='None', vmin=0, cmap=cmap)
    
    # Make it look good
    for ax in axs:
        overplot_shell(ax, origin=(tomo1.major,0))
        ax.set_xlabel(r'$R$')
        ax.set_ylabel(r'$Z$')
    
    if labels is not None:
        text(0.15, 0.95, labels[0], axs[0], fontsize=14)
        text(0.15, 0.95, labels[1], axs[1], fontsize=14)

    fig.suptitle(r'SXR-ABCD, t = {0:4.1f} sec.'.format(tomo1.time[time_n]))

    # Pack the figures nicely
    #axs[0].get_yaxis().set_visible(False)
    axs[1].get_yaxis().set_visible(False)
    plt.subplots_adjust(wspace=0, top=0.92)
    
    return fig, ax

def animate_emiss(tomo, vmax=550, fps=5, label=None):
    """
    Generate an animation of a tomographic inversion. The input tomo is the dictionary returned
        by mst_ida.data.sxt.invert_brightness.
    """
    fig, ax = plt.subplots(figsize=(5,4))
    plt.xlabel(r'$R$')
    plt.ylabel(r'$Z$')

    # Prepare frames ahead of time
    n_frames = len(tomo.time)
    xs = tomo.xs - tomo.major
    ys = tomo.ys
    xx, yy = np.meshgrid(xs,ys)

    emiss = np.zeros([len(tomo.xs), len(tomo.ys), n_frames])
    for frame in range(len(tomo.time)):
        emiss[:,:,frame] = np.copy(tomo.emiss[:,:,frame])
        emiss[(xx**2 + yy**2) > 0.53**2] = np.nan

    # Function to generate each frame
    im = plt.imshow(emiss[:,:,0].T, extent=[tomo.xs[0],tomo.xs[-1],tomo.ys[0],tomo.ys[-1]],
        interpolation='None', vmin=0, vmax=vmax, cmap=cm.plasma)
    overplot_shell(ax, origin=(tomo.major,0))
    cbar = add_colorbar(im, label=r'Emissivity (W m$^{-3}$ sr$^{-1}$)')
    ax.set_title(r'SXR-ABCD, t = {0:4.1f} sec.'.format(tomo.time[0]))
    
    if label is not None:
        text(0.1, 0.95, label, ax, fontsize=12)

    def animate(i):
        im.set_array(emiss[:,:,i].T)
        ax.set_title(r'SXR-ABCD, t = {0:4.1f} sec.'.format(tomo.time[i]))
        return im

    return animation.FuncAnimation(fig, animate, frames=len(tomo.time), interval=1/fps*1000)

def animate_emiss_2c(tomo1, tomo2, vmax1=550, vmax2=50, fps=5, labels=None):
    """
    Generate an animation of a tomographic inversion. The input tomo is the dictionary returned
        by mst_ida.data.sxt.invert_brightness. This version shows two inversions at the same time.
    """
    fig, axs = plt.subplots(1, 2, figsize=(8,4))
    n_frames = len(tomo1.time)

    # Prepare frames ahead of time - first set
    xs = tomo1.xs - tomo1.major
    ys = tomo1.ys
    xx, yy = np.meshgrid(xs,ys)

    emiss1 = np.zeros([len(tomo1.xs), len(tomo1.ys), n_frames])
    for frame in range(len(tomo1.time)):
        emiss1[:,:,frame] = np.copy(tomo1.emiss[:,:,frame])
        emiss1[(xx**2 + yy**2) > 0.53**2] = np.nan

    # Prepare frames ahead of time - second set
    xs = tomo2.xs - tomo2.major
    ys = tomo2.ys
    xx, yy = np.meshgrid(xs,ys)

    emiss2 = np.zeros([len(tomo2.xs), len(tomo2.ys), n_frames])
    for frame in range(len(tomo2.time)):
        emiss2[:,:,frame] = np.copy(tomo2.emiss[:,:,frame])
        emiss2[(xx**2 + yy**2) > 0.53**2] = np.nan

    # Function to generate each frame
    im1 = axs[0].imshow(emiss1[:,:,0].T, extent=[tomo1.xs[0],tomo1.xs[-1],tomo1.ys[0],tomo1.ys[-1]],
        interpolation='None', vmin=0, vmax=vmax1, cmap=cm.plasma)
    im2 = axs[1].imshow(emiss2[:,:,0].T, extent=[tomo2.xs[0],tomo2.xs[-1],tomo2.ys[0],tomo2.ys[-1]],
        interpolation='None', vmin=0, vmax=vmax2, cmap=cm.plasma)

    for ax in axs:
        overplot_shell(ax, origin=(tomo1.major,0))
        ax.set_xlabel(r'$R - R_0$')
        ax.set_ylabel(r'$Z$')
    
    if labels is not None:
        text(0.15, 0.95, labels[0], axs[0], fontsize=14)
        text(0.15, 0.95, labels[1], axs[1], fontsize=14)

    fig.suptitle(r'SXR-ABCD, t = {0:4.1f} sec.'.format(tomo1.time[0]))

    # Pack the figures nicely
    axs[1].get_yaxis().set_visible(False)
    plt.subplots_adjust(wspace=0, top=0.92)

    def animate(i):
        im1.set_array(emiss1[:,:,i].T)
        im2.set_array(emiss2[:,:,i].T)
        fig.suptitle(r'SXR-ABCD, t = {0:4.1f} sec.'.format(tomo1.time[i]))
        return im1, im2

    return animation.FuncAnimation(fig, animate, frames=n_frames, interval=1/fps*1000)