"""
"""
import numpy as np
import scipy as sp
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# Basic style conventions
colors = ['xkcd:{0:}'.format(col) for col in ['red', 'green', 'blue', 'magenta', 'mustard yellow', 'burgundy',
                                              'dark orange', 'steel blue', 'bluish purple']]

styles = ['-', '--', '-.', ':']

# Functions to clean up my plotting code a little bit
def get_ax(figsize=(8,6), dpi=110):
    fig = plt.figure(1, figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(1,1,1)
    plt.tick_params(labelsize=14)
    return fig, ax

def set_axes(ax, x_set, y_set):
    ax.set_xlabel(x_set[0], fontsize=14)
    ax.set_xlim(x_set[1])
    ax.set_ylabel(y_set[0], fontsize=14)
    ax.set_ylim(y_set[1])

# Useful analysis plots
def plot_confidence(samples, x, ax=None, label='', ylabel='', ylim=[None,None],
        color_avg='xkcd:orange', color_1s='xkcd:sky blue', color_2s='xkcd:light blue', legend=True):
    if ax is None:
        fig, ax = get_ax()

    set_axes(ax, ('MST radius r/a', [0, 0.9]), (ylabel, ylim))
    
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)
    upper_1s = mean + std
    lower_1s = mean - std
    upper_2s = mean + 2.*std
    lower_2s = mean - 2.*std
    
    ax.plot(x, mean, color=color_avg, label=label, zorder=100)
    
    ax.plot(x, upper_1s, color=color_1s)
    ax.plot(x, lower_1s, color=color_1s)
    ax.fill_between(x, upper_1s, lower_1s, color=color_1s, alpha=0.6, label=r'$1 \sigma$')
    
    ax.plot(x, upper_2s, color=color_2s)
    ax.plot(x, lower_2s, color=color_2s)
    ax.fill_between(x, upper_2s, lower_2s, color=color_2s, alpha=0.4, label=r'$2 \sigma$')

    if legend:
        ax.legend(loc='upper right', fontsize=12)
    return ax

def plot_confidence_outline(ax, samples, x, label='', color_avg='xkcd:black', color_1s='xkcd:black', color_2s='xkcd:black'):
    """
    A useful modification of the confidence plot which is useful for visualizing evolution from the previous frame.
    """
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)
    upper_1s = mean + std
    lower_1s = mean - std
    upper_2s = mean + 2.*std
    lower_2s = mean - 2.*std
    
    ax.plot(x, mean, color=color_avg, label=label, linestyle='--', alpha=0.8)
    
    ax.plot(x, upper_1s, color=color_1s, linestyle=':', alpha=0.6)
    ax.plot(x, lower_1s, color=color_1s, linestyle=':', alpha=0.6)
    
    ax.plot(x, upper_2s, color=color_2s, linestyle=':', alpha=0.4)
    ax.plot(x, lower_2s, color=color_2s, linestyle=':', alpha=0.4)
    
def add_MST_info(ax, shot, frame, x=0.2, y=0.1, exp_time=1.0, exp_period=2.0):
    time_start = frame*exp_period
    ax.text(x, y, 'MST #{0:10d}\n{1:.1f}-{2:.1f} ms'.format(shot, time_start, time_start+exp_time),
            color='black', fontsize=14,
            horizontalalignment='center', verticalalignment='center', transform = ax.transAxes,
            bbox=dict(facecolor='none', edgecolor='red', fc='w', boxstyle='round'))
    
def burn_plot(chain, burn_step, nwalkers, indices=[0, 1, 2], labels=[r'$T_{e,0}$', r'$\alpha$', r'$\beta$'], figsize=(8,6)):
    # Plot a time-series plot of the walkers
    fig = plt.figure(1, figsize=figsize, dpi=110)
    
    # Generate the trace plots
    steps = np.arange(len(chain[0, :, 0]))
    num_plts = len(indices)
    ax = []

    for ii in range(num_plts):
        ax.append(fig.add_subplot(num_plts,1,ii+1))
        plt.tick_params(labelsize=14)
        for jj in range(nwalkers):
            ax[-1].plot(steps, chain[jj, :, ii], linewidth=1, alpha=0.5, color='black')

        # Make the axes look good
        ax[-1].set_ylabel(labels[ii], fontsize=14)
        ax[-1].axvline(x=burn_step, color='red', linestyle=':')
        
        if ii != num_plts-1:
            ax[-1].set_xlim([0, steps[-1]])
            ax[-1].xaxis.set_ticklabels([])
        else:
            ax[-1].set_xlim([0, steps[-1]])
            ax[-1].set_xlabel('Step number', fontsize=14)
    
    plt.subplots_adjust(wspace=0, hspace=0)
    
    return fig, ax

def plot_confidence_multiframe(samples_ts, ys, frames=[7,8,9,10,11], ylabel=r'$T_e$ (eV)', ylim=[0,900], scale=1.0):
    fig = plt.figure(1, figsize=(12,4), dpi=110)
    axs = []
    n_frames = len(frames)

    for index in range(n_frames):
        axs.append(fig.add_subplot(1,n_frames,index+1))
        plt.tick_params(labelsize=12)

        # Plot the corresponding profile
        if index == 0:
            plot_confidence(samples_ts[index]/scale, ys/0.52, ax=axs[-1], legend=True)
        else:
            plot_confidence(samples_ts[index]/scale, ys/0.52, ax=axs[-1], legend=False)

        # Plot an outline of the previous profile
        if index > 0:
            plot_confidence_outline(axs[-1], samples_ts[index-1]/scale, ys/0.52)

        if index == 0:
            axs[-1].set_xticks([0.2, 0.5, 0.8])
            axs[-1].set_ylabel(ylabel, fontsize=12)
        else:
            axs[-1].set_xticks([0.2, 0.5, 0.8])
            axs[-1].yaxis.set_ticklabels([])

        axs[-1].set_xlim([0,0.9])
        axs[-1].set_ylim(ylim)
        axs[-1].set_xlabel('Norm. radius', fontsize=12)

        frame = frames[index]
        axs[-1].text(0.25, 0.06, '{0:d}-{1:d} ms'.format(2*frame, 2*frame+1), color='black', fontsize=10,
                horizontalalignment='center', verticalalignment='center', transform = axs[-1].transAxes,
                bbox=dict(facecolor='none', edgecolor='red', fc='w', boxstyle='round'))

    plt.subplots_adjust(wspace=0, hspace=0)
    return fig

def plot_dists(samples, prior_uniform=False, prior_func=None, label='x', units=None, bins=50, xrange=[0, 100], npts=500):
    """
    Use this function to illustrate the marginal posterior distribution for a single parameter. Also allows the user
    to overplot the prior distribution either assuming it is uniform or accepting an arbitrary function.
    """
    fig, ax = get_ax()
    xs = np.linspace(*xrange, num=npts)

    # Plot the prior, normalized for this view
    if prior_func is None and prior_uniform:
        prior = np.ones(xs.shape)*(1/(xrange[1]-xrange[0]))
        plot_prior = True
    elif prior_func is not None and not prior_uniform:
        prior = prior_func(xs)
        plot_prior = True
    elif prior_func is None and not prior_uniform:
        plot_prior = False
    else:
        raise Exception('Selected prior options are incompatible.')
        
    if plot_prior:
        ax.plot(xs, prior, color='red', linestyle='dashed', label='$p({0:}|I)$'.format(label))

    # Plot the histogram
    hist = ax.hist(samples, bins=bins, density=True, color='xkcd:sky blue', alpha=0.6)

    # Plot the kernel density estimate
    kde = sp.stats.gaussian_kde(samples)
    ax.plot(xs, kde(xs), color='black', label=r'$p({0:}|d,\sigma,I)$'.format(label))

    # Make the plot look nice
    if units is not None:
        xlabel = '${0:}$ (${1:}$)'.format(label, units)
    else:
        xlabel = '${0:}$'.format(label)
        
    set_axes(ax, (xlabel, xrange), ('Probability density', [0,None]))
    ax.legend(loc='upper right', fontsize=14)
    
    return fig, ax

# ------------------------------ ME-SXR Plots ------------------------------
from mst_ida.data.mesxr import load_raw_data, get_profiles_data, profiles_from_image

def prof_plot(shot, frame, smooth=False):
    fig, ax = get_ax()
    mesxr_data, mesxr_sigma, signed_p, thresholds = get_profiles_data(shot, frame, smooth=smooth)
    max_counts = np.amax(mesxr_data[np.amin(thresholds)])
    set_axes(ax, ('Impact param. (m)', [-0.45,0.45]), ('Total Counts', [0, 1.2*max_counts]))
    normalize = mcolors.Normalize(vmin=np.amin(thresholds), vmax=np.amax(thresholds))
    colormap = cm.jet
    
    for Ec in thresholds:
        ax.errorbar(signed_p, mesxr_data[Ec], yerr=mesxr_sigma[Ec], color=colormap(normalize(Ec)),
                    capsize=2, ms=3, marker='o', linestyle=':', label='$E_c = {0:.1f}$ keV'.format(Ec))
        
    ax.legend(loc='upper right', fontsize=10)
    
    ax.text(0.17,0.93,'MST #{0:10d}'.format(shot), color='black', fontsize=14,
            horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
    ax.text(0.17,0.88,'Frame {0:02d}'.format(frame), color='black', fontsize=14,
            horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
    
    return fig, ax

def quick_plot(shot, frame, vmax=100):
    fig = plt.figure(1, figsize=(8,6), dpi=110)
    ax = fig.add_subplot(1,1,1)
    
    MST_data = load_raw_data(shot, avg_axis=0)
    cax = ax.imshow(MST_data['images'][:,:,frame].T, vmin=0, vmax=vmax)
    
    ax.set_xlabel('X Pixel')
    ax.set_ylabel('Y Pixel')
    fig.colorbar(cax, label='Counts', orientation='horizontal')
    ax.set_title('MST shot {0:10d}, t = {1:.1f} ms'.format(shot, MST_data['time'][frame]))
    
    return fig

def plot_time(shot, start_frame=0, end_frame=25, vmax=None, contour_ref=2, smooth=True, figsize=(8,10)):
    """
    This function produces the time traces for the x-projection of each threshold in a multi-energy configuration.
    Pixels are sorted according to the supplied threshold vector, which contians the energy threshold as a
    function of x-pixel index. If vmax is not set than each profile is simply plotted relative to its own max
    value.
    """
    # Check for relative plotting
    if vmax is None:
        relative = True
    else:
        relative = False

    # Load the data and fit each threshold
    frames = np.arange(start_frame, end_frame+1)
    num_frames = len(frames)
    
    t_start = 2*start_frame+0.5
    t_end = 2*end_frame+0.5
    tiempo = np.arange(t_start, t_end+2, 2)
    
    # Set the first frame manually to get the thresholds
    mesxr_data = load_raw_data(shot)
    prof, signed_p, thresholds = profiles_from_image(mesxr_data['images'][:,:,0], mesxr_data['thresholds'], smooth=smooth)
    profiles = {Ec:np.zeros([num_frames, 60]) for Ec in thresholds}
    
    for ii,frame in enumerate(frames):
        prof = profiles_from_image(mesxr_data['images'][:,:,frame], mesxr_data['thresholds'], smooth=smooth)[0]
        
        for Ec in thresholds:
            profiles[Ec][ii,:] = prof[Ec]

    # Make the plots
    fig = plt.figure(1, figsize=figsize, dpi=110)

    n_rows = len(thresholds)//2
    axes = []
    contours = []

    for index, Ec in enumerate(thresholds):
        ax = fig.add_subplot(n_rows, 2, index+1)
        
        if relative:
            cax = ax.contourf(tiempo, signed_p, profiles[Ec].T, 100, cmap='plasma')
        else:
            cax = ax.contourf(tiempo, signed_p, profiles[Ec].T, 100, vmin=0, vmax=vmax, cmap='plasma')
        
        # Axis labels
        if index in (0,1):
            # Top Row
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top')
            ax.set_xlabel('Time [ms]')
        elif index in (2*(n_rows-1), 2*n_rows-1):
            # Bottom
            ax.set_xlabel('Time [ms]')
        else:
            # Middle
            ax.xaxis.set_ticklabels([])

        if index % 2 == 1:
            # Right side
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position('right')
            ax.set_ylabel('X (radial)')
        else:
            # Left side
            ax.set_ylabel('X (radial)')

        # Label the threshold
        ax.text(0.9, 0.85, '{0:.2f}'.format(Ec), horizontalalignment='center', color='white',
                verticalalignment='center', transform = ax.transAxes, fontsize=12)

        # Keep up with the axes
        ax.set_xlim([tiempo[0], tiempo[-1]])
        axes.append(ax)
        contours.append(cax)

    # Get rid of space between plots and add a colorbar
    plt.subplots_adjust(wspace=0, hspace=0)
    if not relative:
        cb = fig.colorbar(contours[contour_ref], ax=axes, label=r'Counts Rate (ph/ms)', orientation='horizontal')

    # Add the shot number
    axes[-1].text(0.75, 0.1, '{0:10d}'.format(shot), horizontalalignment='center', color='white',
                  verticalalignment='center', transform = axes[-1].transAxes, fontsize=14)
        
    return fig, axes

# ------------------------------------------ New Plots ------------------------------------------
import mst_ida.analysis.ida as ida

def profile_CIs(ax, stats, ys, ylabel=r'f(r)', ylim=[0,0.52], scale=1.0, legend=True,
    cid=1, fontsize=12):
    """
    cid is the color id, change when overplotting
    """
    if cid == 1:
        color='xkcd:orange'
        shades = ['xkcd:royal blue', 'xkcd:bright blue', 'xkcd:sky blue']
    elif cid == 2:
        color='xkcd:green'
        shades = ['xkcd:crimson', 'xkcd:red', 'xkcd:pink']
    elif cid ==3:
        color='xkcd:red'
        shades = ['xkcd:blue green', 'xkcd:kelly green', 'xkcd:pastel green']
    elif cid == 4:
        color='xkcd:black'
        shades = ['xkcd:dark grey', 'xkcd:steel', 'xkcd:grey']
    
    ax.plot(ys, stats['median']/scale, color=color, zorder=100, label=r'$E[$' + ylabel + '$]$')

    labels = ['68.0%', '95.0%', '99.7%']
    for index, key in enumerate(['1 sigma', '2 sigma', '3 sigma']):
        ax.fill_between(ys, stats[key]['high']/scale, y2=stats[key]['low']/scale,
            color=shades[index], zorder=100-index, alpha=0.5, label=labels[index])

    if legend:
        ax.legend(loc='upper right', fontsize=fontsize)
    set_axes(ax, ('MST radius [m]', ylim), (ylabel, [0,None]))
    
def profile_CIs_log(ax, stats, ys, ylabel=r'$f(r)$', legend=True):
    ax.semilogy(ys, stats['median'], color='xkcd:orange', zorder=100, label=r'$E[{0:}]$'.format(ylabel))

    shades = ['xkcd:royal blue', 'xkcd:bright blue', 'xkcd:sky blue']
    labels = ['68.0%', '95.0%', '99.7%']
    
    for index, key in enumerate(['1 sigma', '2 sigma', '3 sigma']):
        #ax.semilogy(ys, stats[key]['high'], color='black', linestyle='dashed', zorder=100)
        #ax.semilogy(ys, stats[key]['low'], color='black', linestyle='dashed', zorder=100)
        ax.fill_between(ys, stats[key]['high'], y2=stats[key]['low'], color=shades[index],
            zorder=100-index, alpha=0.5, label=labels[index])

    if legend:
        ax.legend(loc='upper right')
    set_axes(ax, ('MST radius [m]', [0,0.52]), (ylabel, [None,None]))
    
def all_profiles_CI(prof_samples, ys):
    """
    """
    fig = plt.figure(1, figsize=(8,6), dpi=110)

    # Temperature
    ax1 = fig.add_subplot(4, 1, 1)

    Te_stats = ida.profile_confidence(prof_samples['Te'])
    profile_CIs(ax1, Te_stats, ys, ylabel=r'$T_e(r)$ [eV]', legend=False)
    ax1.xaxis.set_ticklabels([])
    ax1.set_ylabel(r'$T_e$ (eV)', fontsize=12)

    # Density
    ax2 = fig.add_subplot(4, 1, 2)

    ne_stats = ida.profile_confidence(prof_samples['ne'])
    profile_CIs(ax2, ne_stats, ys, ylabel=r'$n_e(r)$ [eV]', scale=1e19, legend=True)
    ax2.xaxis.set_ticklabels([])
    ax2.set_ylabel(r'$n_e$ ($\times 10^{19}$ m$^{-3}$)', fontsize=12)

    # Aluminum
    ax3 = fig.add_subplot(4, 1, 3)

    nAl_stats = ida.profile_confidence(prof_samples['nZ']['Al'])
    profile_CIs_log(ax3, nAl_stats, ys, ylabel=r'$n_{Al}(r)$ [eV]', legend=False)
    ax3.xaxis.set_ticklabels([])
    ax3.set_ylabel(r'$n_{Al}$ (m$^{-3}$)', fontsize=12)
    ax3.set_ylim([1e13,1e19])

    # Carbon
    ax4 = fig.add_subplot(4, 1, 4)

    nC_stats = ida.profile_confidence(prof_samples['nZ']['C'])
    profile_CIs_log(ax4, nC_stats, ys, ylabel=r'$n_{C}(r)$ [eV]', legend=False)
    ax4.set_ylabel(r'$n_{C}$ (m$^{-3}$)', fontsize=12)

    ax4.set_xlabel('MST radius (m)', fontsize=14)
    ax4.set_ylim([1e13,1e19])

    plt.subplots_adjust(wspace=0, hspace=0)
    
    return fig, [ax1,ax2,ax3,ax4]