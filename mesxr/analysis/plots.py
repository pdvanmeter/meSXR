"""
Make plots to analyze the data from the tree.
"""
import numpy as np
import matplotlib.pyplot as plt
import MDSplus
import data

colors = ['xkcd:{0:}'.format(col) for col in ['red', 'green', 'blue', 'magenta', 'mustard yellow', 'burgundy',
                                              'dark orange', 'steel blue', 'bluish purple']]

def plot_frame(shot, frame, vmax=100, scale=4, remove_edges=False, center_pix=True, xlim=[0,487]):
    """
    Plot the raw image and profiles for each threshold in an image.
    """
    MST_data = data.load_ME(shot, frames=[frame], center_only=True, remove_edges=remove_edges)
    Ec_set = np.sort(MST_data['ME']['thresholds'])
    
    fig = plt.figure(1, figsize=(8,6), dpi=110)
    
    # Plot 1 - Raw image
    ax1 = fig.add_subplot(2,1,1)
    cax1 = ax1.imshow(MST_data['images'][:,:,frame].T, vmax=vmax, aspect='auto')
    ax1.xaxis.set_ticklabels([])
    ax1.set_ylabel('Y pixel (toroidal)')
    ax1.set_xlabel('X pixel')
    ax1.set_xlim(xlim)
    
    # Plot 2 - Profiles
    ax2 = fig.add_subplot(2,1,2)
    max_val = 0

    for index, Ec in enumerate(Ec_set):
        ax2.errorbar(MST_data['ME']['x_index'][Ec], MST_data['ME']['profiles'][0][Ec]/10.**scale,
                     yerr=np.sqrt(MST_data['ME']['profiles'][0][Ec])/10.**scale, color=colors[index],
                    capsize=2, ms=3, marker='o', linestyle=':', label='$E_c = {0:.1f}$ keV'.format(Ec))
        
        if np.amax(MST_data['ME']['profiles'][0][Ec]) > max_val:
            max_val = np.amax(MST_data['ME']['profiles'][0][Ec])
        
    ax2.set_xlabel('X pixel (radial)')
    ax2.set_ylabel(r'Counts ($\times 10^{0:d}$)'.format(scale))
    ax2.grid(axis='x')
    ax2.set_xlim(xlim)
    
    ax2.set_ylim([0, 1.1*max_val/10.**scale])
    
    # Remove the space between plots and add the colorbar
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.colorbar(cax1, ax=[ax1, ax2], label='Counts')
    
    # Add the shot number
    ax2.text(0.15, 0.9, str(shot), horizontalalignment='center',
             verticalalignment='center', transform = ax2.transAxes, fontsize=14)
    
    # Add the time range
    lower_t = MST_data['ME']['time'][0] - 1000.*MST_data['config']['exp_time']/2.0
    upper_t = MST_data['ME']['time'][0] + 1000.*MST_data['config']['exp_time']/2.0
    ax2.text(0.16, 0.8, '{0:.1f} - {1:.1f} ms'.format(lower_t, upper_t), horizontalalignment='center',
             verticalalignment='center', transform = ax2.transAxes, fontsize=14)
    
    ax2.legend(loc='upper right', fontsize=8)
        
    return fig

def plot_time(shot, start_frame=0, end_frame=25, vmax=-1, remove_edges=False, center_only=True, contour_ref=2):
    """
    This function produces the time traces for the x-projection of each threshold in a multi-energy configuration.
    Pixels are sorted according to the supplied threshold vector, which contians the energy threshold as a
    function of x-pixel index. If vmax is not set than each profile is simply plotted relative to its own max
    value.
    """
    # Check for relative plotting
    if vmax == -1:
        relative = True
    else:
        relative = False

    # Load the data and fit each threshold
    frames = np.arange(start_frame, end_frame+1)
    num_frames = len(frames)
    ME_data = data.load_ME(shot, frames=frames, center_only=center_only, remove_edges=remove_edges)
    Ec_set = np.sort(ME_data['ME']['thresholds'])

    # Make the plots
    fig = plt.figure(1, figsize=(8,10), dpi=110)

    n_rows = len(Ec_set)/2
    axes = []
    contours = []

    for index, Ec in enumerate(Ec_set):
        ax = fig.add_subplot(n_rows, 2, index+1)
        
        profile_vs_time = np.array([ME_data['ME']['profiles'][i][Ec] for i in range(num_frames)])
        
        if relative:
            cax = ax.contourf(ME_data['ME']['time'], ME_data['ME']['x_index'][Ec], profile_vs_time.T, 100, cmap='plasma')
        else:
            cax = ax.contourf(ME_data['ME']['time'], ME_data['ME']['x_index'][Ec], profile_vs_time.T, 100, vmin=0, vmax=vmax, cmap='plasma')
        
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
        ax.set_xlim([ME_data['ME']['time'][0], ME_data['ME']['time'][-1]])
        axes.append(ax)
        contours.append(cax)

    # Get rid of space between plots and add a colorbar
    plt.subplots_adjust(wspace=0, hspace=0)
    if not relative:
        cb = fig.colorbar(contours[contour_ref], ax=axes, label=r'Counts Rate (ph/ms)', orientation='horizontal')

    # Add the shot number
    axes[-1].text(0.75, 0.1, '{0:10d}'.format(shot), horizontalalignment='center', color='white',
                  verticalalignment='center', transform = axes[-1].transAxes, fontsize=14)
        
    return fig

def multi_info_plot(shot, en_indices, start_frame=0, end_frame=25, remove_edges=False, center_only=True, label='',
                    vmax=-1, contour_ref=0, x_pixel=250):
    """
    This is most useful for standard and QSH plasmas, hence the n=5 focus.
    """
    # Check for relative plotting
    if vmax == -1:
        relative = True
    else:
        relative = False

    # Assemble the data
    frames = np.arange(start_frame, end_frame+1)
    num_frames = len(frames)
    ME_data = data.load_ME(shot, frames=frames, center_only=center_only, remove_edges=remove_edges)
    Ec_set = np.sort(ME_data['ME']['thresholds'])
    
    # Make the plot
    fig = plt.figure(1, figsize=(8,10), dpi=110)
    num_plts = len(en_indices)
    num_signals = 5
    axes = []
    contours = []
    
    for counter, index in enumerate(en_indices):
        ax = fig.add_subplot(num_signals+num_plts, 1, counter+1)
        Ec = Ec_set[index]
        profile_vs_time = np.array([ME_data['ME']['profiles'][i][Ec] for i in range(num_frames)])
        if relative:
            cax = ax.contourf(ME_data['ME']['time'], ME_data['ME']['x_index'][Ec], profile_vs_time.T, 100, cmap='plasma')
        else:
            cax = ax.contourf(ME_data['ME']['time'], ME_data['ME']['x_index'][Ec], profile_vs_time.T, 100, cmap='plasma', vmin=0, vmax=vmax)
        
        # Axis labels
        if index == 0:
            # First image
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top')
            ax.set_xlabel('Time [ms]')
        else:
            # Middle
            ax.xaxis.set_ticklabels([])
            
        ax.set_ylabel('X (radial)')
            
        # Label the threshold
        ax.axhline(y=x_pixel, color='white', linestyle=':', linewidth=0.8)
        ax.text(0.9, 0.85, '{0:.2f} keV'.format(Ec), horizontalalignment='center', color='white',
                verticalalignment='center', transform = ax.transAxes, fontsize=12)
            
        # Keep up with the axes
        ax.set_xlim([ME_data['ME']['time'][0], ME_data['ME']['time'][-1]])
        axes.append(ax)
        contours.append(cax)
    
    # Add a plot for 1d time traces
    max_val = 0
    axes.append(fig.add_subplot(num_signals+num_plts, 1, num_plts+1))
    for index, Ec in enumerate(ME_data['ME']['thresholds']):
        x_n = np.argmin(np.abs(ME_data['ME']['x_index'][Ec] - x_pixel))
        prof1d = np.array([ME_data['ME']['profiles'][i][Ec] for i in range(num_frames)])[:, x_n]
        axes[-1].plot(ME_data['ME']['time'], prof1d, linestyle=':', marker='o', ms=3, color=colors[index],
                      label='{0:.1f} keV'.format(Ec))

        if np.amax(prof1d) > max_val:
            max_val = np.amax(prof1d)

    axes[-1].set_xlim([ME_data['ME']['time'][0], ME_data['ME']['time'][-1]])
    axes[-1].set_ylim([0, 1.1*max_val])
    axes[-1].set_ylabel('Counts')
    axes[-1].legend(loc='upper right', fontsize=8, ncol=2)
    axes[-1].xaxis.set_ticklabels([])

    # Add the other signals
    mstTree = MDSplus.Tree('mst', shot, 'READONLY')
    
    axes.append(fig.add_subplot(num_signals+num_plts, 1, num_plts+2))
    plot_Ip_axis(axes[-1], mstTree, ME_data['ME']['time'][0], ME_data['ME']['time'][-1])
    
    axes.append(fig.add_subplot(num_signals+num_plts, 1, num_plts+3))
    plot_Bp_amp_axis(axes[-1], mstTree, ME_data['ME']['time'][0], ME_data['ME']['time'][-1], right=False)
    
    axes.append(fig.add_subplot(num_signals+num_plts, 1, num_plts+4))
    plot_Bp_phs_axis(axes[-1], mstTree, ME_data['ME']['time'][0], ME_data['ME']['time'][-1], right=False)
    
    axes.append(fig.add_subplot(num_signals+num_plts, 1, num_plts+5))
    plot_ne_FIR_axis(axes[-1], mstTree, ME_data['ME']['time'][0], ME_data['ME']['time'][-1])
    
    # Get rid of space between plots
    plt.subplots_adjust(wspace=0, hspace=0)

    #if not relative:
    #    cb = fig.colorbar(contours[contour_ref], ax=axes, label=r'Counts Rate (ph/ms)', orientation='vertical')
    
    # Add the shot number to the plot
    axes[0].text(0.88, 0.15, '{0:10d}'.format(shot), horizontalalignment='center', color='white',
                 verticalalignment='center', transform = axes[0].transAxes, fontsize=14)
    
    return fig

# -----------------------------------------------------------------------------------------------
# Modules to read in additional signals and plot them
# -----------------------------------------------------------------------------------------------

def plot_Ip_axis(ax, mstTree, start_t, end_t):
    mst_Ip = mstTree.getNode('\MST_OPS::ip').getData().data()
    mst_Ip_time = mstTree.getNode('\MST_OPS::ip').getData().dim_of().data()*1000
    ax.plot(mst_Ip_time, mst_Ip)
    ax.set_xlim([start_t, end_t])
    ax.set_ylim([0, 1.2*np.amax(mst_Ip)])
    ax.set_ylabel('Ip (kA)')
    ax.set_xlabel('Time (ms)')
    ax.text(0.88, 0.2, r'max $I_p = {0:.1f}$ kA'.format(np.amax(mst_Ip)), horizontalalignment='center', color='black',
                  verticalalignment='center', transform = ax.transAxes, fontsize=10,
                  bbox=dict(facecolor='none', edgecolor='black', fc='w', boxstyle='round'))
    
def plot_Bp_amp_axis(ax, mstTree, start_t, end_t, right=True, max_mode=5):
    mst_Bp_N05 = mstTree.getNode('\MST_MAG::BP_N05_AMP').getData().data()
    mst_Bp_N06 = mstTree.getNode('\MST_MAG::BP_N06_AMP').getData().data()
    mst_Bp_N07 = mstTree.getNode('\MST_MAG::BP_N07_AMP').getData().data()
    mst_Bp_time = mstTree.getNode('\MST_MAG::BP_N05_AMP').getData().dim_of().data()*1000
    ax.plot(mst_Bp_time, mst_Bp_N05, label='n = 5', color='blue', zorder=100)
    ax.plot(mst_Bp_time, mst_Bp_N06, label='n = 6', color='red')
    ax.plot(mst_Bp_time, mst_Bp_N07, label='n = 7', color='green')
    ax.set_xlim([start_t, end_t])
    if right:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
    ax.set_ylabel('|Bp| (G)')
    ax.set_xlabel('Time (ms)')
    
    ax.text(0.05, 0.8, '5', horizontalalignment='center', color='blue',
                  verticalalignment='center', transform = ax.transAxes, fontsize=12)
    ax.text(0.1, 0.8, '6', horizontalalignment='center', color='red',
                  verticalalignment='center', transform = ax.transAxes, fontsize=12)
    ax.text(0.15, 0.8, '7', horizontalalignment='center', color='green',
                  verticalalignment='center', transform = ax.transAxes, fontsize=12)

    ax.text(0.87, 0.2, r'max $B_{{p,n=5}} = {0:.1f}$ G'.format(np.amax(mst_Bp_N05)), horizontalalignment='center', color='black',
        verticalalignment='center', transform = ax.transAxes, fontsize=10,
        bbox=dict(facecolor='none', edgecolor='black', fc='w', boxstyle='round'))
    
def plot_Bp_phs_axis(ax, mstTree, start_t, end_t, right=False, n05_avg=True):
    mst_Bp_N05 = mstTree.getNode('\MST_MAG::BP_N05_PHS').getData().data()
    mst_Bp_N06 = mstTree.getNode('\MST_MAG::BP_N06_PHS').getData().data()
    mst_Bp_N07 = mstTree.getNode('\MST_MAG::BP_N07_PHS').getData().data()
    mst_Bp_time = mstTree.getNode('\MST_MAG::BP_N05_PHS').getData().dim_of().data()*1000
    ax.plot(mst_Bp_time, mst_Bp_N05, label='n = 5', color='blue', zorder=100)
    ax.plot(mst_Bp_time, mst_Bp_N06, label='n = 6', color='red')
    ax.plot(mst_Bp_time, mst_Bp_N07, label='n = 7', color='green')
    ax.set_xlim([start_t, end_t])
    if right:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
    ax.set_ylabel(r'$\delta_p$ (rad)')
    ax.set_xlabel('Time (ms)')
    
    # Estimate the phase from 30-35 secs
    if n05_avg:
        index_30 = np.argmin(np.abs(mst_Bp_time - 30.0))
        index_35 = np.argmin(np.abs(mst_Bp_time - 35.0))
        phase_avg = np.average(mst_Bp_N05[index_30:index_35+1])

        ax.text(0.87, 0.2, r'$\langle \delta_{{p,n=5}} \rangle = {0:.2f}$ rad'.format(phase_avg), horizontalalignment='center', color='black',
                  verticalalignment='center', transform = ax.transAxes, fontsize=10,
                  bbox=dict(facecolor='none', edgecolor='black', fc='w', boxstyle='round'))
    
def plot_Bp_vel_axis(ax, mstTree, start_t, end_t, right=False):
    mst_Bp_N05 = mstTree.getNode('\MST_MAG::BP_N05_VEL').getData().data()
    mst_Bp_N06 = mstTree.getNode('\MST_MAG::BP_N06_VEL').getData().data()
    mst_Bp_N07 = mstTree.getNode('\MST_MAG::BP_N07_VEL').getData().data()
    mst_Bp_time = mstTree.getNode('\MST_MAG::BP_N05_VEL').getData().dim_of().data()*1000
    ax.plot(mst_Bp_time, mst_Bp_N05, label='n = 5', color='blue', zorder=100)
    ax.plot(mst_Bp_time, mst_Bp_N06, label='n = 6', color='red')
    ax.plot(mst_Bp_time, mst_Bp_N07, label='n = 7', color='green')
    ax.set_xlim([start_t, end_t])
    if right:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
    ax.set_ylabel('vel (m/s)')
    ax.set_xlabel('Time (ms)')
    
def plot_ne_FIR_axis(ax, mstTree, start_t, end_t):
    # P06 is magnetic axis
    try:
        mst_ne = mstTree.getNode('\\fir_fast_N02').getData().data()
        mst_ne_time = mstTree.getNode('\\fir_fast_N02').getData().dim_of().data()
        ax.plot(mst_ne_time, mst_ne)
        ax.text(0.9, 0.85, 'FIR', horizontalalignment='center', color='Black',
                verticalalignment='center', transform = ax.transAxes, fontsize=12,
                bbox=dict(facecolor='none', edgecolor='black', fc='w', boxstyle='round'))
    except:
        # Default to CO2
        mst_ne = mstTree.getNode('\MST_OPS::N_CO2').getData().data() / 1.0e13
        mst_ne_time = mstTree.getNode('\MST_OPS::N_CO2').getData().dim_of().data()*1000
        ax.plot(mst_ne_time, mst_ne)
        ax.text(0.95, 0.85, r'$CO_2$', horizontalalignment='center', color='Black',
                verticalalignment='center', transform = ax.transAxes, fontsize=12)
    
    ax.set_xlim([start_t, end_t])
    ax.set_ylim([0, 1.5])
    ax.set_ylabel(r'$\langle n_e \rangle$ ($\times 10^{19} m^{-3}$)')
    ax.set_xlabel('Time (ms)')