"""
This module contains a variety of methods aimed at producing plots using data collected by the ME-SXR detector.
These methdods present raw data with minimal processing.
This version is outdated. Use the module "plots" instead.
"""
import MDSplus
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import pilatus.analysis as an
import pilatus.calibration as calib
import pilatus.configuration as config

def fit_func(x, mean, std, amplitude):
    """
    This function is simply a Gaussian used in fitting procedures.
    """
    return amplitude*np.exp(-(x - mean)**2/(2*std**2) )

def plot_frame(shot, frame, label, vmin=0, vmax=1100, scale=5, remove_edges=True, save=False, fit_data=False,
                ff_corr=np.ones([calib.M_SIZE_X, calib.M_SIZE_Y])):
    """
    This function produces a figure made up of two plots. The top plot is the raw image at the sppecified
    frame, and the bottom image is the x-projection (sum of all pixels in the y-direction for each x). This
    can optionally be fit to a Gaussian function to show agreement.
    """
    # Assemble the data and create the figure
    base_path = '/home/pdvanmeter/data/meSXR/images/MST_data'
    data = an.load_data(shot, base_path=base_path, remove_edges=remove_edges)
    fig = plt.figure(1, figsize=(8,6), dpi=110)
    label += ', {0:d} - {1:d} ms'.format(frame*2, frame*2+1)

    data_corr = data[:, :, frame]*ff_corr
    
    # Plot 1 - Full Image
    ax1 = fig.add_subplot(2,1,1)
    cax1 = ax1.imshow(data_corr.T, vmin=vmin, vmax=vmax, aspect='auto')
    ax1.xaxis.set_ticklabels([])
    ax1.set_ylabel('Y pixel (toroidal)')
    ax1.set_xlabel('X pixel')
    ax1.set_title(label)

    # Plot 2 - x-axis projection
    ax2 = fig.add_subplot(2,1,2)
    image_x_proj = np.ma.sum(data_corr, axis=1)
    ax2.errorbar(range(calib.M_SIZE_X), image_x_proj / 10.**scale, yerr=np.ma.sqrt(image_x_proj) / 10.**scale,
                 capsize=2, ms=1, marker='o', linestyle='none')
    ax2.set_xlabel('X pixel (radial)')
    ax2.set_ylabel(r'Counts ($\times 10^{0:d}$)'.format(scale))
    ax2.grid(axis='x')
    
    # Remove the space between plots and add the colorbar
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.colorbar(cax1, ax=[ax1, ax2], label='Counts')
    
    # Add the shot number
    ax2.text(0.15, 0.9, str(shot), horizontalalignment='center',
             verticalalignment='center', transform = ax2.transAxes, fontsize=14)
    
    # Test goodness of data by fitting to a Gaussian in the x projection
    if fit_data:
        data_x = []
        x_pixels = []
        for x_index in range(calib.M_SIZE_X):
            if image_x_proj.mask[x_index] == False:
                data_x.append(image_x_proj[x_index] + 0.01)
                x_pixels.append(x_index)

        p0 = [250, 80., data_x[250]]
        popt, pcov = sp.optimize.curve_fit(fit_func, x_pixels, data_x, p0=p0, sigma=np.sqrt(data_x))

        mean = popt[0]
        std = np.abs(popt[1])
        width = 2.355*std

        x_model = np.linspace(0,calib.M_SIZE_X, num=2000)
        y_model = fit_func(x_model, mean, std, popt[2])
        ax2.plot(x_model, y_model/ 10.**scale)

        # Chi-squared in the x projection
        chi_sq = np.sum( (data_x - fit_func(x_pixels, mean, std, popt[2]))**2 / data_x)
        dof = len(data_x)
        red_chi_sq = chi_sq / dof
        ax2.text(0.5, 0.1, r'$\chi^2_\nu = {0:.1f}$'.format(red_chi_sq), horizontalalignment='center',
                            verticalalignment='center', transform = ax2.transAxes, fontsize=14)
    
    # Save the figure if desired, using the caption as the filename
    if save:
        fig.savefig('./figures/{0:10d} - {1:}.png'.format(shot, label))
        
    return fig

def plot_time(shot, start_image, end_image, label='', scale=4, delta_t=1., remove_edges=True, save=False):
    """
    This function produces a time trace of the x-projection for a given shot.
    """
    # Assemble the data - data[x_index, y_index, frame]
    base_path = '/home/pdvanmeter/data/meSXR/images/MST_data'
    data = an.load_data(shot, base_path=base_path, remove_edges=remove_edges)
    frames = np.arange(start_image, end_image+1)
    time = 2.*frames + delta_t/2.
    
    # Project the data to 1D, removing masked pixels
    num_x_points = data.count(axis=0)[5,0]
    time_series_data = np.zeros([num_x_points, len(time)])
    x_pixels = np.zeros(num_x_points)
    for t_index, frame in enumerate(frames):
        image_x_proj = np.ma.sum(data[:, :, frame], axis=1)
        data_x_index = 0
        for x_index in range(calib.M_SIZE_X):
            if image_x_proj.mask[x_index] == False:
                time_series_data[data_x_index, t_index] = image_x_proj[x_index]
                x_pixels[data_x_index] = x_index
                data_x_index += 1
    
    # Make the plot
    fig = plt.figure(1, figsize=(8,8), dpi=110)
    ax1 = fig.add_subplot(3,1,1)
    cax = ax1.contourf(time, x_pixels, time_series_data/10.**scale, 100)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('X (radial)')
    ax1.set_title(label)
    ax1.set_xlim([time[0], time[-1]])
    
    # Include shot number
    ax1.text(0.15, 0.9, str(shot), horizontalalignment='center', color='white',
             verticalalignment='center', transform = ax1.transAxes, fontsize=14)
    ax1.text(0.9, 0.9, label, horizontalalignment='center', color='white',
             verticalalignment='center', transform = ax1.transAxes, fontsize=14)
    
    # Add subplots for the current and density
    mstTree = MDSplus.Tree('mst', shot, 'READONLY')
    
    ax2 = fig.add_subplot(3,1,2)
    plot_Ip_axis(ax2, mstTree, time[0], time[-1])
    
    ax3 = fig.add_subplot(3,1,3)
    plot_ne_FIR_axis(ax3, mstTree, time[0], time[-1])
    
    # Remove space between plots and add a colorbar
    plt.subplots_adjust(wspace=0, hspace=0)
    cb = fig.colorbar(cax, ax=[ax1, ax2, ax3], label=r'Counts Rate ($\times 10^{0:d}$ ph/ms)'.format(scale))
    
    # Save the figure if desired, using the caption as the filename
    if save:
        fig.savefig('./figures/time_{0:10d} - {1:}.png'.format(shot, label))
        
    return fig

def ME_plot_frame(shot, frame, thresholds, label='ME-SXR', scale=3, save=False, remove_edges=True):
    """
    This function produces a plot of the raw image and an x-pixel projection for a multi-energy configuration.
    Pixels are sorted according to the supplied threshold vector, which contians the energy threshold as a
    function of x-pixel index.
    """
    # Load the data
    base_path = '/home/pdvanmeter/data/meSXR/images/MST_data'
    data = an.load_data(shot, base_path=base_path, remove_edges=remove_edges)
    label += ', {0:d} - {1:d} ms'.format(frame*2, frame*2+1)
    
    # Sort the data by threshold
    image_x_proj = np.ma.sum(data[:, :, frame], axis=1)
    data_x_thr = {}
    x_arr_thr = {}
    
    for x_index in xrange(calib.M_SIZE_X):
        if image_x_proj.mask[x_index] == False:
            Ec = thresholds[x_index]
            if Ec not in data_x_thr.keys():
                # Check for new thresholds or skipped pixels
                data_x_thr[Ec] = []
                x_arr_thr[Ec] = []
            elif Ec != 0:
                data_x_thr[Ec].append(image_x_proj[x_index])
                x_arr_thr[Ec].append(x_index)
        
    # Plot the results
    fig = plt.figure(1, figsize=(8,6), dpi=110)
    ax1 = fig.add_subplot(2,1,1)
    cax1 = ax1.imshow(data[:,:,frame].T, aspect='auto')
    
    ax2 = fig.add_subplot(2,1,2)
    for Ec in np.sort(data_x_thr.keys()):
        data_x_thr[Ec] = np.array(data_x_thr[Ec])
        x_arr_thr[Ec] = np.array(x_arr_thr[Ec])
        
        ax2 = fig.add_subplot(2,1,2)
        ax2.errorbar(x_arr_thr[Ec], data_x_thr[Ec] / 10.**scale, yerr=np.sqrt(data_x_thr[Ec]) / 10.**scale,
                     capsize=2, ms=1, marker='o', linestyle='solid', label='{0:.1f}'.format(Ec))
    
    # Make the plots look good
    ax1.xaxis.set_ticklabels([])
    ax1.set_ylabel('Y pixel (toroidal)')
    ax1.set_xlabel('X pixel')
    ax1.set_title(label)
    
    ax2.set_xlabel('X pixel (radial)')
    ax2.set_ylabel(r'Counts ($\times 10^{0:d}$)'.format(scale))
    ax2.grid(axis='x')
    ax2.legend(loc='upper right')
    ax2.set_xlim([0, calib.M_SIZE_X])
    
    # Remove spacing between plots
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.colorbar(cax1, ax=[ax1, ax2], label='Counts')
    
    # Add the shot number
    ax2.text(0.15, 0.9, str(shot), horizontalalignment='center',
             verticalalignment='center', transform = ax2.transAxes, fontsize=14)
    
    # Save the figure if desired, using the caption as the filename
    if save:
        fig.savefig('./figures/ME_{0:10d} - {1:}.png'.format(shot, label))
        
    return fig

def ME_time_plot(shot, start_image, end_image, thresholds, vmax=5, scale=4, label='', delta_t=1., remove_edges=True, save=False):
    """
    This function produces the time traces for the x-projection of each threshold in a multi-energy configuration.
    Pixels are sorted according to the supplied threshold vector, which contians the energy threshold as a
    function of x-pixel index.
    """
    # Assemble the data - data[x_index, y_index, frame]
    time_series_data, x_pixels, time, Ec_list = get_ME_data(shot, start_image, end_image, thresholds, delta_t=delta_t, remove_edges=remove_edges)
    
    # Make the plot
    fig = plt.figure(1, figsize=(8,8), dpi=110)
    axes = []
    contours = []
    
    for en_index, Ec in enumerate(Ec_list):
        ax = fig.add_subplot(5,2,en_index+1)
        cax = ax.contourf(time, x_pixels[Ec], time_series_data[Ec]/10.**scale, 50, vmin=0, vmax=vmax)
        
        # Axis labels
        if en_index in (0,1):
            # Top Row
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top')
            ax.set_xlabel('Time [ms]')
        else:
            # Middle
            ax.xaxis.set_ticklabels([])
            
        if en_index % 2 == 1:
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
        ax.set_xlim([time[0], time[-1]])
        axes.append(ax)
        contours.append(cax)
        
    # Add in two additional frames for Ip and Bp
    mstTree = MDSplus.Tree('mst', shot, 'READONLY')
    
    axes.append(fig.add_subplot(5,2,9))
    plot_Ip_axis(axes[-1], mstTree, time[0], time[-1])
    
    axes.append(fig.add_subplot(5,2,10))
    plot_Bp_amp_axis(axes[-1], mstTree, time[0], time[-1])
    
    # Get rid of space between plots and add a colorbar
    plt.subplots_adjust(wspace=0, hspace=0)
    cb = fig.colorbar(contours[0], ax=axes, label=r'Counts Rate ($\times 10^{0:d}$ ph/ms)'.format(scale), orientation='horizontal')
    
    # Add labels to the plot
    plt.suptitle(str(shot) + ': ' + label)
    
    # Save the figure if desired, using the caption as the filename
    if save:
        fig.savefig('./figures/time_{0:10d} - {1:}.png'.format(shot, label))
        
    return fig


def get_ME_data(shot, start_image, end_image, thresholds, delta_t=1., remove_edges=True):
    # Assemble the data - data[x_index, y_index, frame]
    base_path = '/home/pdvanmeter/data/meSXR/images/MST_data'
    data = an.load_data(shot, base_path=base_path, remove_edges=remove_edges)
    frames = np.arange(start_image, end_image+1)
    time = 2.*frames + delta_t/2.
    
    # Determine the energy list
    Ec_list = []
    num_x_points = {}
    for x_index in xrange(calib.M_SIZE_X):
            Ec = thresholds[x_index]
            
            # Record the value of each threshold
            if Ec not in Ec_list and Ec != 0:
                Ec_list.append(Ec)
                num_x_points[Ec] = 0
            
            # Count up the number of points for each threshold
            if data.mask[x_index, 5, 0] == False:
                num_x_points[Ec] += 1
                
    Ec_list = np.sort(Ec_list)
    
    # Sort the data by threshold
    time_series_data = {}
    x_pixels = {}
    
    for Ec in Ec_list:
        time_series_data[Ec] = np.zeros([num_x_points[Ec], len(time)])
        x_pixels[Ec] = np.zeros(num_x_points[Ec])
    
    for f_index, frame in enumerate(frames):
        image_x_proj = np.ma.sum(data[:, :, frame], axis=1)
        x_data_index = {x:0 for x in Ec_list}
        for x_index in xrange(calib.M_SIZE_X):
            if image_x_proj.mask[x_index] == False:
                Ec = thresholds[x_index]
                time_series_data[Ec][x_data_index[Ec], f_index] = image_x_proj[x_index]
                x_pixels[Ec][x_data_index[Ec]] = x_index
                x_data_index[Ec] += 1
                
    return time_series_data, x_pixels, time, Ec_list

def plot_Ip_axis(ax, mstTree, start_t, end_t):
    mst_Ip = mstTree.getNode('\MST_OPS::ip').getData().data()
    mst_Ip_time = mstTree.getNode('\MST_OPS::ip').getData().dim_of().data()*1000
    ax.plot(mst_Ip_time, mst_Ip)
    ax.set_xlim([start_t, end_t])
    ax.set_ylim([0, 1.2*np.amax(mst_Ip)])
    ax.set_ylabel('Ip [kA]')
    ax.set_xlabel('Time [ms]')
    ax.text(0.5, 0.1, r'max $I_p = {0:.1f}$ kA'.format(np.amax(mst_Ip)), horizontalalignment='center', color='black',
                  verticalalignment='center', transform = ax.transAxes, fontsize=12)
    
def plot_Bp_amp_axis(ax, mstTree, start_t, end_t, right=True):
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
    ax.set_ylabel('|Bp| [G]')
    ax.set_xlabel('Time [ms]')
    
    ax.text(0.05, 0.8, '5', horizontalalignment='center', color='blue',
                  verticalalignment='center', transform = ax.transAxes, fontsize=12)
    ax.text(0.1, 0.8, '6', horizontalalignment='center', color='red',
                  verticalalignment='center', transform = ax.transAxes, fontsize=12)
    ax.text(0.15, 0.8, '7', horizontalalignment='center', color='green',
                  verticalalignment='center', transform = ax.transAxes, fontsize=12)
    
def plot_Bp_phs_axis(ax, mstTree, start_t, end_t, right=True):
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
    ax.set_ylabel('phs [rad]')
    ax.set_xlabel('Time [ms]')
    
def plot_Bp_vel_axis(ax, mstTree, start_t, end_t, right=True):
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
    ax.set_ylabel('vel [m/s]')
    ax.set_xlabel('Time [ms]')
    
def plot_ne_FIR_axis(ax, mstTree, start_t, end_t):
    try:
        mst_ne = mstTree.getNode('\\fir_fast_N02').getData().data()
        mst_ne_time = mstTree.getNode('\\fir_fast_N02').getData().dim_of().data()
        ax.plot(mst_ne_time, mst_ne)
        ax.text(0.9, 0.85, 'FIR', horizontalalignment='center', color='Black',
                verticalalignment='center', transform = ax.transAxes, fontsize=12)
    except:
        # Default to CO2
        mst_ne = mstTree.getNode('\MST_OPS::N_CO2').getData().data() / 1.0e13
        mst_ne_time = mstTree.getNode('\MST_OPS::N_CO2').getData().dim_of().data()*1000
        ax.plot(mst_ne_time, mst_ne)
        ax.text(0.9, 0.85, 'CO2', horizontalalignment='center', color='Black',
                verticalalignment='center', transform = ax.transAxes, fontsize=12)
    
    ax.set_xlim([start_t, end_t])
    ax.set_ylim([0, 1.5])
    ax.set_ylabel(r'$\langle n_e \rangle$ ($\times 10^{19} m^{-3}$)')
    ax.set_xlabel('Time [ms]')
    
def multi_info_plot(shot, en_indices, start_image, end_image, thresholds, vmax=5, scale=4, label='', delta_t=1., remove_edges=True, save=False):
    # Assemble the data - data[x_index, y_index, frame]
    time_series_data, x_pixels, time, Ec_list = get_ME_data(shot, start_image, end_image, thresholds, delta_t=delta_t, remove_edges=remove_edges)
    
    # Make the plot
    fig = plt.figure(1, figsize=(8,10), dpi=110)
    num_plts = len(en_indices)
    num_signals = 4
    axes = []
    contours = []
    
    for counter, index in enumerate(en_indices):
        ax = fig.add_subplot(num_signals+num_plts, 1, counter+1)
        Ec = Ec_list[index]
        cax = ax.contourf(time, x_pixels[Ec], time_series_data[Ec]/10.**scale, 50, vmin=0, vmax=vmax)
        
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
        ax.text(0.9, 0.85, '{0:.2f}'.format(Ec), horizontalalignment='center', color='white',
                verticalalignment='center', transform = ax.transAxes, fontsize=12)
            
        # Keep up with the axes
        ax.set_xlim([time[0], time[-1]])
        axes.append(ax)
        contours.append(cax)
        
    # Add the other signals
    mstTree = MDSplus.Tree('mst', shot, 'READONLY')
    
    axes.append(fig.add_subplot(num_signals+num_plts, 1, num_plts+1))
    plot_Ip_axis(axes[-1], mstTree, time[0], time[-1])
    
    axes.append(fig.add_subplot(num_signals+num_plts, 1, num_plts+2))
    plot_Bp_amp_axis(axes[-1], mstTree, time[0], time[-1], right=False)
    
    axes.append(fig.add_subplot(num_signals+num_plts, 1, num_plts+3))
    plot_Bp_phs_axis(axes[-1], mstTree, time[0], time[-1], right=False)
    
#    axes.append(fig.add_subplot(num_signals+num_plts, 1, num_plts+4))
#    plot_Bp_vel_axis(axes[-1], mstTree, time[0], time[-1], right=False)
    
    axes.append(fig.add_subplot(num_signals+num_plts, 1, num_plts+4))
    plot_ne_FIR_axis(axes[-1], mstTree, time[0], time[-1])
    
    # Get rid of space between plots and add a colorbar
    plt.subplots_adjust(wspace=0, hspace=0)
    cb = fig.colorbar(contours[0], ax=axes, label=r'Counts Rate ($\times 10^{0:d}$ ph/ms)'.format(scale), orientation='horizontal',
                      fraction=0.046, pad=0.04)
#    divider = make_axes_locatable(axes[0])
#    cax = divider.append_axes("right", size="5%", pad=0.05)
#    plt.colorbar(contours[0], cax=cax)
    
    # Add labels to the plot
    plt.suptitle(str(shot) + ': ' + label)
    
    # Save the figure if desired, using the caption as the filename
    if save:
        fig.savefig('./figures/signals_{0:10d} - {1:}.png'.format(shot, label))
        
    return fig