#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Package: mesxr.calibration
Module: plots
Author: Patrick VanMeter
Affiliation: Department of Physics, University of Wisconsin-Madison
Last Updated: November 2018

Description:
    This module is used to generate common plots to analyse the ressults of the ME-SXR calibration procedure.
Usage:
    TBD
"""
import numpy as np
import scipy as sp
import scipy.signal
import matplotlib.pyplot as plt

# Some colors to use for plotting
colors = ['xkcd:{0:}'.format(col) for col in ['red', 'green', 'blue', 'magenta', 'mustard yellow', 'burgundy',
                                              'dark orange', 'steel blue', 'bluish purple']]

def s_curves_plot(det, x, y):
    """
    Desctription:
        Creates a plot of the calibration S-curves for the pixel at the selected coordinates. Responses are normalized so
        that the last trimbit has a response of 0 and the first trimbit a response of 1.
    Inputs:
        - det = (Pilatus_Detector) The calibrated PILATUS detector object, found in the mesxr.calibration.trimscan module.
        - x = (int) The global x coordinate for the pixel to plot.
        - y = (int) The global y coordinate for the pixel to plot.
    Returns:
        - fig = (pyplot figure) The figure containing the plot.
    """
    pixels = det.get_pixels()
    trimscan_data = pixels[x,y].data

    # Create the plot with the appropriate fine sizes for Jupyter notebooks
    fig = plt.figure(1, figsize=(8,6), dpi=110)
    ax = fig.add_subplot(1,1,1)
    plt.tick_params(labelsize=14)

    for index, elem in enumerate(pixels[x,y].elements):
        # Scale for nicer plotting of the data
        bottom_adj = np.amin(trimscan_data[index])
        data_adj = trimscan_data[index] - bottom_adj
        scale_fact = np.amax(data_adj)
        data_scaled = data_adj / scale_fact
        sigma_scaled = np.sqrt(trimscan_data[index]) / scale_fact
        
        label = r'{0:}, $\chi^2 = {1:.1f}$'.format(elem, pixels[x,y].trimfit_chi2[index])
        ax.errorbar(pixels[x,y].trimbits[index], data_scaled, yerr=sigma_scaled, label=label,
                    capsize=1, ms=2, marker='o', linestyle=':', color=colors[index])
        
        # Scale and plot the model
        model_trims = np.linspace(0, 64, num=200)
        model_adj = pixels[x,y].s_curve_model(elem, model_trims) - bottom_adj
        model_scale = model_adj / scale_fact
        
        ax.plot(model_trims, model_scale, color=colors[index])
        
        # Plot the a0 points
        ax.axvline(x=pixels[x,y].trimfit_params[index, 0], color=colors[index], linestyle='--', linewidth=0.8)

    # Annotate the plot
    ax.legend(fontsize=12)
    ax.set_xlim([0,64])
    ax.set_ylim([0,1])
    ax.set_xlabel('Trimbit', fontsize=14)
    ax.set_ylabel('Response (norm.)', fontsize=14)

    # Add coordinates
    ax.text(0.5, 0.95, '(x,y) = {0:}'.format(pixels[x,y].coords), color='black',
        fontsize=16, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)

    return fig

def trimscan_curve(det, x, y, xlim=[2,15]):
    """
    Description:
        Creates a plot of the mapping between trimbit setting and the resulting threshold energy for a calibrated Pilatus_Detector
        object. Also includes uncerainty bands based on Poisson statistics.
    Inputs:
        - det = (Pilatus_Detector) The calibrated PILATUS detector object, found in the mesxr.calibration.trimscan module.
        - x = (int) The global x coordinate for the pixel to plot.
        - y = (int) The global y coordinate for the pixel to plot.
        - xlim = (tuple of numbers) The Argument to be passed along to the plot set_xlim function. Sets the limits
                on the x axis.
    Returns:
        - fig = (pyplot figure) The figure containing the plot.
    """
    pixels = det.get_pixels()

    # Create the plot with the appropriate fine sizes for Jupyter notebooks
    fig = plt.figure(1, figsize=(8,6), dpi=110)
    ax = fig.add_subplot(1,1,1)
    plt.tick_params(labelsize=14)

    # Plot the data with error bars
    ax.errorbar(pixels[x,y].energies, pixels[x,y].trimfit_params[:, 0], yerr=np.sqrt(pixels[x,y].trimfit_cov[:, 0, 0]),
            capsize=4, ms=4, marker='o', linestyle='none', color='xkcd:royal blue', label='Data')

    # Plot the fit
    en_model = np.linspace(xlim[0], xlim[1], num=200)
    trim_model = pixels[x,y].en_curve_model(en_model)
    ax.plot(en_model, trim_model, color='xkcd:orange', linewidth=1, label='Fit', zorder=100)

    # Plot the uncertainty
    model_sigma = pixels[x,y].en_curve_uncertainty(en_model)
    ax.plot(en_model, trim_model+model_sigma, color='xkcd:light blue', label=r'$1\sigma$', alpha=0.75)
    ax.plot(en_model, trim_model-model_sigma, color='xkcd:light blue', alpha=0.75)
    ax.fill_between(en_model, trim_model-model_sigma, trim_model+model_sigma, alpha=0.5, color='xkcd:light blue')

    ax.set_xlim(xlim)
    ax.set_ylim([0, det.num_trimbits])
    ax.set_xlabel(r'Threshold $E_c$ (keV)', fontsize=14)
    ax.set_ylabel('Trimbit', fontsize=14)
    ax.legend(loc='lower right', fontsize=14)

    ax.text(0.165, 0.95, '(x,y) = {0:}'.format(pixels[x,y].coords), color='black',
            fontsize=16, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
    ax.text(0.14, 0.88, r'$\chi^2 = {0:.2f}$'.format(pixels[x,y].enfit_chi2), color='black',
            fontsize=16, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)

    return fig

def trimscan_chi2_plot(det, element, chips, bins=500, plot_stdev=True, chi_range=(0,1000), cutoff=1000):
    """
    """
    # Create the plot with the appropriate fine sizes for Jupyter notebooks
    fig = plt.figure(1, figsize=(8,6), dpi=110)
    ax = fig.add_subplot(1,1,1)
    plt.tick_params(labelsize=14)

    # Load in the chi^2 values
    chi2_set = []

    for det_chip in det.modules[0,0].chips.ravel():
        if det_chip.number in chips:
            try:
                chi2_set.extend([pixel.trimfit_chi2[pixel.elements.index(element)] for pixel in det_chip.pixels.ravel() if element in pixel.elements])
            except:
                pass

    chi2_set = np.array(chi2_set)

    # Count and remove outliers
    rem_indices = np.where(chi2_set > cutoff)[0]
    num_removed = len(rem_indices)
    slice_indices = [i for i in range(len(chi2_set)) if i not in rem_indices]
    chi2_set = chi2_set[slice_indices]

    # Make the histogram
    hist = ax.hist(chi2_set, bins=bins, density=True, range=chi_range, color='xkcd:light blue')

    # Statistical info
    mean = np.nanmean(chi2_set)
    sigma = np.nanstd(chi2_set)

    ax.axvline(x=mean, color='xkcd:brick red', linestyle='dashed', label=r'$\chi^2$ mean')
    if plot_stdev:
        ax.axvline(x=mean+sigma, color='xkcd:red', label=r'$\chi^2$ stdev')
        ax.axvline(x=mean-sigma, color='xkcd:red')

    ax.set_xlabel(r'Trimbit fit $\chi^2$', fontsize=14)
    ax.set_ylabel('Probability density', fontsize=14)

    ax.legend(loc='upper right')

    # Add statistics lables
    ax.text(0.9, 0.8, element, color='black',
            fontsize=16, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
    ax.text(0.88, 0.72, r'$\langle \chi^2 \rangle = {0:.1f}$'.format(mean), color='black',
            fontsize=16, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
    if plot_stdev:
        ax.text(0.88, 0.64, r'$\sigma_{{\chi^2}} = {0:.1f}$'.format(sigma), color='black',
                fontsize=16, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)

    # Include outlier information for transparency
    ax.text(0.15, 0.95, '{0:} points removed'.format(num_removed), color='black',
            fontsize=12, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
    ax.text(0.15, 0.90, r'cutoff $\chi^2 > {0:}$'.format(cutoff), color='black',
            fontsize=12, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)

    return fig

def energy_chi2_plot(det, chips, bins=500, plot_stdev=False, chi_range=(0,100), cutoff=100):
    """
    """
    # Create the plot with the appropriate fine sizes for Jupyter notebooks
    fig = plt.figure(1, figsize=(8,6), dpi=110)
    ax = fig.add_subplot(1,1,1)
    plt.tick_params(labelsize=14)

    # Load in the chi^2 values
    chi2_set = []

    for det_chip in det.modules[0,0].chips.ravel():
        if det_chip.number in chips:
            chi2_set.extend([pixel.enfit_chi2 for pixel in det_chip.pixels.ravel()])

    chi2_set = np.array(chi2_set)

    # Count and remove outliers
    rem_indices = np.where(chi2_set > cutoff)[0]
    num_removed = len(rem_indices)
    slice_indices = [i for i in range(len(chi2_set)) if i not in rem_indices]
    chi2_set = chi2_set[slice_indices]

    # Make the histogram
    hist = ax.hist(chi2_set, bins=bins, density=True, range=chi_range, color='xkcd:light blue')

    # Statistical info
    mean = np.nanmean(chi2_set)
    sigma = np.nanstd(chi2_set)

    ax.axvline(x=mean, color='xkcd:brick red', linestyle='dashed', label=r'$\chi^2$ mean')
    if plot_stdev:
        ax.axvline(x=mean+sigma, color='xkcd:red', label=r'$\chi^2$ stdev')
        ax.axvline(x=mean-sigma, color='xkcd:red')

    ax.set_xlabel(r'Energy fit $\chi^2$', fontsize=14)
    ax.set_ylabel('Probability density', fontsize=14)

    ax.legend(loc='upper right')

    # Add statistics lables
    ax.text(0.9, 0.8, r'$E_c$ fits', color='black',
            fontsize=16, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
    ax.text(0.88, 0.72, r'$\langle \chi^2 \rangle = {0:.1f}$'.format(mean), color='black',
            fontsize=16, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
    if plot_stdev:
        ax.text(0.88, 0.64, r'$\sigma_{{\chi^2}} = {0:.1f}$'.format(sigma), color='black',
                fontsize=16, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)

    # Include outlier information for transparency
    ax.text(0.15, 0.95, '{0:} points removed'.format(num_removed), color='black',
            fontsize=12, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
    ax.text(0.15, 0.90, r'cutoff $\chi^2 > {0:}$'.format(cutoff), color='black',
            fontsize=12, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)

    return fig

def corner_plot(data, labels, ranges, bins=250, figsize=(8,6), plt_label='none'):
    """
    """
    fig = plt.figure(1, figsize=figsize, dpi=110)
    num_params = len(data)

    # Draw the scatter plots on the off-diagonals
    for col in range(num_params):
        for row in range(col+1):
            index = row*num_params + (col+1)
            ax = fig.add_subplot(num_params, num_params, index)
            
            # Diagonal histograms
            if row == col:
                counts, edges, patches = ax.hist(data[col], bins=bins, range=ranges[col],
                                                color='xkcd:royal blue')
                
                # Enforce consisntent spacing of grid lines
                delta_x = (ranges[col][1] - ranges[col][0])/5.
                max_counts = np.amax(counts)
                ticks_x = np.arange(ranges[col][0]+delta_x, ranges[col][1], delta_x)
                ticks_y = np.arange(0, 1.2*max_counts, max_counts/4.)

                ax.xaxis.set_ticks(ticks_x)
                ax.yaxis.set_ticks(ticks_y[0:-1])
                ax.set_ylabel('Counts')
                ax.set_xlabel(labels[row])

                ax.set_xlim(ranges[row])
                ax.grid(axis='x')

                    # Add annotation to the first plot, if desired
                if row == 0 and plt_label != 'none':
                    ax.text(0.85, 0.85, plt_label, color='red',
                            fontsize=16, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes,
                            bbox=dict(facecolor='none', edgecolor='black', fc='w', boxstyle='round'))

            # Off-diagonal 2D histograms
            else:
                hbins = [np.linspace(ranges[col][0], ranges[col][1], num=100),
                         np.linspace(ranges[row][0], ranges[row][1], num=100)]
                hist, xedges, yedges = np.histogram2d(data[col], data[row], bins=hbins)

                hist_masked = np.ma.masked_where(hist == 0, hist)
                pmesh = ax.pcolormesh(xedges, yedges, hist_masked.T, cmap='jet')
                ax.set_xlim(ranges[col])
                ax.set_ylim(ranges[row])
                
                ax.grid()

                # Get rid of overlapping tick marks and enforce consistnent spacing
                delta_y = (ranges[row][1] - ranges[row][0])/5.
                delta_x = (ranges[col][1] - ranges[col][0])/5.
                ticks_y = np.arange(ranges[row][0]+delta_y, ranges[row][1], delta_y)
                ticks_x = np.arange(ranges[col][0]+delta_x, ranges[col][1], delta_x)
                ax.yaxis.set_ticks(ticks_y)
                ax.xaxis.set_ticks(ticks_x)

                if row == 0:
                    ax.xaxis.tick_top()
                    ax.set_xlabel(labels[col])
                    ax.xaxis.set_label_position('top') 
                else:
                    ax.xaxis.set_ticklabels([])
                if col == num_params-1:
                    ax.yaxis.tick_right()
                    ax.set_ylabel(labels[row])
                    ax.yaxis.set_label_position('right') 
                else:
                    ax.yaxis.set_ticklabels([])
                
    # Remove spacing between plots
    plt.subplots_adjust(wspace=0, hspace=0)

    return fig

def trimscan_corner_plot(det, element, ranges, chips=range(16)):
    """
    """
    pixels = det.get_pixels()

    labels = [r'$a_0$', r'$a_1$',   r'$a_2$',     r'$a_3$',   r'$a_4$',    r'$a_5$']
    data = []

    # Loop over pixels and do the analysis
    for index in range(len(labels)):
        fit_data = []
        for x in range(pixels.shape[0]):
            for y in range(pixels.shape[1]):
                # Don't include data which is not used - this exludes all gap pixels
                if pixels[x, y].good_enfit and pixels[x,y].chip in chips:
                    if element in pixels[x,y].elements:
                        elem_index = pixels[x, y].elements.index(element)
                        
                        if pixels[x, y].good_trimfits[elem_index]:
                            fit_data.append(pixels[x, y].trimfit_params[elem_index, index])
        data.append(fit_data)

    # Make the plot using the general corner plot function
    fig = corner_plot(data, labels, ranges, figsize=(16,12), plt_label=element)
    
    return fig

def energy_corner_plot(det, ranges, chips=range(16)):
    """
    """
    pixels = det.get_pixels()

    labels = [r'$c_0$', r'$c_1$',   r'$c_2$']
    data = []

    # Loop over pixels and do the analysis
    for index in range(len(labels)):
        fit_data = []
        for x in range(pixels.shape[0]):
            for y in range(pixels.shape[1]):
                # Don't include data which is not used - this exludes all gap pixels
                if pixels[x, y].good_enfit and pixels[x,y].chip in chips:
                    fit_data.append(pixels[x, y].enfit_params[index])
        data.append(fit_data)

    # Make the plot using the general corner plot function
    fig = corner_plot(data, labels, ranges, figsize=(8,6))
    
    return fig

def uniform_treshold_trimbit_maps(det, threshold_set, figsize=(12,12)):
    """
    """
    # Get the trimbits from the calibration data
    pixels = det.get_pixels()
    trimbit_maps = np.zeros([len(threshold_set), pixels.shape[0], pixels.shape[1]])

    for index, thresh in enumerate(threshold_set):
        for x in range(pixels.shape[0]):
            for y in range(pixels.shape[1]):
                if pixels[x,y].good_enfit:
                    trimbit_maps[index, x, y] = pixels[x, y].en_curve_model(thresh)
                else:
                    trimbit_maps[index, x, y] = np.nan

    # Create the plots
    fig = plt.figure(1, figsize=figsize, dpi=110)
    num_col = len(threshold_set)/2 + len(threshold_set)%2

    for index, thresh in enumerate(threshold_set):
        ax = fig.add_subplot(num_col, 2, index+1)
        image = ax.imshow(trimbit_maps[index, :, :].T, vmin=0, vmax=63)
        
        # Format the plot to remove overlap
        # Only label the axes on the edge
        if index % 2 == 0:
            ax.set_ylabel('Y Index')
        else:
            ax.yaxis.set_ticklabels([])

        if index == 0 or index == 1:
            ax.set_xlabel('X Index')
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top') 
        else:
            ax.xaxis.set_ticklabels([])
            
        ax.text(0.85, 0.85, '{0:.0f} keV'.format(thresh), color='red',
                fontsize=14, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes,
                bbox=dict(facecolor='none', edgecolor='black', fc='w', boxstyle='round'))

    #plt.tick_params(labelsize=14)
    plt.subplots_adjust(wspace=0.025, hspace=0.05)
    cax = fig.add_axes([0.1, 0.05, 0.9, 0.02])
    fig.colorbar(image, cax=cax, orientation='horizontal', label=r'Trimbit settings')

    return fig

def uniform_treshold_trimbit_distributions(det, threshold_set, chips=range(16), figsize=(8, 6)):
    """
    """
    # Create a figure to plot all the data
    fig = plt.figure(1, figsize=figsize, dpi=110)
    num_col = len(threshold_set)/2 + len(threshold_set)%2

    # Don't include bad pixels or chip boundaries
    pixels = det.get_pixels()

    for index, thresh in enumerate(threshold_set):
        ax = fig.add_subplot(num_col, 2, index+1)
        trim_data = []

        for x in range(pixels.shape[0]):
            for y in range(pixels.shape[1]):
                if pixels[x,y].good_enfit:
                    # Make sure that the result is within the valid range
                    tbit = pixels[x, y].en_curve_model(thresh)
                    if tbit < 0:
                        tbit = 0.
                    elif tbit > 63:
                        tbit = 63.
                    
                    trim_data.append(tbit)
        
        trim_data = np.array(trim_data)
        
        # Make the histogram
        hist, bins, patches = ax.hist(trim_data, bins=250, range=[0,64], color='xkcd:light blue')
        
        # Include some basic statistics
        stdev = np.std(trim_data)
        mean = np.mean(trim_data)
        ax.axvline(x=mean, color='black', linestyle='dashed')
        ax.axvline(x=mean+stdev, color='green', linestyle='dashed')
        ax.axvline(x=mean-stdev, color='green', linestyle='dashed')

        # Position the text legibly
        if mean <=25:
            text_x_loc = mean + stdev + 2
        else:
            text_x_loc = mean - stdev - 20
        
        ax.text(text_x_loc, 0.55*max(hist), r'$\sigma =$ {:.2f}'.format(stdev), fontsize=10, color='green')
        ax.text(text_x_loc, 0.8*max(hist), r'$\langle t \rangle =$ {:.2f}'.format(mean), fontsize=10)

        # Put the threshold label somewhere it will not overlap with other text
        if mean < 44:
            ax.text(0.85, 0.8, '{0:.1f} keV'.format(thresh), horizontalalignment='center', verticalalignment='center',
                    transform = ax.transAxes, fontsize=12, color='red',
                    bbox=dict(facecolor='none', edgecolor='black', fc='w', boxstyle='round'))
        else:
            ax.text(0.15, 0.8, '{0:.1f} keV'.format(thresh), horizontalalignment='center', verticalalignment='center',
                    transform = ax.transAxes, fontsize=12, color='red',
                    bbox=dict(facecolor='none', edgecolor='black', fc='w', boxstyle='round'))

        max_counts = max(hist)
        ticks = np.arange(0, 1.2*max_counts, max_counts/4)
        ax.yaxis.set_ticks(ticks[0:-1])
        ax.xaxis.set_ticks(range(4,65,8))
        ax.set_xlim([0,63])
        ax.grid(axis='x')

        # Only label the axes on the edge
        if index % 2 == 0:
            ax.set_ylabel('Counts')
        else:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position('right')

        if index == len(threshold_set)-1 or index == len(threshold_set)-2:
            ax.set_xlabel(r'Requested Trimbit')
        else:
            ax.xaxis.set_ticklabels([])

    # Remove spacing between plots
    plt.subplots_adjust(wspace=0, hspace=0)
    #plt.tick_params(labelsize=14)

    return fig

def uniform_treshold_delta_E(det, threshold_set, chips=range(16), figsize=(8, 6), xrange=[-0.25,0.25]):
    """
    """
    # Create a figure to plot all the data
    fig = plt.figure(1, figsize=figsize, dpi=110)
    num_col = len(threshold_set)/2 + len(threshold_set)%2

    # Don't include bad pixels or chip boundaries
    pixels = det.get_pixels()

    # Keep up with threshold widths
    width_set = np.zeros(len(threshold_set))

    for index, thresh in enumerate(threshold_set):
        ax = fig.add_subplot(num_col, 2, index+1)
        thresh_data = []

        # MODIFY: loop over pixels, find the trimbit, round, and map back to Ec
        for x in range(pixels.shape[0]):
            for y in range(pixels.shape[1]):
                if pixels[x,y].good_enfit:
                    # Make sure that the result is within the valid range
                    tbit = pixels[x, y].trimbit_from_threshold(thresh)
                    if tbit < 0:
                        tbit = 0.
                    elif tbit > 63:
                        tbit = 63.
                    else:
                        tbit = np.round(tbit)
                    
                    thresh_data.append( pixels[x,y].threshold_from_trimbit(tbit) )
        
        thresh_data = np.array(thresh_data)

        en_diff = thresh_data - thresh
        hist, bins, patches = ax.hist(en_diff, bins=200, range=xrange, color='xkcd:light blue')
        ax.text(0.85, 0.8, '{0:.1f} keV'.format(thresh), horizontalalignment='center', verticalalignment='center',
                transform = ax.transAxes, fontsize=12, color='red',
                bbox=dict(facecolor='none', edgecolor='black', fc='w', boxstyle='round'))
        
        # Smooth the data to determine the width of the spread
        bin_w = (max(bins) - min(bins)) / (len(bins) - 1.)
        bin_points = np.arange(min(bins)+bin_w/2., max(bins)+bin_w/2., bin_w)
        hist_smooth = sp.signal.savgol_filter(hist, 11, 2)
        
        max_count = np.amax(hist_smooth)
        zero_index = np.argmin(np.abs(bin_points))
        half_index_l = np.argmin(np.abs(hist_smooth[:zero_index] - max_count/2.))
        half_index_r = zero_index + np.argmin(np.abs(hist_smooth[zero_index:] - max_count/2.))
        width_l = bin_points[zero_index] - bin_points[half_index_l]
        width_r = bin_points[half_index_r] - bin_points[zero_index]
        delta_eV = (width_r + width_l)*1000./2.
        width_set[index] = delta_eV

        ax.axvline(x=bin_points[half_index_l], color='red', linewidth=0.8)
        ax.axvline(x=bin_points[half_index_r], color='red', linewidth=0.8)
        ax.axvline(x=bin_points[zero_index], color='black', linestyle='dashed', linewidth=0.5)
        ax.text(0.5, 0.1, r'$\Delta E = $ {0:3.0f} eV'.format(delta_eV), horizontalalignment='center',
                verticalalignment='center', transform = ax.transAxes, fontsize=12, color='black')

        max_counts = max(hist)
        ticks = np.arange(0, 1.2*max_counts, max_counts/4)
        ax.yaxis.set_ticks(ticks[0:-1])
        ax.grid(axis='x')

        # Only label the axes on the edge
        if index % 2 == 0:
            ax.set_ylabel('Counts')
        else:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position('right')

        if index == len(threshold_set)-1 or index == len(threshold_set)-2:
            ax.set_xlabel(r'Threshold - Avg. (keV)')
        else:
            ax.xaxis.set_ticklabels([])

    # Remove spacing between plots
    plt.subplots_adjust(wspace=0, hspace=0)

    return fig, width_set