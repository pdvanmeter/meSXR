"""
"""
from __future__ import division
import numpy as np
import MDSplus
from mst_ida.data.nickal2 import smooth_signal

# Constants for the MST FIR diagnostic system
fir_chord_names =           ['N32', 'N24', 'N17', 'N09', 'N02', 'P06', 'P13', 'P21', 'P28', 'P36', 'P43']
fir_chord_radius = np.array([-32.0, -24.0, -17.0,  -9.0,  -2.0,   6.0,  13.0,  21.0,  28.0,  36.0,  43.0])/100.
fir_chord_angle =  np.array([255.0, 250.0, 255.0, 250.0, 255.0, 250.0, 255.0, 250.0, 255.0, 250.0, 255.0])
fir_chord_length = np.array([81.97, 92.26, 98.29, 102.4, 103.9, 103.3, 100.7, 95.14, 87.64, 75.04, 58.48])/100.
fir_chord_angle_rad = fir_chord_angle*np.pi/180.

def get_FIR_data(shot, time, delta_t=1.0):
    """
    Time is in ms. Note that this only loads a single time point.
    """
    # Open the tree and get the desired time point
    mstTree = MDSplus.Tree('mst', shot, 'READONLY')
    fir_time = mstTree.getNode('\\fir_fast_N02').getData().dim_of().data()
    t_index = np.argmin(np.abs(time - fir_time))
    dt = fir_time[1] - fir_time[0]
    t_end = t_index + np.round(delta_t / dt).astype(int)
    
    ne_avg_data = np.zeros(len(fir_chord_names))
    ne_avg_err = np.zeros(len(fir_chord_names))
    
    for ii, name in enumerate(fir_chord_names):
        fir_data = mstTree.getNode('\\fir_fast_{0:}'.format(name)).getData().data()
        ne_avg_data[ii] = np.average(fir_data[t_index:t_end])
        ne_avg_err[ii] = np.std(fir_data[t_index:t_end])
    
    print('Retriving FIR data for t = {0:.2f} to t = {1:.2f} ms.'.format(time, time+delta_t))
    return ne_avg_data, ne_avg_err

def get_fir_signals(shot):
    """
    Time is in ms. This returns an array corresponding to measurements in time.
    """
    mstTree = MDSplus.Tree('mst', shot, 'READONLY')
    time = mstTree.getNode('\\fir_fast_N02').getData().dim_of().data()
    
    # Find t=0
    n_start = np.argmin(np.abs(time))
    fir_time = time[n_start:]
    
    fir_data = np.zeros([len(fir_chord_names), len(fir_time)])
    for ii, name in enumerate(fir_chord_names):
        fir_data[ii,:] = mstTree.getNode('\\fir_fast_{0:}'.format(name)).getData().data()[n_start:]
        
    return fir_data, fir_chord_radius, fir_time

def get_fir_resampled(shot, t_start=8.0, t_end=28.0, delta=0.5):
    """
    """
    fir_data, fir_radius, fir_time = get_fir_signals(shot)
    ne_sm, t_sm, err_sm = smooth_signal(fir_data.T, fir_time, dt=0.5)
    n_start = np.argmin(np.abs(fir_time - t_start))
    n_end = np.argmin(np.abs(fir_time - t_end))+1

    fir_data_temp = []
    fir_err_temp = []
    for index in range(len(fir_radius)):
        ne_sm, t_sm, err_sm = smooth_signal(fir_data[index,n_start:n_end], fir_time[n_start:n_end], dt=delta)
        fir_data_temp.append(ne_sm)
        fir_err_temp.append(err_sm)

    fir_data_sm = np.zeros([len(fir_radius), len(t_sm)])
    fir_err_sm = np.zeros([len(fir_radius), len(t_sm)])

    for index in range(len(fir_radius)):
        fir_data_sm[index,:] = fir_data_temp[index]
        fir_err_sm[index,:] = fir_err_temp[index]

    return fir_data_sm, fir_err_sm, fir_radius, t_sm