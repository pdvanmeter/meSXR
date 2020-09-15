"""
"""
from __future__ import division
import numpy as np
import scipy as sp
import scipy.signal
import MDSplus
from mst_ida.utilities.structures import AttrDict

def get_magnetics(shot, t_start=10.0, t_end=60.0, delta_t=0.1, components=['BP', 'BT'], modes=range(5,16), dom_mode='N05'):
    # Container for the results
    mag = {}

    # Format mode numbers to string from. "N05', 'N06', etc.
    modes_str = ['N'+str(x).zfill(2) for x in modes]

    # Connect to the tree
    try:
        mstTree = MDSplus.Tree('mst', shot, 'READONLY')
    except:
        print('ERROR: Shot not found')

    # Variables to assist with loading the time base
    time_loaded = False
    n_start = 0
    n_end = 0
    delta_n = 0

    # Iterate over the mode strings and load the desired data
    for comp in components:
        mag[comp] = {}
        for mode in modes_str:
            # Load in the data
            try:
                phs_node = mstTree.getNode('\\MST_MAG::'+ comp +'_'+ mode + '_PHS')
                amp_node = mstTree.getNode('\\MST_MAG::'+ comp +'_'+ mode + '_AMP')
                vel_node = mstTree.getNode('\\MST_MAG::'+ comp +'_'+ mode + '_VEL')
            except:
                print('Error loading node. Please check shot number.')
            else:
                mag[comp][mode] = {}
                phs = phs_node.getData().data()
                amp = amp_node.getData().data()
                vel = vel_node.getData().data()

                # Record the time base for the first iteration only, and downsample it
                if not time_loaded:
                    t_mag = phs_node.getData().dim_of().data()*1000.      	# in ms
                    t_smooth = np.around(np.float64(t_mag),3)    		# Deal with rounding problems
                    n_start = np.argmin(np.abs(t_smooth - t_start)) 	# Find starting index
                    n_end = np.argmin(np.abs(t_smooth - t_end))		# Find the ending index
                    delta_n = int((delta_t)/(t_smooth[1] - t_smooth[0]))	# The spacing between samples
                    
                    mag['Time'] = t_smooth[n_start:n_end+1:delta_n]
                    time_loaded = True

                # Downsample the magnetic field data
                mag[comp][mode]['Phase']     = phs[n_start:n_end+1:delta_n]
                mag[comp][mode]['Amplitude'] = amp[n_start:n_end+1:delta_n]
                mag[comp][mode]['Velocity']  = vel[n_start:n_end+1:delta_n]
                
    # Compute the spectral index
    total_mode_energy = 0.0
    for mode in modes_str:
        total_mode_energy += (mag['BP'][mode]['Amplitude']**2 + mag['BT'][mode]['Amplitude']**2)
    
    spec = 0.0
    for mode in modes_str:
        spec += ( (mag['BP'][mode]['Amplitude']**2 + mag['BT'][mode]['Amplitude']**2) / total_mode_energy )**2

    # Include this into the magnetics dictionary
    mag['Index'] = 1./spec
    mag['Avg Index'] = np.average(mag['Index'])
    
    # Compute the total dominant and secondary mode amplitudes
    if dom_mode in modes_str:
        mag['Dominant'] = np.sqrt(mag['BP'][dom_mode]['Amplitude']**2 + mag['BT'][dom_mode]['Amplitude']**2)
        mag['Secondary'] = np.zeros(len(mag['Dominant']))
        for mode in modes_str:
            if mode != dom_mode:
                mag['Secondary'] += mag['BP'][mode]['Amplitude']**2 + mag['BT'][mode]['Amplitude']**2
        mag['Secondary'] = np.sqrt(mag['Secondary'])

    # Include equilibrium measurements
    bp_node = mstTree.getNode('\\MST_MAG::BP_TORARR_EQUIL')
    bp = bp_node.data()
    bp_time = bp_node.dim_of().data() * 1000.
    mag['BP']['EQ'] = bp[n_start:n_end+1:delta_n]

    bt_node = mstTree.getNode('\\MST_MAG::BT_TORARR_EQUIL')
    bt = bt_node.data()
    bt_time = bt_node.dim_of().data() * 1000.
    mag['BT']['EQ'] = bt[n_start:n_end+1:delta_n]
    
    return AttrDict(mag)
    
def get_Ip(shot, t_start, t_end, delta_t=None):
    """
    """
    mstTree = MDSplus.Tree('mst', shot, 'READONLY')
    ip_node = mstTree.getNode('\\ip')
    ip = ip_node.data()
    t_ip = ip_node.dim_of().data() * 1000.
    
    t_smooth = np.around(np.float64(t_ip), 3)            # Deal with rounding problems
    n_start = np.argmin(np.abs(t_smooth - t_start))      # Find starting index
    n_end = np.argmin(np.abs(t_smooth - t_end))          # Find the ending index
    
    if delta_t is None:
        delta_n = 1
    else:
        delta_n = int((delta_t)/(t_smooth[1] - t_smooth[0])) # The spacing between samples
    
    return ip[n_start:n_end+1:delta_n], t_smooth[n_start:n_end+1:delta_n]

def get_vtg(shot, t_start, t_end):
    """
    Since the purpose of looking at VTG is typically to identify sawtooth events, I decided
    not to innclude any resampling options.
    """
    mstTree = MDSplus.Tree('mst', shot, 'READONLY')
    vtg_node = mstTree.getNode('\\vtg')
    vtg = vtg_node.data()
    vtg_time = vtg_node.dim_of().data() * 1000.

    n_start = np.argmin(np.abs(vtg_time - t_start))      # Find starting index
    n_end = np.argmin(np.abs(vtg_time - t_end))          # Find the ending index

    return vtg[n_start:n_end], vtg_time[n_start:n_end]

def get_vpg(shot, t_start, t_end):
    """
    Since the purpose of looking at VTG is typically to identify sawtooth events, I decided
    not to innclude any resampling options.
    """
    mstTree = MDSplus.Tree('mst', shot, 'READONLY')
    vpg_node = mstTree.getNode('\\vpg')
    vpg = vpg_node.data()
    vpg_time = vpg_node.dim_of().data() * 1000.

    n_start = np.argmin(np.abs(vpg_time - t_start))      # Find starting index
    n_end = np.argmin(np.abs(vpg_time - t_end))          # Find the ending index

    return vpg[n_start:n_end], vpg_time[n_start:n_end]

def get_energy(shot, t_start, t_end):
    """
    """
    mstTree = MDSplus.Tree('mst', shot, 'READONLY')
    en_node = mstTree.getNode('\\MST_OPS::ENER_ALPHA')
    energy = en_node.getData().data()
    en_time = en_node.getData().dim_of().data()*1000

    n_start = np.argmin(np.abs(en_time - t_start))      # Find starting index
    n_end = np.argmin(np.abs(en_time - t_end))          # Find the ending index

    return energy[n_start:n_end], en_time[n_start:n_end]

def get_reversal(shot, t_start, t_end):
    """
    """
    mstTree = MDSplus.Tree('mst', shot, 'READONLY')
    F_node = mstTree.getNode('\\MST_OPS::F')
    F = F_node.getData().data()
    F_time = F_node.getData().dim_of().data()*1000

    n_start = np.argmin(np.abs(F_time - t_start))      # Find starting index
    n_end = np.argmin(np.abs(F_time - t_end))          # Find the ending index

    return F[n_start:n_end], F_time[n_start:n_end]

# --------------------------------------- Summary Stats ---------------------------------------
def get_max_Ip(shot, t_end=80):
    ip, t = get_Ip(shot, 0, t_end)
    return np.amax(ip)

def get_avg_phs(shot, t_start=20, t_end=35, mode='N05'):
    mag = get_magnetics(shot, t_start=t_start, t_end=t_end)
    return np.average(mag['BP'][mode]['Phase']), np.std(mag['BP'][mode]['Phase'])

def get_avg_amp(shot, t_start=20, t_end=35, mode='N05'):
    mag = get_magnetics(shot, t_start=t_start, t_end=t_end)
    amp = np.sqrt(mag['BP'][mode]['Amplitude']**2 + mag['BT'][mode]['Amplitude']**2)
    return np.average(amp), np.std(amp)

def get_persistence(shot, t_start=20., t_end=35., delta_t=0.01):
    mag = get_magnetics(shot, t_start=20., t_end=35., delta_t=0.01)
    ns = mag['Index']
    return np.sum(ns < 2) / len(ns) * 100.

def count_sawteeth(shot, t_start, t_end):
    """
    Use VTG to count the number sawteeth between a given time interval.
    """
    vtg, vtg_time = ops.get_vtg(shot, t_start, t_end)
    indices = np.where(vtg > 50)
    times = vtg_time[indices]

    counts = 0
    event_t = 0.0
    for t in times:
        if t > event_t + 0.5:
            counts += 1
            event_t = t
    
    return counts