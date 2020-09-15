"""
Load in NICKAL2 data from the MDSplus tree.
"""
from __future__ import division
import numpy as np
import scipy as sp
import scipy.signal
import MDSplus
import pidly

from mst_ida.utilities.structures import AttrDict

filters =  ['AlBe', 'ZrMylar', 'SiBe']

# ------------------------------------------- Native Python -------------------------------------------
eta = 4.2809982e-09 # m^2 * sr

nickal2_probes = {
    'AlBe':('B', 10),
    'ZrMylar':('B', 19),
    'SiBe':('B',20)
    }

nickal2_nodes = {
    'AlBe':'\\MRAW_MISC::SXRTOMO_TR1612_5_06',
    'ZrMylar':'\\MRAW_MISC::SXRTOMO_TR1612_4_08',
    'SiBe':'\\MRAW_MISC::SXRTOMO_TR1612_5_08'
    }

n2amp_nodes = {
    'AlBe':'\\DIFF_30_GAIN',
    'ZrMylar':'\\DIFF_39_GAIN',
    'SiBe':'\\DIFF_40_GAIN'
}

def smooth_signal(x, t, dt=0.1, window='hann'):
    """
    """
    # Get the number of points in the resampled signal
    scale = dt / (t[1] - t[0])
    num = int(len(t) / scale)
    
    #  Resample the data
    x_rs, t_rs =  sp.signal.resample(x, num, t=t, window=window)
    
    # Use the original signal to calculate error bars from the variance
    stdev = np.zeros(num)
    for ii, tn in enumerate(t_rs):
        ns = np.argmin(np.abs(t - (tn - dt/2.) ))
        nf = np.argmin(np.abs(t - (tn + dt/2.) ))
        stdev[ii] = np.std(x[ns:nf+1])
        
    return x_rs, t_rs, stdev

def get_NickAl2_data(shot, t_start=0.0, t_end=60.0, off_start=75.0, off_end=80.0, dt=0.5):
    """
    """
    nickal2_data = {'bright':{}, 'sigma':{}}
    mstTree = MDSplus.Tree('MST', shot, 'READONLY')
    
    for filt in filters:
        node_str = '\\MST_MISC::DD_{0:}_{1:02d}'.format(*nickal2_probes[filt])
        #node_str = nickal2_nodes[filt]
        sxr_node = mstTree.getNode(node_str)
        sxr_data = sxr_node.getData().data()
        sxr_time = sxr_node.getData().dim_of().data() * 1000.
        sxr_gain = mstTree.getNode(node_str+'_AMP').getData().data()
        #sxr_gain = mstTree.getNode(n2amp_nodes[filt]).getData().data()

        # Convert to brightness units [W / m^2 /sr]
        sxr_data = ( (sxr_data / sxr_gain)*3.63 ) / eta

        # Subtract off the offset using the end of the shot (much cleaner than before)
        ns = np.argmin(np.abs(sxr_time - off_start))
        nf = np.argmin(np.abs(sxr_time - off_end))
        sxr_data -= np.average(sxr_data[ns:nf+1])

        # Restrict data to desired interval
        ns = np.argmin(np.abs(sxr_time - t_start))
        nf = np.argmin(np.abs(sxr_time - t_end))
        sxr_data = sxr_data[ns:nf+1]
        sxr_time = sxr_time[ns:nf+1]

        # Smooth approriately
        data_rs, t_rs, sigma = smooth_signal(sxr_data, sxr_time, dt=dt)
        nickal2_data['bright'][filt] = data_rs
        nickal2_data['sigma'][filt] = sigma
        
    # Time should be the same for each signal
    nickal2_data['time'] = t_rs
    return AttrDict(nickal2_data)

def get_NickAl2_data_point(shot, tiempo, filters=['AlBe', 'ZrMylar', 'SiBe'], **kwargs):
    """
    Use this to return only the data from a single specified time point. Useful for IDA.
    """
    n2_data = get_NickAl2_data(shot, **kwargs)
    index = np.argmin(np.abs(tiempo - n2_data['time']))
    
    dt = kwargs['dt'] if 'dt' in kwargs else 0.5
    print('Retriving NICKAL2 data for t = {0:.2f} to t = {1:.2f} ms.'.format(tiempo-dt, tiempo+dt))
    return {f:n2_data[f]['brightness'][index] for f in filters}, {f:n2_data[f]['sigma'][index] for f in filters}

# ------------------------------------------- IDL wrapper -------------------------------------------
def load_brightness(shot_num, t_start=8.0, t_end=28.0, delta=0.1, smooth=10.0):
    """
    Function: st = load_brightness(shot_num, t_start, t_end, delta, smooth)
        This version of load_brightness interfaces directly with the IDL implementation, via the pidly interface.
    Inputs:
        - shot_num = [INT] The MST shot ID for the desired set of data
        - t_start = [FLOAT] The start time for the desired interval of SXR data.
        - t_end = [FLOAT] The end time for the desired interval of SXR data.
        - delta = [FLOAT] The desired sampling window for SXR data.
        - smooth = [FLOAT] The size of the smoothing window (10.0 is standard).
    Outputs:
        - st['key'] = [DICT] Nested dictionary containing the SXR tomography diagnostic data, indexed by camera label.
    """
    # Access (and initialize, if needed) the pidly object and assemble the command string
    idl = pidly.IDL()
    idl_str = "n2d = NICKAL2_signal("+ str(shot_num) + ", tstart=" + str(t_start)
    idl_str += ", tend=" + str(t_end) + ", delta=" + str(delta) + ", sm=" + str(smooth) + ")"
    idl('cd, "/home/pdvanmeter/lib/idl"')
    idl('.r NICKAL2_signal')
    idl(idl_str)
    
    # Extract the data from IDL and format
    data = {
        'AlBe':idl.ev('n2d.data.al'),
        'SiBe':idl.ev('n2d.data.si'),
        'ZrMylar':idl.ev('n2d.data.zr')
    }

    error = {
        'AlBe':idl.ev('n2d.err.al'),
        'SiBe':idl.ev('n2d.err.si'),
        'ZrMylar':idl.ev('n2d.err.zr')
    }

    noise = {
        'AlBe':idl.ev('n2d.noise.al'),
        'SiBe':idl.ev('n2d.noise.si'),
        'ZrMylar':idl.ev('n2d.noise.zr')
    }

    tiempo = idl.ev('n2d.time')

    # Put the result together
    st = {
        'bright':data,
        'sigma':error,
        'noise':noise,
        'time':tiempo
    }
    
    idl.close()
    return AttrDict(st)