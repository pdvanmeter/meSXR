"""
"""
from __future__ import division
import numpy as np
import scipy as sp
import MDSplus as mds

def get_Te(shot):
    mst = mds.Tree('mst', shot, 'READONLY')
    te_node = mst.getNode(r'\mst_tsmp::top.proc_tsmp.t_e_bayesian')
    te= te_node.data()
    t= te_node.dim_of(0).data()
    r= te_node.dim_of(1).data()[:,0]
    te_err= te_node.getError().data()
    qual_node= mst.getNode('\\mst_tsmp::top.proc_tsmp.quality')
    qualFlags= qual_node.data()
    return te, te_err, t, r, qualFlags

def clean_TS(te, te_err, roa, qualFlags, upper_Te=2100):
    # Pick out data points with odd quality flags or 20+% error abve 300 eV TODO
    flag_indices = np.where((qualFlags % 2) != 0)
    perc_indices = np.where(te_err/(te+0.01) > 0.2)
    high_indices = np.where(te > upper_Te)
    zero_indices = np.where(te == 0)
    
    bad_indices = np.union1d(flag_indices, perc_indices)
    bad_indices = np.union1d(bad_indices, high_indices)
    bad_indices = np.union1d(bad_indices, zero_indices)
    good_indices = [n for n in range(len(te)) if n not in bad_indices]

    return te[good_indices], te_err[good_indices], roa[good_indices], qualFlags[good_indices]

def clear_TS_noFlag(data_ts, sigmas_ts, radius_ts, qualFlags, upper_Te=2100):
    # The same thing, but ignore flags since they don't seem very reliable
    indices = np.where((data_ts < upper_Te) & (data_ts > 0) & (sigmas_ts/(data_ts+0.1) <= 0.2))[0]
    return data_ts[indices], sigmas_ts[indices], radius_ts[indices], qualFlags[indices]

def get_clean_Te(shot, upper_Te=2100):
    te, te_err, ts_time, r, qualFlags = get_Te(shot)

    ts_data = []
    ts_err = []
    ts_radius = []
    ts_flags = []

    for index in range(len(ts_time)):
        data, err, radius, flags = clear_TS_noFlag(te[:,index], te_err[:,index], r, 
            qualFlags[:,index], upper_Te=upper_Te)
        ts_data.append(data)
        ts_err.append(err)
        ts_radius.append(radius)
        ts_flags.append(flags)
    return ts_data, ts_err, ts_time, ts_radius, ts_flags