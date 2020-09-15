"""
"""
import os
import pickle
import numpy as np
import scipy as sp
import mst_ida.models.base.physical_profiles as prof
import mst_ida.models.plasma as plasma
import mst_ida.models.mesxr as mesxr

def identify_outliers(samples, param=0, zlim=3.0):
    """
    The purpose of this function is to clean the data saved in the samples array based on a simple z-score identification method.
    The method is iterative.
    """
    dist = samples[:,param]
    z_scores = np.abs(sp.stats.zscore(dist))
    good_indices = np.arange(len(dist))
    
    # Determine the expected number of values above the specified sigma
    perc_above = 2*(1 - sp.stats.norm.cdf(zlim, loc=0, scale=1))
    num_z_above = len(dist)*perc_above
    
    while sum(z_scores > zlim) > num_z_above:
        good_indices = np.where(z_scores < zlim)[0]
        samples = samples[good_indices,:]
        dist = samples[:,param]
        
        z_scores = np.abs(sp.stats.zscore(dist))
        num_z_above = len(dist)*perc_above
    
    return samples

# Profile specific samples
def get_Te_prof_samples(samples, ys, n_samples=500, delta_a=0.06, delta_h=0.01):
    Te0_samples = samples[:,0]
    alpha_samples = samples[:,1]
    beta_samples = samples[:,2]
    
    Te_prof_samples = np.zeros([n_samples, len(ys)])
    for index, n in enumerate(np.random.randint(len(Te0_samples), size=n_samples)):
        Te0, alpha, beta = Te0_samples[n], alpha_samples[n], beta_samples[n]
        Te = prof.Temperature_Alpha(Te0, alpha=alpha, beta=beta, delta_a=delta_a, delta_h=delta_h)
        Te_prof_samples[index, :] = [Te(0,y) for y in ys]
        
    return Te_prof_samples

def get_ne_prof_samples(samples, ys, n_samples=500, delta_a=0.06, delta_h=0.01):
    ne0_samples = samples[:,3]
    alpha_samples = samples[:,4]
    beta_samples = samples[:,5]
    
    ne_prof_samples = np.zeros([n_samples, len(ys)])
    for index, n in enumerate(np.random.randint(len(ne0_samples), size=n_samples)):
        ne0, alpha_ne, beta_ne = ne0_samples[n], alpha_samples[n], beta_samples[n]
        ne = prof.Electron_Density_Alpha(ne0, alpha=alpha_ne, beta=beta_ne, delta_a=delta_a, delta_h=delta_h)
        ne_prof_samples[index, :] = [ne(0,y) for y in ys]
        
    return ne_prof_samples

def get_nAl_prof_samples(samples, ys, n_samples=500, delta_a=0.06, delta_h=0.01, alpha_Z=12., beta_Z=4.):
    nAl0_samples = samples[:,6]
    delta_nAl_samples = samples[:,8]
    peak_r_samples = samples[:,10]
    delta_r_samples = samples[:,11]
    
    nAl_prof_samples = np.zeros([n_samples, len(ys)])
    for index, n in enumerate(np.random.randint(len(nAl0_samples), size=n_samples)):
        nAl0, delta_nAl, peak_r, delta_r = nAl0_samples[n], delta_nAl_samples[n], peak_r_samples[n], delta_r_samples[n]
        nAl = prof.Ion_Density_Alpha(nAl0, alpha=alpha_Z, beta=beta_Z, element='Al', delta_a=delta_a, delta_h=delta_h)
        nAl += prof.Density_Hollow(delta_nAl, peak_r=peak_r, delta_r=delta_r, delta_a=delta_a, delta_h=delta_h)
        nAl_prof_samples[index, :] = [nAl(0,y) for y in ys]
        
    return nAl_prof_samples

def get_nC_prof_samples(samples, ys, n_samples=500, delta_a=0.06, delta_h=0.01, alpha_Z=12., beta_Z=4.):
    nC0_samples = samples[:,7]
    delta_nC_samples = samples[:,9]
    peak_r_samples = samples[:,10]
    delta_r_samples = samples[:,11]
    
    nC_prof_samples = np.zeros([n_samples, len(ys)])
    for index, n in enumerate(np.random.randint(len(nC0_samples), size=n_samples)):
        nC0, delta_nC, peak_r, delta_r = nC0_samples[n], delta_nC_samples[n], peak_r_samples[n], delta_r_samples[n]
        nC = prof.Ion_Density_Alpha(nC0, alpha=alpha_Z, beta=beta_Z, element='C', delta_a=delta_a, delta_h=delta_h)
        nC += prof.Density_Hollow(delta_nC, peak_r=peak_r, delta_r=delta_r, delta_a=delta_a, delta_h=delta_h)
        nC_prof_samples[index, :] = [nC(0,y) for y in ys]
        
    return nC_prof_samples

# Model specific samples
from mst_ida.models.plasma import get_plasma
from mst_ida.models.fir import fir_chord_names, get_fir_model

def get_mesxr_samples(samples, p3det, n_samples=200):
    mesxr_samples = {Ec:np.zeros([n_samples, 60]) for Ec in p3det.thresholds}
    
    for index, n in enumerate(np.random.randint(samples.shape[0], size=n_samples)):
        model_plasma = get_plasma(*samples[n,:])
        model_prof = p3det.take_data(model_plasma)
        
        for Ec in mesxr_samples.keys():
            mesxr_samples[Ec][index,:] = model_prof[Ec]
            
    return mesxr_samples

def get_fir_samples(samples, n_samples=200, delta_a=0.06, delta_h=0.01):
    ne0_samples = samples[:,3]
    alpha_samples = samples[:,4]
    beta_samples = samples[:,5]
    
    fir_samples = np.zeros([n_samples, len(fir_chord_names)])
    for index, n in enumerate(np.random.randint(samples.shape[0], size=n_samples)):
        ne0, alpha_ne, beta_ne = ne0_samples[n], alpha_samples[n], beta_samples[n]
        fir_samples[index, :] = get_fir_model(ne0, alpha_ne, beta_ne, delta_a=delta_a, delta_h=delta_h)
            
    return fir_samples

def get_Zeff_samples(samples, ys, p3det, n_samples=500, delta_a=0.06, delta_h=0.01, n0_time=17.5):
    Zeff_samples = np.zeros([n_samples, len(ys)])
    for index, n in enumerate(np.random.randint(samples.shape[0], size=n_samples)):
        model_plasma = plasma.get_plasma(*samples[n,:], n0_time=n0_time, delta_a=delta_a, delta_h=delta_h)
        Zeff, nDs = mesxr.get_Zeff(np.zeros(ys.shape), ys, model_plasma, p3det.avg_Z, return_nD=True)
        Zeff_samples[index,:] = Zeff
    return Zeff_samples

# Load saved pickle files
def load_profile_samples(shot, frames=[7,8,9,10], mcmc_dir = '/home/pdvanmeter/data/meSXR/MCMC', fname_prefix='MESXR_10p_v2'):
    samples_ts = []
    ts = [2.0*frame+0.5 for frame in frames]

    for frame in frames:
        load_path = os.path.join(mcmc_dir, fname_prefix+'_{0:10d}-{1:02d}.pkl'.format(shot, frame))
        samples = pickle.load(open(load_path, 'rb'))
        samples = identify_outliers(samples)
        samples_ts.append(samples)

    ys = np.linspace(0, 0.52, num=100)

    Te_samples_ts = [get_Te_prof_samples(samp, ys) for samp in samples_ts]
    ne_samples_ts = [get_ne_prof_samples(samp, ys) for samp in samples_ts]
    nAl_samples_ts = [get_nAl_prof_samples(samp, ys) for samp in samples_ts]
    
    return {'Te':Te_samples_ts, 'ne':ne_samples_ts, 'nAl':nAl_samples_ts, 'params':samples_ts, 'frames':frames}