"""
Useful functions used throughout the models.
"""
import time
import numpy as np
import scipy as sp
import scipy.stats

def stringify_float(val):
    # Helper function for loading file names
    return '{0:d}p{1:d}'.format(int(val), int(10*val - int(val)*10))

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

def get_MLE(ln_likelihood, x0, bounds=None, verbose=True):
    """
    Estimate the maximum of a supplied log-likelihood function.
    """
    nll = lambda theta: -ln_likelihood(theta)

    t_start = time.time()
    soln = sp.optimize.minimize(nll, x0, bounds=bounds)
    t_end = time.time()
    dt = t_end - t_start
    theta0 = soln.x
    
    if verbose:
        print('MLE required {0:} seconds.'.format(dt))
        print(soln)
        
    return theta0