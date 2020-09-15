"""
These base functions generalize the Bayesian computations that I generally use. They will help make
my code more compact and consistent.
"""
import numpy as np
import scipy as sp
import mst_ida.models.plasma as plasma

# ------------------------------------------ Base PDFs ------------------------------------------
def gaussian_likelihood(data, model, sigma):
    """
    Multivariate gaussian for use in likelihood calculations.
    """
    return -0.5*np.sum((data - model)**2 / sigma**2 + np.log(2*np.pi*sigma**2))

def gaussian_prior(x, mean, sigma):
    """
    Univariate gaussian for use in priors.
    """
    return -0.5*( ((x - mean) / sigma)**2 + np.log(2*np.pi*sigma**2) )

# ------------------------------------------ Likelihoods ------------------------------------------

# Let's make a general function to define all of these likelihoods
def ln_likelihood_det(model_plasma, det, data, sigma, labels=None):
    """
    A simple interface to avoid repeating code when defining the log-likelihoods for each diagnostic.
    This assumes that the noise is normally-distributed and uncorrelated, so if these assumptions do
    not hold it is bes to define a new function.
    """
    try:
        model_data = det.take_data(model_plasma)
    except:
        return -np.inf
    
    if labels is None:
        ln_p = gaussian_likelihood(data, model_data, sigma)
    else:
        ln_p = 0.0
        for label in labels:
            ln_p += gaussian_likelihood(data[label], model_data[label], sigma[label])
    
    if not np.isfinite(ln_p):
        return -np.inf
    else:
        return ln_p
    
def ln_likelihood(theta, detectors, data, sigmas, labels, **kwargs):
    """
    Combine the likelihoods into one
    """
    num_det = len(detectors)
    model_plasma = plasma.get_plasma(*theta, **kwargs)
    ln_p = 0.0
    for ii in range(num_det):
        ln_p += ln_likelihood_det(model_plasma, detectors[ii], data[ii], sigmas[ii], labels=labels[ii])
    return ln_p

# ------------------------------------------ Priors ------------------------------------------
def ln_uniform_prior(x, min_x, max_x):
    """
    A basic uniform prior
    """
    if min_x < x < max_x:
        return -np.log(max_x - min_x)
    else:
        return -np.inf
    
def log10_uniform_prior(y, min_exp, max_exp):
    """
    """
    if y <= 0:
        return -np.inf
    else:
        x = np.log10(y)
        if min_exp < x < max_exp:
            return -np.log(y) - np.log((max_exp - min_exp)*np.log(10))
        else:
            return -np.inf
    
def ln_gaussian_prior(x, mean, sigma):
    """
    A basic Gaussian prior - also enofrces that the parameter must be positive. Generally this should not
    make much difference as the distribution should be far from zero in order for a Gaussian to be a good
    assumption. However this functions as a failsafe to keep the likelihood from attempting to compute a
    non-physical parameter.
    """
    if x >= 0:
        return gaussian_prior(x, mean, sigma)
    else:
        return -np.inf
    
def ln_prior(theta, hyperparameters):
    """
    Combine all the priors into one. The input hyperparameters is a list of tuples. The first entry in each
    tuple is the name of the prior distribution and the second entry is a tuple of parameters for that prior.
    For example, hyperparameters = [('unifrom', (0, 1)), ('normal', (2, 0.4))]
    """
    ln_p = 0.0
    for x, (name, params) in zip(theta, hyperparameters):
        if name == 'uniform':
            ln_p += ln_uniform_prior(x, *params)
        elif name == 'normal':
            ln_p += ln_gaussian_prior(x, *params)
        elif name == 'log_uniform':
            ln_p += log10_uniform_prior(x, *params)
        else:
            raise RuntimeError('Prior distribution type name not supported.')
    return ln_p

# ------------------------------------------ Posterior ------------------------------------------
def ln_prob(theta, hyperparameters, *args, **kwargs):
    """
    Combine all priors and likelihoods into one function.
    """
    lp = ln_prior(theta, hyperparameters)
    lk = ln_likelihood(theta, *args, **kwargs)
    return lp + lk