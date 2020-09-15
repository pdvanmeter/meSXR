"""
This module contians the set of tools needed to properly perform IDA on MST.
This includes a reparameterizatin in order to properly fit scale parameters.
"""
import numpy as np

# ------------------------------------------ Posterior ------------------------------------------
def ln_prob(theta, hyperparameters, *args, **kwargs):
    """
    Combine all priors and likelihoods into one function.
    """
    lp = ln_prior(theta, hyperparameters)
    lk = ln_likelihood(theta, *args, **kwargs)
    return lp + lk

# ------------------------------------------ Likelihoods ------------------------------------------
def gaussian_likelihood(data, model, sigma):
    """
    Multivariate gaussian for use in likelihood calculations.
    """
    return -0.5*np.sum((data - model)**2 / sigma**2 + np.log(2*np.pi*sigma**2))

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
    model_plasma = get_plasma(*theta, **kwargs)
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
        return 0.0
    else:
        return -np.inf
    
def ln_gaussian_prior(x, mean, sigma):
    """
    A basic Gaussian prior - note that this will not necessarily enforce that the parameter
    should be positive. This could potentially be an issue if the model underlying the likelihood
    requires a positive input paramter.
    """
    return -0.5*( ((x - mean) / sigma)**2 + np.log(2*np.pi*sigma**2) )
    
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
        else:
            raise RuntimeError('Prior distribution type name not supported.')
    return ln_p

# ------------------------------------------ Modified Plasma Function ------------------------------------------
import mst_ida.models.base.physical_profiles as prof
from mst_ida.models.base.profile_base import *
from mst_ida.models.plasma import Plasma
a = 0.52 # MST minor radius in [m]

class Ion_Density_Alpha(Profile_Alpha_Beta_Rho):
    """
    The only difference between this class and the one in the physical_profiles module is that I got rid of x10^19 scaling, which
    will make it easier to specify the density by its base 10 logarithm.
    """
    def __init__(self, nZ0, alpha=12., beta=4., element='Al', delta_a=0.06, delta_h=0.01):
        self.symbol = element
        super(Ion_Density_Alpha, self).__init__('{0:} Density'.format(element), nZ0, alpha, beta,
                                                dim_units=['m', 'm'], units=r'$m^{-3}$', delta_a=delta_a, delta_h=delta_h)
        
class Density_Hollow(Profile_Hollow_Flux):
    """
    The only difference between this class and the one in the physical_profiles module is that I got rid of x10^19 scaling, which
    will make it easier to specify the density by its base 10 logarithm.
    """
    def __init__(self, delta_nZ, peak_r=0.33/a, width_r=0.09/a, delta_a=0.06, delta_h=0.01):
        super(Density_Hollow, self).__init__('Density Hollow Profile', delta_nZ, peak_r, width_r, units=r'$m^{-3}$', dim_units=['m', 'm'],
              delta_a=delta_a, delta_h=delta_h)
        
def get_plasma(Te0, alpha_Te, beta_Te, ne0, alpha_ne, beta_ne, X_Al, X_C, alpha_Z, beta_Z, X_dAl, X_dC, peak_r, width_r,
               n0_time=17.5, delta_a=0.06, delta_h=0.01):
    """
    This plasma is meant to represent a PPCD or rotating standard RFP plasma. The neutral denisty is based on high-current
    PPCD and has not been well-characterized in other cases.
    """
    # Define the plasma
    Te_prof = prof.Temperature_Alpha(Te0, alpha=alpha_Te, beta=beta_Te, delta_a=delta_a, delta_h=delta_h)
    ne_prof = prof.Electron_Density_Alpha(ne0, alpha=alpha_ne, beta=beta_ne, delta_a=delta_a, delta_h=delta_h)
    n0_prof = prof.Neutral_Density_Profile(n0_time)
    
    # Impurity base profile parameters
    nAl0 = 10.0**X_Al
    nC0 = 10.0**X_C
    nZ0 = {'Al':nAl0, 'C':nC0, 'N':0.3*nC0, 'B':0.3*nC0, 'O':0.9*nC0}

    # PPCD hollow profile parameters
    delta_nAl = 10.0**X_dAl
    delta_nC = 10.0**X_dC
    nZ_delta = {'Al':delta_nAl, 'C':delta_nC, 'N':0.3*delta_nC, 'B':0.3*delta_nC, 'O':0.75*delta_nC}

    nZ_set = {}
    for ion in nZ0.keys():
        nZ_set[ion] = Ion_Density_Alpha(nZ0[ion], alpha=alpha_Z, beta=beta_Z, element=ion, delta_a=delta_a, delta_h=delta_h)
        nZ_set[ion] += Density_Hollow(nZ_delta[ion], peak_r=peak_r, width_r=width_r, delta_a=delta_a, delta_h=delta_h)

    return Plasma(Te_prof, ne_prof, n0_prof, nZ_set=nZ_set, delta_a=delta_a, delta_h=delta_h)

# ------------------------------------------------ Samples ------------------------------------------------
import mst_ida.models.mesxr as mesxr

def get_prof_samples(samples, ys, n_samples=1000, delta_a=0.06, delta_h=0.01, n0_time=17.5, impurities=['Al', 'C', 'N', 'B', 'O']):
    """
    """
    prof_samples = {
        'Te':np.zeros([n_samples, len(ys)]),
        'ne':np.zeros([n_samples, len(ys)]),
        'nZ':{Z:np.zeros([n_samples, len(ys)]) for Z in impurities}
    }
    
    for index, n in enumerate(np.random.randint(samples.shape[0], size=n_samples)):
        plasma = get_plasma(*samples[n,:], delta_a=delta_a, delta_h=delta_h, n0_time=n0_time)
        prof_samples['Te'][index, :] = [plasma.Te(0,y) for y in ys]
        prof_samples['ne'][index, :] = [plasma.ne(0,y) for y in ys]
        for Z in impurities:
            prof_samples['nZ'][Z][index, :] = [plasma.nZ[Z](0,y) for y in ys]
            
    return prof_samples

def get_Zeff_samples(samples, ys, p3det, n_samples=500, delta_a=0.06, delta_h=0.01, n0_time=17.5):
    Zeff_samples = np.zeros([n_samples, len(ys)])
    for index, n in enumerate(np.random.randint(samples.shape[0], size=n_samples)):
        plasma = get_plasma(*samples[n,:], n0_time=n0_time, delta_a=delta_a, delta_h=delta_h)
        Zeff, nDs = mesxr.get_Zeff(np.zeros(ys.shape), ys, plasma, p3det.avg_Z, return_nD=True)
        Zeff_samples[index,:] = Zeff
    return Zeff_samples

# ------------------------------------------------- Derived quantities -------------------------------------------------

# Physical constants
e_charge = 1.602e-19 # C
me = 9.109e-31 # kg
eV_to_J = 1.602e-19 # J/eV
eps0 = 8.854e-12 # F/m
mu0 = 1.2566e-6 # N/A^2
a = 0.52 # m

def plasma_lambda(Te, ne, Z):
    return np.log(((Te*eV_to_J)**(3/2)) / (np.sqrt(np.pi)*Z*(e_charge**3)*np.sqrt(ne)))

def spitzer_F(Z):
    return (1 + 1.198*Z + 0.222*Z**2) / (1 + 2.966*Z + 0.753*Z**2)

def spitzer_resistivity(Te, ne, Zeff):
    F = spitzer_F(Zeff)
    ln_lambda = 17.0 #plasma_lambda(Te, ne, Zeff)
    top = np.sqrt(2*me)*Zeff*(e_charge**2)*ln_lambda*F
    bot = 12*(np.pi**(3/2))*(eps0**2)*((Te*eV_to_J)**(3/2))
    return top / bot

# Ion masses - kg
mass = {
    'Al':4.5e-26,
    'D':3.344e-27,
    'C':2e-26,
    'B':1.8e-26,
    'N':2.3e-26,
    'O':2.7e-26
}

def lundquist(Te, ne, Zeff, nAl, nC, B=0.15):
    nN = 0.3*nC
    nB = 0.3*nC
    nO = 0.9*nC
    rho = ne*mass['D'] + nAl*mass['Al'] + nC*mass['C'] + nB*mass['B'] + nN*mass['N'] + nO*mass['O']
    eta = spitzer_resistivity(Te, ne, Zeff)
    return (mu0*a**2 / eta)*(B /(a*np.sqrt(mu0*rho)))

def get_resistivity_samples(prof_samples, Zeff_samples, ys):
    n_edge = np.argmin(np.abs(ys/0.52 - 0.8))
    etas = np.zeros(prof_samples['Te'].shape)
    for ii in range(len(ys)):
        Tes = prof_samples['Te'][:,ii]
        nes = prof_samples['ne'][:,ii]
        Zs = Zeff_samples[:,ii]
        etas[:,ii] = spitzer_resistivity(Tes, nes, Zs)
    return etas

def get_core_lundquist_samples(prof_samples, Zeff_samples, B=0.15):
    # Lundquist number
    Tes = prof_samples['Te'][:, 0]
    nes = prof_samples['ne'][:, 0]
    nAls = prof_samples['nZ']['Al'][:, 0]
    nCs = prof_samples['nZ']['C'][:, 0]
    Zs = Zeff_samples[:, 0]

    return lundquist(Tes, nes, Zs, nAls, nCs)

# ------------------------------------------------ Confidence intervals ------------------------------------------------

def confidence_interval(vals, bins=100, c=0.68):
    """
    """
    #hist, bin_edges = np.histogram(vals, bins=bins, density=True)
    #x = np.mean(np.vstack([bin_edges[0:-1], bin_edges[1:]]), axis=0)
    #cdf = sp.integrate.cumtrapz(hist, x=x, initial=0)
    x, counts = np.unique(vals, return_counts=True)
    cdf = np.cumsum(counts) / np.sum(counts)
    
    x_median = x[np.argmin(np.abs(cdf - 0.5))]
    x_lower = x[np.argmin(np.abs( cdf - (0.5-c/2) ))]
    x_upper = x[np.argmin(np.abs( cdf - (0.5+c/2) ))]
    
    return x_median, x_lower, x_upper

def profile_confidence(samples):
    """
    """
    n_pts = samples.shape[1]
    func_y = {
        'median':np.zeros(n_pts),
        '1 sigma':{'low':np.zeros(n_pts), 'high':np.zeros(n_pts)},
        '2 sigma':{'low':np.zeros(n_pts), 'high':np.zeros(n_pts)},
        '3 sigma':{'low':np.zeros(n_pts), 'high':np.zeros(n_pts)}
    }
    for ii in range(n_pts):
        vals = samples[:,ii]
        
        median, lower_1s, upper_1s = confidence_interval(vals, c=0.68)
        func_y['median'][ii] = median
        func_y['1 sigma']['low'][ii] = lower_1s
        func_y['1 sigma']['high'][ii] = upper_1s
        
        median, lower_2s, upper_2s = confidence_interval(vals, c=0.95)
        func_y['2 sigma']['low'][ii] = lower_2s
        func_y['2 sigma']['high'][ii] = upper_2s
        
        median, lower_3s, upper_3s = confidence_interval(vals, c=0.997)
        func_y['3 sigma']['low'][ii] = lower_3s
        func_y['3 sigma']['high'][ii] = upper_3s
    
    return func_y