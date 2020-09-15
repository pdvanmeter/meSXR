"""
Implementation of the Ratio Distribution, which is the distribution of two non-central, uncorrelated, 
normally-distributed random variables. This implementation was designed to follow the format established
by the scipy.stats modules.

Note that as the mean goes to zero this distribution becomes a Cauchy distribution, which is well-known as a
"pathological" distribution of undefined mean and variance. It is advised not to use this module when the mean
of the denominator is too close to zero (mu2 >= 5 is a good limit) in order to ensure proper behavior. The
method used for sampling relies on the existence of a finite mean and variance in order to set the sampling
bounds. However, the pdf function is still valid for this case.

Read more: https://en.wikipedia.org/wiki/Ratio_distribution
"""
import numpy as np
import scipy as sp
from scipy.special import ndtr

def pdf(z, mu1=1, mu2=1, sigma1=1, sigma2=1):
    """
    Returns the normalized probability density function for the random variable Z = X/Y,
    where X ~ N(mu1,sigma1) and Y ~ N(mu2,sigma2).
    Args:
    	z: (array-like) The values of z = x/y at which to evaluate the pdf.
    Optional:
    	mu1 = (float) The mean of the distribution of X.
    	mu2 = (float) The mean of the distribution of Y.
    	sigma1 = (float) The standard deviation of the distribution of X.
    	sigma2 = (float) The standard deviation of the distribution of Y.
    Returns:
    	pdf: (array-like) The probability density evaluated at the points z.
    """
    a,b,c,d = coefficients(z, mu1=mu1, mu2=mu2, sigma1=sigma1, sigma2=sigma2)
    f1 = (b*d) / a**3
    f2 = 1 / (np.sqrt(2*np.pi)*sigma1*sigma2)
    f3 = ndtr(b/a) - ndtr(-b/a)
    f4 = np.exp(-c/2) / ((a**2)*np.pi*sigma1*sigma2)

    return f1*f2*f3 + f4

def logpdf(z, **kwargs):
    """
    The logarithm of the probability density function. See pdf() for details about the
    arguments and keywords.
    """
    return np.log(pdf(z, **kwargs))

def cdf(zs, **kwargs):
    """
    Generate the cdf using numerical integration. See pdf() for details about the
    arguments and keywords.
    """
    func = lambda x: pdf(x, **kwargs)
    return np.array([sp.integrate.quad(func, -np.inf, z)[0] for z in zs])

def moment(n, **kwargs):
    """
    Returns the order-n moment of the distribution.
    """
    func = lambda x: (x**n)*pdf(x, **kwargs)
    return sp.integrate.quad(func, -np.inf, np.inf, full_output=True)[0]

def mean(**kwargs):
    """
    Returns the mean (expected value) of the distribution.
    """
    return moment(1, **kwargs)

def var(**kwargs):
    """
    Returns the variance (E[X^2] - E[X]^2) of the distribution.
    """
    return moment(2, **kwargs) - moment(1, **kwargs)**2

def std(**kwargs):
    """
    Returns the variance (var[X]^1/2) of the distribution.
    """
    return np.sqrt(var(**kwargs))

def rvs(n_samples, **kwargs):
    """
    Generate samples using inverse transform on an interpolated cdf.
    Uses an approximation for sigma to determine the approximate range of consideration.
    """
    mu = mean(**kwargs)
    sigma = std(**kwargs)
    zs = np.linspace(mu-4*sigma, mu+4*sigma, num=200)
    cdfs = cdf(zs, **kwargs)
    us = np.random.rand(n_samples)
    return np.interp(us, cdfs, zs)

# ----------------------------------- Internal methods -----------------------------------
def coefficients(z, mu1=1, mu2=1, sigma1=1, sigma2=1):
    a = np.sqrt((z**2 / sigma1**2) + (1 / sigma2**2))
    b = z*mu1 / sigma1**2 + mu2 / sigma2**2
    c = mu1**2 / sigma1**2 + mu2**2 / sigma2**2
    d = np.exp((b**2 - c*a**2) / (2*a**2))
    return a,b,c,d