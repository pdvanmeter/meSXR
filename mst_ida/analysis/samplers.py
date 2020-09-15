"""
Classes to simplify and standardize the process of drawing samples from the posterior distribution in
Bayesian inference problems.
"""
import numpy as np
import scipy as sp

class Quad_Sampler(object):
    """
    Class for drawing samples from an arbitrary one-dimensional probability distribution using numerical integration 
    and interpolation. In general this will be superior to more sophisticated sampling methods for 1D problems.
    Assumes that priors are uniform.
    Args:
        ln_likelihood: Function which takes the independent variable x as its first argument and returns the log
            of the likelihood function, p(d|x,I), up to a constant. May take other *args or **kwargs.
        priors: List-type of the form [a,b], where a and b define the upper and lower bounds of the uniform
            prior p(x|I).
    optioinal:
        vect: (bool) Set to true if the log-likelihood accepts a vectorized input.
    """
    def __init__(self, ln_likelihood, priors, vect=False):
        self._ln_likelihood = ln_likelihood
        self._a, self._b = priors
        self._vect = vect

        # Default values
        self.ln_Z = np.nan
        self.mean = np.nan
        self.std  = np.nan
    
    def fit(self, n_pts=200, args=(), **kwargs):
        """
        Perform the fit.
        Optional:
            n_pts: (int) Number of evenly-spaced points over which to compute the probability.
            args: (tuple) All additional arguments to be passed on the the likelihood function.
            **kwargs: All other keywords are passed on the the likelihood function.
        """
        # Evaluate the pdf
        self.xs = np.linspace(self._a, self._b, num=n_pts)
        if self._vect:
            self.ln_pdf = self._ln_likelihood(self.xs, *args, **kwargs)
        else:
            self.ln_pdf = np.array([self._ln_likelihood(x, *args, **kwargs) for x in self.xs])
        
        # Rescale with the maxima
        ln_C = np.amax(self.ln_pdf)
        pdf_scaled = np.exp(self.ln_pdf - ln_C)

        # Compute the evidence and rescale
        Z_scaled = np.trapz(pdf_scaled, x=self.xs)
        self.ln_Z = np.log(Z_scaled) + ln_C
        self.pdf = pdf_scaled / Z_scaled
        self.cdf = sp.integrate.cumtrapz(self.pdf, x=self.xs, initial=0)

        # Estimate summary statistics - assuming a normal distribution
        samples = self.get_samples(1000)
        self.mean = np.mean(samples)
        self.std = np.std(samples)
        
    def get_samples(self, n_samples):
        """
        """
        u_samp = np.random.rand(n_samples)
        return np.interp(u_samp, self.cdf, self.xs)


class Quad_Sampler_ND(object):
    """
    Class for drawing samples from an arbitrary N-dimensional probability distribution using numerical integration 
    and interpolation. This can be useful for problems with a low number of dimensions (~3) for which the likelihood
    function can be computed quickly (<< 1 second).
    Assumes that priors are uniform. Currently does not support vectorized likelihoods.
    Args:
        ln_likelihood: Function which takes the independent variables (x1, x2, ..., xN) as its first argument and returns
            the log of the likelihood function, p(d|x1,...,I), up to a constant. May take other *args or **kwargs.
        priors: List of tuples, of the form [(a1,b1), (a2,b2), ..., (aN,bN)] where a and b define the upper and lower bounds
            of the uniform prior p(x1,...|I).
    optioinal:
        vect: (bool) Set to true if the log-likelihood accepts a vectorized input.
    """
    def __init__(self, ln_likelihood, ndim, priors):
        self._ln_likelihood = ln_likelihood
        self.ndim = ndim
        self._a = np.zeros(self.ndim)
        self._b = np.zeros(self.ndim)
        for n in range(self.ndim):
            self._a[n], self._b[n] = priors[n]

        # Default values
        self.ln_Z = np.nan
        self.mean = np.nan
        self.std  = np.nan
    
    def fit(self, n_pts=200, args=(), **kwargs):
        """
        Perform the fit.
        Optional:
            n_pts: (int) Number of evenly-spaced points over which to compute the probability.
            args: (tuple) All additional arguments to be passed on the the likelihood function.
            **kwargs: All other keywords are passed on the the likelihood function.

        This doesn't work yet.
        """
        # Construct the evaluation grid
        self.xs = np.zeros([self.ndim, n_pts])
        for n in range(slef.ndim):
            self.xs[n,:] = np.linspace(self._a[n], self._b[n], num=n_pts)

        # Evaluate the pdf
        self.ln_pdf = np.zeros([self.ndim, n_pts])
        for n in range(slef.ndim):
            self.ln_pdf[n] = np.array([self._ln_likelihood(x, *args, **kwargs) for x in self.xs[n]])
        
        # Rescale with the maxima
        ln_C = np.amax(self.ln_pdf)
        pdf_scaled = np.exp(self.ln_pdf - ln_C)

        # Compute the evidence and rescale
        Z_scaled = np.trapz(pdf_scaled, x=self.xs)
        self.ln_Z = np.log(Z_scaled) + ln_C
        self.pdf = pdf_scaled / Z_scaled
        self.cdf = sp.integrate.cumtrapz(self.pdf, x=self.xs, initial=0)

        # Estimate summary statistics - assuming a normal distribution
        samples = self.get_samples(1000)
        self.mean = np.mean(samples)
        self.std = np.std(samples)
        
    def get_samples(self, n_samples):
        """
        """
        u_samp = np.random.rand(n_samples)
        return np.interp(u_samp, self.cdf, self.xs)

# ------------------------------------------------ MESXR Sampler ------------------------------------------------
import time
import emcee
import mst_ida.models.mesxr3 as m3
import mst_ida.data.mesxr as mesxr
import mst_ida.analysis.ida as ida
import mst_ida.analysis.emissivity as em
import mst_ida.models.base.response as rsp
from mst_ida.utilities.functions import identify_outliers
from mst_ida.models.base.geometry import flux_coords, sunflower_points

default_priors = {
    'alpha':((10, 14), (0.1,18), (0.1,18))
}

class MESXR_Emiss_Sampler(object):
    """
    """
    def __init__(self, shot, frame, flux=None, Ec_ref=3.0, priors=None, indices=np.arange(5,55), Ew=300.,
        method='alpha', nwalkers=32, center=True, delta_a=0.06, delta_h=0.01, manual=None):
        # Load the data
        self.shot = shot
        if manual is not None:
            self.mesxr_data = manual['data']
            self.mesxr_sigmas = manual['sigmas']
            self.signed_ps = manual['impact_p']
            self.thresholds = manual['thresholds']
        else:
            self.frame = frame
            self.mesxr_data, self.mesxr_sigmas, self.signed_ps, self.thresholds = mesxr.get_8c_data(self.shot, self.frame, center=center)

        # Model and geometry
        if flux is None:
            self.flux = flux_coords(delta_a=delta_a, delta_h=delta_h)
        else:
            self.flux = flux
        
        self.method = method
        self.p3det = m3.MESXR(shot=self.shot, center=center)
        self.gij_set = {}
        self.ss_set = {}
        for Ec in self.thresholds:
            self.gij_set[Ec], self.ss_set[Ec] = em.get_geometry_matrix(self.flux, self.p3det)

        # Include specified data points
        self.indices = np.arange(6,55)
        z = {Ec:np.maximum(self.mesxr_data[Ec][self.indices]+1, 1) for Ec in self.thresholds}
        self.ln_data_fact = {Ec:-np.sum(sp.special.loggamma(z[Ec])) for Ec in self.thresholds}
        
        # Set up the priors
        if priors is None:
            self.priors = default_priors[self.method]
        else:
            self.priors = priors

        # Sampler parameters
        self.nwalkers = nwalkers
        self.pos0 = self.get_pos0()
        self.ndim = self.pos0.shape[1]

        # Set up the samplers
        moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2),]
        self.samplers = {}

        for index, Ec in enumerate(self.thresholds):
            self.samplers[Ec] = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.ln_prob, moves=moves, kwargs={'Ec':Ec})
        
        # Set up ratio curves
        self.Ew = Ew
        self.Ec_ref = Ec_ref
        self.temps = np.linspace(10, 3000, num=6000)
        self.ratios= {}
        
        for Ec in self.thresholds:
            if Ec != self.Ec_ref:
                self.ratios[Ec] = np.array([self.model_ratio(Te,Ec*1000.) for Te in self.temps])
        
    def fit(self, nsteps=10000, remove_outliers=True, resume=False, burn_step=3000, n_samples=5000, progress=True):
        """
        """
        # MCMC sampling
        if nsteps is not None:
            for Ec in self.thresholds:
                if not resume:
                    #print('Beginning sampling for Ec = ' + str(Ec) + ' keV')
                    time.sleep(1)
                    self.samplers[Ec].run_mcmc(self.pos0, nsteps, progress=progress)
                else:
                    #print('Resuming sampling for Ec = ' + str(Ec) + ' keV')
                    time.sleep(1)
                    self.samplers[Ec].run_mcmc(None, nsteps, progress=progress)
            
        self.samples = {Ec:self.samplers[Ec].get_chain(discard=burn_step, flat=True) for Ec in self.thresholds}
        
        # Remove points from poorly-converged walkers
        if remove_outliers:
            for Ec in self.thresholds:
                self.samples[Ec] = identify_outliers(self.samples[Ec])
        
        # Save the average fit parameters
        self.theta_avg = {Ec:np.average(self.samples[Ec], axis=0) for Ec in self.thresholds}
        
        # Get the emissivity profile samples
        self.n_samples = n_samples
        self.emiss_samples = {Ec:self.get_emiss_samples(self.samples[Ec], Ec=Ec, n_samples=n_samples) for Ec in self.thresholds}
        self.emiss_CIs = {Ec:ida.profile_confidence(self.emiss_samples[Ec]) for Ec in self.thresholds}
        
    def get_Te_samples(self, slim=0.7, include=[4.0, 5.0]):
        """
        """
        ss = self.ss_set[self.Ec_ref].ravel()
        sn = np.argmin(np.abs(ss - slim))
        s_vals = self.ss_set[self.Ec_ref].ravel()[:sn]
        Te_avg_prof_samples = np.zeros([self.n_samples, sn])

        for s_index in range(sn):
            ratios = {Ec:self.emiss_samples[Ec][:,s_index]/self.emiss_samples[self.Ec_ref][:,s_index] for Ec in include}
            Te_samples = {Ec:self.Te_from_R(ratios[Ec], Ec=Ec) for Ec in include}
            Te_avg_prof_samples[:,s_index] = sum([Te_samples[Ec] for Ec in include]) / len(include)

        Te_avg_CI = ida.profile_confidence(Te_avg_prof_samples)
        return s_vals, Te_avg_prof_samples, Te_avg_CI
    
    # ----------------------------------------------- Emissivity Model -----------------------------------------------
    def emiss_model_alpha(self, ss, Xem, alpha, beta):
        return (10.**Xem)*(1 - ss**alpha)**beta

    def emiss_model(self, *args):
        if self.method == 'alpha':
            return self.emiss_model_alpha(*args)
        else:
            raise KeyError('Please select a valid fitting method.')
    
    def get_model(self, theta, Ec=3.0):
        gij = self.gij_set[Ec]
        ss = self.ss_set[Ec]
        emiss = self.emiss_model(ss, *theta)
        bright = np.dot(gij, emiss).squeeze()
        return self.p3det.etendue[Ec]*bright
    
    # ----------------------------------------------- Bayesian Methods -----------------------------------------------
    def ln_prob(self, theta, Ec=3.0):
        lp = self.ln_prior(theta)
        if np.isfinite(lp):
            return lp + self.ln_likelihood(theta, Ec=Ec)
        else:
            return -np.inf
    
    def ln_likelihood(self, theta, Ec=3.0):
        data = self.mesxr_data[Ec][self.indices]
        model = self.get_model(theta, Ec=Ec)[self.indices]
        return -np.sum(model - data*np.log(model)) + self.ln_data_fact[Ec]
    
    def ln_prior(self, theta):
        if self.method == 'alpha':
            return self.ln_prior_alpha(*theta)
        else:
            raise KeyError('Method not recognized.')
        
    def ln_prior_alpha(self, Xem, alpha, beta):
        X_min, X_max = self.priors[0]
        al_min, al_max = self.priors[1]
        bt_min, bt_max = self.priors[2]

        if (X_min < Xem < X_max) and (al_min < alpha < al_max) and (bt_min < beta < bt_max):
            return 0.0
        else:
            return - np.inf
    
    def get_pos0(self):
        if self.method == 'alpha':
            X_min, X_max = self.priors[0]
            al_min, al_max = self.priors[1]
            bt_min, bt_max = self.priors[2]
            pos0 = np.zeros([self.nwalkers, 3])
            pos0[:,0] = (X_max - X_min)*np.random.random(size=self.nwalkers) + X_min
            pos0[:,1] = (al_max - al_min)*np.random.random(size=self.nwalkers) + al_min
            pos0[:,2] = (bt_max - bt_min)*np.random.random(size=self.nwalkers) + bt_min
        return pos0
    # ----------------------------------------------- Ratio Model -----------------------------------------------
    def Te_from_R(self, rs, Ec=4.0):
        if Ec > self.Ec_ref:
            return np.interp(rs, self.ratios[Ec], self.temps)
        else:
            # Reverse to avoid interpolation error
            return np.interp(rs, np.flip(self.ratios[Ec]), np.flip(self.temps))
    
    def get_en_int(self, Te, Ec):
        """
        Model the local emissivity, to a constant factor.
        """
        en = np.linspace(1500, 20000, num=1000)
        resp = rsp.Pilatus_Response(Ec, self.Ew)
        return np.trapz(resp(en)*np.exp(-en/Te)/np.sqrt(Te)/en, x=en)

    def model_ratio(self, Te, Ec):
        """
        Get the ratio for a given Te and Ec relative to the reference
        """
        en_int = self.get_en_int(Te, Ec)
        ref = self.get_en_int(Te, self.Ec_ref*1000.)
        return en_int / ref
        
    # ----------------------------------------------- Analysis Methods -----------------------------------------------
    def get_emiss_samples(self, samples, n_samples=5000, Ec=3.0):
        """
        """
        ss = self.ss_set[Ec]

        emiss_samples = np.zeros([n_samples, len(ss)])
        for index,n in enumerate(np.random.randint(samples.shape[0]-1, size=n_samples)):
            emiss_samples[index,:] = self.emiss_model(ss, *samples[n,:]).squeeze()

        return emiss_samples
    
    def get_params(self):
        """
        Return LaTeX fromatted strings for each parameter.
        """
        if self.method == 'alpha':
            return [r'$X_{\varepsilon}$', r'$\alpha$', r'$\beta$']
        else:
            return []
        
# ----------------------------------------------- Diagnostic Plots -----------------------------------------------
import mst_ida.analysis.plots as idap
from matplotlib.lines import Line2D
from mst_ida.utilities.graphics import *

def burn_plot(sampler, Ec=3.0, burn_step=30000):
    params = sampler.get_params()
    return idap.burn_plot(sampler.samplers[Ec].chain, burn_step, sampler.nwalkers, indices=range(len(params)), labels=params, figsize=(8,6))
    
def distplot(sampler, Ec=3.0):
    return distplot(sampler.samples[Ec], labels=sampler.get_params())
    
def emiss_xs_plot(sampler, Ec=3.0, shax=False):
    """
    Plot the cross-section (XS) using the average parameter values.
    """
    fig, ax = plt.subplots(1,1)
    xs, ys = sunflower_points(5000)
    s_xs = sampler.flux.rho(xs, ys)
    emiss_xs_samples = np.zeros([sampler.n_samples, len(s_xs)])
    for n in range(sampler.n_samples):
        emiss_xs_samples[n,:] = sampler.emiss_model(s_xs, *sampler.samples[Ec][n,:])

    emiss_xs = np.average(emiss_xs_samples, axis=0)

    cax = plt.tricontourf(xs, ys, emiss_xs, 50, cmap='plasma')
    plt.colorbar(cax, label=r'SXR emission (ph ms$^{-1}$ m$^{-3}$ sr$^{-1}$)')
    plt.xlabel(r'$R - R_0$ (m)')
    plt.ylabel(r'$Z$ (m)')
    overplot_shell(plt.gca())
    text(0.11, 0.96, r'$E_c = {0:.1f}$ keV'.format(Ec), plt.gca())

    if shax:
        levels = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        cax = plt.tricontour(sampler.flux.xs, sampler.flux.ys, sampler.flux.ss, levels, colors='black')

    return fig, ax

def emiss_prof_plot(sampler):
    fig, ax = plt.subplots(1,1)
    labels = ['{0:.1f} keV'.format(Ec) for Ec in sampler.thresholds]
    nlabels = len(labels)
    custom_lines = [Line2D([0], [0], color='xkcd:bright blue', lw=4),
                    Line2D([0], [0], color='xkcd:red', lw=4),
                    Line2D([0], [0], color='xkcd:kelly green', lw=4),
                    Line2D([0], [0], color='xkcd:steel', lw=4),
                    Line2D([0], [0], color='xkcd:bright blue', lw=4),
                    Line2D([0], [0], color='xkcd:red', lw=4),
                    Line2D([0], [0], color='xkcd:kelly green', lw=4),
                    Line2D([0], [0], color='xkcd:steel', lw=4)]

    for index,Ec in enumerate(sampler.thresholds):
        ss = sampler.ss_set[Ec].ravel()
        idap.profile_CIs(ax, sampler.emiss_CIs[Ec], ss, ylim=[0,1], ylabel=r'$\varepsilon(s)$', cid=(index%4)+1, legend=False)

    ax.set_ylabel(r'SXR emission (ph ms$^{-1}$ m$^{-3}$ sr$^{-1}$)')
    ax.set_xlabel('Flux $\psi$ (norm.)')
    ax.legend(custom_lines[:nlabels], labels)
    return fig, ax

def Te_prof_plot(sampler, ax=None, legend=True, **kwargs):
    s_vals, Te_avg_prof_samples, Te_avg_CI = sampler.get_Te_samples(**kwargs)
    
    if ax is None:
        fig, ax = plt.subplots(1,1)

    idap.profile_CIs(ax, Te_avg_CI, s_vals, ylim=[0,1], ylabel=r'$T_e$', legend=legend)
    ax.set_ylabel('Electron temperature (eV)')
    ax.set_xlabel(r'Flux surface label $\psi$')
    return ax
    
def get_fit_plot(sampler, n_samples=1000):
    fig, ax = plt.subplots(1,1)
    
    for Ec in sampler.thresholds:
        model_samples = np.zeros([n_samples, 60])

        for index,n in enumerate(np.random.randint(sampler.samples[Ec].shape[0]-1, size=n_samples)):
            theta = sampler.samples[Ec][n,:]
            model_samples[index,:] = sampler.get_model(theta, Ec=Ec)

        model_CI = ida.profile_confidence(model_samples)
        ax.errorbar(sampler.signed_ps[Ec], sampler.mesxr_data[Ec], yerr=sampler.mesxr_sigmas[Ec], marker='o', ms=3, capsize=3,
                    linestyle='none', label='Data', zorder=100)
        idap.profile_CIs(plt.gca(), model_CI, sampler.p3det.p[Ec], ylim=[-0.45,0.45], ylabel=r'$N_\gamma$', legend=False, cid=4)

    plt.xlabel('Chord radius (m)')
    plt.ylabel('Counts (photons/ms)')
    plt.xlim([-0.45, 0.45])
    plt.ylim([0,None])
    return fig, ax

def get_hist_plot(sampler, s_index=0, include=None, xlim=(500,2000)):
    if include is None:
        include = [Ec for Ec in sampler.thresholds if Ec != sampler.Ec_ref]

    ratios = {Ec:sampler.emiss_samples[Ec][:,s_index]/sampler.emiss_samples[sampler.Ec_ref][:,s_index] for Ec in include}
    Te_samples = {Ec:sampler.Te_from_R(ratios[Ec], Ec=Ec) for Ec in include}
    s_vals, Te_avg_samples, Te_avg_CI = sampler.get_Te_samples(include=include)

    fig, ax = plt.subplots(1,1)
    norm = Normalize(min(include), max(include))
    cmap = cm.jet

    for Ec in include:
        hout = plt.hist(Te_samples[Ec], bins=100, histtype='step', density=True, color='black', range=xlim)

        kde = sp.stats.gaussian_kde(Te_samples[Ec])
        tes = np.linspace(*xlim, num=2000)
        plt.plot(tes, kde(tes), label=Ec, color=cmap(norm(Ec)))

    hout = plt.hist(Te_avg_samples[:,s_index], bins=100, histtype='step', density=True, color='black', range=xlim)
    kde = sp.stats.gaussian_kde(Te_avg_samples[:,s_index])
    tes = np.linspace(*xlim, num=2000)
    plt.plot(tes, kde(tes), color='black', linestyle='--', label='Average')
        
    plt.legend()
    plt.xlabel('Electron temperature (eV)')
    plt.ylabel(r'$p(T_e|d)$')
    return fig, ax