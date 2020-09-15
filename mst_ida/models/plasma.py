"""
"""
from __future__ import division
import os
import pickle
import numpy as np
import scipy as sp
import scipy.interpolate
import mst_ida.models.base.physical_profiles as prof

class Plasma(object):
    def __init__(self, Te_prof, ne_prof, n0_prof, nZ_set={}, avg_Z=None):
        """
        This class functions as a container for the Profile objects which define a plasma. The new Plasma class
        is simple and lightweight, so objects of this class can be created rapidly when performing iterative analyses.
        
        Use the avg_Z keyword to save computation time and memory when performing calculations that involve a very large
        number of Plasma objects.
        """
        self.Te = Te_prof
        self.ne = ne_prof
        self.n0 = n0_prof
        
        self.impurities = nZ_set.keys()
        self.nZ = nZ_set
        
        if avg_Z is None:
            self.avg_Z = get_avg_Z(impurities=self.impurities)
        else:
            self.avg_Z = avg_Z
        
    def add_ion(self, symbol, nZ_prof):
        self.impurities.append(symbol)
        self.nZ[symbol] = nZ_prof
        
    def nD(self, xs, ys):
        """
        Calculate the deutrium density for some specified spatial points.
        """
        # Evaluate the profiles
        Te_xs = np.maximum(self.Te(xs, ys), 10.0)
        ne_xs = np.maximum(self.ne(xs, ys), 1e15)
        n0_xs = self.n0(xs, ys)
        nZ_xs = {ion:self.nZ[ion](xs,ys) for ion in self.impurities}
        
        # Get the average charge numbers
        pts = list( zip( Te_xs.ravel(), ne_xs.ravel()/1e19, n0_xs.ravel()/1e14 ) )
        avg_Z_xs = {ion:np.reshape(self.avg_Z[ion](pts), xs.shape) for ion in self.impurities}
        
        # Evaluate quasi-neutrality
        nD_xs = np.ones(xs.shape)
        for ion in self.impurities:
            nD_xs -= (nZ_xs[ion]/ne_xs)*avg_Z_xs[ion]
        nD_xs *= ne_xs
        nD_xs[nD_xs < 0] = 0.0
        
        return nD_xs
    
    def Zeff(self, xs, ys):
        """
        Calculate the ion-effective charge for some specified spatial points.
        """
        # Evaluate the profiles
        Te_xs = np.maximum(self.Te(xs, ys), 10.0)
        ne_xs = np.maximum(self.ne(xs, ys), 1e15)
        n0_xs = self.n0(xs, ys)
        nZ_xs = {ion:self.nZ[ion](xs,ys) for ion in self.impurities}
        nD_xs = self.nD(xs, ys)
        
        # Get the average charge numbers
        pts = list( zip( Te_xs.ravel(), ne_xs.ravel()/1e19, n0_xs.ravel()/1e14 ) )
        avg_Z_xs = {ion:np.reshape(self.avg_Z[ion](pts), xs.shape) for ion in self.impurities}
        
        # Evaluate the ion-effective charge
        Zeff_xs = (nD_xs / ne_xs)
        for ion in self.impurities:
            Zeff_xs += (nZ_xs[ion]/ne_xs)*avg_Z_xs[ion]**2
            
        return Zeff_xs
    
    def nD_rv(self, rs):
        """
        A modified version of the self.nD function which allows for 1D coordinate
        input, assuming all underlying profiles implement a univariate value function.
        """
        # Evaluate the profiles
        Te_rs = np.maximum(self.Te.value(rs), 10.0)
        ne_rs = np.maximum(self.ne.value(rs), 1e15)
        n0_rs = self.n0.value(rs)
        nZ_rs = {ion:self.nZ[ion].r_value(rs) for ion in self.impurities}
        
        # Get the average charge numbers
        pts = list( zip( Te_rs, ne_rs/1e19, n0_rs/1e14 ) )
        avg_Z_rs = {ion:self.avg_Z[ion](pts) for ion in self.impurities}
        
        # Evaluate quasi-neutrality
        nD_rs = np.ones(rs.shape)
        for ion in self.impurities:
            nD_rs -= (nZ_rs[ion]/ne_rs)*avg_Z_rs[ion]
        nD_rs *= ne_rs
        nD_rs[nD_rs < 0] = 0.0
        
        return nD_rs
    
    def Zeff_rv(self, rs):
        """
        A modified version of the self.Zeff function which allows for 1D coordinate
        input, assuming all underlying profiles implement a univariate value function.
        """
         # Evaluate the profiles
        Te_rs = np.maximum(self.Te.value(rs), 10.0)
        ne_rs = np.maximum(self.ne.value(rs), 1e15)
        n0_rs = self.n0.value(rs)
        nZ_rs = {ion:self.nZ[ion].r_value(rs) for ion in self.impurities}
        nD_rs = self.nD_rv(rs)
        
        # Get the average charge numbers
        pts = list( zip( Te_rs, ne_rs/1e19, n0_rs/1e14 ) )
        avg_Z_rs = {ion:self.avg_Z[ion](pts) for ion in self.impurities}
        
        # Evaluate the ion-effective charge
        Zeff_rs = (nD_rs / ne_rs)
        for ion in self.impurities:
            Zeff_rs += (nZ_rs[ion]/ne_rs)*avg_Z_rs[ion]**2
            
        return Zeff_rs

# ---------------------------------------------- Get plasma ----------------------------------------------

def get_plasma(Te0, alpha_Te, beta_Te, ne0, alpha_ne, beta_ne, nAl0, nC0, delta_nAl, delta_nC, peak_r, width_r, alpha_Z, beta_Z,
               n0_params=np.array([2.0, 10.0, 450.0, 0.6]), delta_a=0.06, delta_h=0.01, flux=None, avg_Z=None):
    """
    This plasma is meant to represent a rotating standard RFP plasma, or locked PPCD.
    This version uses a direct parameterization for ion densities.
    """
    # Define the plasma
    Te_prof = prof.Temperature_Alpha(Te0, alpha=alpha_Te, beta=beta_Te, delta_a=delta_a, delta_h=delta_h, flux=flux)
    ne_prof = prof.Electron_Density_Alpha(ne0, alpha=alpha_ne, beta=beta_ne, delta_a=delta_a, delta_h=delta_h, flux=flux)
    n0_prof = prof.Neutral_Density_Model(*n0_params, flux=flux)
    
    # Impurity base profile parameters
    nZ0 = {'Al':nAl0, 'C':nC0, 'N':0.3*nC0, 'B':0.3*nC0, 'O':0.9*nC0}

    # PPCD hollow profile parameters
    nZ_delta = {'Al':delta_nAl, 'C':delta_nC, 'N':0.3*delta_nC, 'B':0.3*delta_nC, 'O':0.75*delta_nC}

    nZ_set = {}
    for ion in nZ0.keys():
        nZ_set[ion] = prof.Ion_Density_Alpha(nZ0[ion], alpha=alpha_Z, beta=beta_Z, element=ion, delta_a=delta_a, delta_h=delta_h, flux=flux)
        nZ_set[ion] += prof.Density_Hollow(nZ_delta[ion], peak_r=peak_r, width_r=width_r, delta_a=delta_a, delta_h=delta_h, flux=flux)

    return Plasma(Te_prof, ne_prof, n0_prof, nZ_set=nZ_set, avg_Z=avg_Z)

def get_plasma2(Te0, alpha_Te, beta_Te, ne0, alpha_ne, beta_ne, XAl, XC, YAl, YC, peak_r, width_r, alpha_Z, beta_Z,
               n0_params=np.array([2.0, 10.0, 450.0, 0.6]), delta_a=0.06, delta_h=0.01, flux=None, avg_Z=None):
    """
    This plasma is meant to represent a rotating standard RFP plasma, or locked PPCD.
    This version uses an exponentail parameterization for ion densities.
    """
    # Define the plasma
    Te_prof = prof.Temperature_Alpha(Te0, alpha=alpha_Te, beta=beta_Te, delta_a=delta_a, delta_h=delta_h, flux=flux)
    ne_prof = prof.Electron_Density_Alpha(ne0, alpha=alpha_ne, beta=beta_ne, delta_a=delta_a, delta_h=delta_h, flux=flux)
    n0_prof = prof.Neutral_Density_Model(*n0_params, flux=flux)
    
    # Impurity base profile parameters
    nAl0 = (10.0**XAl) / 1e19
    nC0 = (10.0**XC) / 1e19
    nZ0 = {'Al':nAl0, 'C':nC0, 'N':0.3*nC0, 'B':0.3*nC0, 'O':0.9*nC0}

    # PPCD hollow profile parameters
    delta_nAl = (10.0**YAl) / 1e19
    delta_nC = (10.0**YC) / 1e19
    nZ_delta = {'Al':delta_nAl, 'C':delta_nC, 'N':0.3*delta_nC, 'B':0.3*delta_nC, 'O':0.75*delta_nC}

    nZ_set = {}
    for ion in nZ0.keys():
        nZ_set[ion] = prof.Ion_Density_Alpha(nZ0[ion], alpha=alpha_Z, beta=beta_Z, element=ion, delta_a=delta_a, delta_h=delta_h, flux=flux)
        nZ_set[ion] += prof.Density_Hollow(nZ_delta[ion], peak_r=peak_r, width_r=width_r, delta_a=delta_a, delta_h=delta_h, flux=flux)

    return Plasma(Te_prof, ne_prof, n0_prof, nZ_set=nZ_set, avg_Z=avg_Z)

# --------------------------------------------- MSTfit functions ---------------------------------------------
class ne_prof_flux(object):
    """
    A simple wrapper to ensure that the density profile implements the necessary
    functions to work in my framework.
    """
    def __init__(self, flux):
        self.flux = flux
    
    def __call__(self, *args):
        return self.flux.dens(*args)
    
    def value(self, *args):
        return self.flux.dens_rho(*args)

def get_plasma_MSTfit(Te0, alpha_Te, beta_Te, nAl0, nC0, delta_nAl, delta_nC, peak_r, width_r, alpha_Z, beta_Z,
               n0_params=np.array([2.0, 10.0, 450.0, 0.6]), flux=None, avg_Z=None):
    """
    This plasma is meant to represent a rotating standard RFP plasma, or locked PPCD. Assumes an MSTfit flux surface
    recnstruction and uses that density profile. To reduce the total number of varaibles write a new function which
    calls this function.
    This version uses a direct parameterization for ion densities.
    """
    if flux is None:
        raise ValueError('Flux map  not provided.')

    # Define the plasma
    Te_prof = prof.Temperature_Alpha(Te0, alpha=alpha_Te, beta=beta_Te, flux=flux)
    ne_prof = ne_prof_flux(flux)
    n0_prof = prof.Neutral_Density_Model(*n0_params, flux=flux)
    
    # Impurity base profile parameters
    nZ0 = {'Al':nAl0, 'C':nC0, 'N':0.3*nC0, 'B':0.3*nC0, 'O':0.9*nC0}

    # PPCD hollow profile parameters
    nZ_delta = {'Al':delta_nAl, 'C':delta_nC, 'N':0.3*delta_nC, 'B':0.3*delta_nC, 'O':0.75*delta_nC}

    nZ_set = {}
    for ion in nZ0.keys():
        nZ_set[ion] = prof.Ion_Density_Alpha(nZ0[ion], alpha=alpha_Z, beta=beta_Z, element=ion, flux=flux)
        nZ_set[ion] += prof.Density_Hollow(nZ_delta[ion], peak_r=peak_r, width_r=width_r, flux=flux)

    return Plasma(Te_prof, ne_prof, n0_prof, nZ_set=nZ_set, avg_Z=avg_Z)

def get_plasma2_MSTfit(Te0, alpha_Te, beta_Te, XAl, XC, YAl, YC, peak_r, width_r, alpha_Z, beta_Z,
               n0_params=np.array([2.0, 10.0, 450.0, 0.6]), flux=None, avg_Z=None):
    """
    This plasma is meant to represent a rotating standard RFP plasma, or locked PPCD. Assumes an MSTfit flux surface
    recnstruction and uses that density profile. To reduce the total number of varaibles write a new function which
    calls this function.
    This version uses an exponentail parameterization for ion densities.
    """
    if flux is None:
        raise ValueError('Flux map  not provided.')

    # Define the plasma
    Te_prof = prof.Temperature_Alpha(Te0, alpha=alpha_Te, beta=beta_Te, flux=flux)
    ne_prof = ne_prof_flux(flux)
    n0_prof = prof.Neutral_Density_Model(*n0_params, flux=flux)
    
    # Impurity base profile parameters
    nAl0 = (10.0**XAl) / 1e19
    nC0 = (10.0**XC) / 1e19
    nZ0 = {'Al':nAl0, 'C':nC0, 'N':0.3*nC0, 'B':0.3*nC0, 'O':0.9*nC0}

    # PPCD hollow profile parameters
    delta_nAl = (10.0**YAl) / 1e19
    delta_nC = (10.0**YC) / 1e19
    nZ_delta = {'Al':delta_nAl, 'C':delta_nC, 'N':0.3*delta_nC, 'B':0.3*delta_nC, 'O':0.75*delta_nC}

    nZ_set = {}
    for ion in nZ0.keys():
        nZ_set[ion] = prof.Ion_Density_Alpha(nZ0[ion], alpha=alpha_Z, beta=beta_Z, element=ion, flux=flux)
        nZ_set[ion] += prof.Density_Hollow(nZ_delta[ion], peak_r=peak_r, width_r=width_r, flux=flux)

    return Plasma(Te_prof, ne_prof, n0_prof, nZ_set=nZ_set, avg_Z=avg_Z)

# --------------------------------------------- V3Fit functions ---------------------------------------------   
_Te0 = 600.
_Te1 = 100.
_alpha_Te = 4.0
_delta_Te = 0.2
_beta_Te = 5.0

_ne0 = 0.5
_ne1 = 0.1
_alpha_ne = 4.0
_delta_ne = 0.2
_beta_ne = 5.0

_nAl0 = 2.0e-3
_nC0 = 6.8e-3
_alpha_Z = 12.0
_beta_Z = 4.0
_n0_params=np.array([2.0, 10.0, 450.0, 0.6])

_XAl = 16.5
_XC = 16.83

def get_plasma_QSH(flux, Te0=_Te0, Te1=_Te1, alpha_Te=_alpha_Te, delta_Te=_delta_Te, beta_Te=_beta_Te,
                         ne0=_ne0, ne1=_ne1, alpha_ne=_alpha_ne, delta_ne=_delta_ne, beta_ne=_beta_ne,
                         nAl0=_nAl0, nC0=_nC0, alpha_Z=_alpha_Z, beta_Z=_beta_Z,
                         n0_params=_n0_params, avg_Z=None, XAl=None, XC=None):
    """
    Return the appropriate plasma object for a SHAx plasma. Profiles allow core peaking.
    TODO: Update ion profiles.
    """
    Te_prof = prof.Temperature_SHAx(Te0, Te1, alpha=alpha_Te, delta=delta_Te, beta=beta_Te, flux=flux)
    ne_prof = prof.Electron_Density_SHAx(ne0, ne1, alpha=alpha_ne, delta=delta_ne, beta=beta_ne, flux=flux)
    n0_prof = prof.Neutral_Density_Model(*n0_params, flux=flux)
    
    # Impurity base profile parameters
    if (XAl is None) and (XC is None):
        nZ0 = {'Al':nAl0, 'C':nC0, 'N':0.3*nC0, 'B':0.3*nC0, 'O':0.9*nC0}
    else:
        if XAl is None:
            XAl = _XAl
        if XC is None:
            XC = _XC
        nAl0 = (10.0**XAl) / 1e19
        nC0 = (10.0**XC) / 1e19
        nZ0 = {'Al':nAl0, 'C':nC0, 'N':0.3*nC0, 'B':0.3*nC0, 'O':0.9*nC0}

    nZ_set = {}
    for ion in nZ0.keys():
        nZ_set[ion] = prof.Ion_Density_Alpha(nZ0[ion], alpha=alpha_Z, beta=beta_Z, element=ion, flux=flux)

    return Plasma(Te_prof, ne_prof, n0_prof, nZ_set=nZ_set, avg_Z=avg_Z)

class QSH_Equilibrium(object):
    """
    Uses an underlying SHAx model to map shared plasma parameters to the appropriate toroidal angles for each diagnostic.
    It is essentially just a collection of Plasma objects united by shared properties.
    """
    def __init__(self, shax, impurities=['Al','C','N','B','O'], **kwargs):
        self.shax = shax
        self.diagnostics = shax.diagnostics
        self.impurities = impurities
        self.avg_Z = get_avg_Z(impurities=self.impurities)
        self.kwargs = kwargs
        self.plasma = {}
        
        for name in self.diagnostics:
            self.plasma[name] = get_plasma_QSH(self.shax.flux[name], avg_Z=self.avg_Z, **self.kwargs)
            
    def add_diagnostic(self, name, phi):
        """
        Easily add another diagnostic to the plasma equilibrium for quick access.
        """
        self.shax.add_diagnostic(name, phi)
        self.plasma[name] = get_plasma_QSH(self.shax.flux[name], avg_Z=self.avg_Z, **self.kwargs)
            
    def get_plasma(self, phi=None, v=None):
        """
        Use this to get a plasma object at an arbitrary toroidal positional phi or orientation v.
        Note that for repeated access it is better to add a diagnostic to the equilibrium.
        """
        if (phi is None) and (v is None):
            raise ValueError('Must select an angle phi or orientation v.')
        elif (phi is not None) and (v is not None):
            raise ValueError('Please select only one of the angle phi or orientation v.')
        elif phi is not None:
            theta_op = np.mod(np.deg2rad(5*phi + 241.0) - self.delta_n, 2*np.pi)
            if theta_op > np.pi:
                theta_op -= 2*np.pi
            v = theta_op/5
            
        return get_plasma_QSH(self.shax.get_flux(phi=phi,v=v), avg_Z=self.avg_Z, **self.kwargs)

# ---------------------------------------------------- Database functions ----------------------------------------------------
def get_avg_Z(impurities=['Al', 'C', 'O', 'N', 'B'], fname='ADAS_avgZ.pkl', emiss_dir='/home/pdvanmeter/data/ADAS/'):
    """
    """
    fpath = os.path.join(emiss_dir, fname)
    with open(fpath, 'rb') as f:
        avgZ_db = pickle.load(f, encoding='latin1')
    
    Te_set = avgZ_db['axes']['Te']
    ne_set = avgZ_db['axes']['ne']
    n0_set = avgZ_db['axes']['n0']
    
    # Build the interpolation functions
    avg_Z = {ion:{} for ion in impurities}
    for ion in impurities:
        avg_Z[ion] = sp.interpolate.RegularGridInterpolator((Te_set, ne_set, n0_set), avgZ_db[ion], bounds_error=False, fill_value=1.0)

    return avg_Z

def get_csf(impurities=['Al', 'C', 'O', 'N', 'B'], fname='nickal2_sxt_emiss_full.pkl', emiss_dir='/home/pdvanmeter/data/ADAS/'):
    """
    """
    fpath = os.path.join(emiss_dir, fname)
    with open(fpath, 'rb') as f:
        emiss_db = pickle.load(f, encoding='latin1')
    
    Te_set = emiss_db['axes']['Te']
    ne_set = emiss_db['axes']['ne']
    n0_set = emiss_db['axes']['n0']
    
    # Build the interpolation functions
    csf = {ion:{} for ion in impurities}
    for ion in impurities:
        num_Zs = emiss_db['frac'][ion].shape[-1]
        for z1 in range(num_Zs):
            csf[ion][z1] = sp.interpolate.RegularGridInterpolator((Te_set, ne_set, n0_set), 
                                                                  np.swapaxes(emiss_db['frac'][ion], 0, 1)[:,:,:,z1],
                                                                  bounds_error=False, fill_value=0.0)

    return csf

# ----------------------------------------------- Deprecated functions -----------------------------------------------
def get_plasma_old(Te0, alpha_Te, beta_Te, ne0, alpha_ne, beta_ne, nAl0, nC0, delta_nAl, delta_nC, peak_r, width_r, alpha_Z, beta_Z,
               n0_time=17.5, delta_a=0.06, delta_h=0.01):
    """
    This plasma is meant to represent a rotating standard RFP plasma. Neutral density is probably not quite right.
    """
    # Define the plasma
    Te_prof = prof.Temperature_Alpha(Te0, alpha=alpha_Te, beta=beta_Te, delta_a=delta_a, delta_h=delta_h)
    ne_prof = prof.Electron_Density_Alpha(ne0, alpha=alpha_ne, beta=beta_ne, delta_a=delta_a, delta_h=delta_h)
    n0_prof = prof.Neutral_Density_Profile(n0_time)
    
    # Impurity base profile parameters
    nZ0 = {'Al':nAl0, 'C':nC0, 'N':0.3*nC0, 'B':0.3*nC0, 'O':0.9*nC0}

    # PPCD hollow profile parameters
    nZ_delta = {'Al':delta_nAl, 'C':delta_nC, 'N':0.3*delta_nC, 'B':0.3*delta_nC, 'O':0.75*delta_nC}

    nZ_set = {}
    for ion in nZ0.keys():
        nZ_set[ion] = prof.Ion_Density_Alpha(nZ0[ion], alpha=alpha_Z, beta=beta_Z, element=ion, delta_a=delta_a, delta_h=delta_h)
        nZ_set[ion] += prof.Density_Hollow(nZ_delta[ion], peak_r=peak_r, width_r=width_r, delta_a=delta_a, delta_h=delta_h)

    return Plasma(Te_prof, ne_prof, n0_prof, nZ_set=nZ_set, delta_a=delta_a, delta_h=delta_h)