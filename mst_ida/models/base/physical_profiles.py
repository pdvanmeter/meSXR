"""
This module contains a number of physical profiles which implement the Profile class
derivatives for use in realistic plasma modeling. Most of these classes are simple wrappers
for the defined profiles which handles the labels, units, and provides default values.
"""
from __future__ import division
import os
import pickle
import numpy as np
from mst_ida.models.base.profile_base import *

# Module-wide constants
DENSITY_SCALE_FACT_SI = 1.0e19     # Density is in units of 1x10^19 m^-3
NH_DENSITY_SCALE_FACT_SI = 1.0e14  # Neutral density is in units of 1x10^14 m^-3
MST_MAJOR_RADIUS = 1.5

class Temperature_Alpha(Profile_Alpha_Beta_Rho):
    """
    """
    def __init__(self, core_Te, alpha=2.5, beta=1.2, delta_a=0.06, delta_h=0.01, flux=None):
        super(Temperature_Alpha, self).__init__('Electron Temperature', core_Te, alpha, beta, units='eV', dim_units=['m', 'm'],
                                                delta_a=delta_a, delta_h=delta_h, flux=flux)


class Temperature_Island(Profile_Island_Flux):
    """
    """
    def __init__(self, delta_Te, r_0 = 0.5, theta_0 = 0.6, delta_r = 0.2, delta_theta = np.pi/4, delta_a=0.06, delta_h=0.01, flux=None):
        super(Temperature_Island, self).__init__('Temperature Island', delta_Te, r_0, theta_0, delta_r, delta_theta,
                                                 units='eV', dim_units=['m', 'm'], delta_a=delta_a, delta_h=delta_h, flux=flux)

class Temperature_SHAx(Profile_Peaked_Rho):
    """
    """
    def __init__(self, Te0, Te1, alpha=4.0, delta=0.2, beta=5.0, flux=None):
        super(Temperature_SHAx, self).__init__('Electron Temperature', Te0, alpha, Te1, delta, beta,
              units='eV', dim_units=['m', 'm'], flux=flux)


class Electron_Density_Alpha(Profile_Alpha_Beta_Rho):
    """
    """
    def __init__(self, core_ne, alpha=4.2, beta=4.0, delta_a=0.06, delta_h=0.01, flux=None):
        core_ne *= DENSITY_SCALE_FACT_SI
        super(Electron_Density_Alpha, self).__init__('Electron Density', core_ne, alpha, beta, dim_units=['m', 'm'],
                                                     units=r'$m^{-3}$', delta_a=delta_a, delta_h=delta_h, flux=flux)

class Electron_Density_SHAx(Profile_Peaked_Rho):
    """
    """
    def __init__(self, ne0, ne1, alpha=4.0, delta=0.2, beta=5.0, flux=None):
        ne0 *= DENSITY_SCALE_FACT_SI
        ne1 *= DENSITY_SCALE_FACT_SI
        super(Electron_Density_SHAx, self).__init__('Electron Density', ne0, alpha, ne1, delta, beta,
              units=r'$m^{-3}$', dim_units=['m', 'm'], flux=flux)

class Density_Constant(Profile_Const):
    """
    Simple object to add a constant offset to a density within the domain. Will work with either electron or ion
    density profiles.
    """
    def __init__(self, dens):
        dens *= DENSITY_SCALE_FACT_SI
        super(Density_Constant, self).__init__('Density Offset', units=r'$m^{-3}$', dim_units=['m', 'm'], const=dens)

class Ion_Density_Alpha(Profile_Alpha_Beta_Rho):
    """
    """
    def __init__(self, core_ni, alpha=15, beta=2, element='Al', delta_a=0.06, delta_h=0.01, flux=None):
        self.symbol = element
        core_ni *= DENSITY_SCALE_FACT_SI
        super(Ion_Density_Alpha, self).__init__('{0:} Density'.format(element), core_ni, alpha, beta, dim_units=['m', 'm'],
                                                units=r'$m^{-3}$', delta_a=delta_a, delta_h=delta_h, flux=flux)

class Ion_Density_SHAx(Profile_Peaked_Rho):
    """
    """
    def __init__(self, ne0, ne1, alpha=4.0, delta=0.2, beta=5.0, element='Al', flux=None):
        self.symbol = element
        ne0 *= DENSITY_SCALE_FACT_SI
        ne1 *= DENSITY_SCALE_FACT_SI
        super(Ion_Density_SHAx, self).__init__('{0:} Density'.format(element), ne0, alpha, ne1, delta, beta,
              units=r'$m^{-3}$', dim_units=['m', 'm'], flux=flux)

class Density_Island(Profile_Island_Flux):
    """
    Defines the profile for an density island, generally to be added on top of an alpha-beta profile. Note that this
    is intended to be added to both electron and ion density profiles.
    """
    def __init__(self, delta_n, r_0 = 0.5, theta_0 = 0.6, delta_r = 0.2, delta_theta = np.pi/4, delta_a=0.06, delta_h=0.01, flux=None):
        delta_n *= DENSITY_SCALE_FACT_SI
        super(Density_Island, self).__init__('Density Island', delta_n, r_0, theta_0, delta_r, delta_theta, units=r'$m^{-3}$',
                                             dim_units=['m', 'm'], delta_a=delta_a, delta_h=delta_h, flux=flux)


class Density_Hollow(Profile_Hollow_Flux):
    """
    Defines the profile for an hollow feature, generally to be added on top of an alpha-beta profile. Note that this
    is intended to be added to both electron and ion density profiles.
    """
    def __init__(self, delta_n, peak_r=0.33/0.52, width_r=0.09, delta_a=0.06, delta_h=0.01, flux=None):
        delta_n *= DENSITY_SCALE_FACT_SI
        super(Density_Hollow, self).__init__('Density Hollow Profile', delta_n, peak_r, width_r, units=r'$m^{-3}$', dim_units=['m', 'm'],
              delta_a=delta_a, delta_h=delta_h, flux=flux)

class CSF_Profile(Profile_2D_Spline):
    """
    This class handles the creation of a 2D spline-based profile for the charge state fraction of a given ion based
    on the supplied samples.
    """
    def __init__(self, element, atomic_number, sample_coords, sample_data):
        self.element = element
        self.z1 = atomic_number
        self.data = sample_data
        self.coords = sample_coords
        super(CSF_Profile, self).__init__('{0:}+{1:d} CSF'.format(self.element, self.z1), self.coords, self.data,
                                          units='N/A', dim_units=['m', 'm'], method='linear')


class Neutral_Density_Profile(Profile_2D_Spline):
    """
    """
    def __init__(self, time, type='PPCD'):
        self.time = time
        
        # Load the neutral profile data from file
        if type == 'PPCD':
            fpath = os.path.join(os.path.dirname(__file__), 'files', 'PPCD_neutral_data.pkl')
        else:
            raise TypeError('Plasma type not yet supported.')

        with open(fpath, 'rb') as f:
            neutral_data = pickle.load(f, encoding='latin1')

        # Find the key most closely matching the supplied time period
        if type == 'PPCD' and time >=18.0:
            # Long-lived PPCD plasmas shoud use the profile for 18.0 ms
            time = 18.0
        else:
            keys = np.array(sorted(neutral_data.keys()))
            time = keys[np.abs(keys - time).argmin()]

        self.coords = list(zip(neutral_data[time]['R'] - MST_MAJOR_RADIUS, neutral_data[time]['Z']))
        self.data = neutral_data[time]['nH']
        super(Neutral_Density_Profile, self).__init__('Neutral Density', self.coords, self.data,
                                          units=r'$m^{-3}$', dim_units=['m', 'm'], method='linear')      

class Neutral_Density_Model(Profile_Power_Flux):
    """
    This is a simplistic parametric model for neutral density, which can be useful for not overfitting the inhomogeneities
    present in Anthony's reconstructed neutral model.
    """
    def __init__(self, nH_0, nH_rm, nH_1, rm, delta_a=0.06, delta_h=0.01, flux=None):
        # Calculate the parameters based on the supplied constraints
        self.core_value = nH_0
        self.amp = nH_1 - nH_0
        self.power = np.log( (nH_rm - nH_0)/(nH_1 - nH_0) ) / np.log(rm)
        
        self.core_value *= NH_DENSITY_SCALE_FACT_SI
        self.amp *= NH_DENSITY_SCALE_FACT_SI
        super().__init__('Neutral Density Profile', self.core_value, self.amp, self.power, units=r'$m^{-3}$',
                         dim_units=['m', 'm'], delta_a=delta_a, delta_h=delta_h, flux=flux)
        
        # Set the boundary to the edge value instead of zero
        self.boundary = nH_1 * NH_DENSITY_SCALE_FACT_SI