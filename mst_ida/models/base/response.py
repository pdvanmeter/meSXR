"""
"""
import os
import pickle
import numpy as np
import scipy as sp
import scipy.special

# Load the mu coefficients for filter transmission calculations
MODULE_PATH = os.path.dirname(__file__)
with open(os.path.join(MODULE_PATH, 'filter_mu.pkl'), 'rb') as f:
        MU_DICT = pickle.load(f, encoding='latin1')

# Common material densities - g/cm^3
DENS = {'Si':2.330,
        'Be':1.848,
        'mylar':1.39}

# Constants for the ME-SXR detector - cm
BE_THICK = 0.0025
SI_THICK = 0.045
MYLAR_THICK = 0.0012 + 0.0100 # was 0.0012 + 0.0050

# -------------------------------------- Response Classes -------------------------------------- #

class Response(object):
    """
    This class is used to make general spectral response objects. For implementations see
    the S_Curve, Filter, and Absorber classes.
    """
    def __init__(self, label, en_lims=[0,30000], units='eV'):
        self.label = label
        self.lims = en_lims
        self.units = units

    def __str__(self):
        return self.label

    def __call__(self, en):
        return self.evaluate(en)

    def __add__(self, other_resp):
        return Composite_Response(self, other_resp, operation='add')

    def __mul__(self, other_resp):
        return Composite_Response(self, other_resp, operation='multiply')

    def __radd__(self, other):
        return self

    def __rmul__(self, other):
        return self

    def domain(self, en):
        return np.amin(en) >= self.lims[0] and np.amax(en) <= self.lims[1]

    def value(self, en):
        return 1

    def evaluate(self, en):
        if self.domain(en):
            return self.value(en)
        else:
            return np.zeros(en.shape)


class Composite_Response(Response):
    """
    This class permits multiple response objects to be combined together into a single composite.
    """
    def __init__(self, resp1, resp2, operation='multiply'):
        self.resp1 = resp1
        self.resp2 = resp2
        self.operation = operation

        # Set the limits to be the intersection of the two supplied profiles
        en_lims = [np.amin([self.resp1.lims[0], self.resp2.lims[0]]), np.amax([self.resp1.lims[1], self.resp2.lims[1]])]

        # Check for unit compatibility
        if self.resp1.units == self.resp2.units:
            units = resp1.units
        else:
            raise ValueError('Profiles have incompatible units.')

        # Generate the appropriate label
        if self.operation == 'add':
            label = '{0:} + {1:}'.format(str(self.resp1), str(self.resp2))
        elif self.operation == 'multiply':
            label = '({0:} x {1:})'.format(str(self.resp1), str(self.resp2))
        else:
            raise ValueError('Operation not recognized.')

        super(Composite_Response, self).__init__(label, en_lims=en_lims, units=units)

    def value(self, en):
        if self.operation == 'add':
            return self.resp1(en) + self.resp2(en)
        elif self.operation == 'multiply':
            return self.resp1(en) * self.resp2(en)
        else:
            raise ValueError('Operation not recognized.')


class S_Curve(Response):
    """
    This simple class allows for the computation of the pixel S-curve response.
    """
    def __init__(self, E_c, E_w, en_lims=[0,30000], units='eV'):
        self.E_c = E_c
        self.E_w = E_w
        super(S_Curve, self).__init__('S-curve', en_lims=en_lims, units=units)
        
    def value(self, en):
        return 0.5*sp.special.erfc(-1.*(en - self.E_c)/(np.sqrt(2)*self.E_w))


class Filter(Response):
    """
    This simple class allows for the computation of the transmission through a solid filter
    (i.e. Be or mylar).
    """
    def __init__(self, element, thickness, en_lims=[0,30000], units='eV'):
        self.element = element
        self.thickness = thickness
        self.density = DENS[element]
        self.mu = MU_DICT['mu'][self.element]
        self.en_mu = MU_DICT['energy']
        super(Filter, self).__init__('{0:} Filter'.format(self.element), en_lims=en_lims, units=units)
        
    def value(self, en):
        return np.exp(-np.interp(en, self.en_mu, self.mu)*self.density*self.thickness)


class Absorber(Response):
    """
    This simple class allows for the computation of the absorption in a solid layer
    (i.e. an Si photodiode).
    """
    def __init__(self, element, thickness, en_lims=[0,30000], units='eV'):
        self.element = element
        self.thickness = thickness
        self.density = DENS[element]
        self.mu = MU_DICT['mu'][self.element]
        self.en_mu = MU_DICT['energy']
        super(Absorber, self).__init__('{0:} Filter'.format(self.element), en_lims=en_lims, units=units)
        
    def value(self, en):
        return 1.0 - np.exp(-np.interp(en, self.en_mu, self.mu)*self.density*self.thickness)

class Charge_Sharing(Response):
    """
    This function allows the implementation of charge sharing in the detector response.

    UPDATE: This version now uses the correct implementation of charge sharing
    """
    def __init__(self, E_c, cs_frac=0.266, en_lims=[0,30000], units='eV'):
        self.E_c = E_c
        self.frac = cs_frac
        super(Charge_Sharing, self).__init__('Charge-sharing', en_lims=en_lims, units=units)

    def value(self, en):
        return (1. + 2.*self.frac*(1. - self.E_c/en) ) / (1. + self.frac)

class Pilatus_Response(Response):
    """
    The class defines the total response for the Pilatus 3 detector. It is simply a wrapper to define
    the whole response in a single line. This is convenient because the filter and Si layer properties
    are generally not mutable.
    """
    def __init__(self, E_c, E_w, Si_thickness=SI_THICK, Be_thickness=BE_THICK, mylar_thickness=MYLAR_THICK, en_lims=[0,30000],
                 charge_sharing=True, cs_frac=0.266):
        self.Si = Absorber('Si', Si_thickness, en_lims=en_lims, units='eV')
        self.Be = Filter('Be', Be_thickness, en_lims=en_lims, units='eV')
        self.mylar = Filter('mylar', mylar_thickness, en_lims=en_lims, units='eV')
        self.scurve = S_Curve(E_c, E_w, en_lims=en_lims, units='eV')
        
        if charge_sharing:
            self.charge_share = Charge_Sharing(E_c, cs_frac=cs_frac, en_lims=en_lims)
            self.total = self.Si * self.Be * self.scurve * self.charge_share * self.mylar
        else:
            self.total = self.Si * self.Be * self.scurve *  self.mylar
        
        super(Pilatus_Response, self).__init__('Total Response', en_lims=en_lims, units='eV')
        
    def value(self, en):
        return self.total(en)