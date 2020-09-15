"""
This module contains the Detector class, which defines the detector and it's response to a plasma. This
modules also contains classes and methods related to the response function (S-curves, absorprion from Si,
transmission through Be, and charge-sharing).
"""
from __future__ import division
import os
import cPickle as pickle
import multiprocessing as mp
import numpy as np
import scipy as sp
import scipy.io
import scipy.special
import pilatus.configuration as cfg

# Other modules in this library
import geometry
import plasma
import physical_profiles as prof

# Get module path
MODULE_PATH = os.path.dirname(__file__)
LoS_FNAME = os.path.join(MODULE_PATH, 'mesxr_mst_los.csv')

# Load the mu coefficients for filter transmission calculations
MU_DICT = pickle.load(open(os.path.join(MODULE_PATH, 'filter_mu.pkl'), 'rb'))

# Common material densities - g/cm^3
DENS = {'Si':2.330,
        'Be':1.848,
        'mylar':1.39}

# Constants for the ME-SXR detector - cm
BE_THICK = 0.0025
SI_THICK = 0.045
MYLAR_THICK = 0.0012 + 0.0050
#MYLAR_THICK = 0.0012 # Old configuration

# Detector geometry constants
NUM_PIX_Y = 195
NUM_PIX_X = 487
PIXEL_DIM = 0.172               # Pixel dimension in mm
DET_DIST = 30.5                 # Distance from detector screen to pinhole, in mm
AREA_PIX = PIXEL_DIM*PIXEL_DIM  # Area of the pixel, in mm^2
AREA_PIN = 4                    # Area of the pinhole opening, in mm^2
mm_2_to_m_2 = 1.0e-6            # Convert mm^2 to m^2, i.e. for etendue

def take_data_mp(pixel):
    """
    This function exists to enable multiporcessing.
    """
    return pixel.get_counts()

# --------------------------------------- Detector Classes --------------------------------------- #

class Detector(object):
    """
    A detector is a collection of pixels exposed and a plasma.
    This model fundamentally assumes the plasma is symmetric in the toroidal
    (second index) direction.
    """
    def __init__(self, pixel_array, etendue=1.012e-11, num_cores=16):
        self.pixel_array = pixel_array
        self.etendue = etendue
        self.cores = num_cores
        
        # Get the los evaluation points
        self.eval_points = []
        for pix in self.pixel_array:
            for ell in pix.ell_array:
                self.eval_points.append(pix.los.get_xy(ell))
        
    def look_at_plasma(self, plasma_obj):
         # Preload emission at the evaluation points
        self.plasma_in_view = plasma_obj
        self.plasma_in_view.preload_points(self.eval_points)
        
        for x_index in range(self.pixel_array.shape[0]):
            self.pixel_array[x_index].look_at_plasma(self.plasma_in_view)

        # Set up the relative line emission profiles
        if self.plasma_in_view.include_excit:
            coords = plasma.sunflower_points(180)
            self.plasma_in_view.preload_points(coords)

            # Break the calculation into two parts since line_emission seems to crash with 100+ points
            coords1 = coords[:90]
            coords2 = coords[90:]
            amp_set1, en_set, mz_set = self.plasma_in_view.line_emission(coords1)
            amp_set2, en_set, mz_set = self.plasma_in_view.line_emission(coords2)

            amp_set = []
            for index in range(len(amp_set1)):
                amp_set.append(np.vstack([amp_set1[index], amp_set2[index]]))

            # Build a profile for each response function
            Ec_map = [pix.response.scurve.E_c for pix in self.pixel_array]
            thresholds = np.array([x for x in np.unique(Ec_map) if x !=0])
            self.observed_line_profiles = {}
            
            for Ec in thresholds:
                index = np.where(Ec_map == Ec)[0][0]
                self.observed_line_profiles[Ec] = prof.Observed_Lines(coords, amp_set, en_set, self.pixel_array[index].response)

            # Preload the eval points
            for Ec in thresholds:
                self.observed_line_profiles[Ec].build_lookup_table(self.eval_points)

            # Add the profiles to the pixels
            for x_index in range(self.pixel_array.shape[0]):
                Ec = Ec_map[x_index]
                if Ec != 0:
                    self.pixel_array[x_index].include_lines(self.observed_line_profiles[Ec])
        
    def change_pixel(self, x_index, y_index, new_pixel):
        new_pixel.look_at_plasma(self.plasma_in_view)
        self.pixel_array[x_index, y_index] = new_pixel
        
    def take_data(self, exp_time):
        """
        Given a plasma take data along all ines of sight - now with multiprocessing.
        The input exp_time is the exposure time in ms. If only one core is selected,
        the multiprocessing library will not be used at all.
        """
        if self.cores == 1:
            measurements = [pix.get_counts() for pix in self.pixel_array]
        else:
            pool = mp.Pool(self.cores)
            measurements = pool.map(take_data_mp, self.pixel_array)
            pool.close()

        return np.vstack([measurements]*NUM_PIX_Y).T*self.etendue*exp_time*get_boundary_mask()


class Pilatus3_Detector(Detector):
    """
    Build a default PILATUS 3 detector based on spatial calibration results. For now
    we will consider only a single row of pixels.
    The "include" array allows the user to specify which pixels are active for a given
    computation. All inactive pixels will measure zero and require no computational
    resources.
    """
    def __init__(self, Ec_map, Ew_map, cs_slope=np.zeros(10), cs_en=np.linspace(0, 30000, num=10), num_cores=16, mlyar_thick=MYLAR_THICK):
        self.Ec_map = Ec_map
        self.Ew_map = Ew_map
        
        # Continers to hold various pixel properties
        self.los_set = np.empty(NUM_PIX_X, dtype=object)
        self.response_set = np.empty(NUM_PIX_X, dtype=object)
        pixel_array = np.empty(NUM_PIX_X, dtype=object)
        etendue = np.zeros([NUM_PIX_X, NUM_PIX_Y])
        
        # Build the filter responses, which are the same for all pixels
        self.impact_params = self.get_impact_params()
        for x_index in xrange(NUM_PIX_X):
            # Only need to calculate the plasma emission in 1D
            self.los_set[x_index] = geometry.line_of_sight(*self.impact_params[x_index])
            self.response_set[x_index] = Pilatus_Response(self.Ec_map[x_index], self.Ew_map[x_index],
                                                           Si_thickness=SI_THICK/np.cos(self.theta_i(x_index, NUM_PIX_Y/2.)),
                                                           Be_thickness=BE_THICK, mylar_thickness=mlyar_thick, cs_slope=cs_slope, cs_en=cs_en)
            pixel_array[x_index] = Pixel(self.los_set[x_index], self.response_set[x_index])
            
            for y_index in xrange(NUM_PIX_Y):
                # Must evaluate the etendue factors in 2D
                etendue[x_index, y_index] = self.get_etendue(x_index, y_index)
                
        super(Pilatus3_Detector, self).__init__(pixel_array, etendue=etendue, num_cores=num_cores)
        
    def get_impact_params(self):
        """
        Get the impact parameters p and zeta from 
        """
        impact_params = np.loadtxt(LoS_FNAME, delimiter=',')
        impact_phi = impact_params[0, :]
        impact_p = impact_params[1, :]
        return zip(impact_p, impact_phi)
    
    def theta_i(self, x_index, y_index):
        """
        Returns the incidence angle used to calculate the etendue of each pixel.
        """
        return np.arctan2(PIXEL_DIM*np.sqrt((x_index + 0.5 - NUM_PIX_X/2.)**2 + (y_index + 0.5 - NUM_PIX_Y/2.)**2), DET_DIST)
        
    def get_etendue(self, x_index, y_index):
        """
        Return the appropriate etendue for the specified pixel.
        """
        return AREA_PIX*AREA_PIN*mm_2_to_m_2/(4*np.pi*DET_DIST**2)*np.cos(self.theta_i(x_index, y_index))**4


class Pixel(object):
    """
    A pixel is defined by a single line-of-sight and a detector response function. Given a plasma
    it will produce a synthetic measurement.
    """
    def __init__(self, los, response, en_lower=1000, en_upper=15000, delta_en=100, delta_ell = 0.05):
        self.los = los
        self.response = response
        self.plasma_in_view = None
        
        self.en_lower = en_lower
        self.en_upper = en_upper
        self.delta_en = delta_en
        
        # Generate the ell array - points along the line of sight to integrate over
        self.delta_ell = delta_ell
        self.ell_max = self.los.intercept_with_circle(0.52)
        self.ell_array = np.arange(-self.ell_max, self.ell_max, self.delta_ell)
        
    def look_at_plasma(self, plasma_obj):
        """
        This designates the Plasma object that the Pixel is looking at. It then creates the observed profile
        of the plasma emissivity sepctrum convolved with the response function of this pixel.
        """
        self.plasma_in_view = plasma_obj
        self.observed_emiss = prof.Observed_Emissivity(self.plasma_in_view.continuum, self.response)

        # Include a containers so that emission can be stored later
        self.emiss_cont = np.zeros(len(self.ell_array))
        self.emiss_lines = np.zeros(len(self.ell_array))

    def include_lines(self, observed_prof):
        """
        Pulls out line emissions at the evaluation points and stores them for later. This is a more lightweight
        approach than trying to save the emission profile to every pixel object, especially when multiprocessing
        is considered.
        """
        self.emiss_lines = np.zeros(len(self.ell_array))

        for index, ell in enumerate(self.ell_array):
            self.emiss_lines[index] = observed_prof(*self.los.get_xy(ell))
        
    def get_counts(self):
        if self.plasma_in_view == None:
            print('No plasma is currently in view.')
        
        else:
            for index, ell in enumerate(self.ell_array):
                self.emiss_cont[index] = self.observed_emiss.integrate(*self.los.get_xy(ell), en_lower=self.en_lower,
                                                                       en_upper=self.en_upper, delta_en=self.delta_en)

        return np.trapz(self.emiss_cont + self.emiss_lines, x=self.ell_array)

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
    This function allows the implementation of charge sharing in the detector response. This is achieved by defining
    the slope k_cs at some energies and then interpolating between those points. The effect of charge sharing vanishes
    entirely when the slope is set to zero everywhere within the energy range.
    """
    def __init__(self, slope_data, slope_energy, E_c, en_lims=[0,30000], units='eV'):
        self.data = slope_data
        self.energy = slope_energy
        self.E_c = E_c
        super(Charge_Sharing, self).__init__('Charge-sharing', en_lims=en_lims, units=units)

    def value(self, en):
        return 1 + np.interp(en, self.energy, self.data)*(en - self.E_c)

class Pilatus_Response(Response):
    """
    The class defines the total response for the Pilatus 3 detector. It is simply a wrapper to define
    the whole response in a single line. This is convenient because the filter and Si layer properties
    are generally not mutable.
    """
    def __init__(self, E_c, E_w, Si_thickness=SI_THICK, Be_thickness=BE_THICK, mylar_thickness=MYLAR_THICK, en_lims=[0,30000],
                 cs_slope=np.zeros(10), cs_en=np.linspace(0, 30000, num=10)):
        self.Si = Absorber('Si', Si_thickness, en_lims=en_lims, units='eV')
        self.Be = Filter('Be', Be_thickness, en_lims=en_lims, units='eV')
        self.mylar = Filter('mylar', mylar_thickness, en_lims=en_lims, units='eV')
        self.scurve = S_Curve(E_c, E_w, en_lims=en_lims, units='eV')
        self.charge_share = Charge_Sharing(cs_slope, cs_en, E_c, en_lims=en_lims)
        self.total = self.Si * self.Be * self.scurve * self.charge_share * self.mylar
        super(Pilatus_Response, self).__init__('Total Response', en_lims=en_lims, units='eV')
        
    def value(self, en):
        return self.total(en)

# -------------------------------------- Analysis ------------------------------------- #

def prfoile_by_threshold(measurements, Ec_map, center_only=True):
    """
    This function sorts data by threshold according to the supplied threshold map. This will
    work for real data, but is included here for use with synthetic data.
    """
    data = {Ec:[] for Ec in np.unique(Ec_map)}
    x_indices = {Ec:[] for Ec in data.keys()}
    
    data_1d = np.sum(measurements, axis=1)
    for x_index, Ec in enumerate(Ec_map):
        if Ec != 0:
            data[Ec].append(data_1d[x_index])
            x_indices[Ec].append(x_index)
        
    for Ec in data.keys():
        data[Ec] = np.array(data[Ec])
        x_indices[Ec] = np.array(x_indices[Ec])
    
    return data, x_indices

def get_boundary_mask():
    """
    Returns a boundary mask which simulates the regions between pixels with no response.
    """
    mask = np.ones([NUM_PIX_X, NUM_PIX_Y])
    for x in range(NUM_PIX_X):
        for y in range(NUM_PIX_Y):
            if cfg.get_chip_coords(x,y)[0] == -1:
                mask[x,y] = 0

    return mask