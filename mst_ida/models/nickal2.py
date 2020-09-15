"""
"""
from __future__ import division
import os
import pickle
import numpy as np
import scipy as sp
import scipy.io
import mst_ida.models.base.geometry as geo

# Parameters for the NICKAL2 geometry - these are fixed
theta = 67.5 + 90.
theta *= np.pi/180.
los_n2 = geo.line_of_sight(0.0, theta)

class NICKAL2(object):
    """
    Used to model the NICKAL2 measurements in [W /m^2 /sr]. Based on the emission model created by Lisa Reusch.
    """
    def __init__(self, filters=['AlBe', 'ZrMylar', 'SiBe'], fname='nickal2_sxt_emiss.pkl'):
        self.filters = filters
        
        # Set up the geometry - this is fixed
        theta = 67.5 + 90.
        theta *= np.pi/180.
        self.los = geo.line_of_sight(0.0, theta)
        
        # Get the NICKAL2 emission interpolators
        self.emiss = get_emiss_NICKAL2(filters=self.filters)
        
    def take_data(self, plasma, num_pts=100):
        """
        Call with a plasma object to generate synthetic measurements.
        """
        return get_pwr_NICKAL2(plasma, self.emiss, filters=self.filters, num_pts=num_pts)

def get_pwr_NICKAL2(plasma, emiss, filters=['AlBe', 'ZrMylar', 'SiBe'], num_pts=100):
    """
    """
    # Generate the evaulation points
    ell_pts = np.linspace(-0.5, 0.5, num=num_pts)
    xs = np.zeros(num_pts)
    ys = np.zeros(num_pts)
    xs[:], ys[:] = list(zip(*[los_n2.get_xy(ell) for ell in ell_pts]))

    # Evaluate the profiles
    Te_xs = np.maximum(plasma.Te(xs, ys), 10.0)
    ne_xs = np.maximum(plasma.ne(xs, ys), 1e15)
    n0_xs = plasma.n0(xs, ys)
    pts = list( zip( Te_xs, ne_xs/1e19, n0_xs/1e14 ) )
        
    # Evaluate deuterium using quasi-netrality
    nZ_xs = {ion:plasma.nZ[ion](xs,ys) for ion in plasma.impurities}
    nZ_xs['D'] = plasma.nD(xs, ys)

    # Calculate the emission array
    emiss_xs = {filt:np.zeros(xs.shape) for filt in filters}
    for filt in filters:
        emiss_xs[filt][:] = ne_xs*nZ_xs['D']*emiss['D'][filt](pts)
        for ion in plasma.impurities:
            emiss_xs[filt][:] += ne_xs*nZ_xs[ion]*emiss[ion][filt](pts)

    # Integrate with the trapezoidal rule
    return {filt:np.trapz(emiss_xs[filt], x=ell_pts) for filt in filters}
    
# -------------------------------------- Emissivity databases --------------------------------------
def get_emiss_NICKAL2(filters=['AlBe', 'ZrMylar', 'SiBe'], impurities=['Al', 'C', 'O', 'N', 'B'],
                      fname='nickal2_sxt_emiss.pkl', emiss_dir='/home/pdvanmeter/data/ADAS/'):
    """
    """
    fpath = os.path.join(emiss_dir, fname)
    with open(fpath, 'rb') as f:
        emiss_db = pickle.load(f, encoding='latin1')
    
    Te_set = emiss_db['axes']['Te']
    ne_set = emiss_db['axes']['ne']
    n0_set = emiss_db['axes']['n0']
    
    # Build the interpolation functions
    emiss_n2 = {ion:{} for ion in impurities}
    for ion in impurities:
        for filt in filters:
            emiss_n2[ion][filt] = sp.interpolate.RegularGridInterpolator((Te_set, ne_set, n0_set),
                                                                         np.swapaxes(emiss_db['NICKAL2'][filt][ion], 0, 1),
                                                                         bounds_error=False, fill_value=0.0)

    # And the majority
    emiss_n2['D'] = {}
    for filt in filters:
        emiss_n2['D'][filt] = sp.interpolate.RegularGridInterpolator((Te_set, ne_set, n0_set),
                                                                     np.swapaxes(emiss_db['NICKAL2'][filt]['H'], 0, 1),
                                                                     bounds_error=False, fill_value=0.0)
        
    return emiss_n2