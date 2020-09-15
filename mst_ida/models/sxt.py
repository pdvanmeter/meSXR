"""
Create a model SXT detector for use in integrated data analysis applications.
"""
import os
import pickle
import pandas as pd
import numpy as np
import scipy as sp
import scipy.interpolate

import mst_ida.models.base.geometry as geo

# Get the geometry file
MODULE_PATH = os.path.dirname(__file__)
sxt_geo_path = os.path.join(MODULE_PATH, 'sxt_geometry.csv')
sxt_geo = pd.read_csv(sxt_geo_path, index_col='logical')

# Various filter labels which might be useful
labels_full =  ['A thick', 'B thick', 'C thick', 'D thick', 'A thin', 'B thin', 'C thin', 'D thin']
labels_thick = ['A thick', 'B thick', 'C thick', 'D thick']
labels_thin =  ['A thin',  'B thin',  'C thin',  'D thin']
labels_AB =    ['A thick', 'B thick', 'A thin',  'B thin']
labels_CD =    ['C thick', 'D thick', 'C thin',  'D thin']

class SXT(object):
    """
    Used to model the SXT measurements in [W /m^2 /sr]. Based on the emission model created by Lisa Reusch.
    """
    def __init__(self, filters=None, labels=labels_full, exclude = [20,59,60], prb=False):
        # Set the desired defualt behavior
        if filters is None and not prb:
            filters = {'thick':'172um', 'thin':'45um'}
        elif filters is None and prb:
            filters = {'thick':'be172', 'thin':'be45'}
        
        self.filters = filters
        self.labels = labels
        
        # Set up the geometry - this is fixed
        self.los = {}
        self.p = {label:[] for label in self.labels}
        self.alpha = {label:[] for label in self.labels}
        for label in self.labels:
            probe, filt = label.split()
            df = sxt_geo.query("label == '{0:}' and filter == '{1:}' and logical not in {2:}".format(probe, filt, exclude))
            ps = df['p'].tolist()
            phis = df['phi'].tolist()
            self.los[label] = define_LoS(ps, phis)
            self.p[label] = df['signed_p'].to_numpy()
            self.alpha[label] = df['alpha'].to_numpy()
        
        # Get the SXT emission interpolators
        if prb:
            self.emiss = get_prb_SXT(filters=self.filters)
        else:
            self.emiss = get_emiss_SXT(filters=self.filters)
        
    def take_data(self, plasma, num_pts=100):
        """
        Call with a plasma object to generate synthetic measurements.
        """
        return get_pwr_SXT(self.los, plasma, self.emiss, num_pts=num_pts, labels=self.labels)

def get_pwr_SXT(sxt_los, plasma, emiss, num_pts=100, labels=labels_full):
    """
    """
    pwr_int = {}
    for ll in labels:
        # Get the appropriate database label
        filt = ll.split()[1]
        pix_los = sxt_los[ll]
        
        # Get the spatial points along the line of sight
        num_pixels = len(pix_los)
        ell_pts = np.linspace(-0.5, 0.5, num=num_pts)
        
        xs = np.zeros([num_pixels, num_pts])
        ys = np.zeros([num_pixels, num_pts])   
        for index,los in enumerate(pix_los):
            #xs[index,:], ys[index,:] = list(zip(*[los.get_xy(ell) for ell in ell_pts]))
            xs[index,:], ys[index,:] = los.get_xy(ell_pts)
        
        # Evaluate the profiles
        Te_xs = np.maximum(plasma.Te(xs, ys), 10.0)
        ne_xs = np.maximum(plasma.ne(xs, ys), 1e15)
        n0_xs = plasma.n0(xs, ys)
        pts = list( zip( Te_xs.ravel(), ne_xs.ravel()/1e19, n0_xs.ravel()/1e14 ) )
        
        # Evaluate deuterium using quasi-netrality
        nZ_xs = {ion:plasma.nZ[ion](xs,ys) for ion in plasma.impurities}
        nZ_xs['D'] = plasma.nD(xs, ys)
        
        # Calculate the emission array
        emiss_xs = np.zeros(xs.shape)
        emiss_xs = ne_xs*nZ_xs['D']*np.reshape(emiss['D'][filt](pts), xs.shape)
        for ion in plasma.impurities:
            emiss_xs += ne_xs*nZ_xs[ion]*np.reshape(emiss[ion][filt](pts), xs.shape)
            
        # Integrate with the trapezoidal rule
        dl = np.ones([num_pts,1])*(ell_pts[1] - ell_pts[0])
        dl[0] *= 0.5
        dl[-1] *= 0.5
        pwr_int[ll] = np.squeeze(np.dot(emiss_xs, dl))
    
    return pwr_int

def define_LoS(impact_p, impact_phi):
    return [geo.line_of_sight(*p) for p in zip(impact_p, impact_phi)]

# ------------------------------------------- Alpha Version -------------------------------------------
class SXT_Alpha(SXT):
    """
    """
    def __init__(self, **kwargs):
        super().__init__(prb=True, **kwargs)
        self.emiss = get_prb_alpha_SXT(filters=self.filters)

    def take_data(self, plasma, num_pts=100):
        """
        Call with a plasma object to generate synthetic measurements.
        """
        return get_pwr_SXT_Alpha(self.los, plasma, self.emiss, self.alpha, num_pts=num_pts, labels=self.labels)


def get_pwr_SXT_Alpha(sxt_los, plasma, emiss, alpha_set, num_pts=100, labels=labels_full):
    """
    """
    pwr_int = {ll:np.zeros(alpha_set[ll].shape) for ll in labels}
    for ll in labels:
        # Get the appropriate database label
        filt = ll.split()[1]
        pix_los = sxt_los[ll]

        for index,(los,alpha) in enumerate(zip(pix_los,alpha_set[ll])):
            # Get the spatial points along the line of sight
            num_pixels = len(pix_los)
            ells = np.linspace(-0.5, 0.5, num=num_pts)
            xs, ys = los.get_xy(ells)
            
            # Evaluate the profiles
            Te_xs = np.maximum(plasma.Te(xs, ys), 10.0)
            ne_xs = np.maximum(plasma.ne(xs, ys), 1e15)
            n0_xs = plasma.n0(xs, ys)
            pts = list( zip( Te_xs, ne_xs/1e19, n0_xs/1e14 ) )
            
            # Evaluate deuterium using quasi-netrality
            nZ_xs = {ion:plasma.nZ[ion](xs,ys) for ion in plasma.impurities}
            nZ_xs['D'] = plasma.nD(xs, ys)
            
            # Calculate the emission array
            emiss_xs = np.zeros(xs.shape)
            emiss_xs = ne_xs*nZ_xs['D']*emiss['D'][filt][alpha](pts)
            for ion in plasma.impurities:
                emiss_xs += ne_xs*nZ_xs[ion]*emiss[ion][filt][alpha](pts)
                
            # Integrate with the trapezoidal rule
            pwr_int[ll][index] = np.trapz(emiss_xs, x=ells)
    
    return pwr_int

# -------------------------------------- Emissivity databases --------------------------------------
def get_emiss_SXT(filters={'thick':'172um', 'thin':'45um'}, impurities=['Al', 'C', 'O', 'N', 'B'],
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
    emiss_sxt = {ion:{} for ion in impurities}
    for ion in impurities:
        for filt in filters.keys():
            key = filters[filt]
            emiss_sxt[ion][filt] = sp.interpolate.RegularGridInterpolator((Te_set, ne_set, n0_set), 
                                                                          np.swapaxes(emiss_db['SXT'][key][ion], 0, 1),
                                                                          bounds_error=False, fill_value=0.0)

    # And the majority
    emiss_sxt['D'] = {}
    for filt in filters.keys():
        key = filters[filt]
        emiss_sxt['D'][filt] = sp.interpolate.RegularGridInterpolator((Te_set, ne_set, n0_set),
                                                                      np.swapaxes(emiss_db['SXT'][key]['H'], 0, 1),
                                                                      bounds_error=False, fill_value=0.0)
    return emiss_sxt

def get_prb_SXT(filters={'thick':'be172', 'thin':'be45'}, ions=['Al', 'C', 'O', 'N', 'B', 'D'],
                  fname='sxr_tomo_pfm.pkl', emiss_dir='/home/pdvanmeter/data/emiss_tables/'):
    """
    """
    fpath = os.path.join(emiss_dir, fname)
    with open(fpath, 'rb') as f:
        prb = pickle.load(f)
    
    Te_set = prb['axes']['Te']
    ne_set = prb['axes']['ne']
    n0_set = prb['axes']['n0']
    
    # Build the interpolation functions
    emiss_sxt = {ion:{} for ion in ions}
    for ion in ions:
        for filt in filters.keys():
            key = filters[filt]
            emiss_sxt[ion][filt] = sp.interpolate.RegularGridInterpolator((Te_set, ne_set, n0_set), prb[key][ion],
                                                                          bounds_error=False, fill_value=0.0)
    
    return emiss_sxt

def get_prb_alpha_SXT(filters={'thick':'be172', 'thin':'be45'}, ions=['Al', 'C', 'O', 'N', 'B', 'D'],
                      fname='sxr_tomo_pfm_alpha.pkl', emiss_dir='/home/pdvanmeter/data/emiss_tables/'):
    """
    """
    fpath = os.path.join(emiss_dir, fname)
    with open(fpath, 'rb') as f:
        prb = pickle.load(f)
    
    Te_set = prb['axes']['Te']
    ne_set = prb['axes']['ne']
    n0_set = prb['axes']['n0']
    alphas = prb['axes']['alpha']
    
    # Build the interpolation functions
    emiss_sxt = {ion:{} for ion in ions}
    for ion in ions:
        for filt in filters.keys():
            key = filters[filt]
            emiss_sxt[ion][filt] = {}
            for alpha in alphas:
                emiss_sxt[ion][filt][alpha] = sp.interpolate.RegularGridInterpolator((Te_set, ne_set, n0_set), prb[key][alpha][ion],
                                                                              bounds_error=False, fill_value=0.0)
    
    return emiss_sxt