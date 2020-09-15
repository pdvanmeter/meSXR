"""
"""
import os
import pickle
import numpy as np
import scipy as sp
import h5py
import mst_ida.models.base.geometry as geo
import mst_ida.data.mesxr as data
from mst_ida.utilities.functions import stringify_float

# Get module path
MODULE_PATH = os.path.dirname(__file__)
LoS_FNAME = os.path.join(MODULE_PATH, 'mesxr_mst_los.csv')

class MESXR(object):
    """
    This class provides a lightweight framework to tie everything else together. It keeps track of the detector properties
    and settings as well as the appropriate emiss and avg_Z databases.

    This object can be created either by passing an Ec_map (a full 2D map of the thresold in keV for each pixel) or by providing
    an associated shot number. If neither is provided the code will throw an error.
    """
    def __init__(self, Ec_map=None, shot=None, A_pin=2.0, mode='lowE', pfm=True, center=False):
        # Load in the thresholds
        if Ec_map is not None:
            self.Ec_map = Ec_map
        elif shot is not None:
            self.Ec_map = data.load_raw_data(shot)['thresholds']
        else:
            raise RuntimeError('User must supply Ec_map or shot.')
        
        ps, phis, thresholds = get_geometry(self.Ec_map)
        self.mode = mode
        self.thresholds = thresholds

        # Define the geometry
        self.p = {Ec:np.sign(np.sin(phis[Ec]))*ps[Ec] for Ec in self.thresholds}
        self.los = {Ec:define_LoS(ps[Ec], phis[Ec]) for Ec in self.thresholds}

        # Load the appropriate databases
        if pfm:
            self.emiss = get_emiss_pfm(thresholds=self.thresholds, mode=self.mode)
        else:
            if mode == 'lowE':
                fname='sxr_emission_100um_corrected.h5'
            elif mode == 'midE':
                fname='sxr_emission_100um_midE.h5'
            else:
                raise KeyError('Supplied mode is not currently available')
            self.emiss = get_emiss_adas(self.thresholds, fname=fname)

        # Get the etendue
        self.A_pin = A_pin
        #self.etendue = get_etendue_sum(self.Ec_map, self.A_pin)
        self.etendue = get_etendue_sum(self.Ec_map, A_pin=self.A_pin, center=center)

    def take_data(self, plasma, num_pts=100, delta_t=0.001):
        """
        """
        return get_counts(self.los, self.thresholds, plasma, self.emiss, self.etendue, num_pts=num_pts, delta_t=delta_t)

def get_counts(los_set, thresholds, plasma, emiss, etendue, num_pts=100, delta_t=0.001):
    """
    The new version
    """
    counts = {Ec:0.0 for Ec in thresholds}
    
    for Ec in thresholds:
        pix_los = los_set[Ec]
        num_pixels = len(pix_los)
        ell_pts = np.linspace(-0.5, 0.5, num=num_pts)

        xs = np.zeros([num_pixels, num_pts])
        ys = np.zeros([num_pixels, num_pts])
        for index,los in enumerate(pix_los):
            xs[index,:], ys[index,:] = list(zip(*[los.get_xy(ell) for ell in ell_pts]))

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
        emiss_xs = ne_xs*nZ_xs['D']*np.reshape(emiss['D'][Ec](pts), xs.shape)
        for ion in plasma.impurities:
            emiss_xs += ne_xs*nZ_xs[ion]*np.reshape(emiss[ion][Ec](pts), xs.shape)
            
        # Integrate with the trapezoidal rule
        dl = np.ones([num_pts,1])*(ell_pts[1] - ell_pts[0])
        dl[0] *= 0.5
        dl[-1] *= 0.5
        counts[Ec] = np.squeeze(np.dot(emiss_xs, dl))*etendue[Ec]*delta_t
    
    return counts

# -------------------------------------- Simplified model --------------------------------------
class MESXR_Avg(object):
    """
    This class provides a lightweight framework to tie everything else together. It keeps track of the detector
    properties and settings as well as the appropriate emiss and avg_Z databases.

    This object can be created either by passing an Ec_map (a full 2D map of the thresold in keV for each pixel) or by providing
    an associated shot number. If neither is provided the code will throw an error.

    This 'simplified' version of the model uses the average lines of sight for each set of thresholds instead of
    the exact values. This version requires fewer loops and is therefore a little faster. It is also useful for data
    which has been interpolated, as in the ratio analysis.
    """
    def __init__(self, Ec_map=None, shot=None, A_pin=2.0, mode='lowE', pfm=True, center=False):
        # Load in the thresholds
        if Ec_map is not None:
            self.Ec_map = Ec_map
        elif shot is not None:
            self.Ec_map = data.load_raw_data(shot)['thresholds']
        else:
            raise RuntimeError('User must supply Ec_map or shot.')
            
        ps, phis, thresholds = get_geometry(self.Ec_map, average_pix=True)
        self.mode = mode
        self.thresholds = thresholds

        # Define the geometry
        self.p = np.sign(np.sin(phis))*ps
        self.los = define_LoS(ps, phis)

        # Load the appropriate databases
        if pfm:
            self.emiss = get_emiss_pfm(thresholds=self.thresholds, mode=self.mode)
        else:
            if mode == 'lowE':
                fname='sxr_emission_100um_corrected.h5'
            elif mode == 'midE':
                fname='sxr_emission_100um_midE.h5'
            else:
                raise KeyError('Supplied mode is not currently available')
            self.emiss = get_emiss_adas(self.thresholds, fname=fname)

        # Get the etendue
        self.A_pin = A_pin
        self.etendue = get_etendue_sum(self.Ec_map, self.A_pin, center=center)
        #self.etendue = get_etendue_2d_threshold(self.Ec_map, A_pin=self.A_pin)

    def take_data(self, plasma, num_pts=100, delta_t=0.001):
        """
        """
        return get_counts_avg(self.los, self.thresholds, plasma, self.emiss, self.etendue,
            num_pts=num_pts, delta_t=delta_t)

def get_counts_avg(avg_los, thresholds, plasma, emiss, etendue, num_pts=100, delta_t=0.001):
    """
    Use the same set of lines-of-sight for each threshold.
    """
    counts = {Ec:0.0 for Ec in thresholds}
    num_pixels = len(avg_los)
    ell_pts = np.linspace(-0.5, 0.5, num=num_pts)

    xs = np.zeros([num_pixels, num_pts])
    ys = np.zeros([num_pixels, num_pts])
    for index,los in enumerate(avg_los):
        xs[index,:], ys[index,:] = list(zip(*[los.get_xy(ell) for ell in ell_pts]))

    # Evaluate the profiles
    Te_xs = np.maximum(plasma.Te(xs, ys), 10.0)
    ne_xs = np.maximum(plasma.ne(xs, ys), 1e15)
    n0_xs = plasma.n0(xs, ys)
    pts = list( zip( Te_xs.ravel(), ne_xs.ravel()/1e19, n0_xs.ravel()/1e14 ) )
    
    # Evaluate deuterium using quasi-netrality
    nZ_xs = {ion:plasma.nZ[ion](xs,ys) for ion in plasma.impurities}
    nZ_xs['D'] = plasma.nD(xs, ys)

    for Ec in thresholds:
       # Calculate the emission array
        emiss_xs = np.zeros(xs.shape)
        emiss_xs = ne_xs*nZ_xs['D']*np.reshape(emiss['D'][Ec](pts), xs.shape)
        for ion in plasma.impurities:
            emiss_xs += ne_xs*nZ_xs[ion]*np.reshape(emiss[ion][Ec](pts), xs.shape)
            
        # Integrate with the trapezoidal rule
        dl = np.ones([num_pts,1])*(ell_pts[1] - ell_pts[0])
        dl[0] *= 0.5
        dl[-1] *= 0.5
        counts[Ec] = np.squeeze(np.dot(emiss_xs, dl))*etendue[Ec]*delta_t
    
    return counts

# -------------------------------------- Detector geometry --------------------------------------

# Detector geometry constants
NUM_PIX_Y = 195
NUM_PIX_X = 487
PIXEL_DIM = 0.172               # Pixel dimension in mm
DET_DIST = 30.5                 # Distance from detector screen to pinhole, in mm
AREA_PIX = PIXEL_DIM*PIXEL_DIM  # Area of the pixel, in mm^2
AREA_PIN = 2.0                  # Area of the pinhole opening, in mm^2 (used to be 4.0)
mm_2_to_m_2 = 1.0e-6            # Convert mm^2 to m^2, i.e. for etendue

# The default central etendue
eta0 = AREA_PIX*AREA_PIN*mm_2_to_m_2/(4*np.pi*DET_DIST**2)

def etendue_2d(x_index, A_pin=AREA_PIN):
    """
    Get the etendue for a stripe across the face of the detector - analytic method.
    """
    eta0 = AREA_PIX*A_pin*mm_2_to_m_2/(4*np.pi*DET_DIST**2)
    d = DET_DIST
    L = NUM_PIX_Y*PIXEL_DIM
    xi = (x_index + 0.5 - 0.5*NUM_PIX_X)*PIXEL_DIM
    a = np.sqrt(xi**2 + d**2)
    
    f1 = NUM_PIX_Y*(d**3) / (2*a**3)
    f2 = (a*d) / (a**2 + L**2/4)
    f3 = (2*d / L)*np.arctan2(L,2*a)
    
    return eta0*f1*(f2 + f3)

def get_etendue_2d_threshold(Ec_map, div=8, A_pin=AREA_PIN):
    """
    Organize the 2D etendue profile by threshold. This should work for
    both 8-color and 4-color configurations when div=8
    """
    x_indices = np.arange(487)
    etas = etendue_2d(x_indices, A_pin=A_pin)
    
    # Sort the new calculation by threshold given an Ec_map
    pixel_gaps = np.arange(60, 487, 61)
    thresholds = np.sort([Ec for Ec in np.unique(Ec_map) if Ec != 0])
    
    etendue_Ec = {Ec:np.zeros(60) for Ec in thresholds}
    index = 0

    for x in range(487):
        if x not in pixel_gaps:
            etendue_Ec[Ec_map[x, 0]][index//div] += etas[x]
            index += 1

    return etendue_Ec

def theta_i(x,y):
    return np.arctan2(PIXEL_DIM*np.sqrt((x + 0.5 - NUM_PIX_X/2.)**2 + (y + 0.5 - NUM_PIX_Y/2.)**2), DET_DIST)
    
def get_etendue(x, y, A_pin, dist=DET_DIST):
    """
    Etendue for a single pixel.
    """
    eta = AREA_PIX*A_pin*mm_2_to_m_2/(4*np.pi*dist**2)
    return eta*np.cos(theta_i(x, y))**4

def get_etendue_sum(Ec_map, A_pin=AREA_PIN, div=8, center=False):
    """
    Old method, calculate each pixel and sum.
    """
    if center:
        y_pix = np.array([y for y in range(65,130) if y != 97])
    else:
        y_pix = np.array([y for y in range(0,195) if y != 97])
    pixel_gaps = np.arange(60, 487, 61)
    
    thresholds = np.sort([Ec for Ec in np.unique(Ec_map) if Ec != 0])
    etendue_sum = {Ec:np.zeros(60) for Ec in thresholds}
    index = 0
    
    for x in range(487):
        if x not in pixel_gaps:
            etendue_sum[Ec_map[x, 0]][index//div] += np.sum([get_etendue(x, y, A_pin) for y in y_pix])
            index += 1
            
    return etendue_sum

def edge_factor(theta, a=0.52, R0=1.5):
    theta_crit = np.arccos(2*np.sqrt(a*R0)/(R0+a))
    theta = np.atleast_1d(theta)
    theta[theta > theta_crit] = theta_crit
    dist = ((R0+a)*np.cos(theta) - np.sqrt(((R0+a)**2)*np.cos(theta)**2 - 4*a*R0))/(2*a)
    return np.squeeze(dist)

def get_edge_factor_grid(**kwargs):
    theta_grid = np.zeros([NUM_PIX_X, NUM_PIX_Y])

    for i in range(NUM_PIX_X):
        for j in range(NUM_PIX_Y):
            theta_grid[i,j] = theta_i(243,j)
    return edge_factor(theta_grid, **kwargs)

def get_geometry(Ec_map, average_pix=False):
    """
    Make this into a more accurate version which treats each threshold independently
    """
    # Reaslistic lines of sight, sorted by threshold "8 detector" model
    impact_p, impact_phi = get_MST_LoS()

    # Sort p and phi into clusters of 8 pixels, skipping over the gaps
    pixel_gaps = np.arange(60, 487, 61)
    thresholds = np.sort([Ec for Ec in np.unique(Ec_map) if Ec !=0])

    xpix_p = np.zeros([60, len(thresholds)])
    xpix_phi = np.zeros([60, len(thresholds)])
    
    ps = {Ec:np.zeros(60) for Ec in thresholds}
    phis = {Ec:np.zeros(60) for Ec in thresholds}

    index = 0
    for x in range(487):
        if x not in pixel_gaps:
            Ec = Ec_map[x, 0]
            ps[Ec][index//8] = impact_p[x]
            phis[Ec][index//8] = impact_phi[x]
            
            Ec_index = np.where(thresholds == Ec)[0][0]
            xpix_p[index//8, Ec_index] = impact_p[x]
            xpix_phi[index//8, Ec_index] = impact_phi[x]
            index += 1
            
    if average_pix:
        # Compute the average p and phi for each cluster
        p_avg = np.average(xpix_p, axis=1)
        phi_avg = np.average(xpix_phi, axis=1)
        return p_avg, phi_avg, thresholds

    else:
        return ps, phis, thresholds

def get_geometry_2(Ec_map, average_pix=False):
    """
    Make this into a more accurate version which treats each threshold independently
    """
    # Reaslistic lines of sight, sorted by threshold "8 detector" model
    impact_p, impact_phi = get_MST_LoS()

    # Sort p and phi into clusters of 8 pixels, skipping over the gaps
    pixel_gaps = np.arange(60, 487, 61)
    thresholds = np.sort([Ec for Ec in np.unique(Ec_map) if Ec !=0])

    xpix_p = np.zeros([60, len(thresholds)])
    xpix_phi = np.zeros([60, len(thresholds)])
    
    ps = {Ec:np.zeros(60) for Ec in thresholds}
    phis = {Ec:np.zeros(60) for Ec in thresholds}

    index = 0
    counts = np.zeros(60)
    for x in range(487):
        if x not in pixel_gaps:
            Ec = Ec_map[x, 0]
            counts[index//8] += 1
            ps[Ec][index//8] = (ps[Ec][index//8] + impact_p[x]) / counts[index//8]
            phis[Ec][index//8] = (phis[Ec][index//8] + impact_phi[x]) / counts[index//8]
            
            Ec_index = np.where(thresholds == Ec)[0][0]
            xpix_p[index//8, Ec_index] = impact_p[x]
            xpix_phi[index//8, Ec_index] = impact_phi[x]
            index += 1
            
    if average_pix:
        # Compute the average p and phi for each cluster
        p_avg = np.average(xpix_p, axis=1)
        phi_avg = np.average(xpix_phi, axis=1)
        return p_avg, phi_avg, thresholds

    else:
        return ps, phis, thresholds

def get_MST_LoS(fname=LoS_FNAME):
    impact_params = np.loadtxt(fname, delimiter=',')
    impact_phi = impact_params[0, :]
    impact_p = impact_params[1, :]
    return impact_p, impact_phi
    
def define_LoS(impact_p, impact_phi):
    return [geo.line_of_sight(*p) for p in zip(impact_p, impact_phi)]

# -------------------------------------- Emissivity databases --------------------------------------
emiss_dir='/home/pdvanmeter/data/emiss_tables/'

def get_emiss_pfm(thresholds=[2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5], ions=['Al','C','O','N','B','D'], mode='lowE'):
    """
    """
    fname = 'mesxr_emiss_{0:}.pkl'.format(mode)
    fpath = os.path.join(emiss_dir, fname)
    with open(fpath, 'rb') as f:
        emiss_db = pickle.load(f, encoding='latin1')
    
    Te_set = emiss_db['axes']['Te']
    ne_set = emiss_db['axes']['ne']
    n0_set = emiss_db['axes']['n0']

    # Build the interpolation functions
    emiss_mesxr = {ion:{} for ion in ions}
    for ion in ions:
        for Ec in thresholds:
            emiss_mesxr[ion][Ec] = sp.interpolate.RegularGridInterpolator((Te_set, ne_set, n0_set), emiss_db[Ec][ion],
                                                                            bounds_error=False, fill_value=0.0)
    return emiss_mesxr

def get_emiss_adas(thresholds, ions=['Al','C','O','N','B','Ar','D'], fname='sxr_emission_100um_corrected.h5'):
    """
    """
    fname = '/home/pdvanmeter/data/ADAS/' + fname
    file = h5py.File(fname, 'r')

    # Load the axis arrays
    Te_set = file['/Information/Dimensions/te'][...]
    ne_set = file['/Information/Dimensions/ne'][...]
    ln_n0_set = file['/Information/Dimensions/ln_n0'][...]
    
    # Scale to match the other convention
    ne_set /= 1e19
    n0_set = np.exp(ln_n0_set) / 1e14

    # Load the emission databases for impurities at the specified thresholds and make the interp functions
    emiss = {ion:{} for ion in ions}

    for ion in ions:
        for Ec in thresholds:
            emiss_db = file['/{0:}/emiss/emiss_{1:}'.format(ion, stringify_float(Ec))][...]*1000.
            emiss[ion][Ec] = sp.interpolate.RegularGridInterpolator((Te_set, ne_set, n0_set), emiss_db,
                                                                    bounds_error=False, fill_value=0.0)

    file.close()
    return emiss