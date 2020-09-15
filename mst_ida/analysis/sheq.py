"""
Functions for interfacing with the SHEq code.
"""
import numpy as np
import pidly
import scipy.interpolate
from mst_ida.models.base.profile_base import Profile_2D_Spline
from mst_ida.utilities.structures import AttrDict

deg2rad = np.pi/180.
R0 = 1.5

def get_flux_grid(shot, time, phi=222.5*deg2rad):
    """
    Use SHEq to directly access the NCT output to evaluate rho on a grid of (R,Z) points.
    
    Inputs:
        - shot = MST shot number, corresponding to an existing NCT run.
        - time = Time point in ms, corresponding to an existing NCT run.
        - phi = Toroidal angle in radians.
    """
    idl = pidly.IDL()
    idl('.r /home/pdvanmeter/lib/idl/startup.pro')
    idl('.r run_sheq')
    idl('flux = run_sheq({0:}, {1:}, phi0={2:})'.format(shot, time, phi))
    flux_grid = idl.ev('flux')
    idl.close()
    
    flux_grid['rho'] = flux_grid['rho'].T
    return AttrDict(flux_grid)

def get_flux(shot, time, phis=[222.5*deg2rad], labels=['TS']):
    """
    """
    flux_grids = {}
    flux = {}
    for label,phi in zip(labels,phis):
        flux_grids[label] = get_flux_grid(shot, time, phi=phi)
        
    # Build the interpolation functions
    for label in labels:
        sample_coords = list(zip(flux_grids[label].r.ravel()-R0, flux_grids[label].z.ravel()))
        flux[label] = Profile_2D_Spline(r'$\rho$', sample_coords, flux_grids[label].rho.T.ravel(), units='norm.', dim_units=['m', 'm'],
                                   method='linear', boundary=1.0)
        
    return flux

def sheq_averages(shot, time, return_all=False):
    """
    Returns flux-surface averaged quantities of physical significance.
    
    Inputs:
        - shot = MST shot number, corresponding to an existing NCT run.
        - time = Time point in ms, corresponding to an existing NCT run.
        - return_all = Set to True to instead return the full sh_averages.pro structure.
    """
    idl = pidly.IDL()
    idl('.r /home/pdvanmeter/lib/idl/startup.pro')
    idl('.r run_sheq')
    idl('averages = sheq_flux_avg({0:}, {1:})'.format(shot, time))
    avg = AttrDict(idl.ev('averages'))
    idl.close()
    
    if return_all:
        return avg
    else:
        return AttrDict({
            'rhop':avg.rhop,
            'rhoh':avg.rhoh,
            'J_pol':avg.jpol/1e6,
            'J_tor':avg.jtor/1e6,
            'B_pol':avg.bpol,
            'B_tor':avg.btor,
            'B':avg.b,
            'B2':avg.b2,
            'JB':-avg.jb/1e6,
            'J_para':-avg.jb/avg.b/1e6,
            'mua':-avg.mua,
            'q':-avg.q,
            'delta':float(avg.delta_axis)
        })