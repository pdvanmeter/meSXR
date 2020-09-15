"""
This module enables the user to load in MSTfit and V3Fit reconstruction for use
in IDA analyses. Remember to build the lookup table.
"""
import os
import numpy as np
import scipy as sp
import pandas as pd
from mst_ida.models.base.profile_base import Profile_2D_Spline
import scipy.interpolate

# Geometric constants
R0 = 1.5
a = 0.52

# ------------------------------------------------ MSTfit ------------------------------------------------
mstfit_dir = '/home/pdvanmeter/datasets/equilibria'

def load_MSTfit_eq(shot, frame):
    """
    Load a reconstruction from my home directory.
    """
    fname = os.path.join(mstfit_dir, 's{0:10d}/MST_s{0:10d}_f{1:02d}.csv'.format(shot, frame))
    df = pd.read_csv(fname)

    eq = {}
    eq['x'] = df['R (m)'].to_numpy() - R0
    eq['y'] = df['Z (m)'].to_numpy()
    eq['psi'] = df['psi_norm'].to_numpy()
    eq['rho'] = df['rho (m)'].to_numpy()
    eq['theta'] = df['theta_surf (rad)'].to_numpy()
    eq['shift'] = df['Rsurf_shaf (m)'].to_numpy() - R0
    eq['ne'] = df['eq ne (10^19m-3)'].to_numpy()
    eq['q'] = df['q'].to_numpy()
    
    return eq

class MSTfit_Flux(object):
    """
    A simple object to simplify the analysis of plasma equilibria on MST. Can call directly with a shot and a frame.
    The keyword argument norm is the normalization constant for rho.
    """
    def __init__(self, eq=None, shot=None, frame=10, norm=0.52):
        self.norm = norm
        
        if eq is not None:
            self.eq = eq
        elif shot is not None:
            self.eq = load_MSTfit_eq(shot, frame)
        else:
            raise Exception('Equilibrium not supplied.')
        
        # Set up the profile objects
        sample_coords = list(zip(self.eq['x'], self.eq['y']))
        self.psi = Profile_2D_Spline(r'$\hat{\psi}$', sample_coords, self.eq['psi'], units='norm.', dim_units=['m', 'm'], method='linear', boundary=1)
        self.rho = Profile_2D_Spline(r'$\rho$', sample_coords, self.eq['rho']/self.norm, units='norm.', dim_units=['m', 'm'], method='linear', boundary=0.52/self.norm)
        self.theta = Profile_2D_Spline(r'$\theta_{\rho}$', sample_coords, self.eq['theta'], units='rad.', dim_units=['m', 'm'], method='linear', boundary=0)
        self.shift = Profile_2D_Spline(r'$\Delta r$', sample_coords, self.eq['shift'], units='m', dim_units=['m', 'm'], method='linear', boundary=0)
        self.dens = Profile_2D_Spline(r'$n_e$', sample_coords, self.eq['ne']*1e19, units=r'm$^{-3}$', dim_units=['m', 'm'], method='linear', boundary=0)
        
        # Mapping between various profiles and psi
        self.rho_1d = sp.interpolate.interp1d(self.eq['psi'], self.eq['rho'], bounds_error=False, fill_value=(0, self.norm))
        self.shift_1d = sp.interpolate.interp1d(self.eq['psi'], self.eq['shift'], bounds_error=False, fill_value=(np.amax(self.eq['shift']), np.amin(self.eq['shift'])))
        self.q_1d = sp.interpolate.interp1d(self.eq['psi'], self.eq['q'], bounds_error=False, fill_value=(np.amax(self.eq['q']), np.amin(self.eq['q'])))
        
        # Mapping between various profiles and rho
        self.dens_rho = sp.interpolate.interp1d(self.eq['rho']/self.norm, self.eq['ne']*1e19, bounds_error=False,
            fill_value=(np.amax(self.eq['ne']*1e19), np.amin(self.eq['ne']*1e19)))
        self.q_rho = sp.interpolate.interp1d(self.eq['rho']/self.norm, self.eq['q'], bounds_error=False,
            fill_value=(np.amax(self.eq['q']), np.amin(self.eq['q'])))
        
    def __call__(self, xs, ys):
        return (self.rho(xs,ys), self.theta(xs,ys))

# ----------------------------------------------- V3Fit/VMEC -----------------------------------------------
import boguski.QSH_analysis as qsh
import boguski.vmec_analysis as vmec
import mst_ida.data.mst_ops as ops
from mst_ida.models.base.geometry import sunflower_points
from mst_ida.utilities.structures import AttrDict

default_diagnostics = {'sxt':90.0, 'ts':222.0, 'fir_m':250.0, 'fir_p':255.0, 'mesxr':110.0}

def get_woufile(shot, time_ms, fit_name='fit1', v3fit_dir='/home/v3fit/fits'):
    """
    Use this to load the woutfile (V3Fit reconstruction output) for a particular shot and time point.
    """
    fname = os.path.join(v3fit_dir, str(shot), '{0:.1f}'.format(time_ms), fit_name, 'wout_{0:}.nc'.format(fit_name))
    if not os.path.isfile(fname):
        raise FileNotFoundError('Could not find woutfile for requested shot.')
    else:
        return fname

class QSH_Map(object):
    """
    This class handles the mapping of a V3Fit equilibrium between multiple diagnostics on MST.
    """
    def __init__(self, shot, time_ms, delta_t=1.0, diagnostics=None, load_shot=True, woutfile='/home/boguski/IDSII_local/wout_inb.nc'):
        # Load the VMEC reconstruction file
        self.woutfile = woutfile
        grid = vmec.get_VMEC_grid(woutfile=self.woutfile)
        self.R0 = 1.5 #grid['R0']
        
        # Set up the VMEC grid
        self.R_flux = grid['R_flux']
        self.Z_flux = grid['Z_flux']
        self.v_arr = grid['v_arr']
        self.u_arr = grid['u_arr'][0,:,0]
        self.s_arr = grid['s_arr']
        
        # Get the magnetic phase
        mags = ops.get_magnetics(shot, delta_t=delta_t)
        tn = np.argmin(np.abs(mags['Time'] - time_ms))
        self.delta_n = mags['BP']['N05']['Phase'][tn]
        
        # Check for default diagnostics
        self.diagnostics = {}
        if diagnostics is None:
            diagnostics = default_diagnostics
        
        # Add flux surface profiles
        self.flux = AttrDict({})
        for name in diagnostics:
            self.add_diagnostic(name, diagnostics[name])
        
    def add_diagnostic(self, name, phi):
        """
        """
        # Choose v to be in the permitted range [-pi/5, +pi/5]
        theta_op = np.mod(np.deg2rad(5*phi + 241.0) - self.delta_n, 2*np.pi)
        if theta_op > np.pi:
            theta_op -= 2*np.pi
        v0 = theta_op/5
        
        xs, ys = sunflower_points(1000)
        rs = xs + self.R0
        ss, us = self.get_vmec_coords2(rs, ys, v=v0)
        self.flux[name] = VMEC_Flux(xs, ys, ss, us)
        self.diagnostics[name] = phi
        
    def get_flux(self, phi=None, v=None):
        """
        Returns a flux object for an arbitrary angle phi. May instead specify the flux orientation v.
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
            
        xs, ys = sunflower_points(1000)
        rs = xs + self.R0
        ss, us = self.get_vmec_coords2(rs, ys, v=v)
        return VMEC_Flux(xs, ys, ss, us)
    
    def get_vmec_coords(self, Rs, Zs, v=0.0):
        """
        Older version which uses the VMEC grid directly.
        """
        # Get the toroidal index
        i_v = np.argmin(np.abs(self.v_arr - v))

        # Get the points
        s_out = np.zeros(Rs.size, dtype=np.float64)
        u_out = np.zeros(Rs.size, dtype=np.float64)

        for index, (R,Z) in enumerate(zip(Rs,Zs)):
            RZ_flux = (self.R_flux[:,:,i_v] - R)**2 + (self.Z_flux[:,:,i_v] - Z)**2
            try:
                i_s, i_u = np.where(RZ_flux == RZ_flux.min())
                s_out[index] = self.s_arr[i_s]
                u_out[index] = self.u_arr[i_u]
            except:
                ii, jj = np.where(RZ_flux == RZ_flux.min())
                i_s = ii[0]
                i_u = jj[0]
                s_out[index] = self.s_arr[i_s]
                u_out[index] = self.u_arr[i_u]

        return s_out, u_out
    
    def get_vmec_coords2(self, Rs, Zs, v=0.0):
        """
        Modified version which uses interpolation to improve accuracy in s. Interpolates the s-grid
        to improve smoothness of the reconstructed flux surfaces.
        """
        # Get the toroidal index
        i_v = np.argmin(np.abs(self.v_arr - v))
        
        # Interpolate the points in s for higher resolution
        R_su = sp.interpolate.RegularGridInterpolator((self.s_arr, self.u_arr), self.R_flux[:,:,i_v],
                                                      bounds_error=False, fill_value=None)
        Z_su = sp.interpolate.RegularGridInterpolator((self.s_arr, self.u_arr), self.Z_flux[:,:,i_v],
                                                      bounds_error=False, fill_value=None)
        
        s_hd = np.linspace(self.s_arr[0], self.s_arr[-1], num=1000)
        ss, uu = np.meshgrid(s_hd, self.u_arr)
        pts = list( zip(ss.ravel(), uu.ravel()) )
        R_hd = np.reshape(R_su(pts), ss.shape).T
        Z_hd = np.reshape(Z_su(pts), ss.shape).T

        # Get the points
        s_out = np.zeros(Rs.size, dtype=np.float64)
        u_out = np.zeros(Rs.size, dtype=np.float64)

        for index, (R,Z) in enumerate(zip(Rs,Zs)):
            RZ_flux = (R_hd - R)**2 + (Z_hd - Z)**2
            try:
                i_s, i_u = np.where(RZ_flux == RZ_flux.min())
                s_out[index] = s_hd[i_s]
                u_out[index] = self.u_arr[i_u]
            except:
                ii, jj = np.where(RZ_flux == RZ_flux.min())
                i_s = ii[0]
                i_u = jj[0]
                s_out[index] = s_hd[i_s]
                u_out[index] = self.u_arr[i_u]

        return s_out, u_out

class VMEC_Flux(object):
    """
    Maps the radial flux label s and poloidal-like angle u to phyiscal locations in the machine.
    Note that this function also implements the flux.rho() functionality in order to retain compatibility
    with existing code.
    TODO: support for reconstructed density profile.
    """
    def __init__(self, xs, ys, ss, us):
        self.xs = xs
        self.ys = ys
        self.ss = ss
        self.us = us
       
        sample_coords = list(zip(self.xs, self.ys))
        self.s = Profile_2D_Spline(r'$s$', sample_coords, self.ss, units='norm.', dim_units=['m', 'm'],
                                   method='linear', boundary=1.0)
        self.u = Profile_2D_Spline(r'$u$', sample_coords, self.us, units='norm.', dim_units=['m', 'm'],
                                   method='linear', boundary=np.nan)
        
    def __call__(self, x, y):
        return self.s(x,y), self.u(x,y)
    
    def rho(self, x, y):
        """
        A wrapper to enable backwards compatibility with existing mst_ida code.
        """
        return self.s(x, y)
        
    def zeta(self, x, y):
        """
        A wrapper to enable backwards compatibility with existing mst_ida code.
        """
        return self.u(x, y)