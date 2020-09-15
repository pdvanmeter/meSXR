"""
This module collects several standard analysis routines.
"""
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

import mst_ida.models.mesxr3 as m3
import mst_ida.models.base.flux_maps as fm
import mst_ida.data.mesxr as mesxr
from mst_ida.analysis.samplers import Quad_Sampler
from mst_ida.models.base.geometry import flux_coords

lowE_fname = '/home/pdvanmeter/data/ratio_tables/rt_mesxr_lowE_ppcd.pkl'
midE_fname = '/home/pdvanmeter/data/ratio_tables/rt_mesxr_midE_ppcd.pkl'

class Ratio_Analyzer_midE(object):
    """
    """
    def __init__(self, shot, frame, priors=[200,2400], center=True, smooth=True, fname=midE_fname,
                 delta_a=0.06, delta_h=0.01):
        self.shot = shot
        self.frame = frame
        self._center = center
        self._smooth = smooth
        self._priors = priors
        
        # Load the ratio data
        self.ratios, self.sigmas, self.ps, self.thresholds = mesxr.get_ratio_data(self.shot, self.frame,
                                                                                  smooth=self._smooth, center=self._center)
        
        # Load the ratio tables
        with open(fname, 'rb') as f:
            self.r_table = pickle.load(f)
            
        # Load the flux surface mapping
        try:
            print('MSTfit reconstruction loaded.')
            self._flux = fm.MSTfit_Flux(shot=self.shot, frame=self.frame)
        except:
            print('MSTfit reconstruction not available. Defaulting to model.')
            self._flux = flux_coords(delta_a=delta_a, delta_h=delta_h)

        # Load the model geometry
        self._det = m3.MESXR_Avg(shot=shot, mode='midE', pfm=True)
        nlos = len(self._det.los)
        self._rhos_all = np.zeros(nlos)

        for ii,los in enumerate(self._det.los):
            ell_max = los.intercept_with_circle(0.52)
            ells = np.linspace(-ell_max, ell_max, num=100)
            xs, ys = los.get_xy(ells)
            self._rhos_all[ii] = np.amin(self._flux.rho(xs,ys))
            
        # Set up the Quad_Samplers
        self._samplers = [Quad_Sampler(lambda x, n=index: self._ln_likelihood(x,n), self._priors, vect=True) for index in range(nlos)]
    
    def fit(self, indices):
        """
        """
        # Fit the data for all specified indices
        for index in indices:
            self._samplers[index].fit()
            
        # Get summary statistics
        self._Tes_chord = np.array([self._samplers[index].mean for index in indices])
        self._Te_errs_chord = np.array([self._samplers[index].std for index in indices])
        self._rhos_chord = self._rhos_all[indices]
        
        # Sort the results by flux surfaces
        sorted_indices = np.argsort(self._rhos_chord)
        self.Tes = self._Tes_chord[sorted_indices]
        self.Te_errs = self._Te_errs_chord[sorted_indices]
        self.rhos = self._rhos_chord[sorted_indices]
        
        # Set up the Profile predictor
        X = np.atleast_2d(self.rhos).T
        y = np.maximum(self.Tes, 0)
        alpha = self.Te_errs**2
        kernel = C(1000, (1, 1e4)) * RBF(0.05, (1e-3, 1))
        self._gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, alpha=alpha)
        self._gp.fit(X,y)
    
    def predict(self, rhos):
        """
        """
        # Predict, then convert back to normal dimensions
        xs = np.atleast_2d(rhos).T
        y_pred, y_sigma = self._gp.predict(xs, return_std=True)
        xs = np.squeeze(xs)
        y_pred = np.squeeze(y_pred)
        y_sigma = np.squeeze(y_sigma)

        return y_pred, y_sigma
    
    def get_data(self):
        """
        Returns the ME-SXR data used to generate the ratios, either smoothed or raw. This is just a wrapper meant to simplify access.
        Returns:
            mesxr_data: (dict) ME-SXR 1D profiles, sorted by threshold.
            mesxr_sigma: (dict) Uncertainties associated with each measurement, sorted by threshold.
            mesxr_ps: (numpy.ndarray*) Average chord radius for each set of measurements. *If smooth=False, this will be a dictionary
                indexed in the same was as mesxr_data.
            thresholds: (numpy.ndarray) List of thresholds in the detector configuration.
        """
        if self._smooth:
            return mesxr.get_smooth_data(self.shot, self.frame, center=self._center)
        else:
            return mesxr.get_8c_data(self.shot, self.frame, center=self._center)
        
    def _ln_likelihood(self, Te, index):
        """
        Defines the log of the likelihood function for each chord.
        Args:
            Te = (np.ndarray) Array of values (in eV) for which to evaluate the likelihood function.
            index = (int) Index specifying a particular chord.
        """
        Te = np.atleast_1d(Te)
        r_model = np.zeros([len(Te), len(self.thresholds)])
        for n,Ec in enumerate(self.thresholds):
            r_model[:,n] = np.interp(Te, self.r_table.temp[:,index], self.r_table.ratio[Ec][:,index])
            
        chi2 = np.zeros([len(Te), len(self.thresholds)])
        for ii in range(len(Te)):
            chi2[ii,:] = (self.ratios[index,:] - r_model[ii,:])**2 / self.sigmas[index,:]**2
            
        return -0.5*np.sum(chi2, axis=1)