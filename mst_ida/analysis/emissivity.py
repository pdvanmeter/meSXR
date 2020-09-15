"""
"""
import numpy as np
import scipy as sp

def get_geometry_matrix(flux, p3det, rho_bins=None, Ec=2.0, M=20000, N=None):
    """
    This method should be totally general and therefore work for both MSTfit and
    V3fit reconstructions.
    """
    # Determine N from the number of chords, unless set manually
    if N is None:
        N = len(p3det.p[Ec])

    # Generate the psi bins
    if rho_bins is None:
        rho_bins = np.linspace(0, 1, num=N+1)
    rhos = np.array([0.5*(rho_bins[n]+rho_bins[n+1]) for n in range(N)]).reshape(-1,1)

    # Build the matrix
    g = np.zeros([N, N])

    for i in range(0,N):
        # Generate the spatial points
        z_max = p3det.los[Ec][i].intercept_with_circle(0.52)
        zs = np.linspace(-z_max, z_max, num=M)

        # Evaluate psi at each point
        xs, ys = p3det.los[Ec][i].get_xy(zs)
        rho_samples = flux.rho(xs, ys)

        # Bin the values
        hist = np.histogram(rho_samples, bins=rho_bins)[0]
        delta_Z = (2*z_max) / M
        g[i,:] = hist * delta_Z
        
    return g, rhos

def get_psis_from_rhos(rhos, mst):
    """
    This method is specific to MSTfit reconstructions. Note that for now I have found this method to work poorly.
    It seems to make the geometry more symmetric, rendering the result non-invertible.
    """
    psis = np.zeros(rhos.shape)
    
    for i, rho in enumerate(rhos):
        func = lambda x: np.abs(mst.rho_1d(x) - rho)
        opt = sp.optimize.minimize_scalar(func, method='bounded', bounds=[0,1])
        psis[i] = opt['x']
        
    return psis