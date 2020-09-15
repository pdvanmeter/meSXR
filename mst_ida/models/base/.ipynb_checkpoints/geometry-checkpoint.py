"""
This file contains classes and methods to define the geometry of the ME-SXR detector, define and
interpret lines of sight, and related convert between various coordinate systems.
"""
from __future__ import division
import numpy as np
import scipy as sp
import scipy.constants

# Module-wide constants
MST_MINOR_RADIUS = 0.52

class line_of_sight(object):
    """
    This class define a line of sight by the impact parameterization. The 1D coordinate ell then
    defines positions along the LoS.
    Inputs:
        - impact_p = (float) The impact parameter, which is the distance from the MST machine origin to
                the line of sight along a line which is perpendicular to that line of sight. Measured in
                meters.
        - impact_phi = (float) The angle of the line which is perpendicular to the line of sight and runs
                through the machine origin. This is measured relative to the outboard midplane and increases
                upwards. Measured in radians.
    """
    def __init__(self, impact_p, impact_phi):
        self.impact_p = np.abs(impact_p)
        self.impact_phi = impact_phi
        
    def get_xy(self, ell):
        """
        Convert the 1D parameter of the lines of sight
        """
        x_dim = self.impact_p*np.cos(self.impact_phi) + ell*np.sin(self.impact_phi)
        y_dim = self.impact_p*np.sin(self.impact_phi) - ell*np.cos(self.impact_phi)
        return (x_dim, y_dim)
    
    def get_ell_from_x(self, x_dim):
        """
        Given a value of x, what is the corresponding ell on the line of sight?
        """
        return (x_dim - self.impact_p*np.cos(self.impact_phi))/np.sin(self.impact_phi)
    
    def get_ell_from_y(self, y_dim):
        """
        Given a value of y, what is the corresponding ell on the line of sight?
        """
        return (self.impact_p*np.sin(self.impact_phi) - y_dim)/np.cos(self.impact_phi)
    
    def get_impact_param(self):
        return np.sign(np.sin(self.impact_phi))*self.impact_p, self.impact_phi
    
    def intercept_with_circle(self, radius):
        """
        The line of sight intersects a circle of the given radius at positive and minus
        this return value.
        """
        return np.sqrt(radius**2 - self.impact_p**2)
    
    def equation(self, x_dim):
        """
        Given an array of points in the x-direction, return y(x) points on the line of sight.
        This allows easy plotting of the LoS.
        """
        return -1.*x_dim/np.tan(self.impact_phi) + self.impact_p/np.sin(self.impact_phi)


def los_from_pixel_coord_1D(x_index, num_pixels, pixel_thick, theta_detector, minor_radius=0.52, det_dist=0.0306, signed=True):
    """
    This function takes the pixel x-index as an input and returns the signed impact parameters
    associated with its line-of-sight. Must also supply the number of pixels, the thickness of each
    pixel, and the poloidal position of the detector. All distances are in meters, angles in radians.
    """
    x_dist = pixel_thick*(x_index+0.5)
    impact_p = minor_radius*(x_dist - 0.5*num_pixels*pixel_thick) / np.sqrt(det_dist**2 + (x_dist - 0.5*num_pixels*pixel_thick)**2 )
    impact_phi = theta_detector - np.arccos(impact_p/minor_radius)
    
    if impact_p < 0:
        impact_phi += np.pi
    
    if not signed:
        impact_p = np.abs(impact_p)
        
    return impact_p, impact_phi


# ------------------------------------ MST Flux Coordinates ------------------------------------ #

class flux_coords(object):
    """
    This class provides a convenient interface for converting between (x,y) coordinates and shifted
    flux surface coordinates, of the sort common in MST plasmas.

    Inputs:
        - delta_a = (float) Magnitude of the Shafranov shift, in meters.
        - delta_h = (float) Magnitude of the LCFS shift, in meters.
    """
    def __init__(self, delta_a=0.06, delta_h=0.01, norm=MST_MINOR_RADIUS):
        self.norm = norm
        self.delta_a = delta_a / self.norm
        self.delta_h = delta_h / self.norm
        self.alpha = (self.delta_h - self.delta_a) / ( 1 - np.abs(self.delta_h) )**2

    def __call__(self, x, y):
        return self.rho(x,y), self.zeta(x,y)

    def rho(self, x, y):
        """
        This function defines the coordinate transformation betweeen the machine-frame (x,y) coordinates
        and the Shafranov-shifted polar coordinate system used to parameterize the profile. This returns the
        flux radius. It is normalized to the provided radius. Set norm=1 to get the "true" value of rho.
        """
        x = np.array(x)/self.norm
        y = np.array(y)/self.norm
        
        # Prevent bad coordinates from being used
        x = np.clip(x, -1.2, 1.2)
        y = np.clip(y, -1.2, 1.2)

        if x.shape != y.shape:
            raise ValueError('Inputs "x" and "y" must have the same length.')

        return np.sqrt(self.rho_x(x,y)**2 + y**2)  / (1 - np.abs(self.delta_h))

    def zeta(self, x, y, norm=MST_MINOR_RADIUS):
        """
        This function defines the coordinate transformation betweeen the machine-frame (x,y) coordinates
        and the Shafranov-shifted polar coordinate system used to parameterize the profile. This returns the
        flux angle.
        """
        x = np.array(x)/self.norm
        y = np.array(y)/self.norm
        
        # Prevent bad coordinates from being used
        x = np.clip(x, -1.2, 1.2)
        y = np.clip(y, -1.2, 1.2)

        if self.delta_a == self.delta_h:
            return np.arctan2(y / (x - self.delta_a))
        else:
            return np.arctan2(y, self.rho_x(x,y))

    def rho_x(self, x, y):
        """
        Returns the x-component of the rho calculation. Note that x and y are normalized.
        """
        if self.delta_a == self.delta_h:
            return (x - self.delta_a)
        else:
            return ( np.sqrt(self.delta(x,y)) - 1.0 ) / (2.0*self.alpha)

    def delta(self, x, y):
        """
        This is a vector used internally by multiple calculations. Note that x and y are normalized.
        """
        return 1.0 - 4.0*self.alpha*(-x + self.alpha*y**2 + self.delta_a) 

# ------------------------------------------------ Utilities ------------------------------------------------
    
def sunflower_points(num_points, radius=MST_MINOR_RADIUS, cartesian=True):
    """
    This functions allows one to somewhat uniformly place points in a circle. This will mainly be used
    for interpolating the charge state fraction profile in two dimensions.
    """
    r_set = np.zeros(num_points)
    theta_set = np.zeros(num_points)
    x_set = np.zeros(num_points)
    y_set = np.zeros(num_points)
    
    # Calculate the sunflower points and scale to the supplied radius
    for k in range(num_points):
        r = radius*np.sqrt(k+1)/np.sqrt(num_points)
        theta = 2*np.pi*(k+1)/sp.constants.golden**2
        
        r_set[k] = r
        theta_set[k] = theta
        x_set[k] = r*np.cos(theta)
        y_set[k] = r*np.sin(theta)
    
    # Return points in the specified coordinate system
    if cartesian:
        return x_set, y_set
    else:
        return r_set, theta_set