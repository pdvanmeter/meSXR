"""
This module contains an updated version of the old Profile family of classes.
The modifications are mostly philosophical, clarifying the inheritence
relationships between the various Profile classes. This was also necessary
in order to allow derived objects to be pickled.
"""
from __future__ import division
import numpy as np
import scipy as sp
import scipy.integrate
import mst_ida.models.base.geometry as geometry

# Module-wide constants
MST_MINOR_RADIUS = 0.52

class Profile(object):
    """
    This is a generic class to represent all profiles used to define any number of generic profiles. By
    default no number of dimensions is specified, as this will be set by descended classes. All other classes
    in this module descend from this class.
    """
    def __init__(self, label, units='N/A', dim_units=['N/A'], boundary=1e-8):
        self.label = label
        self.units = units
        self.n_dims = 0
        self.dim_units = dim_units
        self.boundary = boundary
    
    def __call__(self, *args):
        # Properly evaluate 
        return self.evaluate(*args)

    def __str__(self):
        if self.units is not 'N/A':
            return self.label + ' (' + self.units + ')'
        else:
            return self.label
    
    def __len__(self):
        return self.n_dims

    def __add__(self, second_profile):
        return Composite_Profile(self, second_profile, operation='add')
            
    def __radd__(self, other):
        return self
    
    def __mul__(self, second_profile):
        return Composite_Profile(self, second_profile, operation='multiply')
            
    def __rmul__(self, other):
        return self

    def transform(self, *args):
        """
        Set the default coordinate transformation, used to convert between the coordinates used to define the
        profile and the shared coordinate system used to link together different profiles. For example, when using
        members of the 2D_Profile family one may wish to define a profile in terms of a radial variable, but then
        convert back to (x,y) Cartesian coordinates for analysis. This takes the shared coordinates as an argument
        and returns the profile coordinates.
        
        Derived classes should override this method.
        """
        return args
        
    def domain(self, *args):
        """
        The domain function which defains the valid domain of the profile. If the supplied point (x,y)
        is within the allowed domain the function returns True, otherwise it returns False.

        Derived classes should override this method.
        """
        return True

    def value(self, *args):
        """
        Returns the value of the profile at each point specified by the supplied coordinates. Returns
        zero if outside the bounds of the plasma. This is generally defined in terms of the profile coordinates.

        Derived classes should override this method.
        """
        return 0

    def evaluate(self, *args):
        """
        This method returns the value of the profile at the specified point if the point is within the domain, and 0
        if it is not within the domain. This is the preferred way to evaluate the value of the profile (instead of the
        self.value method), and is the method triggered by the __call__ method. Derived classes generally do not need
        to override this method.
        """
        #if self.domain(*self.transform(*args)): 
        #    return self.value(*self.transform(*args))
        #else:
        #    return self.boundary
        # Now handle arrays naturally, as long as self.value does not throw an exception
        vals = self.value(*self.transform(*args))
        doms = self.domain(*self.transform(*args))
        return vals*doms + np.logical_not(doms)*self.boundary


class Composite_Profile(Profile):
    """
    This class allows one to create a new profile with a response defined as the sum of two previously
    defined profiles, i.e. h(x,y) = f(x,y) + g(x,y). The profiles must have consistent units and dim_units.
    The domain is taken to be the intersection of the domains of the two defining functions (that is,
    the overlap between the two domains).
    """
    def __init__(self, prof1, prof2, operation='add'):
        self.prof1 = prof1
        self.prof2 = prof2
        self.operation = operation
        
        # Check for unit compatibility
        if self.operation == 'add':
            label = '{0:} + {1:}'.format(str(self.prof1), str(prof2))
            if (self.prof1.units != self.prof2.units) or (self.prof1.dim_units != self.prof2.dim_units):
                raise ValueError('Profiles have incompatible units.')
            else:
                super(Composite_Profile, self).__init__(label, units=self.prof1.units, dim_units=self.prof1.dim_units)

        elif self.operation == 'multiply':
            label = '({0:} x {1:})'.format(str(self.prof1), str(prof2))
            if self.prof1.dim_units != self.prof2.dim_units:
                raise ValueError('Profiles have incompatible units.')
            else:
                if prof1.units == 'N/A':
                    units = prof2.units
                elif prof2.units == 'N/A':
                    units = prof1.units
                else:
                    units = '({0:} x {1:})'.format(self.prof1.units, self.prof2.units)
                super(Composite_Profile, self).__init__(label, units=units, dim_units=self.prof1.dim_units)
        
    def domain(self, *args):
        """
        The new domain is defined to be the intersection of the domains of the defining profiles.
        """
        return self.prof1.domain(*self.prof1.transform(*args))*self.prof2.domain(*self.prof2.transform(*args))

    def value(self, *args):
        """
        The value of the function is determined by the specified operation.
        """
        if self.operation == 'add':
            return self.prof1(*args) + self.prof2(*args)
        elif self.operation == 'multiply':
            return self.prof1(*args) * self.prof2(*args)
        else:
            print('Warning: composite profile operation not recognized.')
            return 0
        
    def r_value(self, *args):
        """
        Provide a method for directly accessing the underlying value functions of each profile. This is especially useful
        if both profiles share a coordinate transformation (i.e. 1D radial profiles) and the user wishes to plot the
        composite as a function of that coordinate.
        """
        return self.prof1.value(*args) + self.prof2.value(*args)

# --------------------------------------- 2D Profile Classes --------------------------------------- #

class Profile_2D(Profile):
    """
    This implementation of the Profile class specifies 2D (x,y) coordinates as the shared system.
    All 2D Profiles should derive from this class.
    """
    def __init__(self, label, units='N/A', dim_units=['m', 'm'], boundary=1e-8):
        super(Profile_2D, self).__init__(label, units=units, dim_units=dim_units, boundary=boundary)
        self.n_dims = 2

    def transform(self, x, y):
        return (x,y)

    def domain(self, x, y):
        return np.sqrt(x**2 + y**2) < MST_MINOR_RADIUS

    def value(self, x, y):
        return 0


class Profile_Polar(Profile_2D):
    """
    Any Profile using 2D polar coordinates should descend from this class. For an example, see
    Profile_Island.
    """
    def __init__(self, label, units='N/A', dim_units=['m', 'm'], norm=MST_MINOR_RADIUS, rlim=[0, 1]):
        super(Profile_Polar, self).__init__(label, units=units, dim_units=dim_units)
        self.norm = norm
        self.rlim = rlim

    def transform(self, x, y):
        return (np.sqrt(x**2 + y**2)/self.norm, np.arctan2(y,x))

    def domain(self, r, theta):
        #return self.rlim[0] <= r <= self.rlim[1]
        return (r > self.rlim[0]) & (r < self.rlim[1])

    def value(self, r, theta):
        return 0

    def dist_theta(self, theta1, theta2):
        """
        Helper function useful for defining the distance between two angles. This always defaults to the shorter
        option. This is mostly useful since theta wraps around at +/- pi.
        """
        return min(np.abs(theta1-theta2), 2*np.pi-np.abs(theta2-theta1))


class Profile_Radial(Profile_2D):
    """
    Any Profile using 2D polar coordinates with angular symmetry should descend from this class. For an
    example, see Profile_Alpha_Beta.
    """
    def __init__(self, label, units='N/A', dim_units=['m', 'm'], norm=MST_MINOR_RADIUS, rlim=[0, 1]):
        super(Profile_Radial, self).__init__(label, units=units, dim_units=dim_units)
        self.norm = norm
        self.rlim = rlim

    def transform(self, x, y):
        return (np.sqrt(x**2 + y**2)/self.norm, )

    def domain(self, r):
        #return self.rlim[0] <= r <= self.rlim[1]
        return (r > self.rlim[0]) & (r < self.rlim[1])

    def value(self, r):
        return 0

    
class Profile_Alpha_Beta(Profile_Radial):
    """
    This is a symmetric 2D profile with a nontrivial value as defined by an alpha-beta parameterization.
    """
    def __init__(self, label, core_value, alpha, beta, units='N/A', dim_units=['m', 'm'], norm=MST_MINOR_RADIUS, rlim=[0., 1.]):
        super(Profile_Alpha_Beta, self).__init__(label, units=units, dim_units=dim_units, rlim=rlim, norm=norm)
        self.core_value = core_value
        self.alpha = alpha
        self.beta = beta

    def value(self, r):
        return self.core_value*(1 - r**self.alpha)**self.beta


class Profile_Island(Profile_Polar):
    """
    This is a 2D profile for an island feature. This will typically be added on top of a base profile, i.e. such as a
    Profile_Alpha_Beta object.
    """
    def __init__(self, label, delta_val, r_0, theta_0, delta_r, delta_theta, rlim=[0, 1], units='N/A', dim_units=['m', 'm'], norm=MST_MINOR_RADIUS):
        super(Profile_Island, self).__init__(label, units=units, dim_units=dim_units, rlim=rlim, norm=norm)
        self.r_0 = r_0
        self.theta_0 = theta_0
        self.delta_val = delta_val
        self.delta_r = delta_r
        self.delta_theta = delta_theta

    def value(self, r, theta):
        return self.delta_val*np.exp(-(r - self.r_0)**2/(2*self.delta_r**2))*np.exp(-self.dist_theta(theta, self.theta_0)**2/(2*self.delta_theta**2))


class Profile_Hollow(Profile_Radial):
    """
    This is a symmetric 2D profile for a hollow profile feature. This will typically be added on top of a base profile, i.e. such as a
    Profile_Alpha_Beta object.
    """
    def __init__(self, label, amp, peak, width, units='N/A', dim_units=['m', 'm'], norm=MST_MINOR_RADIUS, rlim=[0., 1.]):
        super(Profile_Hollow, self).__init__(label, units=units, dim_units=dim_units, rlim=rlim, norm=norm)
        self.amp = amp
        self.peak = peak
        self.width = width

    def value(self, r):
        #return (self.amp / (np.sqrt(2*np.pi)*self.width))*np.exp(-(self.peak - r)**2 / (2*self.width**2))
        return self.amp*np.exp(-(self.peak - r)**2 / (2*self.width**2))
    
class Profile_Power(Profile_Radial):
    """
    This profile is designed to facilitate a simplistic neutral density model which is nearly flat in the core and steep
    near the edges. In general it can be used to implement any profile which scalres like r^a for some finite a.
    """
    def __init__(self, label, core_value, amp, power, units='N/A', dim_units=['m', 'm'], norm=MST_MINOR_RADIUS, rlim=[0., 1.]):
        super().__init__(label, units=units, dim_units=dim_units, rlim=rlim, norm=norm)
        self.core_value = core_value
        self.amp = amp
        self.power = power

    def value(self, r):
        return self.core_value + self.amp*( r**(self.power) )

class Profile_Radial_Spline(Profile_Radial):
    """
    This class creates a nontrivial symmetric 2D profile by interpolating between the supplied data points.
    """
    def __init__(self, label, r_array, data_points, units='N/A', dim_units=['m', 'm'], rlim=[0., 1.], norm=MST_MINOR_RADIUS):
        super(Profile_Radial_Spline, self).__init__(label, units=units, dim_units=dim_units, rlim=rlim, norm=norm)
        self.r_array = r_array
        self.data_points = data_points

    def value(self, r):
        return np.interp(r, self.r_array, self.data_points)


class Profile_2D_Spline(Profile_2D):
    """
    This class creates a nontrivial arbitrary 2D profile by interpolating between the supplied data points.
    Because this interpolation can be computationally expensive in real time, the build_lookup_table
    method is supplied to pre-evaluate the profile at some specified points.
    """
    def __init__(self, label, sample_coords, sample_data, units='N/A', dim_units=['m', 'm'], method='linear', boundary=0):
        super(Profile_2D_Spline, self).__init__(label, units=units, dim_units=dim_units, boundary=boundary)
        self.sample_coords = sample_coords
        self.sample_data = sample_data
        self.method = method
        self.lookup = {pt:self.sample_data[index] for index,pt in enumerate(self.sample_coords)}
        self.fill = boundary

    def build_lookup_table(self, coords):
        """
        This function builds the lookup table which is checked before the function is called again.
        """
        values = self.calculate(coords)
        self.update(values, coords)
        
    def update(self, values, coords):
        """
        Add new values to the lookup table.
        """
        self.lookup.update({pt:values[index] for index,pt in enumerate(coords)})
        
    def calculate(self, coords):
        """
        Use the interpolation function to evaluate new coordinates.
        """
        return sp.interpolate.griddata(self.sample_coords, self.sample_data, coords, fill_value=self.fill, method=self.method)
    
    def preload(self, x, y):
        """
        Load a large number of points into the lookup table. This can be useful if the points need to be accessed in a
        non-vectorized way, such as a loop over coordinates. This is the same as self.build_lookup_table, but allows for
        vectorized inputs.
        """
        xs = np.atleast_1d(x)
        ys = np.atleast_1d(y)
        
        if xs.shape != ys.shape:
            raise ValueError('Input coordinates must have the same shape.')
        
        coords = list(zip(xs.ravel(), ys.ravel()))
        self.build_lookup_table(coords)
    
    def value(self, x, y):
        """
        Updated to support scalar and vectorized input coordinates.
        Inputs x and y must have the same shape so that f(x,y) = [f1, f2, ...] = [f(x1,y1), f(x2,y2), ...].
        """
        xs = np.atleast_1d(x)
        ys = np.atleast_1d(y)
        
        if xs.shape != ys.shape:
            raise ValueError('Input coordinates must have the same shape.')
        
        coords = list(zip(xs.ravel(), ys.ravel()))
        
        # Determine what has already been calculated
        vals = np.zeros(len(coords))
        new_coords = []
        indices = []
        
        for ii, xy in enumerate(coords):
            if xy in self.lookup.keys():
                vals[ii] = self.lookup[xy]
            else:
                new_coords.append(xy)
                indices.append(ii)
                
        # Calculate the new entries
        if len(new_coords) > 0:
            vals[indices] = self.calculate(new_coords)
            self.update(vals[indices], new_coords)
            
        # Cast as the input shape - the factor of one and np.squeeze casts scalar inputs correctly
        return 1*np.squeeze(np.reshape(vals, xs.shape))


# --------------------------------------- 2D Magnetic Flux Profiles --------------------------------------- #

class Transform_Flux(object):
    """
    This is an extension for the standard 2D polar profile which uses the magnetic flux coordinates rho and zeta
    instead of the polar r and theta. This is defined by the Shafranov shift delta_a and the LCFS shift delta_h. This
    can be easily applied to any class descended from Profile_Polar via multiple inheritence.
    """
    def __init__(self, delta_a=0.06, delta_h=0.01, norm=MST_MINOR_RADIUS, flux=None):
        # change to pass function
        if func is None:
            self.flux = geometry.flux_coords(delta_a=delta_a, delta_h=delta_h, norm=norm)
        else:
            self.flux = flux

    def transform(self, x, y):
        return self.flux(x, y)


class Transform_Rho(object):
    """
    This is an extension for the 2D symmetric radial profile which uses magnetic flux coordinate rho instead
    of the geometric radius. This is defined by the Shafranov shift delta_a and the LCFS shift delta_h. This
    can be easily applied to any class descended from Profile_Radial via multiple inheritence.
    
    Assumes that the underlying flux_coords object implements a rho(x,y) function.
    """
    def __init__(self, delta_a=0.06, delta_h=0.01, norm=MST_MINOR_RADIUS, flux=None):
        if flux is None:
            self.flux = geometry.flux_coords(delta_a=delta_a, delta_h=delta_h, norm=norm)
        else:
            self.flux = flux

    def transform(self, x, y):
        return (self.flux.rho(x, y), )


class Profile_Alpha_Beta_Rho(Transform_Rho, Profile_Alpha_Beta):
    """
    Modification of the standard alpha-beta profile to work with the radial flux coordinate rho.
    """
    def __init__(self, label, core_value, alpha, beta, units='N/A', dim_units=['m', 'm'], norm=MST_MINOR_RADIUS, rlim=[0., 1.],
                 delta_a=0.06, delta_h=0.01, flux=None):
        Transform_Rho.__init__(self, delta_a=delta_a, delta_h=delta_h, norm=norm, flux=flux)
        Profile_Alpha_Beta.__init__(self, label, core_value, alpha, beta, units=units, dim_units=dim_units, norm=norm, rlim=rlim)


class Profile_Island_Flux(Transform_Flux, Profile_Island):
    """
    Modification of the standard island profile to work with the flux coordinates rho, zeta.
    """
    def __init__(self, label, delta_val, r_0, theta_0, delta_r, delta_theta, rlim=[0, 1], units='N/A', dim_units=['m', 'm'], norm=MST_MINOR_RADIUS,
                 delta_a=0.06, delta_h=0.01, flux=None):
        Transform_Flux.__init__(self, delta_a=delta_a, delta_h=delta_h, norm=norm, flux=flux)
        Profile_Island.__init__(self, label, delta_val, r_0, theta_0, delta_r, delta_theta, rlim=rlim, units=units, dim_units=dim_units, norm=norm)


class Profile_Hollow_Flux(Transform_Rho, Profile_Hollow):
    """
    Modification of the standard island profile to work with the flux coordinates rho, zeta.
    """
    def __init__(self, label, amp, peak, width, units='N/A', dim_units=['m', 'm'], norm=MST_MINOR_RADIUS, rlim=[0., 1.],
                 delta_a=0.06, delta_h=0.01, flux=None):
        Transform_Rho.__init__(self, delta_a=delta_a, delta_h=delta_h, norm=norm, flux=flux)
        Profile_Hollow.__init__(self, label, amp, peak, width, rlim=rlim, units=units, dim_units=dim_units, norm=norm)
        
class Profile_Power_Flux(Transform_Rho, Profile_Power):
    """
    Modification of the standard power profile to work with the flux coordinates rho, zeta. Useful for implementing a neutral
    density model.
    """
    def __init__(self, label, core_value, amp, power, units='N/A', dim_units=['m', 'm'], norm=MST_MINOR_RADIUS, rlim=[0., 1.],
                 delta_a=0.06, delta_h=0.01, flux=None):
        Transform_Rho.__init__(self, delta_a=delta_a, delta_h=delta_h, norm=norm, flux=flux)
        Profile_Power.__init__(self, label, core_value, amp, power, rlim=rlim, units=units, dim_units=dim_units, norm=norm)

# --------------------------------------- Spectral Profile Classes --------------------------------------- #

class Profile_Spectral(Profile):
    """
    This class extends the standard profile class in order to permit the value of the profile at
    each point in space to be a single-variable function. In some sense this makes the space 3D,
    but the spectral dimension is not treated equally to the two spatial dimenstions.
    
    The main purpose of this class is to define a spectrum which itself depends on the local values
    of other profiles, such as how the x-ray emission sepctrum (as a function of photon energy)
    depens on the local values of T_e and n_e.

    The user can designate discontinuities, which will be accounted for in the integration

    The current version of the model does not depend on this class, rendering it deprecated.
    """
    def __init__(self, label, units='N/A', dim_units=['m', 'm', 'eV']):
        super(Profile_Spectral, self).__init__(label, units=units, dim_units=dim_units)
        self.n_dims = 3

    def __add__(self, second_profile):
        return Composite_Spectrum(self, second_profile, operation='add')
    
    def __mul__(self, second_profile):
        return Composite_Spectrum(self, second_profile, operation='multiply')

    def transform(self, x, y, en):
        return (x, y, en)

    def domain(self, x, y, en):
        """
        Make sure the point is within MST and that all energies are positive.
        """
        return (np.sqrt(x**2 + y**2) <= MST_MINOR_RADIUS) and np.amin(en) >= 0

    def value(self, x, y, en):
        """
        Inputs x and y must be scalar values, but en is permitted to be an array.
        """
        return np.zeros(np.array(en).shape)

    def integrate(self, x, y, en_lower=100, en_upper=20000, delta_en=100):
        """
        Return the integrated value of the spectrum at the specified point. This is computed via
        a trapezoidal rule computation.
        """
        if self.domain(x, y, [en_lower, en_upper]):
            en_array = np.arange(en_lower, en_upper, delta_en)
            return np.trapz(self.evaluate(x, y, en_array), x=en_array)
        else:
            return 0


class Composite_Spectrum(Composite_Profile, Profile_Spectral):
    """
    This allows composite spectra to be combined. It is essentially the same as the Composite_Profile,
    but inherits the integrate method from Profile_Spectral.
    """
    def __init__(self, prof1, prof2, operation='add'):
        super(Composite_Spectrum, self).__init__(prof1, prof2, operation=operation)