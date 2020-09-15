#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Package: mesxr.calibration
Module: trimscan
Author: Patrick VanMeter
Affiliation: Department of Physics, University of Wisconsin-Madison
Last Updated: November 2018

Description:
    This module is a significant overhaul to the overiginal ME-SXR calibration software. The
    goals of the re-write are to improve usability and transperency, and to more naturally
    facilitate features which were added on to the original code. 
Acknowledgements:
    - Novimir Pablant and Jacob Maddox, for the original calibration code.
    - Luis Felipe Delgado-Aparicio, for heading the PPPL/MST collaboration.
    - Daniel Den Hartog and Lisa Reusch, for advising me.
"""
import copy
import os
import multiprocessing as mp
import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
from scipy.special import erf
import mesxr.calibration.utilities as utilities


def calibrate_mp(pixel):
    """
    This is a helper function in order to use multiprocessing to invoke the calibrate() method for an array
    of Pilatus_Pixel objects.
    """
    pixel.calibrate()
    return copy.copy(pixel)

def calibrate_module_mp(chip):
    """
    This helper function is used to calibrate an entire module using multiprocessing. This ensures that each individual chip does
    not itself attempt to spawn additional processes.
    """
    chip.calibrate(num_cores=1)
    return copy.copy(chip)

class Pilatus_Pixel(object):
    """
    Description:
        TBD
    Arguments:
        - calibration_data = (list of np.array) Contains the calibration data for this pixel. The list should have
                a number of entries equal to the length of self.energies, and each entry should be an array with a
                length equal to the length of the corresponding entry in self.trimbits.
        - calibration_energies = (np.array) The photon energy for each calibration source element.
        - elements = (list) Each entry should be a string which identifies the element corresponding to the
                same index in calibration_energies.
        - trimbits = (list of np.array) Identifies the trimbit values which correspond to the arrays in calibration_data.
                This allows the user to remove certain trimbit settings from the calibration, if necessary.
        - coords = (tuple of ints) The global (x,y) coordinates of this pixel.
        - size = (tuple of float) The physical dimensions (x,y) of the pixel, in micrometers.
    """
    def __init__(self, calibration_data, calibration_energies, elements, trimbits, coords, chip_number, module_number, size=(172.0, 172.0)):
        # Physical parameters
        self.data = calibration_data
        self.energies = calibration_energies
        self.elements = elements
        self.trimbits = trimbits
        self.coords = coords
        self.chip = chip_number
        self.module = module_number
        self.size = size
        self.edge_pixel = False

        # Calibration parameters to be fit to the data
        self.num_elements = len(self.elements)
        self.trimfit_params = np.zeros([self.num_elements, 6])
        self.trimfit_cov    = np.zeros([self.num_elements, 6, 6])

        self.enfit_params = np.zeros(3)
        self.enfit_cov    = np.zeros([3, 3])

        self.trimfit_chi2 = np.zeros(self.num_elements)
        self.enfit_chi2 = 0.0

        # Keep track of failed fits
        self.good_trimfits = np.array([False for x in range(self.num_elements)])
        self.good_enfit = False

    def __str__(self):
        return str(self.coords)

    def s_curve(self, trim, A0, A1, A2, A3, A4, A5):
        """
        This function parameterizes the detector response to a given trimbit setting. This returns the predicted
        photon counts given the calibration parameters and supplied trimbit value. This function permits non-integer
        trimbit values, which in reality must be rounded to the nearest integer.
        """
        return 0.5*(erf( -1*(trim - A0)/(np.sqrt(2)*A1) ) + 1.)*(A2 + A3*(trim - A0)) + A4 + A5*(trim - A0)

    def s_curve_model(self, element, trim):
        """
        Use the fit results to return values from the analytic model for the S-curve fit.
        """
        elem_index = self.elements.index(element)
        return self.s_curve(trim, *self.trimfit_params[elem_index, :])

    def en_curve(self, energy, C0, C1, C2):
        """
        This function parameterizes the mapping between the trimbit of the S-curve inflection point and the calibration
        line energies.
        """
        return C0*energy**2 + C1*energy + C2

    def en_curve_model(self, energy):
        """
        Use the fit results to return values from the analytic model for the energy fit.
        """
        return self.en_curve(energy, *self.enfit_params)

    def en_curve_uncertainty(self, energy):
        """
        Returns the uncertainty in the trimbit required to set the threshold to a given energy, based on the
        energy fit results.
        """
        var_2 = (energy**4)*self.enfit_cov[0,0] + (energy**2)*self.enfit_cov[1,1] + self.enfit_cov[2,2]
        cov_2 = 2*energy*( (energy**2)*self.enfit_cov[0,1] + energy*self.enfit_cov[0,2] + self.enfit_cov[1,2] )
        return np.sqrt(var_2 + cov_2)

    def trimbit_from_threshold(self, energy):
        """
        This is just a wrapper for the en_curve_model function which provides a more consistent notation for some
        use cases.
        """
        return self.en_curve_model(energy)

    def trimbit_uncertainty(self, energy):
        """
        This is just a wrapper for the en_curve_model function which provides a more consistent notation for some
        use cases.
        """
        return self.en_curve_uncertainty(energy)
    
    def threshold_from_trimbit(self, trimbit):
        """
        Invert the threshold fit. That is, get the lower threshold energy corresponding to a specific trimbit value. This
        is useful when considering the impact of trimbit rounding.
        """
        c0, c1, c2 = self.enfit_params
        return (-c1 + np.sqrt(c1**2 - 4*c0*(c2 - trimbit))) / (2*c0)

    def threshold_uncertainty(self, trimbit):
        """
        """
        return 0

    def exclude_trimbits(self, remove_bits):
        """
        Remove the given trimbits and corresponding data so that they will not be used in the calibration.
        """
        for elem_index in range(self.num_elements):
            if set(remove_bits).issubset(set(self.trimbits[elem_index])):
                exclude_indices = [np.where(tbit == self.trimbits[elem_index])[0][0] for tbit  in remove_bits]
                slice_indices = [i for i in range(len(self.trimbits[elem_index])) if i not in exclude_indices]
                self.trimbits[elem_index] = self.trimbits[elem_index][slice_indices]
                self.data[elem_index] = self.data[elem_index][slice_indices]

    def exclude_elements(self, remove_elem):
        """
         Remove the given elements and corresponding data so that they will not be used in the calibration.
        """
        # Determine the indices of the entries to keep
        try:
            remove_indices = [self.elements.index(elem) for elem in remove_elem]
        except:
            print('ERROR: Supplied element(s) not in the elements array.')
            remove_indices = []

        slice_indices = [i for i in range(len(self.elements)) if i not in remove_indices]

        # Remove the unwanted elements
        self.elements = [self.elements[i] for i in slice_indices]
        self.data = [self.data[i] for i in slice_indices]
        self.trimbits = [self.trimbits[i] for i in slice_indices]
        self.energies = self.energies[slice_indices]
        self.num_elements = len(self.elements)

        # Remake the calibration result arrays which depend on the number of elements
        self.trimfit_params = np.zeros([self.num_elements, 6])
        self.trimfit_cov    = np.zeros([self.num_elements, 6, 6])
        self.trimfit_chi2   = np.zeros(self.num_elements)
        self.good_trimfits  = np.array([False for x in range(self.num_elements)])
    
    # Calibration functions
    def trimbit_fit(self):
        """
        Use the calibration data to determine the best-fit parameters and covariance matrix for this pixel.
        """
        for elem_index in range(self.num_elements):
            # Use a gradient method to guess the trimbit of the inflection point
            data_slope = np.gradient(self.data[elem_index], 1)
            index = np.argmin(data_slope) + self.trimbits[elem_index][0]

            # Initial guesses
            p0 = [float(index),                     # A0 - Inflection trimbit
                1.0,                              # A1 - S-curve width
                np.amax(self.data[elem_index]),   # A2 - Response amplitude
                0.0,                              # A3 - CX slope
                np.amin(self.data[elem_index]),   # A4 - BG amplitude
                0.0]                              # A5 - BG CX slope

            # Do the fit
            bounds = (np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]),   # Lower bounds
                      np.array([np.inf, np.inf, np.inf, 0.0, np.inf, 0.0]))               # Upper bounds

            sigma = np.sqrt(self.data[elem_index])

            try:
                self.trimfit_params[elem_index, :], self.trimfit_cov[elem_index, :, :] = curve_fit(self.s_curve, self.trimbits[elem_index], self.data[elem_index],
                                                                                                p0=p0, bounds=bounds, sigma=sigma, absolute_sigma=True)

                # Determine the reduced Chi^2 of the fit
                model_data = self.s_curve(self.trimbits[elem_index], *self.trimfit_params[elem_index, :])
                self.trimfit_dof = len(self.data[elem_index]) - 6.0
                self.trimfit_chi2[elem_index] = np.sum( (self.data[elem_index] - model_data)**2/(sigma**2) )

                self.good_trimfits[elem_index] = True
            except:
                # The fit did not work - consider including fallback options here
                self.good_trimfits[elem_index] = False
                self.trimfit_params[elem_index, :] = np.nan
                self.trimfit_cov[elem_index, :, :] = np.nan
                self.trimfit_chi2[elem_index] = np.nan
                self.trimfit_dof = np.nan

    def energy_fit(self):
        """
        Use the results of the trimbit_fit to fit the en_curve.
        """
        # Only use the successful fits
        if np.sum(self.good_trimfits) >= 4:
            trim_data = self.trimfit_params[self.good_trimfits, 0]
            trim_sigma = np.sqrt(self.trimfit_cov[self.good_trimfits, 0, 0])
            trim_energies = self.energies[self.good_trimfits]

            #self.enfit_params, self.enfit_cov = np.polyfit(self.energies, trim_data, 2, w=1/trim_sigma, cov=True)
            p0 = [np.mean(trim_data), 0.0, 0.0]

            try:
                self.enfit_params[:], self.enfit_cov[:,:] = curve_fit(self.en_curve, trim_energies, trim_data, p0=p0,
                                                                      sigma=trim_sigma, absolute_sigma=True)

                # Determine the Chi^2 of the fit
                trim_model = self.en_curve(trim_energies, *self.enfit_params)
                self.enfit_dof = len(trim_data) - 3.0
                self.enfit_chi2 = np.sum( (trim_data - trim_model)**2/(trim_sigma**2) )
                self.good_enfit = True
            except:
                # The fit did not work - consider including fallback options here
                self.good_enfit = False
                self.enfit_params[:] = np.nan
                self.enfit_cov[:] = np.nan
                self.enfit_chi2 = np.nan
                self.enfit_dof = np.nan
        else:
            # There is too little good data for a reliable fit
            self.good_enfit = False
            self.enfit_params[:] = np.nan
            self.enfit_cov[:,:] = np.nan
            self.enfit_chi2 = np.nan
            self.enfit_dof = np.nan

    def calibrate(self):
        """
        Performs both the trimbit fit and the energy fit in a single command.
        """
        self.trimbit_fit()
        self.energy_fit()


class Pilatus_Null_Pixel(Pilatus_Pixel):
    """
    This class exists to allow the creation of placeholder Pixel objects which do nothing and hold no data, but share the
    same methods. This is useful as a placeholder and for representing the empty row between pixels.
    """
    def __init__(self):
        self.coords = (-1,-1)
        self.chip = -1
        self.module = -1
        self.edge_pixel = False

        # Calibration parameters to be fit to the data
        self.elements       = ['none']
        self.num_elements   = len(self.elements)
        self.trimfit_params = np.array([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]])
        self.trimfit_cov    = np.zeros([1,6,6])
        self.trimfit_cov[:,:,:] = np.nan

        self.enfit_params   = np.array([np.nan, np.nan, np.nan])
        self.enfit_cov      = np.nan

        self.trimfit_chi2   = np.array([np.nan])
        self.enfit_chi2     = np.nan
        
        self.good_trimfits = np.array([False])
        self.good_enfit = False

    def s_curve_model(self, element, trim):
        return -1.0

    def en_curve_model(self, energy):
        return -1.0
    
    def trimbit_fit(self):
        pass

    def energy_fit(self):
        pass


class Pilatus_Chip(object):
    """
    Description:
        The chip is the basic organizational unit for pixels in the PILATUS series of X-ray detectors. Chips essentially
        consist of an array of pixels and some additional properties.
    Inputs:
        - pixel_dims = (tuple of int) The size n x m of the array of pixels on the chip.
        - chip_number = (int) ID number for the chip.
        - calibration_data = (list of np.array) Each array in the list corresponds to the calibration data for the
                corresponding entry in elements. The arrays are 3D with indices corresponding to [x, y, trimbit].
        - calibration_energies = (np.array) The photon energy for each calibration source element.
        - elements = (list) Each entry should be a string which identifies the element corresponding to the
                same index in calibration_energies.
        - trimbits = (list of np.array) The trimbits to include for the calibration of this chip. This is expected to vary
                on a chip-by-chip basis. This should be the same length as the third axis of each element in the
                calibration_data list.
        - pixel_coords = (np.array) This array assigns the global coordinates for each pixel on the chip. The array
                is indexed by [x, y, coord] where coord=0 gives the global X coordinate and coord=1 gives the global Y.
                The indices x,y refer to the local (chip-level) coordinates.
        - vcmp = (float) The Vcmp value for this chip. This is the global value with the offset applied.
    """
    def __init__(self, pixel_dims, chip_number, module_number, calibration_data, calibration_energies, elements, trimbits, pixel_coords, vcmp):
        self.pixel_dims = pixel_dims
        self.number = chip_number
        self.module = module_number
        self.trimbits = trimbits

        # Create the array of pixels
        self.pixels = np.empty(self.pixel_dims, dtype=object)

        for y in range(self.pixel_dims[1]):
            for x in range(self.pixel_dims[0]):
                self.pixels[x, y] = Pilatus_Pixel([data[x, y, :] for data in calibration_data], calibration_energies,
                                                  elements, copy.copy(self.trimbits), tuple(pixel_coords[x,y,:]), self.number, self.module)
                
                if x == 0 or y == 0 or x == self.pixel_dims[0]-1 or y == self.pixel_dims[1]-1:
                    self.pixels[x, y].edge_pixel = True

        # Store chip-level detector properties
        self.vcmp = vcmp

    def __str__(self):
        return str(self.number)

    def exclude_trimbits(self, remove_bits):
        """
        Remove the specified trimbits and associated data for all pixels on the chip.
        """
        for pixel in self.pixels.ravel():
            pixel.exclude_trimbits(remove_bits)

    def remove_elements(self, remove_elem):
        """
        Remove the specified elements and associated data for all pixels on the chip.
        """
        for pixel in self.pixels.ravel():
            pixel.exclude_elements(remove_elem)
    
    def calibrate(self, num_cores=16):
        """
        Calibrate all pixels on a chip. Returns the reduced chi^2 metric for the energy fits. This is mostly of interest when
        investigating individual modules. Otherwise, see the detector-level calibrate() function.
        """
        if num_cores == 1:
            [pixel.calibrate() for pixel in self.pixels.ravel()]
        else:
            pool = mp.Pool(num_cores)
            pixels = pool.map(calibrate_mp, self.pixels.ravel())
            self.pixels = np.array(pixels).reshape(self.pixel_dims)


class Pilatus_Module(object):
    """
    Description:
        TBD
    Inputs:
        - chip_dims = (tuple of int) The global (X, Y) layout of chips on the module.
        - pixel_dims = (tuple of int) The local layout (x,y) of pixels on a chip.
        - module_number = (int) Identifier for the module. This is important for multi-module detectors.
        - calibration_data = (list of np.array) Each array in the list corresponds to the calibration data for the
                corresponding entry in elements. The arrays are 3D with indices corresponding to [X, Y, trimbit], where
                (X,Y) are the pixel coordinates in the module frame. It is assumed that no trimbit data has been dropped
        - calibration_energies = (np.array) The photon energy for each calibration source element.
        - elements = (list) Each entry should be a string which identifies the element corresponding to the
                same index in calibration_energies.
        - trimbit_start = (np.array) The trimbit to start the scan for each element for each module. This array is indexed
                by [elem, chip_num]. Any value above zero will cause some data to be omitted from the calibration procedure.
        - chip_coords = (np.array) Entries in this array define the starting pixel (upper-left) for each chip. This is indexed
                by [chip_num, dim] where dim=0 is x and dim=1 is y. For example, on the Pilatus3 chip 0 start with pixel (0,0)
                and chip 1 starts with (61, 0) so chip_coords[0,:] = [0,0] and chip_coords[1,:] = [61,0].
        - num_trimbits = [keyword](int) Specify the number of trimbits available to each pixel.
    """
    def __init__(self, chip_dims, pixel_dims, module_number, calibration_data, calibration_energies, elements,
                 trimbit_start, chip_coords, num_trimbits=64):
        self.number = module_number
        self.chip_dims = chip_dims
        self.pixel_dims = pixel_dims
        self.num_trimbits = num_trimbits

        # Initialize the chips on the module
        self.chips = np.empty(self.chip_dims, dtype=object)
        chip_num = 0

        for y in range(self.chip_dims[1]):
            for x in range(self.chip_dims[0]):
                vcmp = 0

                # Determine the pixel coordinates based on the supplied dimensions and the start coordinates
                pixel_coords = np.zeros([self.pixel_dims[0], self.pixel_dims[1], 2], dtype=int)
                for pix_x in range(self.pixel_dims[0]):
                    for pix_y in range(self.pixel_dims[1]):
                        pixel_coords[pix_x, pix_y, :] = chip_coords[chip_num, :] + [pix_x, pix_y]
                
                # Trim the calibration data to the appropriate size
                trimbits = [np.arange(trimbit_start[elem, chip_num], self.num_trimbits) for elem in range(len(elements))]

                start_x = chip_coords[chip_num, 0]
                end_x   = chip_coords[chip_num, 0]+self.pixel_dims[0]
                start_y = chip_coords[chip_num, 1]
                end_y   = chip_coords[chip_num, 1]+self.pixel_dims[1]
                pixel_data = [calibration_data[elem][start_x:end_x, start_y:end_y, trimbits[elem][0]:] for elem in range(len(elements))]

                self.chips[x, y] = Pilatus_Chip(self.pixel_dims, chip_num, self.number, pixel_data, calibration_energies, elements,
                                                trimbits, pixel_coords, vcmp)
                chip_num += 1

    def __str__(self):
        return str(self.number)

    def get_pixels(self):
        """
        Return an array with all pixels organized by module coordinates.
        """
        x_dim = 487
        y_dim = 195
        pixels_mod = np.array([[Pilatus_Null_Pixel() for y in range(y_dim)] for x in range(x_dim)])

        for chip in self.chips.ravel():
            for pixel in chip.pixels.ravel():
                pixels_mod[pixel.coords] = pixel

        return pixels_mod

    def calibrate(self, num_cores=16):
        """
        Use multiprocessing to calibrate all pixels. If possible it is advisable to set num_cores equal to the
        number of chips on the module.
        """
        if num_cores == 1:
            chips = [chip.calibrate() for chip in self.chips.ravel()]
        else:
            pool = mp.Pool(num_cores)
            chips = pool.map(calibrate_module_mp, self.chips.ravel())

        self.chips = np.array(chips).reshape(self.chip_dims)


class Pilatus_Detector(object):
    """
    Description:
        A detector is an object which holds an array of Modules and the associated global settings. Objects
        of this class also provide methods for initiating the calibration process.
    Usage:
        This class is intended to be inhereted by classes representing the implementation of specific Pilatus
        detectors. As such, direct instances of this class will be of limited use. For an example implementation,
        see the Pilatus3_100k class below.
    Parameters:
        - TBD
    """
    def __init__(self, *args, **kwargs):
        # Detector properties
        self.global_pixel_dims = (0, 0)     # Dimensions of all pixels on the detector face, including fake pixels
        self.num_chips    = 0               # The number of chips which compose the module
        self.module_dims  = (0, 0)          # The layout of modules on the detector
        self.chip_dims    = (0, 0)          # The layout of chips on the module
        self.pixel_dims   = (0, 0)          # The layout of pixels on a chip
        self.num_trimbits = 0               # The number of trimbits available to each pixel
        self.trimbit_start = [[0]]          # Value to start trimbit data at, indexed by [elem, chip_num]
        self.calibrated = False             # Flag to tell whether the detector has been calibrated already
        self.bad_pixels = []                # Keep track of which pixels are problematic

        # Initialize the modules and global settings
        self.load_trimscan_data([], [])
        self.load_global_settings(0)
        self.init_modules()

        # Make pixels easily accessible by global coordinates
        self.pixels = self.get_pixels()

    def init_modules(self):
        """
        Run this to initialize the modules which compose the detector. This is separated out from the __init__ method
        since it does generally not need to be overwritten for inhereted classes.
        """
        self.modules = np.empty(self.module_dims, dtype=object)
        module_number = 0

        for y in range(self.module_dims[1]):
            for x in range(self.module_dims[0]):
                self.modules[x,y] = Pilatus_Module(self.chip_dims, self.pixel_dims, module_number, self.trimscan_data, self.trimscan_energy,
                                                   self.elements, self.trimbit_start, self.get_chip_coords(module_number),
                                                   num_trimbits=self.num_trimbits)
                module_number += 1

    def load_global_settings(self, settings_path):
        """
        Load the detector settings in from the data file. This may require slight modification for multi-module
        detectors since I do not know which settings are shared across modules. I have separated out the VCMP settings
        due to its unique chip-level variation.
        """
        self.vcmp = np.zeros(self.num_chips)
        self.settings = {}

        # Load the settings data from the file
        if settings_path != 0:
            try:
                fname = os.path.join(settings_path, 'setdacs_b01_m01.dat')
                settings_file = open(fname, 'r')

                for line in settings_file.readlines():
                    line_elems = line.split()
                    if line_elems[0] == 'set':
                        key = line_elems[1].split('_')[2]
                        self.settings[key] = float(line_elems[2])

                settings_file.close()

                # Pull out the VCMP settings
                for key in self.settings.keys():
                    if 'VCMP' in key:
                        index = int(key.split('VCMP')[1])
                        self.vcmp[index] = self.settings[key]
            
            except:
                print('ERROR: Failed to load supplied settings file. Check path name.')
        else:
            pass

    def load_trimscan_data(self, elements, trimscan_paths):
        """
        """
        self.elements = elements

        self.trimscan_data = [np.zeros([self.global_pixel_dims[0], self.global_pixel_dims[1], self.num_trimbits]) for i in self.elements]
        self.trimscan_energy = utilities.get_line_energy(self.elements)
        self.trimbits = [np.arange(self.num_trimbits) for elem in self.elements]
        
        for elem, fname in enumerate(trimscan_paths):
            self.trimscan_data[elem] = utilities.load_calibration_data(fname)

    def generate_settings_file(self, output_path):
        """
        Used to generate the settings file for use in ME-SXR operation. This removes the requirement of keeping
        up with the original file used in the calibration procedure.
        """
        pass

    def pixel_mapping(self, x, y):
        """
        Returns the mapping between between global (x,y) coordinates and local (mod_num, chip_num, (pix_x, pix_y))
        coordinates. This is important for mapping the trimbit scan data to the Detector data structure.
        """
        return (0, 0, (x, y))

    def get_chip_coords(self, mod_number):
        """
        Override to set the global coordinates for the first (upper-left) pixel on each chip.
        """
        return np.array([[0,0]])

    def get_pixels(self):
        """
        Return an array of all pixel objects following the global coordinates.
        """
        pass

    def calibrate(self, num_cores=16):
        """
        Use multiprocessing to calibrate all pixels. 
        """
        pass

    def determine_bad_pixels(self, include=[]):
        """
        Update the list of bad pixels. This should generally be called immediately after calibration. Use
        the 'include' keyword to manually add points, even if the fit did not fail. This keyword accepts a
        list of tuples.
        """
        # Ensure that the specified fits are marked as bad
        for coords in include:
            if coords not in self.bad_pixels:
                self.pixels[coords].good_enfit = False

        # Now add all remaining pixels for which a fit was not found
        for pixel in self.pixels.ravel():
            if not pixel.good_enfit and type(pixel) != Pilatus_Null_Pixel:
                if pixel.coords not in self.bad_pixels:
                    self.bad_pixels.append(pixel.coords)
    
    def get_trimbit_map(self, threshold_map):
        """
        Returns the trimbit values for the supplied threshold map (in keV).
        """
        if self.calibrated:
            trimbit_map = np.zeros(self.global_pixel_dims, dtype=int)

            for x in range(self.global_pixel_dims[0]):
                for y in range(self.global_pixel_dims[1]):
                    # Ensure that the trimbit is within the valid range
                    tbit = self.pixels[x,y].trimbit_from_threshold(threshold_map[x,y])
                    if tbit < 0 or np.isnan(tbit):
                        tbit = 0
                    elif tbit > self.num_trimbits - 1:
                        tbit = self.num_trimbits - 1
                    else:
                        tbit = int(round(tbit))
                    
                    trimbit_map[x,y] = tbit
        else:
            print('Detector is not yet calibrated.')
            trimbit_map = -1*np.ones(self.global_pixel_dims, dtype=int)
        return trimbit_map

# ---------------------------------------- Specific Detector Implementations ----------------------------------------

class Pilatus3_100k(Pilatus_Detector):
    """
    Description:
        This is a specific implementation of the Pilatus_Detector which integrates the appropraite pixel geometry
        and device settings. The PILATUS3 100K detector, produced by DECTRIS Ltd., is composed of a single module
        with approximately 100,000 total pixels.
    """
    def __init__(self, elements, trimscan_paths, settings_path, trimbit_start=0, omit_trimbits=False, calib_name='MST'):
        # Pilatus3 pixel properties
        self.global_pixel_dims = (487, 195)
        self.num_chips         = 16
        self.module_dims       = (  1,   1)
        self.chip_dims         = (  8,   2)
        self.pixel_dims        = ( 60,  97)
        self.num_trimbits      = 64
        self.calibrated        = False
        self.bad_pixels        = []
        self.name              = calib_name

        if not omit_trimbits:
            self.trimbit_start = np.zeros([len(elements), self.num_chips], dtype=int)
        else:
            self.trimbit_start = trimbit_start

        # Pilatus3 physical characteristics

        # Load modules and settings
        self.load_trimscan_data(elements, trimscan_paths)
        self.load_global_settings(settings_path)
        self.init_modules()

        # Make pixels easily accessible by global coordinates
        self.pixels = self.get_pixels()

    def pixel_mapping(self, x, y):
        """
        Implement the proper mapping for the PILATUS3 100K detector. This is most easily done using the pre-existing
        mapping function from Novi.
        """
        chip_num, chip_x, chip_y = utilities.get_chip_coords(x, y)
        return (0, chip_num, (chip_x, chip_y))

    def get_chip_coords(self, mod_number):
        """
        Returns the global coordinates for the first (upper-left) pixel on each chip for the PILATUS3 100K.
        """
        dx = 61
        dy = 98
        chip_coords = np.array([[0*dx, 0], [1*dx, 0], [2*dx, 0], [3*dx, 0], [4*dx, 0], [5*dx, 0], [6*dx, 0], [7*dx, 0],
                                [0*dx,dy], [1*dx,dy], [2*dx,dy], [3*dx,dy], [4*dx,dy], [5*dx,dy], [6*dx,dy], [7*dx,dy]])
        return chip_coords

    def write_settings_file(self, filepath):
        """
        Writes the settings used to generate the trimscan back out to a file.
        """
        # Format the settings into a string with linebreaks
        settings = '# /dev/shm/setdacs_b01_m01.dat\n'
        settings += 'set B01_M01_VTRM {0: 2.4f}\n'.format(self.settings['VTRM'])

        for vcmp in range(self.num_chips):
            vcmp_str = 'VCMP{0:}'.format(vcmp)
            settings += 'set B01_M01_{0:} {1: 2.4f}\n'.format(vcmp_str, self.settings[vcmp_str])

        for key in ['VCCA', 'VRF', 'VRFS', 'VCAL', 'VDEL', 'VADJ']:
            settings += 'set B01_M01_{0:} {1: 2.4f}\n'.format(key, self.settings[key])

        # Remove the final unwanted newline
        settings = settings[:-1]

        with open(os.path.join(filepath, 'setdacs_b01_m01.dat'), 'wb') as f:
            f.write(settings)

        print('Settings file written to {0:}.'.format(filepath))

    def write_autotrim_files(self, threshold_map, filepath, filename='autotrim'):
        """
        Generate autotrim files given a specified threshold map using the calibration.
        """
        trimbit_map = self.get_trimbit_map(threshold_map)
        trimbit_sorted = np.zeros([self.num_chips, self.pixel_dims[0], self.pixel_dims[1]])

        for g_x in range(self.global_pixel_dims[0]):
            for g_y in range(self.global_pixel_dims[1]):
                cn, cx, cy = utilities.get_chip_coords(g_x, g_y)
                if cn != -1:
                    trimbit_sorted[cn, cx, cy] = trimbit_map[g_x, g_y]
        
        for cn in range(self.num_chips):
            trim_string = '# AUTOTRIM files for PILATUS3 100K by UW Madison code\n'
            trim_string += 'set B01_M01_CHSEL{0:} 1\n'.format(cn)

            for cx in range(self.pixel_dims[0]):
                for cy in range(self.pixel_dims[1]):
                    trim_string += 'trim {0:} {1:} {2:}\n'.format(cx, cy, hex(int(trimbit_sorted[cn, cx, cy])))

            trim_string += 'settrims'

            # Write the output file
            fname = '{0:}_b01_m01_c{1:02d}.dat'.format(filename, cn)
            with open(os.path.join(filepath, fname), 'wb') as f:
                f.write(trim_string)

            print('Saved {0:} to file.'.format(fname))

    def write_bad_pixels_file(self, filepath):
        """
        Outputs a CSV file containing all of the bad pixels. This is useful in later analysis.
        """
        fname = 'bad_pixels.csv'
        np.savetxt(os.path.join(filepath, fname), self.bad_pixels, delimiter=',', fmt='%d')

    def get_pixels(self):
        """
        Since this detector has only one module, just call that module's get_pixels() method.
        """
        return self.modules[0,0].get_pixels()

    def calibrate(self, num_cores=16, bad_pix=[]):
        """
        Since this detector has only one module, just call that module's calibrate() method. Use the 'bad_pix'
        to manually ensure that the specified pixels get marked as bad, even if the fit does not fail.
        """
        self.modules[0,0].calibrate(num_cores=num_cores)
        self.calibrated = True
        self.pixels = self.get_pixels()
        self.determine_bad_pixels(include=bad_pix)