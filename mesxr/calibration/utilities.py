#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Package: mesxr.calibration
Module: utilities
Author: Patrick VanMeter, Novimir Pablant
Affiliation: Department of Physics, University of Wisconsin-Madison
Last Updated: November 2018

Description:
    This module contains a number of auxilary functions for the main timscan.py module to
    make use of. These are either functions which might be useful elsewhere (such are coordinate
    conversions) or functions which are more likely to change in future implementations (like
    loading in the trimbit scan data from a specific file naming scheme).
Usage:
    TBD
"""
from os import path
import numpy as np
import tifffile

M_SIZE_Y = 195
M_SIZE_X = 487
M_NUM_TRIM = 64
M_NUM_CHIPS_Y = 2
M_NUM_CHIPS_X = 8
M_NUM_CHIPS = M_NUM_CHIPS_X * M_NUM_CHIPS_Y
M_CHIP_SIZE_Y = 97
M_CHIP_SIZE_X = 60

def load_calibration_data(calib_path):
    """
    Description
        Reads in the calibration .tiff files and returns an image indexed by pixel location
        and trimbit setting. These files are photon counts for each pixel with the detector
        exposed to a given emission line, with this measurement repeated for each trimbit
        setting.

        The calibrate_detector routine stores the output of this code under the key "trimscan"
        in the calibration dictionary. This terminology is used throughout the code.
    Parameters:
        - calib_path = (string) Path to the folder containing the calibration images.
    Returns:
        - images = (int[x_index, y_index, trimbit]) Array containing one calibration
                image for each of the 64 possible trimbit settings.
    Credit:
        This function was originally written by Novimir Pablant and/or Jacob Maddox, then
        modified by Patrick VanMeter.
    """
    # Read in the params_b01_m01.txt file - left in for future expansion
    with open(path.join(calib_path, 'config.txt'), 'r') as f:
        params = f.readlines()

    # Container for the calibration data
    images = np.empty([M_SIZE_X, M_SIZE_Y, M_NUM_TRIM])

    # Load in the data one trimbit at a time
    for i in range(0, M_NUM_TRIM):
        try:
            filename = path.join(calib_path, 'scan_image_{:03d}.tif'.format(i))
            images[:, :, 63 - i] = tifffile.imread(filename).transpose()
        except:
            # Also try 5-digit labeling, the camserver default
            filename = path.join(calib_path, 'scan_image_{:05d}.tif'.format(i))
            images[:, :, 63 - i] = tifffile.imread(filename).transpose()

    return images

def get_chip_coords(image_x, image_y):
    """
    Description
        This function takes coordinates in the broader "image" (detector) reference frame and
        determines chat chip this point falls on as well as its local x,y coordinates in the
        chip reference frame.
    Parameters:
        - image_x = (int) X-coordinate of a point on the overall detector image
        - image_y = (int) Y-coordinate of a point on the overall detector image
    Returns:
        - chip_num = (int) The chip number on which the point (image_x, image_y) lies
        - chip_x = (int) The x-coordinate of the point in the frame of chip_num
        - chip_y = (int) The y-coordinate of the point in the frame of chip_num
    Credit:
        This function was originally written by Novimir Pablant. Original note:
            This is copied from pix_add in utils.c for the p2det detector. Given an x and y
            location on the detector, this will return the appropriate chip number and the x
            and y location on that given chip.
    """
    if image_y < M_SIZE_Y/2:
        chip_num = image_x/(M_CHIP_SIZE_X + 1)
        chip_x = (M_CHIP_SIZE_X+1)*(chip_num+1) - image_x - 2
        chip_y = image_y

        if chip_x < 0:
            chip_num = -1
    elif image_y == M_SIZE_Y/2:
        chip_num = -1

    else:
        chip_num = M_NUM_CHIPS/2 + image_x/(M_CHIP_SIZE_X + 1)
        chip_x = image_x % (M_CHIP_SIZE_X+1)
        chip_y = M_SIZE_Y - image_y - 1

        if chip_x >= M_CHIP_SIZE_X:
            chip_num = -1

    # Check if this is a valid chip.
    if chip_num < 0:
        chip_y = -1
        chip_x = -1

    return chip_num, chip_x, chip_y

# Data for the line energies
energies = {'Zr': 2.04,   'Mo': 2.29,   'Ag': 2.98, 'In': 3.29, 'Ti': 4.51,  'V': 4.95,
            'Cr': 5.41,   'Fe': 6.40,   'Cu': 8.05, 'Ge': 9.89, 'Br': 11.92, 'Y': 14.95,
            'MoK': 17.48, 'AgK': 22.16, 'Sn': 25.27}

def get_line_energy(elements):
    """
    Return the appropriate energies for the supplied elements.
    """
    elem_en = np.array([energies[elem] for elem in elements])
    return elem_en