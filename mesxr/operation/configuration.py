#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Package: mesxr.operation
Module: configuration
Author: Patrick VanMeter
Affiliation: Department of Physics, University of Wisconsin-Madison
Last Updated: January 2019

Description:
    This module contains methods to facilitate the generation of trimbit configurations. This
    includes a number of methods to generate different types of threshold configurations.
Usage:
    TBD
"""
import os
import numpy as np
from PIL import Image

def configure_detector(det, threshold_map, config_dir, config_name='default'):
    """
    Description:
        Produce a set of autotrim files to configure the detector for a provided
        threshold map.
    Parameters:
        - det - The calibrated Pilatus_Detector object.
        - threshold_map - A 2D array [N_pix_x, N_pix_y] of threhsolds in keV.
        - config_dir = The directory to save the results in.
        - config_name = The name for the calibration.
    Returns:
        - (none)
    """
    filepath = os.path.join(config_dir, config_name)
    config_det = False

    # Check if the configuration already exists
    if os.path.isdir(filepath):
        user_input = raw_input('This configuration already exists. Overwrite? (y/n): ')
        if user_input.lower()[0] == 'y':
            config_det = True
    else:
        try:
            os.mkdir(filepath)
            config_det = True
        except:
            print('ERROR: Could not create directory. Please check permissions.')

    if config_det:
        # Output the main settings files
        det.write_settings_file(filepath)
        det.write_autotrim_files(threshold_map, filepath)
        det.write_bad_pixels_file(filepath)

        # Also save the trimbit and threshold maps for future reference
        tbit_map = det.get_trimbit_map(threshold_map)
        save_tiff(tbit_map, os.path.join(filepath, 'trimbits.tif'), mode='I')
        save_tiff(threshold_map, os.path.join(filepath, 'thresholds.tif'), mode='F')

def save_tiff(image, filename, mode='F'):
    """
    Quickly save a 2D array to a .tif image in either an float (mode='F') or integer (mode='I') mode.
    This method avoids and problems with image scaling that other built-in algorithms might have.
    """
    data = np.asarray(image)

    if mode == 'F':
        data32 = data.astype(np.float32)
        image = Image.frombytes(mode, data32.shape, data32.transpose().tostring())
    if mode == 'I':
        data32 = data.astype(np.uint32)
        image = Image.frombytes(mode, data32.shape, data32.transpose().tostring())

    image.save(filename)
    print('Saved {0:} to file.'.format(truncate_str(filename)))

def truncate_str(string):
    """
    A useful function to make sure filename strings in the console don't become too unwieldy.
    """
    return string[:50] + '...' + string[-25:] if len(string) > 80 else string

# ------------------------------------------------- Specific Pixel Maps -------------------------------------------------

def uniform_treshold_map(det, threshold):
    """
    Returns a uniform threshold map of the appropriate dimensions. The specified threshold is a float in keV.
    Gaps between the chips are skipped over. This should work well for any multiple of 4 thresholds.
    """
    return np.ones(det.global_pixel_dims)*threshold

def ME_horiz_stripes_map(det, thresholds):
    """
    Generates a threshold map based on repeating the supplied columns (in keV).
    """
    threshold_map = np.zeros(det.global_pixel_dims)
    num_energies = len(thresholds)
    num_cols = int(det.pixel_dims[0]*det.chip_dims[0] / num_energies)

    # Cycle through pixels, skipping over the boundary columns
    dummy_index = 0
    for x_base in range(num_cols):
        for en_index, thresh in enumerate(thresholds):
            chip_num = det.pixels[x_base*num_energies + en_index + dummy_index, 0].chip
            if chip_num == -1:
                dummy_index += 1
            threshold_map[x_base*num_energies + en_index + dummy_index, :] = thresh

    return threshold_map

def ME_vert_stripes_map(det, thresholds):
    """
    Generates a threshold map based on repeating the supplied columns (in keV).
    """
    midpoint = 97
    threshold_map = np.zeros(det.global_pixel_dims)
    num_energies = len(thresholds)
    
    for y in range(det.global_pixel_dims[1]):
        if y < midpoint:
            threshold_map[:, y] = thresholds[y%num_energies]
        elif y > midpoint:
            threshold_map[:, y] = thresholds[(y-1)%num_energies]

    return threshold_map

def ME_metapixel(det, metapixel, max_val=6.5, dim=4):
    """
    Generates a clever mapping of repeating 4x4 metapixels.
    """
    # Use the input metapixel to generate a chip map
    chip_map = np.zeros([60, 97])

    for x in range(60):
        for y in range(97):
            if y == 0:
                chip_map[x,y] = max_val
            else:
                chip_map[x,y] = metapixel[x%dim, (y-1)%dim]

    chip_map_mirror = np.flip(chip_map, 1)

    # Now use this to map onto the overall threshold map
    Nx, Ny = det.global_pixel_dims
    threshold_map = np.zeros([Nx, Ny])

    for chip in det.modules[0,0].chips.ravel():
        cx, cy = chip.pixel_dims
        for x in range(cx):
            for y in range(cy):
                if chip.number < 8:
                    threshold_map[chip.pixels[x,y].coords] = chip_map[x,y]
                else:
                    threshold_map[chip.pixels[x,y].coords] = chip_map_mirror[x,y]

    return threshold_map