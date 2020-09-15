"""
This module contains
"""
import numpy as np
import scipy as sp
import h5py

def stringify_float(val):
    # Helper function for loading file names
    return '{0:d}p{1:d}'.format(int(val), int(10*val - int(val)*10))

def get_emiss(thresholds, impurities=['Al', 'C', 'O', 'N', 'B', 'Ar'], fname='sxr_emission_100um_corrected.h5'):
    """
    """
    fname = '/home/pdvanmeter/data/ADAS/' + fname
    file = h5py.File(fname, 'r')

    # Load the axis arrays
    Te_set = file['/Information/Dimensions/te'][...]
    ne_set = file['/Information/Dimensions/ne'][...]
    ln_n0_set = file['/Information/Dimensions/ln_n0'][...]

    # Load the emission databases for impurities at the specified thresholds and make the interp functions
    avg_Z = {}
    emiss = {ion:{} for ion in impurities}

    for ion in impurities:
        for Ec in thresholds:
            emiss_db = file['/{0:}/emiss/emiss_{1:}'.format(ion, stringify_float(Ec))][...]
            emiss[ion][Ec] = sp.interpolate.RegularGridInterpolator((Te_set, ne_set, ln_n0_set), emiss_db,
                                                                    bounds_error=False, fill_value=0.0)

        avg_Z_db = file['/{0:}/avg_Z'.format(ion)][...]
        avg_Z[ion] = sp.interpolate.RegularGridInterpolator((Te_set, ne_set, ln_n0_set), avg_Z_db, bounds_error=False, fill_value=0.0)

    # Also include deuterium
    emiss['D'] = {}
    for Ec in thresholds:
        emiss_D = file['/D/emiss/emiss_{0:}'.format(stringify_float(Ec))][...]
        emiss['D'][Ec] = sp.interpolate.RegularGridInterpolator((Te_set, ne_set, ln_n0_set), emiss_D, bounds_error=False, fill_value=0.0)

    file.close()
    return emiss, avg_Z