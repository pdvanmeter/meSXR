"""
"""
from __future__ import division
import numpy as np
import scipy as sp
import MDSplus
import mst_ida.models.base.physical_profiles as phys
import mst_ida.models.base.geometry as geo

# Constants for the MST FIR diagnostic system
fir_chord_names =           ['N32', 'N24', 'N17', 'N09', 'N02', 'P06', 'P13', 'P21', 'P28', 'P36', 'P43']
fir_chord_radius = np.array([-32.0, -24.0, -17.0,  -9.0,  -2.0,   6.0,  13.0,  21.0,  28.0,  36.0,  43.0])/100.
fir_chord_angle =  np.array([255.0, 250.0, 255.0, 250.0, 255.0, 250.0, 255.0, 250.0, 255.0, 250.0, 255.0])
fir_chord_length = np.array([81.97, 92.26, 98.29, 102.4, 103.9, 103.3, 100.7, 95.14, 87.64, 75.04, 58.48])/100.
fir_chord_angle_rad = fir_chord_angle*np.pi/180.

class FIR(object):
    """
    A simple interface to unify the FIR model with the other models I have developed for this framework.
    """
    def __init__(self):
        self.channels = fir_chord_names
        self.radius = fir_chord_radius
        self.chords = [FIR_Chord(name) for name in fir_chord_names]
        
    def take_data(self, plasma):
        return np.array([chord.measure(plasma.ne) for chord in self.chords])/1e19

class FIR_Chord(object):
    """
    """
    def __init__(self, chord_name):
        self.name = chord_name
        index = fir_chord_names.index(chord_name)
        
        radius = fir_chord_radius[index]
        if radius > 0:
            self.impact_p = radius
            self.impact_phi = 0.0
        else:
            self.impact_p = -1.0*radius
            self.impact_phi = np.pi
        
        self.los = geo.line_of_sight(self.impact_p, self.impact_phi)
        self.length = fir_chord_length[index]
        
    def measure(self, ne_prof, num_pts=25):
        ells = np.linspace(-0.5, 0.5, num=num_pts)
        xs, ys = self.los.get_xy(ells)
        ne_xs = ne_prof(xs,ys)
        return np.trapz(ne_xs, x=ells)/self.length

def get_fir_model(plasma):
    """
    """
    fir_chords = [FIR_Chord(name) for name in fir_chord_names]
    return np.array([fir.measure(plasma.ne) for fir in fir_chords])/1e19
    
def get_fir_model_old(ne0, alpha, beta, delta_a=0.06, delta_h=0.01):
    """
    """
    ne_prof = phys.Electron_Density_Alpha(ne0, alpha=alpha, beta=beta, delta_a=delta_a, delta_h=delta_h)
    fir_chords = [FIR_Chord(name) for name in fir_chord_names]
    return np.array([fir.measure(ne_prof) for fir in fir_chords])/1e19

# ------------------------------------------------- 3D model -------------------------------------------------

class FIR_3D(object):
    """
    A simple interface to unify the FIR model with the other models I have developed for this framework.
    """
    def __init__(self):
        self.channels = fir_chord_names
        self.radius = fir_chord_radius
        self.chords = [FIR_Chord(name) for name in fir_chord_names]
        self.pluses = [angle == 255.0 for angle in fir_chord_angle]
        
    def take_data(self, eq3d):
        dens = np.zeros(len(self.chords))
        for index, (chord,plus) in enumerate(zip(self.chords, self.pluses)):
            if plus:
                dens[index] = chord.measure(eq3d.plasma['fir_p'].ne)
            else:
                dens[index] = chord.measure(eq3d.plasma['fir_m'].ne)
        return dens/1e19
