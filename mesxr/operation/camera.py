#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Package: mesxr.operation
Module: camera
Author: Patrick VanMeter
Affiliation: Department of Physics, University of Wisconsin-Madison
Last Updated: January 2019

Description:
    This module is an updated version of the camera operations code contained in my old
    PILATUS library. This version has been modified to work with the new naming conventions
    for pixel configurations and contains other feature updates. In general, use this version.

    This code is made specifically for the PILATUS3 100K and might require some modifications
    for use with other detector models.
Usage:
    TBD
Acknowledgements:
    - Novimir Pablant for the original IDL code this module is loosely based upon.
    - Luis Felipe Delgado-Aparicio, for heading the PPPL/MST collaboration.
    - Daniel Den Hartog and Lisa Reusch, for advising me.
"""
import socket
import time
import os
import re
import numpy as np
import tifffile as tif
import MDSplus
import mesxr.calibration.utilities as util

# Module-level constants for the camera - these should eventually be read in
COMP_NAME = 'dec1424'
PORT = 41234
IP_ADDR = '127.0.0.1'
NUM_CHIPS = 16
M_SIZE_Y = 195
M_SIZE_X = 487
BASE_IMAGE_PATH = '/home/det/p2_det/images/MST_data'
TAKE_DATA_ADDR = 'aurora.physics.wisc.edu'

class camserver():
    """
    An instance of this object is helpful in facilitating remote operation
    of the camera via the camserver. This class is currently under development.
    """
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.online = False
        self.set_output_mode('verbose')
        self.set_timeout(None)
        self.connect(IP_ADDR, PORT)

    def __del__(self):
        if self.online:
            print('Error detected. Automatically closing socket connection.')
            self.disconnect()

    def connect(self, ip_addr, port):
        """
        Establish a connection to the PILATUS detector.
        """
        try:
            self.sock.connect((ip_addr, port))
            self.online = True
            print('Camserver connection established with ' + COMP_NAME + '.')
        except:
            print('Camserver connection cannot be established. Check settings.')
            self.online = False

    def disconnect(self):
        """
        Close the connection with the PILATUS detector.
        """
        if self.online:
            self.sock.close()
            self.online = False
            print('Disconnected from the camserver.')
        else:
            print('Alread disconnected from the camserver.')

    def execute(self, command):
        """
        Exectute a command on the camserver and wait for a return code.
        """
        if not self.quiet:
            print('CMD >> ' + command)

        if self.online:
            self.sock.send(command + '\n')
            code, message = self.recieve()
        else:
            code = -1
            message = 'OFFLINE'
            print('Must be online to execute commands.')

        if not self.quiet:
            print('p3det >> (' + str(code) + ') ' + message)

        return code, message

    def wait(self, end_code):
        """
        Wait until a specified end code is returned.
        """
        code, message = self.recieve()
        while code != end_code and code != -1:
            code, message = self.recieve()

        if not self.quiet:
            print('p3det >> (' + str(code) + ') ' + message)

        return code, message

    def recieve(self):
        """
        Read the next socket output, out to the \x18 message termination code.
        Consider implementing a timeout function if this becomes an issue.
        """
        if self.online:
            end_of_message = False
            message = ''
            while not end_of_message:
                next_char = self.sock.recv(1)
                if next_char == '\x18':
                    end_of_message = True
                else:
                    message += next_char

            # Remove the numerical code from the start of the message
            code = int(message.split()[0])
            message = ' '.join(message.split()[1:])
        else:
            code = -1
            message = 'OFFLINE'
            print('Must be online to communicate with the camserver.')

        return code, message

    def set_timeout(self, timeout):
        self.timeout = timeout
        self.sock.settimeout(self.timeout)
        print('Camserver timeout set to ' + str(timeout) + '.')

    def set_output_mode(self, mode):
        """
        Set mode to either "verbose" or "quiet". This controls whether camera commands and
        returned messages are output to the console.
        """
        if mode.lower() == 'quiet':
            self.quiet = True
        elif mode.lower() == 'verbose':
            self.quiet = False

# -----------------------------------------------------------------------------------------

def arm_detector(shot=0, mode='cycle', load_mds=False, write_mds=True,
                 quiet=True, timeout=None, retrigger=False, rate_corr=True, acq_time=10,
                 config_name='ppcd_8_color', calibration_name='midE',
                 n_frames=30, exp_period=0.002, exp_time=0.001, delay=0,
		         image_path='/home/det/p2_det/images/MST_data'):
    """
    Description
    ===============
        The main routine of
    Parameters:
    ===============
        - shot = (int) The shot number corresponding to the given plasma discharge. This
            is used in the resulting image filename.
        - mode = (string) One of the following strings:
            'trigger' - Exposure is initiated by external trigger one time.
            'manual' - Exposure initiated by this script one time.
            'gate' - Exposure is controlled by external gate signal one time.
            'cycle' - Exposure is initiated by external trigger repeatedly. The shot number
                is also overridden by the current shot in the MDSplus tree. This is the mode
                that should be set at the beginning of a run day. Use a keyboard interrupt
                (CRTL+C) to end the loop and close the camserver connection.
        - load_mds = (bool) Set to true to load default exposure settings from the MDSplus
            tree. This overrides any values set manually.
        - write_mds = (bool) When True ME-SXR data is written to the MDSplus tree. When False
                data is only written to the specified output directory.
        - quiet = (bool) Set to true to print all camserver outputs to the console.
        - timeout = (float) Set the camera to time out if an exposure is not taken within the
                specified number of seconds. Default is None, which is no limit.
        - retrigger = (bool) Set to True to enable instant retrigger technology. This increases
                the max count rate but may cause problems with the calibration.
        - rate_corr = (bool) Enable to use the rate correction files loaded during calibration. This
                is not needed used running in the 'dectris' calibration mode.
        - acq_time = (int) Time in seconds to wait while cycling after data is written in the tree
                before reading in the next shot number. This may need to be extended if a long
                many diagnostics are running.
        - calibration_name = (string) The name used for the desired calibration. Use 'dectris'
                to load factory calibrations
        - config_name = (string) The name associated with the desired trimbit configuration. For
               	the 'dectris' calibration this is the command passed to the camserver (e.g.
                'setthr 4000').
        - n_frames = (int) The number of exposures to take once initialized.
        - exp_period = (double) The cycle time for an exposure period, in seconds. This is
            	the time from the start of one exposure until the start of the next exposure.
        - exp_time = (double) The length of time for each exposure, in seconds. This should
            	be shorter than exp_period.
	- image_path = (str) Change this to save files in a different directory. This is generally
	    		useful for testing.
        Returns:
    ===============
        - success = (bool) True if the function executes successfully, false if any errors
            are thrown.
    """
    if load_mds:
        print('Loading default exposure settings from the MDSplus tree.')
        try:
            mesxr = MDSplus.Tree('mst_me_sxr', shot=-1)
            n_frames = mesxr.getNode(r'.CONFIG:N_IMAGES').getData().data()
            exp_time = mesxr.getNode(r'.CONFIG:EXPOSUR_TIME').getData().data()
            exp_period = mesxr.getNode(r'.CONFIG:EXPOSUR_PER').getData().data()
            delay = mesxr.getNode(r'.CONFIG:DELAY').getData().data()

            print('n_frames: ' + str(n_frames))
            print('exp_time: ' + str(exp_time))
            print('exp_period: ' + str(exp_period))
            print('delay: ' + str(delay))
        except:
            print('ERROR: Could not load settings from the model tree.')

    # Make sure the exposure period is permissible
    if exp_period <= exp_time + 0.001:
        exp_period = exp_time + 0.001

    filename = 's' + str(shot) + '.tif'
    settrims_prefix = 'autotrim_'
    setdacs_filename = 'setdacs_b01_m01.dat'
    base_shot = int(shot/1000)

    config_path = os.path.join('/home/det/meSXR/configurations/', calibration_name, config_name)
    setdacs_path = os.path.join(config_path, setdacs_filename)

    # Write settings to the model tree, if desired
    if write_mds:
        set_model_tree(n_frames, exp_period, exp_time, delay, config_path, setdacs_path)

    # Camserver commands
    command_nimages = 'nimages ' + str(n_frames)
    command_expperiod = 'expperiod ' + str(exp_period)
    command_exptime = 'exptime ' + str(exp_time)
    command_delay = 'delay ' + str(delay)

    # Select camserver exposure mode - see function description
    if mode.lower() == 'trigger':
        command_exposure = 'ExtTrigger ' + filename
    elif mode.lower() == 'manual':
        command_exposure = 'Exposure ' + filename
    elif mode.lower() == 'gate':
        command_exposure = 'ExtEnable ' + filename
    elif mode.lower() == 'cycle':
        command_exposure = 'ExtTrigger ' + filename
    else:
        print('ERROR: Mode selection not recognized. Defaulting to manual operation.')
        command_exposure = 'Exposure ' + filename

    # Extablish the connection to the camserver
    try:
        print('Attempting to open socket connection on ' + COMP_NAME + '.')
        cam = camserver()

        # Configure client settings
        if quiet:
            cam.set_output_mode('quiet')

        cam.set_timeout(timeout)

        if calibration_name == 'dectris':
            cam.execute('setCu')
            cam.execute(config_name)

        else:
            # Set up camera settings
            cam.execute('dacoffset off')
            cam.execute('LdCmndFile ' + setdacs_path)

            # Load the trimbit configuration
            for index in range(NUM_CHIPS):
                settrims_filename = settrims_prefix + 'b01_m01_c{0:02d}.dat'.format(index)
                cam.execute('prog b01_m01_chsel 0x0')
                cam.execute('trimfromfile ' + os.path.join(config_path, settrims_filename))

            # Reset the readout chip selection
            cam.execute('prog b01_m01_chsel 0xffff')

            # Manually disable flat-field correction
            cam.execute('ldflatfield 0')

            # Set the pixel rate correction
            if rate_corr:
                print('Enabling rate correction.')
                cam.execute('ratecorrlutdir ContinuousStandard_v1.1')

        # Set up remaining exposure settings
        cam.execute(command_nimages)
        cam.execute(command_expperiod)
        cam.execute(command_exptime)

        if delay != 0:
            cam.execute(command_delay)

        # Disable instant retrigger
        if not retrigger:
            cam.execute('setretriggermode 0')

        if mode == 'cycle':
            # To get the current shot from Aurora
            conn = MDSplus.Connection(TAKE_DATA_ADDR)

            cycle = True
            while cycle:
                try:
                    # Get the next shot number from the tree
                    shot = int(conn.get('current_shot("mst")')) + 1
                    base_shot = int(shot/1000)
                    filename = 's' + str(shot) + '.tif'
                    command_exposure = 'ExtTrigger ' + filename

                    output_path = os.path.join(image_path, str(base_shot), str(shot))
                    command_imgpath = 'imgpath ' + output_path
                    cam.execute(command_imgpath)

                    # Write out the trimbit configuration
                    cam.execute('imgmode p')
                    cam.execute('imgonly trimbit.tif')

                    print('Waiting for trigger for shot ' + str(shot) + '.')
                    cam.execute(command_exposure)
                    cam.wait(7)
                    print('Data collected for shot ' + str(shot) + '.')

                    # Write data to tree if desired
                    if write_mds:
                        write_to_tree(shot, n_frames, exp_period, exp_time, delay, output_path)

                    # Wait an appropriate amount of time for the shot to increment
                    time.sleep(acq_time)
                    
                except KeyboardInterrupt:
                    cam.execute('k ')
                    print('\nME-SXR trigger manually disarmed. Cylcing stopped.')
                    cycle = False
        else:
            # Use the supplied shot number for a single exposure
            output_path = os.path.join(image_path, str(base_shot), str(shot))
            command_imgpath = 'imgpath ' + output_path
            cam.execute(command_imgpath)

            # Write out the trimbit configuration
            cam.execute('imgmode p')
            cam.execute('imgonly trimbit.tif')

            print('Beginning exposure.')
            cam.execute(command_exposure)
            cam.wait(7)

            cam.disconnect()
            return True
    except:
        print('ERROR: Camera comunication error. Check settings.')
        return False

def write_to_tree(shot, n_frames, exp_period, exp_time, delay, imgpath):
    """
    This should be run after every data acquisition period. Given a shot number and time parameters,
    this loads in the output tiff files, generates a time base, and loads the data into the tree.
    """
	# Load shot data from the output tiff files
    data = np.zeros([M_SIZE_X, M_SIZE_Y, n_frames])
    for index in range(n_frames):
        fname = os.path.join(imgpath, 's{0:010d}_{1:05d}.tif'.format(shot, index))
        try:
            data[:, :, index] = tif.imread(fname).T
        except:
            print('ERROR: Data acquisition failed for shot {0:010d} frame {1:05d}. Setting to -1.'.format(shot, index))
            data[:, :, index] = -1

    # Generate the time array
    time = ( np.array([n*exp_period + exp_time/2. for n in range(n_frames)]) + delay )*1000.

    # Write data and settings to mdsPlus
    try:
        # Connect to the tree
        #mesxr = MDSplus.Tree('me_sxr_ext', shot, 'NORMAL')
        mesxr = MDSplus.Tree('me_sxr_ext', shot, 'EDIT')

        # Write image data with time points
        imagesNode = mesxr.getNode(r'.ME_SXR_EXT:IMAGES')
        compiled_data = MDSplus.Data.compile("BUILD_SIGNAL($1,, $2)", data, time)
        imagesNode.putData(compiled_data)

        # Write camera settings and configuration data
        mesxr.write()
        print('Data for shot {0:10d} written to the tree.'.format(shot))
    except Exception as e:
        print('ERROR: Writing to MDSplus tree failed.')
        print(str(e))

def set_model_tree(n_frames, exp_period, exp_time, delay, config_path, setdacs_path):
    """
    This function should be executed once at the beginning of data acquisition. It set various parameters
    in the model tree which will not be changed until the next iteration of the acquisition loop.
    """
    # Read in the detector voltage parameters
    with open(setdacs_path) as dacs_file:
        v_cmp = np.zeros(16)

        for line in dacs_file:
            if 'VTRM' in line:
                v_trm = float(line.split()[-1])
            elif 'VCMP' in line:
                # Extract the chip number and the setting value - sorry this is fairly convoluted
                line_elements = re.split('_|  |  ', line)
                vcmp_chip_index = int(line_elements[-2].split('VCMP')[-1])
                vcmp_chip_value = float(line_elements[-1])
                v_cmp[vcmp_chip_index] = vcmp_chip_value
            elif 'VCCA' in line:
                v_cca = float(line.split()[-1])
            elif 'VRF' in line and not 'VRFS' in line:
                v_rf = float(line.split()[-1])
            elif 'VRFS' in line:
                v_rfs = float(line.split()[-1])
            elif 'VCAL' in line:
                v_cal = float(line.split()[-1])
            elif 'VDEL' in line:
                v_del = float(line.split()[-1])
            elif 'VADJ' in line:
                v_adj = float(line.split()[-1])

    # Read in the trimbit and threshold maps
    trimbit_map = tif.imread(os.path.join(config_path, 'trimbits.tif')).T
    threshold_map = tif.imread(os.path.join(config_path, 'thresholds.tif')).T

    # Generate the bad pixel map
    bad_pixel_map = get_bad_pixel_map(config_path, trimbit_map.shape)

    # Write the data to the tree
    try:
        # Connect to the tree
        mesxr = MDSplus.Tree('mst_me_sxr', -1, 'EDIT')

        # Write the exposure timing settings
        n_images_node = mesxr.getNode(r'.CONFIG:N_IMAGES')
        n_images_node.putData(n_frames)
        exp_period_node = mesxr.getNode(r'.CONFIG:EXPOSUR_PER')
        exp_period_node.putData(exp_period)
        exp_time_node = mesxr.getNode(r'.CONFIG:EXPOSUR_TIME')
        exp_time_node.putData(exp_time)
        delay_node = mesxr.getNode(r'.CONFIG:DELAY')
        delay_node.putData(delay)

        # Write the detector voltage settings
        vtrm_node = mesxr.getNode(r'.CONFIG:V_TRM')
        vtrm_node.putData(v_trm)
        vcmp_node = mesxr.getNode(r'.CONFIG:V_COMP')
        vcmp_node.putData(v_cmp)
        vcca_node = mesxr.getNode(r'.CONFIG:V_CCA')
        vcca_node.putData(v_cca)
        vrf_node = mesxr.getNode(r'.CONFIG:V_RF')
        vrf_node.putData(v_rf)
        vrfs_node = mesxr.getNode(r'.CONFIG:V_RFS')
        vrfs_node.putData(v_rfs)
        vcal_node = mesxr.getNode(r'.CONFIG:V_CAL')
        vcal_node.putData(v_cal)
        vdel_node = mesxr.getNode(r'.CONFIG:V_DEL')
        vdel_node.putData(v_del)
        vadj_node = mesxr.getNode(r'.CONFIG:V_ADJ')
        vadj_node.putData(v_adj)

        # Write the trimbit, threshold, and bad pixel maps
        threshold_node = mesxr.getNode(r'.CONFIG:E_THRESH_MAP')
        threshold_node.putData(threshold_map)
        trimbit_node = mesxr.getNode(r'.CONFIG:TRIMBIT_MAP')
        trimbit_node.putData(trimbit_map)
        bad_pix_node = mesxr.getNode(r'.CONFIG:BAD_PX_MAP')
        bad_pix_node.putData(bad_pixel_map)

        # Write camera settings and configuration data
        mesxr.write()
        print('Data for shot -1 written to the model tree.')

    except Exception as e:
        print('ERROR: Writing to MDSplus model tree failed.')
        print('Output: ' + str(e))

def initialize_model_tree(px_size=172, si_thick=450, be_thick=25):
    """
    This function only needs to be executed once, or when any of the physical parameters of the detector
    are changed.
    """
    try:
        # Connect to the tree
        mesxr = MDSplus.Tree('mst_me_sxr', -1, 'NORMAL')

        # Write data to the model tree
        px_size_node = mesxr.getNode(r'.CONFIG:PX_SIZE')
        px_size_node.putData(px_size)
        si_thick_node = mesxr.getNode(r'.CONFIG:SI_THICK')
        si_thick_node.putData(si_thick)
        filt_thick_node = mesxr.getNode(r'.GEOMETRY:FILT_THICK')
        filt_thick_node.putData(be_thick)

        # Write camera settings and configuration data
        mesxr.write()
        print('Model tree updated.')

    except Exception as e:
        print('ERROR: Writing to MDSplus model tree failed.')
        print('Output: ' + str(e))

def get_bad_pixel_map(config_dir, map_dims):
    """
    This function loads in the bad pixel map from the calibration directory and puts it in the expected format.
    It marks as bad the pixels contianed in the bad_pixels.csv file as well as the gap pixels.
    """
    bad_coords = np.loadtxt(os.path.join(config_dir, 'bad_pixels.csv'), delimiter=',').astype(int).tolist()
    
    bad_pixels = np.full(map_dims, False, dtype=bool)
    for x in range(map_dims[0]):
        for y in range(map_dims[1]):
            if [x,y] in bad_coords:
                bad_pixels[x,y] = True
            elif util.get_chip_coords(x,y)[0] == -1:
                bad_pixels[x,y] = True

    return bad_pixels
