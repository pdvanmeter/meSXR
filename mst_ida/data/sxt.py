"""
"""
from __future__ import division
import numpy as np
import scipy as sp
import pidly

from mst_ida.utilities.structures import AttrDict

# ---------------------------------------------------- IDL Wrapper ----------------------------------------------------

def load_brightness(shot_num, t_start=8.0, t_end=28.0, delta=0.1, smooth=10.0, exclude=[20,59,60], adj=None, cmd_str=''):
    """
    Function: st = load_brightness(shot_num, t_start, t_end, delta, smooth, exclude, adj, cmd_str)
    This version of load_brightness interfaces directly with the IDL implementation, via the pidly interface
        (see the signals README for instructions on setting this up).
    Inputs:
        - shot_num = [INT] The MST shot ID for the desired set of data
        - t_start = [FLOAT] The start time for the desired interval of SXR data.
        - t_end = [FLOAT] The end time for the desired interval of SXR data.
        - delta = [FLOAT] The desired sampling window for SXR data. Set to None to get the full signal. Note that
            sampling error will not be available in that case.
        - smooth = [FLOAT] The size of the smoothing window (10.0 is standard).
        - exclude = [[INT]] List of logicals which were excluded in the data file. By defaults excludes the NickAl2
            channels. For older data make sure to exclude 69
        - adj = [[FLOAT]] The time adjustment due to SXR trigger settings. Set to 0 for data past mid-2016
        - cmd_str = [STR] Additional string to append to the IDL command.
    Outputs:
        - st['key'] = [DICT] Nested dictionary containing the SXR tomography diagnostic data, indexed by camera label.
    """
    # Access (and initialize, if needed) the pidly object and assemble the command string
    idl = pidly.IDL()
    idl_str = "staus = sxr_mst_get_signals(St, /ms, shot="+ str(shot_num) + ", tst = " + str(t_start) + ", excl = " + str(exclude)
    idl_str += ", tend = " + str(t_end)

    # Optional keywods
    if delta is not None:
        idl_str += ", delta = " + str(delta)

    if smooth is not None:
        idl_str += ", sm = " + str(smooth)

    if len(cmd_str) == 0:
        idl_str += ")"
    else:
        idl_str += ", " + cmd_str + ")"
    
    # Create the structure in IDL and begin exporting the contents
    idl(idl_str)
    sxr_data = idl.ev('st.bright.data', use_cache=True)
    sxr_impact = idl.ev('st.bright.prel', use_cache=True)
    sxr_angles = idl.ev('st.bright.phi', use_cache=True)
    sxr_noise = idl.ev('st.bright.off_str.sxr_r_noise', use_cache=True)
    sxr_time = idl.ev('st.bright.time', use_cache=True)

    if delta is not None:
        sxr_error = idl.ev('st.bright.err', use_cache=True)

    # Organize the data by probe label and thick/thin convention. Now automated to account for excluded probes
    filt_list = ['A thick', 'B thick', 'C thick', 'D thick', 'A thin', 'B thin', 'C thin', 'D thin']
    
    brightness = {}
    impact_p = {}
    impact_angle = {}
    sigma = {}
    off_noise = {}
    logicals_inc = {}

    # Manual index in order to allow exclusion of specified logicals
    index = 0

    for filt in filt_list:
        base_logical = filt_list.index(filt)*10 + 1
        
        data = []
        error = []
        impact = []
        angles = []
        noise = []
        logs = []
        for logical in range(base_logical, base_logical + 10):
            if logical not in exclude:
                data.append(np.transpose(sxr_data[:,index]))
                impact.append(sxr_impact[index])
                angles.append(sxr_angles[index])
                noise.append(sxr_noise[index])
                logs.append(logical)

                if delta is not None:
                    error.append(np.transpose(sxr_error[:,index]))
                else:
                    error.append([])

                index += 1
        
        # Store into the dictionary
        brightness[filt] = np.array(data)
        sigma[filt] = np.array(error)
        impact_p[filt] = np.array(impact)
        impact_angle[filt] = np.array(angles)
        off_noise[filt] = np.array(noise)
        logicals_inc[filt] = logs

    # Also store same basic configuration info - logical indexing is fine
    config ={
        'filters':idl.ev('st.bright.FILTERBE_THICK', use_cache=True),
        'alpha':idl.ev('st.bright.alfa', use_cache=True),
        'gain':idl.ev('st.bright.gain', use_cache=True),
        'insertion':idl.ev('st.bright.insertion', use_cache=True)
    }

    # Assemble these into a single dictionary
    st = {
            'bright':brightness,
            'p':impact_p,
            'phi':impact_angle,
            'sigma':sigma,
            'noise':sigma,
            'logical':logicals_inc,
            'shot':shot_num,
            'time':sxr_time,
            'config':config
        }

    # Close the IDL instance to prevent runaway processes
    idl.close()

    return AttrDict(st)

def get_sxt_data_point(shot, index, t_start=0.5, t_end=40.5, delta=1.0):
    """
    """
    struct = load_brightness(shot, t_start=0.5, t_end=30.5, delta=1.0)
    sxt_data = {label:struct['brightness'][label][:,index] for label in struct['brightness'].keys()}
    sxt_sigma = struct['sigma']
    sxt_p = struct['p']
    time = struct['time'][index]
    print('Retriving SXT data for t = {0:.2f} to t = {1:.2f} ms.'.format(time-delta/2, time+delta/2))
    return sxt_data, sxt_sigma, sxt_p

def filter_list(keyword):
    """
    Function: filt_list = filter_list(keyword)
    This function returns some lists that may be useful for looping over certain sets of data. It simply makes
        the code easier to read.
    Inputs:
        - keyword = 'all', 'thick', or 'thin'. Which set of data are you going to iterate over?
    Outputs:
        - filt_list = The desired list of filter keys. If an incorrect keyword is used, this is set to 0.
    """
    if keyword == 'all':
        filt_list = ['A thick', 'B thick', 'C thick', 'D thick', 'A thin', 'B thin', 'C thin', 'D thin']
    elif keyword == 'thick':
        filt_list = ['A thick', 'B thick', 'C thick', 'D thick']
    elif keyword == 'thin':
        filt_list = ['A thin', 'B thin', 'C thin', 'D thin']
    else:
        filt_list = 0

    return filt_list

def phase_diff(data_1, data_2, delta=0.01, radians=False):
    """
    Function: (delta_theta, max_freq) = phase_diff(data_1, data_2)
    Dtermines the phase of the supplied data by performing and FFT and determining the complex phase of the
        coefficients at the max frequency. This is then compared relative to the phase angle for the other set of data.
    Inputs:
        - data_1 = First set of periodic sampled data. Might be the source signal.
        - data_2 = Second set of periodic sampled data. Might be the amplified photodetector signal.
        - delta= The spacing of the smootehed sxr data. Defaults to 0.01 ms.
        - radians = Boolean flag. If set to True, the return value is left in radians.
    Outputs:
        - delta_theta = angle(data_1[f_max]) - angle(data_2[f_max]). The phase that data_1 leads data_2 by, in degrees.
        - freq_max = The frequency that the pahse is evaluated at. By convention this is taken from data_2. In kHz.
    """
    fft_1 = np.fft.rfft(data_1)
    fft_2 = np.fft.rfft(data_2)

    # Determine index of max frequency
    n_max_1 = np.argmax(np.abs(fft_1))
    n_max_2 = np.argmax(np.abs(fft_2))

    # Determine the phase angles
    theta_1 = np.arctan2(np.imag(fft_1[n_max_1]), np.real(fft_1[n_max_1]))
    theta_2 = np.arctan2(np.imag(fft_2[n_max_2]), np.real(fft_2[n_max_2]))

    # Return the difference
    if radians:
        delta_theta = theta_1 - theta_2
    else:
        delta_theta = np.degrees(theta_1 - theta_2)

    # Determine the frequency
    sxr_freq = np.fft.rfftfreq(n=len(data_1), d=delta)

    return (delta_theta, sxr_freq[n_max_2])

# ---------------------------------------------- Tomography ----------------------------------------------

def invert_brightness(shot_num, t_start=8.0, t_end=28.0, delta=0.1, smooth=10.0, exclude=[20,59,60],
        thick=False, thin=False, **kwargs):
    """
    Perform a tomographic inversion using the Cormack-Bessel technique. Makes use of Paolo's IDL library.
    Inputs:
        - shot_num = [INT] The MST shot ID for the desired set of data
    Optional:
        - t_start = [FLOAT] The start time for the desired interval of SXR data.
        - t_end = [FLOAT] The end time for the desired interval of SXR data.
        - delta = [FLOAT] The desired sampling window for SXR data.
        - smooth = [FLOAT] The size of the smoothing window (10.0 is standard).
        - exclude = [[INT]] List of logicals which were excluded in the data file. By default excludes the NickAl2
            channels. For older data make sure to still exclude 69.
        - thick = [BOOL] Include only thick filter data.
        - thin = [BOOL] Include only thick filter data.
        - kwargs = Additional keywords are used to change the inversion

    Inversion keywords: Other keyword arguments will be used to alter the inversion options array passed
        directly to the IDL routine. The allowed arguments, according to the IDL documentation, are:
        ka = $
        [ $
          'name=Cormack'          # inversion method
          'base=Bessel'           # radial function base
          'matname=matrix.dat'    # not important
          'mc=1'                  # n. of angular (poloidal) cos components
          'ms=1'                  # n. of angular (poloidal) sin components
          'ls=6'                  # n. of radial components
          'svd_tol=0.100'         # svd threshold
          'p_ref=[0.0,0,0,0.0]'   # coordinates of the origin of the axis
                                  # where the inversion will be performed
          'n_nch=5'               # n. of added edge lines of sight
          'mst'=1                 # specifies that we are working on Mst SXR data
        ]
        
        Note: The values of p_ref are [x0, y0, z0], signifying the location of the magnetic axis. These
            values are measured in meters, and z0 will typically be set to zero. The defaults arguments are
            p_ref = [0.06, 0.0, 0.0], signifying a 6cm Shafranov shift.
        Note: The value of svd_tol is the tolerance for stopping the iterative SVD inversion process. The
            default is 0.06, but can be increased (up to 0.1) as needed to smooth the reconstruction.
    """
    # Load the data into idl
    idl = pidly.IDL()
    idl_str =  "staus = sxr_mst_get_signals(St, /ms, shot="+ str(shot_num) + ", tst = " + str(t_start)
    idl_str += ", excl = " + str(exclude) + ", tend = " + str(t_end)
    idl_str += ", delta = " + str(delta) + ", sm = " + str(smooth) + ')'
    idl(idl_str)

    # Format keywords - check for supplied arguments and change if necessary
    ka = {
        'name':'Cormack',
        'base':'bessel',
        'matname':'matrix.dat',
        'mc':1,
        'ms':1,
        'ls':6,
        'svd_tol':0.06,
        'p_ref':[0.06,0.00,0.0]
    }

    if len(kwargs) > 0:
        for key,value in kwargs.items():
            ka[key] = value

    ka_str = 'ka = ' + str(['{0:}={1:}'.format(key,value) for key,value in ka.items()])
    idl(ka_str)
    idl_str = "st_out = sxr_MST_get_emiss(st, st_emiss=st_emiss, status=status, ka=ka"
    if thick:
        idl_str += ', /thick'
    elif thin:
        idl_str += ', /thin'
    idl_str += ')'

    # Do the inversion
    idl(idl_str)

    # Import the data into python
    results = {
        'emiss':idl.ev('st_emiss.emiss', use_cache=True).T,
        'time':idl.ev('st_emiss.t', use_cache=True),
        'xs':idl.ev('st_emiss.x_emiss', use_cache=True),
        'ys':idl.ev('st_emiss.y_emiss', use_cache=True),
        'major':idl.ev('st_emiss.majr'),
        'radius':idl.ev('st_emiss.radius'),
        'kwargs':ka
    }

    # Close the IDL instance to prevent runaway processes
    idl.close()

    return AttrDict(results)