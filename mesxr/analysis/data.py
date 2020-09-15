"""
This module contains the primary methods used to load data from the ME-SXR tree into
forms which are easy to process.
"""
import os
import numpy as np
import scipy as sp
import scipy.stats
import MDSplus as mds
import pilatus.calibration as calib
import pilatus.configuration as cfg

CHIP_SIZE_X = 60
NUM_CHIPS_X = 8
NUM_PIX_Y = 195
NUM_PIX_X = 487

def load_raw_data(shot, avg_bad_pix=True, avg_axis=0, remove_edges=False, key='MST'):
    """
    This function loads the raw image data for a shot from the tree. If projection=True then a
    summation is also performed along the specified axis.
    
    TODO: Allow additional masking of specified pixels (i.e. outer pixels).
    """
    # Get the image data
    mesxr = mds.Tree('mst_me_sxr', shot, 'READONLY')
    mesxr_ext = mds.Tree('me_sxr_ext', shot, 'READONLY')
    images_node = mesxr_ext.getNode(r'.ME_SXR_EXT.IMAGES')
    images = images_node.getData().data()
    time = images_node.dim_of().data()
    
    # Get the config data
    exp_time = mesxr.getNode(r'.CONFIG:EXPOSUR_TIME').getData().data()
    exp_period = mesxr.getNode(r'.CONFIG:EXPOSUR_PER').getData().data()
    exp_delay = mesxr.getNode(r'.CONFIG:DELAY').getData().data()
    n_images = mesxr.getNode(r'.CONFIG:N_IMAGES').getData().data()
    thresholds = mesxr.getNode(r'.CONFIG:E_THRESH_MAP').getData().data()
    thresholds = np.around(thresholds.astype(float), decimals=1)
    
    vtrm = mesxr.getNode(r'.CONFIG:V_TRM').getData().data()
    vcmp = mesxr.getNode(r'.CONFIG:V_COMP').getData().data()
    vcca = mesxr.getNode(r'.CONFIG:V_CCA').getData().data()
    vrf = mesxr.getNode(r'.CONFIG:V_RF').getData().data()
    vrfs = mesxr.getNode(r'.CONFIG:V_RFS').getData().data()
    vcal = mesxr.getNode(r'.CONFIG:V_CAL').getData().data()
    vdel = mesxr.getNode(r'.CONFIG:V_DEL').getData().data()
    vadj = mesxr.getNode(r'.CONFIG:V_ADJ').getData().data()
    
    # Deal with bad pixels - set to zero or average with surrounding pixels
    bad_pixels = mesxr.getNode(r'.CONFIG:BAD_PX_MAP').getData().data()
    bad_pixels[155, 21] = 1
    bad_pixels[179, 76] = 1

    if avg_bad_pix:
        for x in range(NUM_PIX_X):
            for y in range(NUM_PIX_Y):
                if bad_pixels[x,y]:
                    if avg_axis == 0:
                        for frame in range(len(time)):
                            if x == 0:
                                images[x, y, frame] = images[x+1, y, frame]
                            elif x == NUM_PIX_X-1:
                                images[x, y, frame] = images[x-1, y, frame]
                            else:
                                images[x, y, frame] = np.average([images[x-1, y, frame], images[x+1, y, frame]])
                    elif avg_axis == 1:
                        for frame in range(len(time)):
                            if y == 0:
                                images[x, y, frame] = images[x, y+1, frame]
                            elif y == NUM_PIX_Y-1:
                                images[x, y, frame] = images[x, y-1, frame]
                            else:
                                images[x, y, frame] = np.average([images[x, y-1, frame], images[x, y+1, frame]])
                    else:
                        print('ERROR: averaging axis not recognized.')

    # if avg_bad_pix:
    #     for x,y in calib.bad_pixels[key]:
    #         if avg_axis == 0:
    #             for frame in range(len(time)):
    #                 images[x, y, frame] = np.average([images[x-1, y, frame], images[x+1, y, frame]])
    #         elif avg_axis == 1:
    #             for frame in range(len(time)):
    #                 images[x, y, frame] = np.average([images[x, y-1, frame], images[x, y+1, frame]])
    #         else:
    #             print('ERROR: averaging axis not recognized.')
    # else:
    #     for x,y in calib.bad_pixels[key]:
    #         images[x, y, :] = 0
    
    # Zero out the boundary pixels
    for coords in get_boundary_coords():
        images[coords[0], coords[1], :] = 0

    # Zero out edge pixels if requested
    if remove_edges:
        for coords in get_edge_coords():
            images[coords[0], coords[1], :] = 0
    
    # TODO - Scale edge pixels by relative area
    
    settings = {'vtrm':vtrm, 'vcmp':vcmp, 'vcca':vcca, 'vrf':vrf,
                'vrfs':vrfs, 'vcal':vcal, 'vdel':vdel, 'vadj':vadj}
    config = {'exp_time':exp_time, 'exp_period':exp_period, 'exp_delay':exp_delay,
              'n_images':n_images, 'bad_pix':calib.bad_pixels[key], 'setdacs':settings}
    
    data_dict = {'images':images, 'time':time, 'shot':shot, 'config':config, 'thresholds':thresholds, 'rm_edges':remove_edges}

    # Load in the impact parameters
    impact_p, impact_phi = get_impact_params()
    data_dict['impact_p'] = impact_p
    data_dict['impact_phi'] = impact_phi
    
    return data_dict
    

def load_ME(shot, frames=range(20), center_only=True, remove_edges=False, direction='horizontal', key='MST'):
    """
    Wrapper for loading ME data from a real shot.
    """
    if direction == 'horizontal':
        data_dict = load_raw_data(shot, avg_bad_pix=True, avg_axis=0, key=key, remove_edges=remove_edges)
        data_dict = load_ME_horiz_2(data_dict, frames=frames, center_only=center_only)
    else:
        print('Vertical data not yet supported.')
        
    return data_dict

def load_ME_sim(measurements, time, thresholds, center_only=True, remove_edges=False, direction='horizontal'):
    """
    Wrapper for loading ME data from meSXR model measurements. This creates a data_dict then runs
    the same code as with real data.
    """
    num_frames = len(measurements)
    data_dict = {'images':np.zeros([NUM_PIX_X, NUM_PIX_Y, num_frames])}
    for frame in range(num_frames):
        data_dict['images'][:,:,frame] = measurements[frame]
    
    impact_p, impact_phi = get_impact_params()
    data_dict['impact_p'] = impact_p
    data_dict['impact_phi'] = impact_phi
    data_dict['time'] = time
    data_dict['thresholds'] = thresholds
    data_dict['rm_edges'] = remove_edges

    if direction == 'horizontal':
        data_dict = load_ME_horiz_2(data_dict, frames=range(num_frames), center_only=center_only)
    else:
        print('Vertical data not yet supported.')
        
    return data_dict

def load_ME_horiz_2(data_dict, frames=range(20), center_only=True):
    """
    A remake of the function to load in multi-energy data which is hopefully somewhat less of a disaster.
    By most accounts, this ambition failed. But at least I can remove edges consistently now.

    Returned dictionary is sorted like:
        - data_dict['ME']['images'][frame][Ec][x, y]
        - data_dict['ME']['profiles'][frame][Ec][x]
        - data_dict['ME']['x_index'][Ec][x]
        - data_dict['ME']['impact_p'][Ec][x]
        - data_dict['ME']['thresholds']
    """
    # Get the unique thresholds, ignoring skipped columns
    thresholds = np.sort(np.unique(data_dict['thresholds']))[1:]

    # Organize the multi-energy data in a sensible way
    ME_images = []
    skip_x = get_ignore_x(remove_edges=data_dict['rm_edges'])
    skip_y = get_ignore_y(remove_edges=data_dict['rm_edges'])

    for frame in frames:
        images_this_frame = {Ec:[] for Ec in thresholds}

        for x_n in range(NUM_PIX_X):
            if x_n not in skip_x:
                Ec = data_dict['thresholds'][x_n, 100]

                # Pull out the data for this given x slice
                x_strip = []
                for y_n in range(NUM_PIX_Y):
                    if y_n not in skip_y:
                        x_strip.append(data_dict['images'][x_n, y_n, frame])

                # Sort the x slices by threshold
                images_this_frame[Ec].append(x_strip)

        for Ec in images_this_frame.keys():
            images_this_frame[Ec] = np.array(images_this_frame[Ec])

        ME_images.append(images_this_frame)

    # Also determine the impact parameters for each image
    impact_p = {Ec:[] for Ec in thresholds}
    x_index = {Ec:[] for Ec in thresholds}
    
    for x_n in range(NUM_PIX_X):
            if x_n not in skip_x:
                Ec = data_dict['thresholds'][x_n, 100]
                impact_p[Ec].append(data_dict['impact_p'][x_n])
                x_index[Ec].append(x_n)

    for Ec in impact_p.keys():
        impact_p[Ec] = np.array(impact_p[Ec])
        x_index[Ec] = np.array(x_index[Ec])

    # Make the 1d profile projections
    ME_profiles = []

    for f_index in range(len(frames)):
        ME_profiles.append({})

        for Ec in thresholds:
            if center_only:
                ME_profiles[f_index][Ec] = np.sum(ME_images[f_index][Ec][:, 65:130], axis=1)
            else:
                ME_profiles[f_index][Ec] = np.sum(ME_images[f_index][Ec], axis=1)

    # Oraganize the data into nice dictionary form
    data_dict['ME'] = {'images':ME_images, 'profiles':ME_profiles, 'impact_p':impact_p, 'x_index':x_index,
                       'thresholds':thresholds, 'time':data_dict['time'][frames]}

    return data_dict

# -------------------------------------------- Geometry Functions --------------------------------------------

# Functions to get lists of special coordinates
def get_boundary_coords():
    """
    'Boundary pixels' refers to single rows/columns of pixels occupying the space between chips. These
    have a chip number of -1.
    """
    coords = []
    for x in range(calib.M_SIZE_X):
        for y in range(calib.M_SIZE_Y):
            if cfg.get_chip_coords(x,y)[0] == -1:
                coords.append((x,y))
    
    return coords

def get_good_x_indices():
    """
    This returns a list of 'good' x indices, that is for pixels which are not on the chip boudnaries.
    """
    x_indices = []
    
    for x in range(calib.M_SIZE_X):
        if cfg.get_chip_coords(x,100)[0] != -1:
            x_indices.append(x)
            
    return np.array(x_indices)

def get_edge_coords():
    """
    'Edge pixels' refers to the single row/columns of pixels on the edge of each chip, adjacent to a boundary.
    These pixels are 50% larger, so the effect of their increased size on count rate should be accounted for.
    """
    coords = []
    for x in range(calib.M_SIZE_X):
        for y in range(calib.M_SIZE_Y):
            if cfg.get_chip_coords(x,y)[1]%(calib.M_CHIP_SIZE_X-1) == 0:
                coords.append((x,y))
            elif cfg.get_chip_coords(x,y)[2]%(calib.M_CHIP_SIZE_Y-1) == 0:
                coords.append((x,y))
    
    return coords

def get_ignore_x(remove_edges=True):
    """
    Return of list of all x coordinates to ignore.
    """
    if remove_edges:
        return np.sort([coords[0] for coords in get_edge_coords() if coords[1]==100] +
                       [coords[0] for coords in get_boundary_coords() if coords[1]==100])
    else:
        return np.sort([coords[0] for coords in get_boundary_coords() if coords[1]==100])

def get_ignore_y(remove_edges=True):
    """
    Return of list of all y coordinates to ignore.
    """
    if remove_edges:
        return np.sort([coords[1] for coords in get_edge_coords() if coords[0]==100] +
                       [coords[1] for coords in get_boundary_coords() if coords[0]==100])
    else:
        return np.sort([coords[1] for coords in get_boundary_coords() if coords[0]==100])

# A function to load in the impact parameters for each pixel
MODULE_PATH = os.path.dirname(__file__)
LoS_FNAME = os.path.join(MODULE_PATH, 'mesxr_mst_los.csv')

def get_impact_params():
    """
    Get the impact parameters p and zeta from 
    """
    impact_params = np.loadtxt(LoS_FNAME, delimiter=',')
    impact_phi = impact_params[0, :]
    impact_p = impact_params[1, :]

    return impact_p*np.sign(np.sin(impact_phi)), impact_phi

# --------------------------------------- 8-Color Te Analysis Functions ---------------------------------------

gaps  = [60, 121, 182, 243, 304, 365, 426]
edges = [0, 59, 61, 120, 122, 181, 183, 242, 244, 303, 305, 364, 366, 425, 427, 486]

def get_k_hat(data_dict, frame=0, num_Ec = 8, start_n = 8, end_n = 50, remove_edges = True):
    """
    Yes, I am tring this again. Hopefully this time is the last.
    """
    # Generate the x coordinates to use for each 8-pixel cluster, making sure to skip over gaps
    num_cluster = (487 - len(gaps)) / num_Ec
    x_sets = [[] for i in range(num_cluster)]

    skips = 0

    for x in range(487):
        if x not in gaps:
            x_sets[(x-skips)/num_Ec].append(x)
        else:
            skips += 1
            
    # Now, we will explicitly remove the edge pixels
    if remove_edges:
        # First determine the pixels to skip
        x_sets_new = [[] for i in range(60)]
        
        for i, x_cluster in enumerate(x_sets):
            for x in x_cluster:
                if x not in edges:
                    x_sets_new[i].append(x)
                    
        x_sets = x_sets_new
        
    # Make them into arrays
    x_sets = [np.array(x_cluster) for x_cluster in x_sets]

    # Generate the corresponding thresholds and counts
    thresh_sets = [data_dict['thresholds'][x_cluster, 10] for x_cluster in x_sets]
    counts_sets = [np.sum(data_dict['images'][x_cluster, 65:130, frame], axis=1) for x_cluster in x_sets]

    # Generate the corresponding coordinates
    p_sets = [data_dict['impact_p'][x_cluster] for x_cluster in x_sets]
    p_avg = np.array([np.average(p_cluster) for p_cluster in p_sets])
    x_avg = np.array([np.average(x_cluster) for x_cluster in x_sets])

    # Do the fits and only for the specified cluster indices (inclusive)
    k_hat = np.zeros(end_n - start_n +1)
    k_hat_sigma = np.zeros(end_n - start_n +1)

    for index, cluster in enumerate(range(start_n, end_n+1)):
        counts = counts_sets[cluster]
        thresh = thresh_sets[cluster]
        
        try:
            params, pcov = np.polyfit(thresh, np.log(counts), 1, w=np.sqrt(counts), cov=True)
            k_hat[index] = -1.0/params[0]
            k_hat_sigma[index] = np.sqrt(pcov[0,0])/(params[0]**2)
            
        except:
            k_hat[index] = np.nan
            k_hat_sigma[index] = np.nan
    
    return {'k_hat':k_hat, 'sigma':k_hat_sigma, 'x_avg':x_avg[start_n:end_n+1], 'p_avg':p_avg[start_n:end_n+1],
            'data':{'counts':counts_sets[start_n:end_n+1], 'thresh':thresh_sets[start_n:end_n+1],
                    'indices':x_sets[start_n:end_n+1], 'impact_p':p_sets[start_n:end_n+1]},
            'start_n':start_n, 'end_n':end_n}

def compute_k_hat(counts, thresholds):
    """
    Computes the full set of k-hat for each cluster of 8 pixels, given the measurements and thresholds. This will work for both synthetic
    and real data. This version is modified from the previous version to fit only a single frame
    
    Inputs:
        - counts = (array of ints/floats) These are the summed 1D counts for some cluster of 8 pixels.
        - thresholds = (array of floats) These are the corresponding cutoff energies.
    """
    try:
        params, pcov = np.polyfit(thresholds, np.log(counts), 1, w=np.sqrt(counts), cov=True)
        k_hat = -1.0/params[0]
        k_hat_err = np.sqrt(pcov[0,0])/(params[0]**2)
    except:
        k_hat = np.nan
        k_hat_err = np.nan
            
    return k_hat, k_hat_err

def load_8_color_sim(measurements, time, thresholds, remove_edges=True, data=False, start_n = 13, end_n = 46, ignore=[22]):
    """
    """
    num_frames = len(measurements)
    data_dict = load_ME_sim(measurements, time, thresholds, center_only=True, remove_edges=False, direction='horizontal')

    k_hat_data = load_8_color(data_dict, frames=range(num_frames), remove_edges=remove_edges, data=data,
                              start_n=start_n, end_n=end_n, ignore=ignore)

    return k_hat_data

def load_8_color_data(shot, frames=range(5,12), remove_edges=True, data=False, start_n = 13, end_n = 46, ignore=[22]):
    """
    This function automatically loads data from the tree and passes it on to the load_8_color function. When loading data
    this is generally the preferred method.
    """
    MST_data = load_ME(shot, frames=frames, direction='horizontal', center_only=True, remove_edges=False)

    k_hat_data = load_8_color(MST_data, frames=frames, remove_edges=remove_edges, data=data,
                              start_n=start_n, end_n=end_n, ignore=ignore)

    return k_hat_data

def load_8_color(MST_data, frames=range(5,12), remove_edges=True, data=False, start_n = 13, end_n = 46, ignore=[22]):
    """
    Load the formatted 8-color data. Use remove_edges=True to remove the edge pixels, which tend to have a messy response.
    Use data=True to also return the full ME-SXR shot data. Use start_n and end_n to restrict the range that is fit into
    """
    # Only process the specified pixels
    include_pix = range(start_n, end_n+1)
    for pix in ignore:
        if pix >= start_n and pix <= end_n:
            index = include_pix.index(pix)
            del include_pix[index]
    
    num_pixels = len(include_pix)
    
    # Data containers
    counts_set = [[] for i in range(len(frames))]
    thresh_set = [[] for i in range(len(frames))]
    k_data = [[] for i in range(len(frames))]
    sigma_data = [[] for i in range(len(frames))]

    edge_pixels = get_ignore_x(remove_edges=True)

    for frame in range(len(frames)):
        counts_frame = [[] for i in range(num_pixels)]
        thresh_frame = [[] for i in range(num_pixels)]
        kd_frame = np.zeros(num_pixels)
        ks_frame = np.zeros(num_pixels)

        for k_n, index in enumerate(include_pix):
            thresholds = np.copy(MST_data['ME']['thresholds'])
            counts = np.array([MST_data['ME']['profiles'][frame][Ec][index] for Ec in thresholds])

            # Remove the edge points, if desired
            if remove_edges:
                x_index = np.array([MST_data['ME']['x_index'][Ec][index] for Ec in thresholds])
                to_remove = []

                for Ec_index, x_n in enumerate(x_index):
                    if x_n in edge_pixels:
                        to_remove.append(Ec_index)

                counts = np.delete(counts, to_remove)
                thresholds = np.delete(thresholds, to_remove)

            # Compute k_hat and k_sigma
            kd, ks = compute_k_hat(counts, thresholds)
            kd_frame[k_n] = kd
            ks_frame[k_n] = ks

            # Store counts for later inspection
            counts_frame[k_n] = counts
            thresh_frame[k_n] = thresholds

        # Store results in the appropriate frame
        k_data[frame] = kd_frame
        sigma_data[frame] = ks_frame
        counts_set[frame] = counts_frame
        thresh_set[frame] = thresh_frame
    
    p_set = np.array([np.average([MST_data['ME']['impact_p'][Ec][index] for Ec in MST_data['ME']['thresholds']]) for index in include_pix])
    
    k_hat_data = {'k_hat':k_data, 'sigma':sigma_data, 'avg_p':p_set, 'time':MST_data['time'][frames],
                  'counts':counts_set, 'Ec':thresh_set}

    if data:
        return k_hat_data, MST_data
    else:
        return k_hat_data



# ---------------------------------------------- Deprecated ----------------------------------------------

def load_ME_horiz(data_dict, col_size=1, key='MST', center_only=True):
    """
    """
    # Reduce the profile to 1D
    num_frames = len(data_dict['time'])
    x_prof_1d = np.zeros([NUM_PIX_X, num_frames])
    thresh_prof_1d = data_dict['thresholds'][:, 100]
    
    for frame in range(num_frames):
        if center_only:
            image = data_dict['images'][:, 65:130, frame]
        else:
            image = data_dict['images'][:, :, frame]
        x_prof_1d[:, frame] = np.sum(image, axis=1)
    
    # Reduce the profiles by combining col_size adjacent columns
    good_pixels = get_good_x_indices()
    num_reduced_pixels = len(np.unique(good_pixels/col_size))
    x_prof_1d_reduced = np.zeros([num_reduced_pixels, num_frames])
    thresh_prof_reduced = np.zeros(num_reduced_pixels)
    x_indices_reduced = np.zeros(num_reduced_pixels)
    p_reduced = np.zeros(num_reduced_pixels)
    
    for frame in range(num_frames):
        for index in range(num_reduced_pixels):
            pix_n = good_pixels[col_size*index:col_size*(index+1)]
            x_prof_1d_reduced[index, frame] = np.sum(x_prof_1d[pix_n, frame])
            x_indices_reduced[index] = np.average(pix_n)
            p_reduced[index] = np.average(data_dict['impact_p'][pix_n])
            thresh_prof_reduced[index] = thresh_prof_1d[pix_n]
            
    # Sort the profiels by threshold
    thresh, counts = np.unique(thresh_prof_reduced, return_counts=True)
    horiz_profiles = {float(thresh[i]):np.zeros([counts[i], num_frames]) for i in range(len(thresh))}
    x_pixels = {float(thresh[i]):np.zeros(counts[i]) for i in range(len(thresh))}
    p_pixels = {float(thresh[i]):np.zeros(counts[i]) for i in range(len(thresh))}
    
    loop_indices = {thresh:0 for thresh in horiz_profiles.keys()}
    for index in range(num_reduced_pixels):
        Ec = float(thresh_prof_reduced[index])
        x_pixels[Ec][loop_indices[Ec]] = x_indices_reduced[index]
        p_pixels[Ec][loop_indices[Ec]] = p_reduced[index]
        
        for frame in range(num_frames):
            horiz_profiles[Ec][loop_indices[Ec], frame] = x_prof_1d_reduced[index, frame]
            
        # Dummy variable to keep track of where we are in the loop
        loop_indices[Ec] += 1
        
    # If desired, methodically look through the data to remove edge pixels
    if data_dict['rm_edges']:
        edge_x = [coords[0] for coords in get_edge_coords() if coords[1]==100]
        to_remove = {Ec:[] for Ec in np.sort(horiz_profiles.keys())}
        tally = {Ec:[] for Ec in np.sort(horiz_profiles.keys())}

        for x_index in range(NUM_PIX_X):
            Ec = thresh_prof_1d[x_index]
            if x_index in edge_x:
                to_remove[Ec] = tally[Ec]
            else:
                tally[Ec] += 1

        for Ec in to_remove.keys():
            pass

    profiles_dict = {'x_index':x_pixels, 'counts':horiz_profiles, 'impact_p':p_pixels, 'thresholds':np.sort(horiz_profiles.keys())}
    data_dict['profiles'] = profiles_dict
    
    return data_dict