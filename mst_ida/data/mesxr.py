"""
"""
from __future__ import division
import os
import numpy as np
import scipy as sp
import scipy.interpolate
import MDSplus as mds
from mst_ida.utilities.structures import AttrDict

# Module-wide detector constants
CHIP_SIZE_X = 60
CHIP_SIZE_Y = 97
NUM_CHIPS_Y = 2
NUM_CHIPS_X = 8
NUM_CHIPS = NUM_CHIPS_X * NUM_CHIPS_Y
NUM_PIX_Y = 195
NUM_PIX_X = 487

def load_raw_data(shot, avg_bad_pix=True, avg_axis=0, remove_edges=False, key='MST', center=False):
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
    
    # Zero out the boundary pixels
    for coords in get_boundary_coords():
        images[coords[0], coords[1], :] = 0

    # Zero out edge pixels if requested
    if remove_edges:
        for coords in get_edge_coords():
            images[coords[0], coords[1], :] = 0

    # Zero out everything but central pixels, if requested
    if center:
        mask = np.zeros([487, 195])
        mask[:,65:130] = 1
        for frame in range(len(time)):
            images[:,:,frame] *= mask
    
    settings = {'vtrm':vtrm, 'vcmp':vcmp, 'vcca':vcca, 'vrf':vrf,
                'vrfs':vrfs, 'vcal':vcal, 'vdel':vdel, 'vadj':vadj}
    config = {'exp_time':exp_time, 'exp_period':exp_period, 'exp_delay':exp_delay,
              'n_images':n_images, 'bad_pix':bad_pixels, 'setdacs':settings}
    
    data_dict = {'images':images, 'time':time, 'shot':shot, 'config':config, 'thresholds':thresholds, 'rm_edges':remove_edges}

    # Load in the impact parameters
    impact_p, impact_phi = get_impact_params()
    data_dict['impact_p'] = impact_p
    data_dict['impact_phi'] = impact_phi
    
    return AttrDict(data_dict)

# -------------------------------------------- Geometry Functions --------------------------------------------

# Functions to get lists of special coordinates
def get_boundary_coords():
    """
    'Boundary pixels' refers to single rows/columns of pixels occupying the space between chips. These
    have a chip number of -1.
    """
    coords = []
    for x in range(NUM_PIX_X):
        for y in range(NUM_PIX_Y):
            if get_chip_coords(x,y)[0] == -1:
                coords.append((x,y))
    
    return coords

def get_edge_coords():
    """
    'Edge pixels' refers to the single row/columns of pixels on the edge of each chip, adjacent to a boundary.
    These pixels are 50% larger, so the effect of their increased size on count rate should be accounted for.
    """
    coords = []
    for x in range(NUM_PIX_X):
        for y in range(NUM_PIX_Y):
            if get_chip_coords(x,y)[1]%(CHIP_SIZE_X-1) == 0:
                coords.append((x,y))
            elif get_chip_coords(x,y)[2]%(CHIP_SIZE_Y-1) == 0:
                coords.append((x,y))
    
    return coords

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

def get_chip_coords(image_x, image_y):
    """
    Description
    ===============
        Original note from Pablant:
            This is copied from pix_add in utils.c for the p2det detector. Given an x and y
            location on the detector, this will return the appropriate chip number and the x
            and y location on that given chip.

        This function takes coordinates in the broader "image" (detector) reference frame and
        determines chat chip this point falls on as well as its local x,y coordinates in the
        chip reference frame.
    Parameters:
    ===============
        - image_x = (int) X-coordinate of a point on the overall detector image
        - image_y = (int) Y-coordinate of a point on the overall detector image
    Returns:
    ===============
        - chip_num = (int) The chip number on which the point (image_x, image_y) lies
        - chip_x = (int) The x-coordinate of the point in the frame of chip_num
        - chip_y = (int) The y-coordinate of the point in the frame of chip_num
    """
    if image_y < NUM_PIX_Y/2:
        chip_num = image_x/(CHIP_SIZE_X + 1)
        chip_x = (CHIP_SIZE_X+1)*(chip_num+1) - image_x - 2
        chip_y = image_y

        if chip_x < 0:
            chip_num = -1
    elif image_y == NUM_PIX_Y/2:
        chip_num = -1

    else:
        chip_num = NUM_CHIPS/2 + image_x/(CHIP_SIZE_X + 1)
        chip_x = image_x % (CHIP_SIZE_X+1)
        chip_y = NUM_PIX_Y - image_y - 1

        if chip_x >= CHIP_SIZE_X:
            chip_num = -1

    # Check if this is a valid chip.
    if chip_num < 0:
        chip_y = -1
        chip_x = -1

    return chip_num, chip_x, chip_y

def get_geometry(Ec_map):
    """
    Get the set of average pixels corresponding to a given shot. In most cases this should
    be the same for all shots. I think this will actually work for four-color data as well.

    This is a little redundant with code below, but removes the need to supply a specific shot
    number. In the end this is mostly behind-the-scenes stuff.
    """

    # Reaslistic lines of sight, sorted by threshold "8 detector" model
    impact_p, impact_phi = get_MST_LoS()

    # Sort p and phi into clusters of 8 pixels, skipping over the gaps
    pixel_gaps = np.arange(60, 487, 61)
    thresholds = np.sort([Ec for Ec in np.unique(Ec_map) if Ec !=0])

    xpix_p = np.zeros([60, len(thresholds)])
    xpix_phi = np.zeros([60, len(thresholds)])

    index = 0
    for x in range(487):
        if x not in pixel_gaps:
            Ec = Ec_map[x, 0]
            Ec_index = np.where(thresholds == Ec)[0][0]
            xpix_p[index//8, Ec_index] = impact_p[x]
            xpix_phi[index//8, Ec_index] = impact_phi[x]
            index += 1
            
    # Compute the average p and phi for each cluster
    p_avg = np.average(xpix_p, axis=1)
    phi_avg = np.average(xpix_phi, axis=1)

    # Create the lines of sight for each meta-pixel
    return p_avg, phi_avg, thresholds

def get_MST_LoS(fname=LoS_FNAME):
    impact_params = np.loadtxt(fname, delimiter=',')
    impact_phi = impact_params[0, :]
    impact_p = impact_params[1, :]
    return impact_p, impact_phi

# -------------------------------- Format data into 1d profiles --------------------------------

y_pix = np.array([y for y in range(0,195) if y != 97])
pixel_gaps = np.arange(60, 487, 61)
pixel_edges = []
for gap in pixel_gaps:
    pixel_edges.extend([gap-1, gap+1])

def get_profiles_data(shot, frame, smooth=True):
    """
    """
    dat = load_raw_data(shot)
    mesxr_data, signed_p, thresholds = profiles_from_image(dat['images'][:,:,frame], dat['thresholds'], smooth=smooth)
    mesxr_sigma = {Ec:np.sqrt(np.maximum(mesxr_data[Ec], np.ones(mesxr_data[Ec].shape))) for Ec in thresholds}
    return mesxr_data, mesxr_sigma, signed_p, thresholds

def profiles_from_image(image, Ec_map, smooth=True):
    """
    """
    p_avg, phi_avg, thresholds = get_geometry(Ec_map)
    signed_p = np.sign(np.sin(phi_avg))*p_avg
    mesxr_prof = {Ec:np.zeros(60) for Ec in thresholds}

    index = 0
    for x in range(487):
        if x not in pixel_gaps:
            mesxr_prof[Ec_map[x, 0]][index//8] += np.sum(image[x,:])
            index += 1

    if smooth:
        mesxr_prof = smooth_edges(mesxr_prof, signed_p, Ec_map)

    return mesxr_prof, signed_p, thresholds

def smooth_edges(mesxr_prof, signed_p, Ec_map):
    """
    """
    edge_indices = get_edge_indices(Ec_map)
    thresholds = sorted(mesxr_prof.keys())
    new_prof = {}
    for Ec in thresholds:
        if len(edge_indices[Ec]) > 0:
            include = np.array([index for index in range(60) if index not in edge_indices[Ec]])
            x_data = np.flip(signed_p[include], axis=0)
            y_data = np.flip(mesxr_prof[Ec][include], axis=0)
            sp_fit = sp.interpolate.UnivariateSpline(x_data, y_data)
            
            new_prof[Ec] = mesxr_prof[Ec]
            new_prof[Ec][edge_indices[Ec]] = sp_fit(signed_p[edge_indices[Ec]])
        else:
            new_prof[Ec] = mesxr_prof[Ec]
            
    return new_prof

def get_edge_indices(Ec_map):
    """
    """
    thresholds = np.sort([Ec for Ec in np.unique(Ec_map) if Ec !=0])
    edge_indices = {Ec:[] for Ec in thresholds}

    index = 0
    for x in range(487):
        if x not in pixel_gaps:
            if x in pixel_edges:
                edge_indices[Ec_map[x, 0]].append(index//8)
            index += 1
    
    return edge_indices

# ----------------------------------------- Analysis -----------------------------------------

# Package this into a convenient function
box1 = [(226, 242), (80, 96)]
box2 = [(245, 261), (80, 96)]
box3 = [(226, 242), (99, 115)]
box4 = [(245, 261), (99, 115)]
boxes = [box1, box2, box3, box4]

def get_count_rate(shot, frame, exp_time=1.0, Ec_target=4.0):
    mst_data = load_raw_data(shot)
    thresholds = np.unique(mst_data['thresholds'])
    Ec = thresholds[np.argmin(np.abs(thresholds - Ec_target))]
    
    coords = []

    for box in boxes:
        for x in range(*box[0]):
            for y in range(*box[1]):
                if mst_data['thresholds'][x,y] == Ec:
                    coords.append((x,y))
                    
    counts = [mst_data['images'][x,y,frame] for x,y in coords]
    return sum(counts) / len(counts) / exp_time * 1000.

def get_profiles_time(shot, start_frame=0, end_frame=25, smooth=True):
    # Load the data and fit each threshold
    frames = np.arange(start_frame, end_frame+1)
    num_frames = len(frames)
    
    t_start = 2*start_frame+0.5
    t_end = 2*end_frame+0.5
    tiempo = np.arange(t_start, t_end+2, 2)
    
    # Set the first frame manually to get the thresholds
    mesxr_data = load_raw_data(shot)
    prof, signed_p, thresholds = profiles_from_image(mesxr_data['images'][:,:,0], mesxr_data['thresholds'], smooth=smooth)
    profiles = {Ec:np.zeros([num_frames, 60]) for Ec in thresholds}
    
    for ii,frame in enumerate(frames):
        prof = profiles_from_image(mesxr_data['images'][:,:,frame], mesxr_data['thresholds'], smooth=smooth)[0]
        
        for Ec in thresholds:
            profiles[Ec][ii,:] = prof[Ec]
            
    return profiles, tiempo, signed_p, thresholds

# ------------------------------------------------------- Routine analysis -------------------------------------------------------
def get_geometry_full(Ec_map, average_pix=False):
    """
    Make this into a more accurate version which treats each threshold independently
    """
    # Reaslistic lines of sight, sorted by threshold "8 detector" model
    impact_p, impact_phi = get_MST_LoS()

    # Sort p and phi into clusters of 8 pixels, skipping over the gaps
    pixel_gaps = np.arange(60, 487, 61)
    thresholds = np.sort([Ec for Ec in np.unique(Ec_map) if Ec !=0])

    xpix_p = np.zeros([60, len(thresholds)])
    xpix_phi = np.zeros([60, len(thresholds)])
    
    ps = {Ec:np.zeros(60) for Ec in thresholds}
    phis = {Ec:np.zeros(60) for Ec in thresholds}

    index = 0
    for x in range(487):
        if x not in pixel_gaps:
            Ec = Ec_map[x, 0]
            ps[Ec][index//8] = impact_p[x]
            phis[Ec][index//8] = impact_phi[x]
            
            Ec_index = np.where(thresholds == Ec)[0][0]
            xpix_p[index//8, Ec_index] = impact_p[x]
            xpix_phi[index//8, Ec_index] = impact_phi[x]
            index += 1
            
    if average_pix:
        # Compute the average p and phi for each cluster
        p_avg = np.average(xpix_p, axis=1)
        phi_avg = np.average(xpix_phi, axis=1)
        return p_avg, phi_avg, thresholds

    else:
        return ps, phis, thresholds

def get_8c_data(shot, frame, smooth=True, **kwargs):
    dat = load_raw_data(shot, **kwargs)
    mesxr_data, signed_p, thresholds = profiles_from_image(dat['images'][:,:,frame], dat['thresholds'], smooth=smooth)
    mesxr_sigma = {Ec:np.sqrt(np.maximum(mesxr_data[Ec], np.ones(mesxr_data[Ec].shape))) for Ec in thresholds}
    
    # Get the full lines of sight
    ps, phis, thresholds = get_geometry_full(dat['thresholds'], average_pix=False)
    signed_ps = {Ec:np.sign(np.sin(phis[Ec]))*ps[Ec] for Ec in thresholds}
    
    return mesxr_data, mesxr_sigma, signed_ps, thresholds

# ------------------------------------- Smooth Profiles -------------------------------------
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Smooth and interpolate with GPR
def smooth_prof_gpr(data, data_sigma, data_ps, sample_ps):
    # Format the inputs
    X = np.atleast_2d(data_ps).T
    y = np.maximum(data, 0)
    alpha = data_sigma**2
    
    # Set up the Gaussian Process Regressor
    kernel = C(1000, (1, 1e4)) * RBF(0.05, (1e-3, 1))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, alpha=alpha)
    gp.fit(X,y)
    
    # Predict, then convert back to normal dimensions
    xs = np.atleast_2d(sample_ps).T
    y_pred, y_sigma = gp.predict(xs, return_std=True)
    xs = np.squeeze(xs)
    y_pred = np.squeeze(y_pred)
    y_sigma = np.squeeze(y_sigma)
    
    return y_pred, y_sigma

def get_smooth_data(shot, frame, **kwargs):
    mesxr_data, mesxr_sigma, mesxr_ps, thresholds = get_8c_data(shot, frame, **kwargs)
    
    # Resample to the approximate lines of sight
    p_avg = np.array([np.average([mesxr_ps[Ec][i] for Ec in thresholds]) for i in range(60)])
    y_smooth = {Ec:np.zeros(p_avg.shape) for Ec in thresholds}
    y_sigma = {Ec:np.zeros(p_avg.shape) for Ec in thresholds}

    for Ec in thresholds:
        ys, sigma = smooth_prof_gpr(mesxr_data[Ec], mesxr_sigma[Ec], mesxr_ps[Ec], p_avg)
        y_smooth[Ec][:] = ys
        y_sigma[Ec][:] = sigma
        
    return y_smooth, y_sigma, p_avg, thresholds

# ------------------------------------- ME-SXR Ratios -------------------------------------
def get_ratio_data(shot, frame, smooth=False, **kwargs):
    if smooth:
        mesxr_data, mesxr_sigmas, mesxr_ps, thresholds = get_smooth_data(shot, frame, **kwargs)
    else:
        mesxr_data, mesxr_sigmas, mesxr_ps, thresholds = get_8c_data(shot, frame, **kwargs)
    
    indices = np.arange(60)
    r_data = np.zeros([len(indices), len(thresholds)-1])
    r_err = np.zeros([len(indices), len(thresholds)-1])
    
    if smooth:
        r_rad = mesxr_ps[indices]
    else:
        r_rad = np.array([np.average([mesxr_ps[Ec][i] for Ec in thresholds]) for i in indices])

    for n,chord_index in enumerate(indices):
        data = np.array([mesxr_data[Ec][chord_index] for Ec in thresholds])
        sigmas = np.array([mesxr_sigmas[Ec][chord_index] for Ec in thresholds])

        r_data[n,:] = data[1:] / data[0]
        r_err[n,:] = r_data[n,:]*np.sqrt((sigmas[1:]/data[1:])**2 + (sigmas[0]/data[0])**2)
        
    return r_data, r_err, r_rad, thresholds[1:]