# all-in-one script for demo notebook

import cv2
try:
    cv2.setNumThreads(4)
except():
    pass

import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.utils.utils import download_demo
from caiman.utils.visualization import nb_view_patches
from skimage.util import montage
import bokeh.plotting as bpl
import holoviews as hv

FORMAT = "%(asctime)s| %(levelname)s [%(filename)s: - %(funcName)20s]  %(message)s"
logging.basicConfig(level=logging.INFO,
                    format=FORMAT)

# I/O PARAMETERS
ca_path = ["/home/wisstock/Bio/scripts/microca/demo_data/E_0002_Ca_300-700_demo.tif"]
output_path = os.path.join(sys.path[0], 'results')
if not os.path.exists(output_path):
    os.makedirs(output_path)

# PARAMETERS
# dataset dependent parameters
fr = 1                      # imaging rate in frames per second
decay_time = 3              # length of a typical transient in seconds (see source/Getting_Started.rst)

# # motion correction parameters
# strides = (80, 80)          # start a new patch for pw-rigid motion correction every x pixels
# overlaps = (20, 20)         # overlap between pathes (size of patch strides+overlaps)
# max_shifts = (10, 10)       # maximum allowed rigid shifts (in pixels)
# max_deviation_rigid = 3     # maximum shifts deviation allowed for patch with respect to rigid shifts
# pw_rigid = True             # flag for performing non-rigid motion correction

# # parameters for source extraction and deconvolution
# p = 2                       # order of the autoregressive system
# gnb = 3                     # number of global background components
# merge_thr = 0.5             # merging threshold, max correlation allowed
# rf =  200                   # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
# stride_cnmf = 100           # amount of overlap between the patches in pixels
# K = 50                      # number of components per patch
# method_init = 'greedy_roi'  # initialization method (if analyzing dendritic data using 'sparse_nmf')
# ssub = 2                    # spatial subsampling during initialization
# tsub = 2                    # temporal subsampling during intialization

# opts_dict = {'fnames': ca_path,
#             'fr': fr,
#             'decay_time': decay_time,
#             'strides': strides,
#             'overlaps': overlaps,
#             'max_shifts': max_shifts,
#             'max_deviation_rigid': max_deviation_rigid,
#             'pw_rigid': pw_rigid,
#             'p': p,
#             'nb': gnb,
#             'rf': rf,
#             'K': K, 
#             'stride': stride_cnmf,
#             'method_init': method_init,
#             'rolling_sum': True,
#             'only_init': True,
#             'ssub': ssub,
#             'tsub': tsub,
#             'merge_thr': merge_thr}

# opts = params.CNMFParams(params_dict=opts_dict)


# dataset dependent parameters
fr = 1                             # imaging rate in frames per second
decay_time = 5                     # length of a typical transient in seconds (see source/Getting_Started.rst)

# motion correction parameters
strides = (48, 48)          # start a new patch for pw-rigid motion correction every x pixels
overlaps = (24, 24)         # overlap between pathes (size of patch strides+overlaps)
max_shifts = (6, 6)         # maximum allowed rigid shifts (in pixels)
max_deviation_rigid = 3     # maximum shifts deviation allowed for patch with respect to rigid shifts
pw_rigid = True             # flag for performing non-rigid motion correction

# parameters for source extraction and deconvolution
p = 0                       # order of the autoregressive system
gnb = 2                     # number of global background components
merge_thr = 0.75            # merging threshold, max correlation allowed
rf =  None                  # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
stride_cnmf = None          # amount of overlap between the patches in pixels
K = 60                      # number of components per patch
method_init = 'graph_nmf'   # initialization method (if analyzing dendritic data using 'sparse_nmf')
ssub = 1                    # spatial subsampling during initialization
tsub = 1                    # temporal subsampling during intialization

opts_dict = {'fnames': ca_path,
            'fr': fr,
            'decay_time': decay_time,
            'strides': strides,
            'overlaps': overlaps,
            'max_shifts': max_shifts,
            'max_deviation_rigid': max_deviation_rigid,
            'pw_rigid': pw_rigid,
            'p': p,
            'nb': gnb,
            'rf': rf,
            'K': K, 
            'stride': stride_cnmf,
            'method_init': method_init,
            'rolling_sum': True,
            'only_init': True,
            'ssub': ssub,
            'tsub': tsub,
            'merge_thr': merge_thr}

opts = params.CNMFParams(params_dict=opts_dict)


# MOTION CORRECTION
if 'dview' in locals():
    cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)

logging.info('otion correction in progress')

tic_0 = time.perf_counter()
mc = MotionCorrect(ca_path, dview=dview, **opts.get_group('motion'))

mc.motion_correct(save_movie=True)
m_els = cm.load(mc.fname_tot_els)
border_to_0 = 0 if mc.border_nan == 'copy' else mc.border_to_0 

# memory map the file in order 'C'
ca_new = cm.save_memmap(mc.mmap_file, base_name='memmap_', order='C',
                        border_to_0=border_to_0) # exclude borders

# now load the file
Yr, dims, T = cm.load_memmap(ca_new)
ca_images = np.reshape(Yr.T, [T] + list(dims), order='F') 

logging.info(f'Ca series shape {np.shape(ca_images)}')

fig, ax = plt.subplots()
ax.imshow(np.mean(ca_images, axis=0), cmap='jet')
ax.axis('off')
plt.title('Ca img avg')
plt.savefig(f'{output_path}/mapping_avg_ctrl.png', dpi=300)
plt.close('all')

tac = time.perf_counter()
logging.info(f'Mapping runtime: {tac - tic_0:0.4f} seconds')


# RUN CNMF
# restart cluster to clean up memory
cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(backend='local',
                                                 n_processes=None,
                                                 single_thread=False)

logging.info('cnm in progress')

tic = time.perf_counter()
cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
cnm = cnm.fit(ca_images)

# CI = cm.local_correlations(images[::1].transpose(1,2,0))
# CI[np.isnan(CI)] = 0

# plot montage array
A = cnm.estimates.A.toarray().reshape(opts.data['dims'] + (-1,), order='F').transpose([2, 0, 1])
Nc = A.shape[0]
grid_shape = (np.ceil(np.sqrt(Nc/2)).astype(int), np.ceil(np.sqrt(Nc*2)).astype(int))
plt.figure(figsize=np.array(grid_shape[::-1])*1.5)
plt.imshow(montage(A, rescale_intensity=True, grid_shape=grid_shape), cmap='jet')
plt.title('Montage of found spatial components')
plt.axis('off')
plt.savefig(f'{output_path}/cnm_ctrl.png', dpi=300)
plt.close('all')

tac = time.perf_counter()
logging.info(f'CNMF runtime: {tac - tic:0.4f} seconds')
logging.info(f'Total runtime: {tac - tic_0:0.4f} seconds')


# # RUN CNMF to sparsify
# # restart cluster to clean up memory
# cm.stop_server(dview=dview)
# c, dview, n_processes = cm.cluster.setup_cluster(backend='local',
#                                                  n_processes=None,
#                                                  single_thread=False)

# logging.info('cnm2 in progress')

# tic = time.perf_counter()
# cnm2 = cnm.refit(ca_images, dview=dview)

# A2 = cnm2.estimates.A.toarray().reshape(opts.data['dims'] + (-1,), order='F').transpose([2, 0, 1])
# Nc = A2.shape[0]
# grid_shape = (np.ceil(np.sqrt(Nc/2)).astype(int), np.ceil(np.sqrt(Nc*2)).astype(int))
# plt.figure(figsize=np.array(grid_shape[::-1])*1.5)
# plt.imshow(montage(A2, rescale_intensity=True, grid_shape=grid_shape))
# plt.title('Montage of found spatial components')
# plt.axis('off')
# plt.savefig(f'{output_path}/cnm2_ctrl.png', dpi=300)
# plt.close('all')

# tac = time.perf_counter()
# logging.info(f'CNMF2 runtime: {tac - tic:0.4f} seconds')


