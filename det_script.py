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

FORMAT = "%(asctime)s| %(levelname)s [%(filename)s: - %(funcName)20s]  %(message)s"
logging.basicConfig(level=logging.WARNING,
                    format=FORMAT)

# I/O PARAMETERS
output_path = os.path.join(sys.path[0], 'results')
if not os.path.exists(output_path):
    os.makedirs(output_path)

Yr, dims, T = cm.load_memmap('/home/wisstock/Bio/scripts/microca/demo_data/memmap__d1_512_d2_512_d3_1_order_C_frames_1500_.mmap')
ca_images = np.reshape(Yr.T, [T] + list(dims), order='F')

logging.warning(f'Ca series shape {np.shape(ca_images)}')

fig, ax = plt.subplots()
ax.imshow(np.mean(ca_images, axis=0), cmap='jet')
ax.axis('off')
plt.title('Ca img avg')
plt.savefig(f'{output_path}/mapping_avg_ctrl.png', dpi=300)
plt.close('all')


# PARAMETERS
# dataset dependent parameters
fr = 0.9                    # imaging rate in frames per second
decay_time = 3              # length of a typical transient in seconds (see source/Getting_Started.rst)

# parameters for source extraction and deconvolution
p = 0                       # order of the autoregressive system
gnb = 2                     # number of global background components
merge_thr = 0.75            # merging threshold, max correlation allowed
rf =  None                  # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
stride_cnmf = None          # amount of overlap between the patches in pixels
K = 60                      # number of components per patch
method_init = 'graph_nmf'   # initialization method (if analyzing dendritic data using 'sparse_nmf')
ssub = 4                    # spatial subsampling during initialization
tsub = 4                    # temporal subsampling during intialization

opts_dict = {'fr': fr,
            'decay_time': decay_time,
            'p': p,
            'nb': gnb,
            'rf': rf,
            'K': K, 
            'stride': stride_cnmf,
            'method_init': method_init,
            'rolling_sum': True,
            'only_init': False,
            'ssub': ssub,
            'tsub': tsub,
            'merge_thr': merge_thr}

opts = params.CNMFParams(params_dict=opts_dict)


# RUN CNMF
# start a cluster for parallel processing (if a cluster already exists it will be closed and a new session will be opened)
if 'dview' in locals():
    cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)

logging.warning('cnm in progress')

tic = time.perf_counter()

cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
cnm = cnm.fit(ca_images)

tac = time.perf_counter()
logging.warning(f'CNMF runtime: {tac - tic:0.4f} seconds')

# contours plot
Cn = cm.local_correlations(ca_images, swap_dim=False)
Cn[np.isnan(Cn)] = 0
cnm.estimates.plot_contours(img=Cn)


