# DATA PARAMETERS (CNMFParams.data)

- __fnames: list[str]__ - list of complete paths to files that need to be processed

- __dims: (int, int), default: computed from fnames__ - dimensions of the FOV in pixels

- __fr: float, default: 30__ - imaging rate in frames per second

- __decay_time: float, default: 0.4__ - length of typical transient in seconds

- __dxy: (float, float)__ - spatial resolution of FOV in pixels per um

- __var_name_hdf5: str, default: 'mov'__ - if loading from hdf5 name of the variable to load

- __caiman_version: str__ - version of CaImAn being used

- __last_commit: str__ - hash of last commit in the caiman repo

- __mmap_F: list[str]__ - paths to F-order memory mapped files after motion correction

- __mmap_C: str__ - path to C-order memory mapped file after motion correction


# PATCH PARAMS (CNMFParams.patch)

- __rf: int or list or None, default: None__ - Half-size of patch in pixels. If None, no patches are constructed and the whole FOV is processed jointly. If list, it should be a list of two elements corresponding to the height and width of patches

- __stride: int or None, default: None__ - Overlap between neighboring patches in pixels.

- __nb_patch: int, default: 1__ - Number of (local) background components per patch

- __border_pix: int, default: 0__ - Number of pixels to exclude around each border.

- __low_rank_background: bool, default: True__ - Whether to update the background using a low rank approximation. If False all the nonzero elements of the background components are updated using hals (to be used with one background per patch)

- __del_duplicates: bool, default: False__ - Delete duplicate components in the overlaping regions between neighboring patches. If False, then merging is used.

- __only_init: bool, default: True__ - whether to run only the initialization

- __p_patch: int, default: 0__ - order of AR dynamics when processing within a patch

- __skip_refinement: bool, default: False__ - Whether to skip refinement of components (deprecated?)

- __remove_very_bad_comps: bool, default: True__ - Whether to remove (very) bad quality components during patch processing

- __p_ssub: float, default: 2__ - Spatial downsampling factor

- __p_tsub: float, default: 2__ - Temporal downsampling factor

- __memory_fact: float, default: 1__ - unitless number for increasing the amount of available memory

- __n_processes: int__ - Number of processes used for processing patches in parallel

- __in_memory: bool, default: True__ - Whether to load patches in memory


# PRE-PROCESS PARAMS (CNMFParams.preprocess)

- __sn: np.array or None, default: None__ - noise level for each pixel

- __noise_range: [float, float], default: [.25, .5]__ - range of normalized frequencies over which to compute the PSD for noise determination

- __noise_method: 'mean'|'median'|'logmexp', default: 'mean'__ - PSD averaging method for computing the noise std

- __max_num_samples_fft: int, default: 3*1024__ - Chunk size for computing the PSD of the data (for memory considerations)

- __n_pixels_per_process: int, default: 1000__ - Number of pixels to be allocated to each process

- __compute_g': bool, default: False__ - whether to estimate global time constant

- __p: int, default: 2__ - order of AR indicator dynamics

- __lags: int, default: 5__ - number of lags to be considered for time constant estimation

- __include_noise: bool, default: False__ - flag for using noise values when estimating g

- __pixels: list, default: None__ - pixels to be excluded due to saturation

- __check_nan: bool, default: True__ - whether to check for NaNs


# INIT PARAMS (CNMFParams.init)

- __K: int, default: 30__ - number of components to be found (per patch or whole FOV depending on whether rf=None)

- __SC_kernel: {'heat', 'cos', binary'}, default: 'heat'__ - kernel for graph affinity matrix

- __SC_sigma: float, default: 1__ - variance for SC kernel

- __SC_thr: float, default: 0,__ - threshold for affinity matrix

- __SC_normalize: bool, default: True__ - standardize entries prior to computing the affinity matrix

- __SC_use_NN: bool, default: False__ - sparsify affinity matrix by using only nearest neighbors

- __SC_nnn: int, default: 20__ - number of nearest neighbors to use

- __gSig: [int, int], default: [5, 5]__ - radius of average neurons (in pixels)

- __gSiz: [int, int], default: [int(round((x * 2) + 1)) for x in gSig],__ - half-size of bounding box for each neuron

- __center_psf: bool, default: False__ - whether to use 1p data processing mode. Set to true for 1p

- __ssub: float, default: 2__ - spatial downsampling factor

- __tsub: float, default: 2__ - temporal downsampling factor

- __nb: int, default: 1__ - number of background components

- __lambda_gnmf: float, default: 1.__ - regularization weight for graph NMF

- __maxIter: int, default: 5__ - number of HALS iterations during initialization

- __method_init: 'greedy_roi'|'corr_pnr'|'sparse_NMF'|'local_NMF' default: 'greedy_roi'__ - initialization method. use 'corr_pnr' for 1p processing and 'sparse_NMF' for dendritic processing.

- __min_corr: float, default: 0.85__ - minimum value of correlation image for determining a candidate component during corr_pnr

- __min_pnr: float, default: 20__ - minimum value of psnr image for determining a candidate component during corr_pnr

- __seed_method: str {'auto', 'manual', 'semi'}__ - methods for choosing seed pixels during greedy_roi or corr_pnr initialization 'semi' detects nr components automatically and allows to add more manually if running as notebook 'semi' and 'manual' require a backend that does not inline figures, e.g. %matplotlib tk

- __ring_size_factor: float, default: 1.5__ - radius of ring (*gSig) for computing background during corr_pnr

- __ssub_B: float, default: 2__ - downsampling factor for background during corr_pnr

- __init_iter: int, default: 2__ - number of iterations during corr_pnr (1p) initialization

- __nIter: int, default: 5__ - number of rank-1 refinement iterations during greedy_roi initialization

- __rolling_sum: bool, default: True__ - use rolling sum (as opposed to full sum) for determining candidate centroids during greedy_roi

- __rolling_length: int, default: 100__ - width of rolling window for rolling sum option

- __kernel: np.array or None, default: None__ - user specified template for greedyROI

- __max_iter_snmf : int, default: 500__ - maximum number of iterations for sparse NMF initialization

- __alpha_snmf: float, default: 100__ - sparse NMF sparsity regularization weight

- __sigma_smooth_snmf : (float, float, float), default: (.5,.5,.5)__ - std of Gaussian kernel for smoothing data in sparse_NMF

- __perc_baseline_snmf: float, default: 20__ - percentile to be removed from the data in sparse_NMF prior to decomposition

- __normalize_init: bool, default: True__ - whether to equalize the movies during initialization

- __options_local_NMF: dict__ - dictionary with parameters to pass to local_NMF initializer


# SPATIAL PARAMS (CNMFParams.spatial)

- __method_exp: 'dilate'|'ellipse', default: 'dilate'__ - method for expanding footprint of spatial components

- __dist: float, default: 3__ - expansion factor of ellipse

- __expandCore: morphological element, default: None(?)__ - morphological element for expanding footprints under dilate

- __nb: int, default: 1__ - number of global background components

- __n_pixels_per_process: int, default: 1000__ - number of pixels to be processed by each worker

- __thr_method: 'nrg'|'max', default: 'nrg'__ - thresholding method

- __maxthr: float, default: 0.1__ - Max threshold

- __nrgthr: float, default: 0.9999__ - Energy threshold

- __extract_cc: bool, default: True__ - whether to extract connected components during thresholding (might want to turn to False for dendritic imaging)

- __medw: (int, int) default: None__ - window of median filter (set to (3,)*len(dims) in cnmf.fit)

- __se: np.array or None, default: None__ - Morphological closing structuring element (set to np.ones((3,)*len(dims), dtype=np.uint8) in cnmf.fit)

- __ss: np.array or None, default: None__ - Binary element for determining connectivity (set to np.ones((3,)*len(dims), dtype=np.uint8) in cnmf.fit)

- __update_background_components: bool, default: True__ - whether to update the spatial background components

- __method_ls: 'lasso_lars'|'nnls_L0', default: 'lasso_lars'__ - 'nnls_L0'. Nonnegative least square with L0 penalty, 'lasso_lars' lasso lars function from scikit learn

- __block_size : int, default: 5000__ - Number of pixels to process at the same time for dot product. Reduce if you face memory problems

- __num_blocks_per_run: int, default: 20__ - Parallelization of A'*Y operation

- __normalize_yyt_one: bool, default: True__ - Whether to normalize the C and A matrices so that diag(C*C.T) = 1 during update spatial


# TEMPORAL PARAMS (CNMFParams.temporal)

- __ITER: int, default: 2__ - block coordinate descent iterations

- __method_deconvolution: 'oasis'|'cvxpy'|'oasis', default: 'oasis'__ - method for solving the constrained deconvolution problem ('oasis','cvx' or 'cvxpy') if method cvxpy, primary and secondary (if problem unfeasible for approx solution)

- __solvers: 'ECOS'|'SCS', default: ['ECOS', 'SCS']__ - solvers to be used with cvxpy, can be 'ECOS','SCS' or 'CVXOPT'

- __p: 0|1|2, default: 2__ - order of AR indicator dynamics

- __memory_efficient: False__

- __bas_nonneg: bool, default: True__ - whether to set a non-negative baseline (otherwise b >= min(y))

- __noise_range: [float, float], default: [.25, .5]__ - range of normalized frequencies over which to compute the PSD for noise determination

- __noise_method: 'mean'|'median'|'logmexp', default: 'mean'__ - PSD averaging method for computing the noise std

- __lags: int, default: 5__ - number of autocovariance lags to be considered for time constant estimation

- __optimize_g: bool, default: False__ - flag for optimizing time constants

- __fudge_factor: float (close but smaller than 1) default: .96__ - bias correction factor for discrete time constants

- __nb: int, default: 1__ - number of global background components

- __verbosity: bool, default: False__ - whether to be verbose

- __block_size : int, default: 5000__ - Number of pixels to process at the same time for dot product. Reduce if you face memory problems

- __num_blocks_per_run: int, default: 20__ - Parallelization of A'*Y operation

- __s_min: float or None, default: None__ - Minimum spike threshold amplitude (computed in the code if used).


# MERGE PARAMS (CNMFParams.merge)
- __do_merge: bool, default: True__ - Whether or not to merge

- __thr: float, default: 0.8__ - Trace correlation threshold for merging two components.

- __merge_parallel: bool, default: False__ - Perform merging in parallel

- __max_merge_area: int or None, default: None__ - maximum area (in pixels) of merged components, used to determine whether to merge components during fitting process


# QUALITY EVALUATION PARAMETERS (CNMFParams.quality)

- __min_SNR: float, default: 2.5__ - trace SNR threshold. Traces with SNR above this will get accepted

- __SNR_lowest: float, default: 0.5__ - minimum required trace SNR. Traces with SNR below this will get rejected

- __rval_thr: float, default: 0.8__ - space correlation threshold. Components with correlation higher than this will get accepted

- __rval_lowest: float, default: -1__ - minimum required space correlation. Components with correlation below this will get rejected

- __use_cnn: bool, default: True__ - flag for using the CNN classifier.

- __min_cnn_thr: float, default: 0.9__ - CNN classifier threshold. Components with score higher than this will get accepted

- __cnn_lowest: float, default: 0.1__ - minimum required CNN threshold. Components with score lower than this will get rejected.

- __gSig_range: list or integers, default: None__ - gSig scale values for CNN classifier. In not None, multiple values are tested in the CNN classifier.


# MOTION CORRECTION PARAMETERS (CNMFParams.motion)

- __border_nan: bool or str, default: 'copy'__ - flag for allowing NaN in the boundaries. True allows NaN, whereas 'copy' copies the value of the nearest data point.

- __gSig_filt: int or None, default: None__ - size of kernel for high pass spatial filtering in 1p data. If None no spatial filtering is performed

- __is3D: bool, default: False__ - flag for 3D recordings for motion correction

- __max_deviation_rigid: int, default: 3__ - maximum deviation in pixels between rigid shifts and shifts of individual patches

- __max_shifts: (int, int), default: (6,6)__ - maximum shifts per dimension in pixels.

- __min_mov: float or None, default: None__ - minimum value of movie. If None it get computed.

- __niter_rig: int, default: 1__ - number of iterations rigid motion correction.

- __nonneg_movie: bool, default: True__ - flag for producing a non-negative movie.

- __num_frames_split: int, default-: 80__ - split movie every x frames for parallel processing

- __num_splits_to_process_els, default: [7, None]__

- __num_splits_to_process_rig, default: None__

- __overlaps: (int, int), default: (24, 24)__ - overlap between patches in pixels in pw-rigid motion correction.

- __pw_rigid: bool, default: False__ - flag for performing pw-rigid motion correction.

- __shifts_opencv: bool, default: True__ - flag for applying shifts using cubic interpolation (otherwise FFT)

- __splits_els: int, default: 14__ - number of splits across time for pw-rigid registration

- __splits_rig: int, default: 14__ - number of splits across time for rigid registration

- __strides: (int, int), default: (96, 96)__ - how often to start a new patch in pw-rigid registration. Size of each patch will be strides + overlaps

- __upsample_factor_grid" int, default: 4__ - motion field upsampling factor during FFT shifts.

- __use_cuda: bool, default: False__ - flag for using a GPU.

- __indices: tuple(slice), default: (slice(None), slice(None))__ - Use that to apply motion correction only on a part of the FOV


# RING CNN PARAMETERS (CNMFParams.ring_CNN)

- __n_channels: int, default: 2__ - Number of "ring" kernels

- __use_bias: bool, default: False__ - Flag for using bias in the convolutions

- __use_add: bool, default: False__ - Flag for using an additive layer

- __pct: float between 0 and 1, default: 0.01__ - Quantile used during training with quantile loss function

- __patience: int, default: 3__ - Number of epochs to wait before early stopping

- __max_epochs: int, default: 100__ - Maximum number of epochs to be used during training

- __width: int, default: 5__ - Width of "ring" kernel

- __loss_fn: str, default: 'pct'__ - Loss function specification ('pct' for quantile loss function, 'mse' for mean squared error)

- __lr: float, default: 1e-3__ - (initial) learning rate

- __lr_scheduler: function, default: None__ - Learning rate scheduler function

- __path_to_model: str, default: None__ - Path to saved weights (if training then path to saved model weights)

- __remove_activity: bool, default: False__ - Flag for removing activity of last frame prior to background extraction

- __reuse_model: bool, default: False__ - Flag for reusing an already trained model (saved in path to model)