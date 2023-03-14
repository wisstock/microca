conda create --name caiman -c conda-forge python>=3.10  scikit-image>=0.19.0 scikit-learn scipy tensorflow>=2.4.0 tifffile tk tqdm yapf z5py>=2.0.15 bokeh coverage cython future h5py holoviews ipykernel ipython ipyparallel jupyter matplotlib mypy nose numpy numpydoc opencv peakutils pims psutil pynwb pyqtgraph



scikit-image>=0.19.0
tensorflow>=2.4.0
z5py>=2.0.15

conda create --name caiman -c conda-forge python=3.10
conda activate caiman
conda install -c conda-forge scikit-image tensorflow z5py cython
conda install -c conda-forge scikit-learn scipy  tifffile tk tqdm yapf  bokeh coverage  future h5py
conda install -c conda-forge holoviews ipykernel ipython ipyparallel jupyter matplotlib mypy nose numpy numpydoc opencv peakutils pims psutil pynwb pyqtgraph