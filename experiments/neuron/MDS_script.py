import scipy.io as sio
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
# import plotly.express as px
import h5py
from sklearn.manifold import MDS # for MDS dimensionality reduction
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from ripser import ripser
from sklearn.neighbors import KernelDensity

from utils import *
from projections import *

f = h5py.File('response_matrix.h5', 'r')

reg_1_response_matrix = np.array(f.get('raster'))

model2d=MDS(n_components=2, 
          metric=True, 
          n_init=4, 
          max_iter=150, 
          verbose=0, 
          eps=0.001, 
          n_jobs=None, 
          random_state=42, 
          dissimilarity='euclidean')

### Step 2 - Fit the data and transform it, so we have 2 dimensions instead of 3
X_trans = model2d.fit_transform(reg_1_response_matrix.T)


# Create an HDF5 file and write the data
with h5py.File('mds.h5', 'w') as file:
    # Create a dataset and write the data to it
    file.create_dataset('data', data=X_trans)
