import os, sys, json 
import numpy as np 
import pandas as pd 
import shapely 
import rasterio
import xarray as xr
import rioxarray as rxr
# from shapely.geometry import Point, Polygon
import datetime
# import warnings
# from shapely.errors import ShapelyDeprecationWarning
# warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 
# import geopandas as gpd
from tqdm import tqdm
# import scipy 
# import scipy.spatial, scipy.cluster
from collections import Counter

import loadpaths
path_dict = loadpaths.loadpaths()
sys.path.append(os.path.join(path_dict['repo'], 'content/'))
import data_utils as du

def correlation_two_matrices(mat1, mat2):
    print(mat1.shape, mat2.shape)
    assert mat1.shape[1] == mat2.shape[1]
    n1 = mat1.shape[0]
    n2 = mat2.shape[0]
    corr_mat = np.zeros((n1, n2), dtype=np.float32)
    for i1 in range(n1):
        for i2 in range(n2):
            corr_mat[i1, i2] = np.corrcoef(mat1[i1, :], mat2[i2, :])[0, 1]
    return corr_mat

def ravel_features(features):
    assert len(features.shape) == 4
    n_patches, n_features, nx, ny = features.shape
    return np.swapaxes(features, 0, 1).reshape((n_features, -1))

def unravel_features(raveled, n_patches, nx, ny):
    n_features = raveled.shape[0]
    return np.swapaxes(raveled.reshape((n_features, n_patches, nx, ny)), 0, 1)