import os, sys
import numpy as np 
from tqdm import tqdm
import numpy as np
from skgstat import Variogram

sys.path.append('../content/')
import data_utils as du
import loadpaths
path_dict_pecl = loadpaths.loadpaths()
# import shapely
# from tqdm import tqdm, tqdm_notebook
# import ast, shutil

def compute_range_for_band(band_array, transform, 
                           model='spherical', bin_func='uniform',
                           maxlag=2000, n_subsample=5000):
    """Compute spatial correlation length (range) in meters."""
    # Get valid pixels
    rows, cols = np.where(~np.isnan(band_array))
    if len(rows) > n_subsample:  # sample to speed up
        idx = np.random.choice(len(rows), n_subsample, replace=False)
        rows, cols = rows[idx], cols[idx]

    # Convert to spatial coordinates
    xs, ys = transform * cols, transform * rows  # assuming 10m resolution
    coords = np.column_stack([xs, ys])
    values = band_array[rows, cols]

    # Compute variogram
    V = Variogram(coords, values, model=model, maxlag=maxlag, 
                  n_lags=20, bin_func=bin_func)
    return V, V.describe()['effective_range']

def compute_range_for_all_patches(model='spherical', n_subsample=500, maxlag=1280, bin_func='uniform',
                    path_folder = '/Users/tplas/data/2025-10 neureo/pecl-100-subsample-30km',   
                    prefix_name = 'pecl176-', save_folder = '/Users/tplas/repos/NeurEO/outputs/'):
    save_name = f'spatial_autocorr_pecl100-30km_{model}_{n_subsample}_{maxlag}.npy'
    assert os.path.exists(path_folder), path_folder
    assert os.path.exists(save_folder), save_folder

    sentinel, features, hypotheses = du.load_all_data(path_folder=path_folder, prefix_name=prefix_name)
    # features_nonnan = [f for f in features if np.sum(np.isnan(f)) == 0]

    n_patches = len(features)
    n_features = len(features[0])
    assert n_features == 64

    mat_effective_range = np.zeros((n_patches, n_features))
    for i in tqdm(range(n_patches)):
        for j in range(n_features):
            band_array = features[i][j]
            V, effective_range = compute_range_for_band(band_array, 10, n_subsample=n_subsample,
                                                        model=model, maxlag=maxlag, bin_func=bin_func)
            mat_effective_range[i, j] = effective_range
        ## temp save:
        np.save(os.path.join(save_folder, save_name), mat_effective_range)

    # Final save
    np.save(os.path.join(save_folder, save_name), mat_effective_range)
    print(f"Saved spatial autocorrelation ranges to {os.path.join(save_folder, save_name)}")

if __name__ == "__main__":
    compute_range_for_all_patches()