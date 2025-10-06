import os, sys 
import numpy as np 
import pandas as pd 
# from dwca.read import DwCAReader
import matplotlib.pyplot as plt
# import seaborn as sns 
import shapely 
import rasterio, rasterio.plot
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
# from matplotlib.colors import ListedColormap
# import scipy.spatial, scipy.cluster
# import json

import loadpaths
path_dict = loadpaths.loadpaths()
sys.path.append(os.path.join(path_dict['repo'], 'content/'))

ONLINE_ACCESS_TO_GEE = False 
if ONLINE_ACCESS_TO_GEE:
    import api_keys
    import ee, geemap 
    ee.Authenticate()
    ee.Initialize(project=api_keys.GEE_API)
    geemap.ee_initialize()
else:
    print('WARNING: ONLINE_ACCESS_TO_GEE is set to False, so no access to GEE')


# def map_str_tuple_to_coords(tup):
#     """Convert a string tuple to a tuple of floats."""
#     return tuple(float(x) for x in ast.literal_eval(tup))

def load_tiff(tiff_file_path, datatype='np', verbose=0):
    '''Load tiff file as np or da'''
    with rasterio.open(tiff_file_path) as f:
        if verbose > 0:
            print(f.profile)
        if datatype == 'np':  # handle different file types 
            im = f.read()
            assert type(im) == np.ndarray
        elif datatype == 'da':
            im = rxr.open_rasterio(f)
            assert type(im) == xr.DataArray
        else:
            assert False, 'datatype should be np or da'

    return im 


def create_timestamp(include_seconds=False):
    dt = datetime.datetime.now()
    timestamp = str(dt.date()) + '-' + str(dt.hour).zfill(2) + str(dt.minute).zfill(2)
    if include_seconds:
        timestamp += ':' + str(dt.second).zfill(2)
    return timestamp

def get_gee_image_from_point(coords, bool_buffer_in_deg=True, buffer_deg=0.01, 
                             verbose=0, year=None, 
                             month_start_str='06', month_end_str='09',
                             image_collection='sentinel2'):
    assert ONLINE_ACCESS_TO_GEE, 'Need to set ONLINE_ACCESS_TO_GEE to True to use this function'
    assert image_collection in ['sentinel2', 'alphaearth'], 'image_collection should be sentinel2 or alphaearth'
    if year is None:
        year = 2024

    point = shapely.geometry.Point(coords)
    if bool_buffer_in_deg:  # not ideal https://gis.stackexchange.com/questions/304914/python-shapely-intersection-with-buffer-in-meter
        polygon = point.buffer(buffer_deg, cap_style=3)  ## buffer in degrees
        xy_coords = np.array(polygon.exterior.coords.xy).T 
        aoi = ee.Geometry.Polygon(xy_coords.tolist())
    else:
        assert False, 'not implemented yet'
    
    if image_collection == 'sentinel2':
        ex_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')

        ## also consider creating a mosaic instead: https://gis.stackexchange.com/questions/363163/filter-out-the-least-cloudy-images-in-sentinel-google-earth-engine
        ex_im_gee = ee.Image(ex_collection 
                            #   .project(crs='EPSG:27700', scale=1)
                            .filterBounds(aoi) 
                            .filterDate(ee.Date(f'{year}-{month_start_str}-01'), ee.Date(f'{year}-{month_end_str}-01')) 
                            .select(['B4', 'B3', 'B2', 'B8'])  # 10m bands, RGB and NIR
                            .sort('CLOUDY_PIXEL_PERCENTAGE')
                            .first()  # get the least cloudy image
                            .clip(aoi))
    elif image_collection == 'alphaearth':
        ex_collection = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        ex_im_gee = ee.Image(ex_collection 
                            #   .project(crs='EPSG:27700', scale=1)
                            .filterBounds(aoi) 
                            .filterDate(ee.Date(f'{year}-01-01'), ee.Date(f'{year}-12-31')) 
                            .first() 
                            .clip(aoi))
    else:
        raise NotImplementedError(image_collection)

    im_dims = ex_im_gee.getInfo()["bands"][0]["dimensions"]
    
    if im_dims[0] < 128 or im_dims[1] < 128:
        print('WARNING: image too small, returning None')
        return None
    
    if verbose:
        print(ex_im_gee.projection().getInfo())
        print(f'Area AOI in km2: {aoi.area().getInfo() / 1e6}')
        print(f'Pixel dimensions: {im_dims}')
        print(ex_im_gee.getInfo()['bands'][3])
    
    return ex_im_gee

def download_gee_image(coords, name: str, bool_buffer_in_deg=True, buffer_deg=0.01, 
                    verbose=0, year=None, 
                    month_start_str='06', month_end_str='09',
                    image_collection='sentinel2',
                    path_save=None):
    if year is None:
        year = 2024

    im_gee = get_gee_image_from_point(coords=coords, bool_buffer_in_deg=bool_buffer_in_deg, buffer_deg=buffer_deg, 
                           verbose=verbose, year=year, 
                           month_start_str=month_start_str, month_end_str=month_end_str,
                           image_collection=image_collection)
    if im_gee is None:  ## if image was too small it was discarded
        return None, None

    if path_save is None:
        path_save = path_dict['data_folder'] 
    if not os.path.exists(path_save):
        os.makedirs(path_save)
        print(f'Created folder {path_save}')

    if image_collection == 'sentinel2':
        filename = f'{name}_sent2-4band_y-{year}_m-{month_start_str}-{month_end_str}.tif'
    elif image_collection == 'alphaearth':
        filename = f'{name}_alphaearth_y-{year}.tif'
    filepath = os.path.join(path_save, filename)
    
    geemap.ee_export_image(
        im_gee, filename=filepath, 
        scale=10,  # 10m bands
        file_per_band=False,# crs='EPSG:32630'
    )

    ## load & save to size correctly (because of buffer): 
    im = load_tiff(filepath, datatype='da')
    remove_if_too_small = True
    if image_collection == 'sentinel2':
        desired_pixel_size = 128  # for sentinel2
    elif image_collection == 'alphaearth':
        desired_pixel_size = 128
    
    if verbose:
        print('Original size: ', im.shape)
    if im.shape[1] < desired_pixel_size or im.shape[2] < desired_pixel_size:
        print('WARNING: image too small, returning None')
        if remove_if_too_small:
            os.remove(filepath)
        return None, None

    ## crop:
    padding_1 = (im.shape[1] - desired_pixel_size) // 2
    padding_2 = (im.shape[2] - desired_pixel_size) // 2
    im_crop = im[:, padding_1:desired_pixel_size + padding_1, padding_2:desired_pixel_size + padding_2]
    assert im_crop.shape[0] == im.shape[0] and im_crop.shape[1] == desired_pixel_size and im_crop.shape[2] == desired_pixel_size, im_crop.shape
    if verbose:
        print('New size: ', im_crop.shape)
    im_crop.rio.to_raster(filepath)

    return im_crop, filepath

def download_list_coord(coord_list, path_save=None, name_group='sample'):
    assert type(coord_list) == list
    for i, coords in enumerate(tqdm(coord_list)):
        name = f'{name_group}-{i}'
        for im_collection in ['sentinel2', 'alphaearth']:
            im, path_im = download_gee_image(coords=coords, name=name, 
                                             path_save=path_save, verbose=0,
                                             image_collection=im_collection)
            if im is None:
                print(f'Image {name} could not be downloaded')

def get_images_from_name(path_folder=path_dict['data_folder'], name='sample-0'):
    assert os.path.exists(path_folder), path_folder
    contents = os.listdir(path_folder)
    contents = [f for f in contents if name in f and f.endswith('.tif')]
    assert len(contents) > 0, f'No files found in {path_folder}'
    assert len(contents) <= 2, f'More than 2 files found in {path_folder}, please specify name more precisely: {contents}'

    file_sent, file_alpha = None, None
    for file in contents:
        if file.split('_')[1].startswith('sent2'):
            file_sent = file
        elif file.split('_')[1].startswith('alphaearth'):
            file_alpha = file 

    assert file_sent or file_alpha

    return (file_sent, file_alpha)


## Plotting images:
def naked(ax):
    '''Remove all spines, ticks and labels'''
    for ax_name in ['top', 'bottom', 'right', 'left']:
        ax.spines[ax_name].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

def plot_image_simple(im, ax=None, name_file=None, use_im_extent=False, verbose=0):
    '''Plot image (as np array or xr DataArray)'''
    if ax is None:
        ax = plt.subplot(111)
    if type(im) == xr.DataArray:
        plot_im = im.to_numpy()
    else:
        plot_im = im
    if verbose > 0:
        print(plot_im.shape, type(plot_im))
    if use_im_extent:
        extent = [im.x.min(), im.x.max(), im.y.min(), im.y.max()]
    else:
        extent = None
    rasterio.plot.show(plot_im, ax=ax, cmap='viridis', 
                       extent=extent)
    naked(ax)
    ax.set_aspect('equal')
    if name_file is None:
        pass 
    else:
        name_tile = name_file.split('/')[-1].rstrip('.tif')
        ax.set_title(name_tile)

def plot_overview_images(path_folder=path_dict['data_folder'], name='sample-0', verbose=0):
    (file_sent, file_alpha) = get_images_from_name(path_folder=path_folder, name=name)
    path_sent = os.path.join(path_folder, file_sent) if file_sent is not None else None
    path_alpha = os.path.join(path_folder, file_alpha) if file_alpha is not None else None

    if path_alpha is not None:
        im_loaded_alpha = load_tiff(path_alpha, datatype='da')
        im_plot_alpha = im_loaded_alpha
        ## normalise:
        im_plot_alpha.values[im_plot_alpha.values == -np.inf] = np.nan
        im_plot_alpha.values[im_plot_alpha.values == np.inf] = np.nan
        im_plot_alpha = (im_plot_alpha - np.nanmin(im_plot_alpha)) / (np.nanmax(im_plot_alpha) - np.nanmin(im_plot_alpha))
        # im_plot_alpha = np.swapaxes(im_plot_alpha, 2, 1)  # change to (bands, height, width) to (height, width, bands)`

    if path_sent is not None:
        im_loaded_s2 = load_tiff(path_sent, datatype='da')
        im_loaded_s2 = np.clip(im_loaded_s2, 0, 3000)
        im_loaded_s2 = im_loaded_s2 / (3000)
        im_plot_s2 = im_loaded_s2[:3, ...]
        im_nir_s2 = im_loaded_s2[1:, ...]
        size_s2 = im_plot_s2.shape[1]
        assert size_s2 == im_plot_s2.shape[2]
        half_size_s2 = size_s2 // 2
        quarter_size_s2 = size_s2 // 4
        # im_plot_s2 = im_plot_s2[:, quarter_size_s2:half_size_s2 + quarter_size_s2, quarter_size_s2:half_size_s2 + quarter_size_s2]
        
    if verbose:
        print(im_loaded_alpha.shape, type(im_loaded_alpha))
        print(im_loaded_s2.shape, type(im_loaded_s2))

    fig, ax = plt.subplots(5, 5, figsize=(15, 15))
    ax = ax.flatten()
    ax_ind = 4
    for ii in range(20):
        ax_ind += 1
        bands_alpha_plot = np.arange(ii * 3, (ii + 1) * 3)
        if bands_alpha_plot.max() >= im_plot_alpha.shape[0]:
            if ax_ind < len(ax):
                naked(ax[ax_ind])
            continue
        plot_image_simple(im_plot_alpha[bands_alpha_plot, ...], ax=ax[ax_ind])
        ax[ax_ind].set_ylim(ax[ax_ind].get_ylim()[::-1])
        ax[ax_ind].set_title(f'AlphaEarth bands {bands_alpha_plot}')

    plot_image_simple(im_plot_s2, ax=ax[0])
    ax[0].set_title('Sentinel-2 RGB')

    plot_image_simple(im_nir_s2, ax=ax[1])
    ax[1].set_title('Sentinel-2 near infrared')

    for ii in range(2, 5):
        naked(ax[ii])
if __name__ == "__main__":
    print('This is a utility script for creating and processing the dataset.')