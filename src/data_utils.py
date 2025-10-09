import os, sys, json 
import numpy as np 
import pandas as pd 
# from dwca.read import DwCAReader
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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
# import scipy.spatial, scipy.cluster
from collections import Counter
import loadpaths
path_dict = loadpaths.loadpaths()
sys.path.append(os.path.join(path_dict['repo'], 'content/'))

ONLINE_ACCESS_TO_GEE = True 
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

def create_cmap_dynamic_world():
    dict_classes = {
        'water': '#419bdf',
        'trees': '#397d49',
        'grass': '#88b053',
        'flooded_vegetation': '#7a87c6',
        'crops': '#e49635',
        'shrub_and_scrub': '#dfc35a',
        'built': '#c4281b',
        'bare': '#a59b8f',
        'snow_and_ice': '#b39fe1'
    }
    return dict_classes

def create_timestamp(include_seconds=False):
    dt = datetime.datetime.now()
    timestamp = str(dt.date()) + '-' + str(dt.hour).zfill(2) + str(dt.minute).zfill(2)
    if include_seconds:
        timestamp += ':' + str(dt.second).zfill(2)
    return timestamp

def get_gee_image_from_point(coords, bool_buffer_in_deg=True, buffer_deg=0.01, 
                             verbose=0, year=None, threshold_size=128,
                             month_start_str='06', month_end_str='09',
                             image_collection='sentinel2'):
    assert ONLINE_ACCESS_TO_GEE, 'Need to set ONLINE_ACCESS_TO_GEE to True to use this function'
    assert image_collection in ['sentinel2', 'alphaearth', 'worldclimbio', 'dynamicworld'], f'image_collection {image_collection} not recognised.'
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
    elif image_collection == 'dynamicworld':
        # ex_collection = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        # ex_im_gee = ee.Image(ex_collection 
        #                     #   .project(crs='EPSG:27700', scale=1)
        #                     .filterBounds(aoi) 
        #                     .filterDate(ee.Date(f'{year}-01-01'), ee.Date(f'{year}-12-31')) 
        #                     .first() 
        #                     .clip(aoi))
        prob_bands = [
            "water", "trees", "grass", "flooded_vegetation",
            "crops", "shrub_and_scrub", "built", "bare", "snow_and_ice"
        ]
        ex_collection = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        ex_im_gee = ee.Image(ex_collection 
                            #   .project(crs='EPSG:27700', scale=1)
                            .filterBounds(aoi) 
                            .filterDate(ee.Date(f'{year}-01-01'), ee.Date(f'{year}-12-31'))
                            .select(prob_bands)  # get all probability bands
                            .mean()  # mean over the year
                            .reproject('EPSG:4326', scale=10)  # reproject to 10m
                            .clip(aoi)
                            )  # mean over the year
    elif image_collection == 'worldclimbio':
        ex_im_gee = ee.Image("WORLDCLIM/V1/BIO").clip(aoi) 
        point = ee.Geometry.Point(coords)  # redefine point for sampling
        values = ex_im_gee.sample(region=point.buffer(1000), scale=1000).first().toDictionary().getInfo()
        return values
    else:
        raise NotImplementedError(image_collection)

    im_dims = ex_im_gee.getInfo()["bands"][0]["dimensions"]
    
    if im_dims[0] < threshold_size or im_dims[1] < threshold_size:
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
                    path_save=None, resize_image=True):
    assert image_collection in ['sentinel2', 'alphaearth', 'worldclimbio', 'dynamicworld'], f'image collection {image_collection} not recognised.'
    if year is None:
        year = 2024

    im_gee = get_gee_image_from_point(coords=coords, bool_buffer_in_deg=bool_buffer_in_deg, buffer_deg=buffer_deg, 
                           verbose=verbose, year=year, 
                           month_start_str=month_start_str, month_end_str=month_end_str,
                           image_collection=image_collection,
                           threshold_size=128)
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
    elif image_collection == 'worldclimbio':
        filename = f'{name}_worldclimbio_v1.json'
    elif image_collection == 'dynamicworld':
        filename = f'{name}_dynamicworld_y-{year}.tif'
    filepath = os.path.join(path_save, filename)

    if image_collection == 'worldclimbio':  # just return values
        dict_save = {**im_gee, **{'coords': coords, 'name': name}}
        with open(filepath, 'w') as f:
            json.dump(dict_save, f)
        return dict_save, filepath
    
    geemap.ee_export_image(
        im_gee, filename=filepath, 
        scale=10,  # 10m bands
        file_per_band=False,# crs='EPSG:32630'
    )

    if resize_image:
        ## load & save to size correctly (because of buffer): 
        im = load_tiff(filepath, datatype='da')
        remove_if_too_small = True
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
        im_gee = im_crop 

    return im_gee, filepath

def download_list_coord(coord_list, path_save=None, name_group='sample'):
    assert type(coord_list) == list
    for i, coords in enumerate(tqdm(coord_list)):
        name = f'{name_group}-{i}'
        for im_collection in ['sentinel2', 'alphaearth', 'dynamicworld', 'worldclimbio']:
            im, path_im = download_gee_image(coords=coords, name=name, 
                                             path_save=path_save, verbose=0,
                                             image_collection=im_collection)
            if im is None:
                print(f'Image {name} could not be downloaded')

def get_images_from_name(path_folder=path_dict['data_folder'], name='sample-0'):
    assert os.path.exists(path_folder), path_folder
    contents = [f for f in os.listdir(path_folder) if name in f]
    assert len(contents) > 0, f'No files found in {path_folder}'
    assert len(contents) <= 4, f'More than 4 files found in {path_folder}, please specify name more precisely: {contents}'

    file_sent, file_alpha, file_dynamic, file_worldclimbio = None, None, None, None
    for file in contents:
        if file.split('_')[1].startswith('sent2'):
            file_sent = file
        elif file.split('_')[1].startswith('alphaearth'):
            file_alpha = file
        elif file.split('_')[1].startswith('dynamicworld'):
            file_dynamic = file
        elif file.split('_')[1].startswith('worldclimbio'):
            file_worldclimbio = file

    if file_sent is None and file_alpha is None and file_dynamic is None and file_worldclimbio is None:
        raise ValueError(f'No recognised files found in {path_folder} with name {name}')
    return (file_sent, file_alpha, file_dynamic, file_worldclimbio)


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

def plot_overview_images(path_folder=path_dict['data_folder'], name='sample-0', 
                         plot_alphaearth=True, plot_dynamicworld_full=True,
                         verbose=0):
    (file_sent, file_alpha, file_dynamic, file_worldclimbio) = get_images_from_name(path_folder=path_folder, name=name)
    path_sent = os.path.join(path_folder, file_sent) if file_sent is not None else None
    path_alpha = os.path.join(path_folder, file_alpha) if file_alpha is not None else None
    path_dynamic = os.path.join(path_folder, file_dynamic) if file_dynamic is not None else None
    path_worldclimbio = os.path.join(path_folder, file_worldclimbio) if file_worldclimbio is not None else None

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
        # im_nir_s2 = im_loaded_s2[1:, ...]
        im_nir_s2 = im_loaded_s2[np.array([0, 1, 3]), ...]  # B4, B3, B8
        ## put B8 band first:
        im_nir_s2 = im_nir_s2[[2, 0, 1], ...]
        size_s2 = im_plot_s2.shape[1]
        assert size_s2 == im_plot_s2.shape[2]
        half_size_s2 = size_s2 // 2
        quarter_size_s2 = size_s2 // 4
        # im_plot_s2 = im_plot_s2[:, quarter_size_s2:half_size_s2 + quarter_size_s2, quarter_size_s2:half_size_s2 + quarter_size_s2]
        
    if path_dynamic is not None:
        im_loaded_dynamic = load_tiff(path_dynamic, datatype='da')
        im_argmax_dynamic = np.argmax(im_loaded_dynamic.values, axis=0)
        print(Counter(im_argmax_dynamic.flatten()))
        ## normalise:
        # im_plot_dynamic.values[im_plot_dynamic.values == -np.inf] = np.nan
        # im_plot_dynamic.values[im_plot_dynamic.values == np.inf] = np.nan
        # im_plot_dynamic = (im_plot_dynamic - np.nanmin(im_plot_dynamic)) / (np.nanmax(im_plot_dynamic) - np.nanmin(im_plot_dynamic))

    if verbose:
        print(im_loaded_alpha.shape, type(im_loaded_alpha))
        print(im_loaded_s2.shape, type(im_loaded_s2))

    n_rows = 1 + 2 * int(plot_dynamicworld_full) + 4 * int(plot_alphaearth)

    fig, ax = plt.subplots(n_rows, 5, figsize=(15, 3 * n_rows))
    ax = ax.flatten()

    ## Top row:
    plot_image_simple(im_plot_s2, ax=ax[0])
    ax[0].set_title('Sentinel-2 RGB')

    plot_image_simple(im_nir_s2, ax=ax[1])
    ax[1].set_title('Sentinel-2 near infrared')

    if path_dynamic is not None:
        dict_classes = create_cmap_dynamic_world()
        cmap_dw = ListedColormap([v for v in dict_classes.values()])
        im = ax[2].imshow(im_argmax_dynamic, cmap=cmap_dw, interpolation='none', origin='upper', vmax=8.5, vmin=-0.5)
        # Place colorbar outside of ax[2] to avoid shrinking the imshow
        cbar = fig.colorbar(im, ax=ax[2], ticks=np.arange(0, 9), location='right', fraction=0.046, pad=0.04)
        cbar.ax.set_yticks(np.arange(0, 9))
        cbar.ax.set_yticklabels([k for k in dict_classes.keys()])
        # ax[2].set_title('Dynamic World land cover')
        naked(ax[2])

    for ii in range(3, 5):
        naked(ax[ii])

    ## dynamic world full:
    if plot_dynamicworld_full and path_dynamic is not None:
        for ii in range(9):
            ax_ind = 5 + ii
            im = ax[ax_ind].imshow(im_loaded_dynamic[ii, ...], cmap='viridis', interpolation='none', 
                                   origin='upper', vmin=0, vmax=1)
            naked(ax[ax_ind])
            ax[ax_ind].set_title(f'DW {ii}: {list(dict_classes.keys())[ii]}')
        ## add cbar to last plot:
        cbar = fig.colorbar(im, ax=ax[ax_ind], ticks=np.linspace(0, 1, 6),
                            location='right', fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('Probability', rotation=270, labelpad=15)
        ax_ind += 1
        naked(ax[ax_ind])

    ## alpha earth:
    if plot_alphaearth and path_alpha is not None:
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

def load_all_modalities_from_name(path_folder=path_dict['data_folder'], name='sample-0', verbose=0):
    (file_sent, file_alpha, file_dynamic, file_worldclimbio) = get_images_from_name(path_folder=path_folder, name=name)
    path_sent = os.path.join(path_folder, file_sent) if file_sent is not None else None
    path_alpha = os.path.join(path_folder, file_alpha) if file_alpha is not None else None
    path_dynamic = os.path.join(path_folder, file_dynamic) if file_dynamic is not None else None
    path_worldclimbio = os.path.join(path_folder, file_worldclimbio) if file_worldclimbio is not None else None

    im_loaded_alpha, im_loaded_s2, im_loaded_dynamic, im_loaded_worldclimbio = None, None, None, None

    if path_sent is not None:
        im_loaded_s2 = load_tiff(path_sent, datatype='da')
        if verbose:
            print('Sentinel-2:', im_loaded_s2.shape, type(im_loaded_s2))
    else:
        if verbose:
            print('No sentinel-2 image found')

    if path_alpha is not None:
        im_loaded_alpha = load_tiff(path_alpha, datatype='da')
        ## vertical flip:
        im_loaded_alpha = im_loaded_alpha[:, ::-1, :]
        if verbose:
            print('AlphaEarth:', im_loaded_alpha.shape, type(im_loaded_alpha))
    else:
        if verbose:
            print('No alphaearth image found')

    if path_dynamic is not None:
        im_loaded_dynamic = load_tiff(path_dynamic, datatype='da')
        if verbose:
            print('Dynamic World:', im_loaded_dynamic.shape, type(im_loaded_dynamic))
    else:
        if verbose:
            print('No dynamic world image found')
    if path_worldclimbio is not None:
        with open(path_worldclimbio, 'r') as f:
            im_loaded_worldclimbio = json.load(f)
        if verbose:
            print('WorldClimBio:', type(im_loaded_worldclimbio), im_loaded_worldclimbio.keys())
    else:
        if verbose:
            print('No worldclimbio data found')

    return im_loaded_s2, im_loaded_alpha, im_loaded_dynamic, im_loaded_worldclimbio
    
def plot_distr_embeddings(path_folder=path_dict['data_folder'], name='sample-0', verbose=0):
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

    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    ax = ax.flatten()

    plot_image_simple(im_plot_s2, ax=ax[0])
    ax[0].set_title('Sentinel-2 RGB')

    for ii in range(8):
        bands = np.arange(ii * 8, (ii + 1) * 8)
        if bands.max() >= im_plot_alpha.shape[0]:
            if ii + 1 < len(ax):
                naked(ax[ii + 1])
            continue

        ax_ind = ii + 1
        curr_ax = ax[ax_ind]
        for jj, band in enumerate(bands):
            if band >= im_plot_alpha.shape[0]:
                continue
            curr_ax.hist(im_plot_alpha[band, ...].to_numpy().flatten(), bins=np.linspace(0, 1, 41), alpha=0.5, label=f'Band {band}',
                         histtype='stepfilled', density=True, color=plt.cm.viridis(jj / len(bands)))
        curr_ax.set_xlim(-0.1, 1.1)
if __name__ == "__main__":
    print('This is a utility script for creating and processing the dataset.')