#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 12:20:07 2025

@author: jbakermans
"""

import data_utils as du
import numpy as np
from matplotlib import pyplot as plt

# Get one example dataset and plot
(data_sent, data_alpha, data_dyn, data_worldclim, data_dsm) = du.load_all_modalities_from_name(name='pecl-fig-0', path_folder='../content/sample_data', verbose=1)
du.plot_overview_images('../content/sample_data', name='pecl-fig-0', plot_alphaearth=True, plot_dynamicworld_full=True)

# Then collect from all patches the alpha and dyn data
patches = 10
hypotheses = []
features = []
for p in range(patches):
    (data_sent, data_alpha, data_dyn, data_worldclim, data_dsm) = du.load_all_modalities_from_name(name=f'pecl-fig-{p}', path_folder='../content/sample_data', verbose=1)
    # Land coverage serves as hypotheses
    hypotheses.append(data_dyn.data)
    # This can definitely be cleaner but I use nan for undefined values
    f_dat = data_alpha.data
    f_dat[~np.isfinite(f_dat)] = np.nan
    features.append(f_dat)    

# Z-score across patches for each feature and each hypothesis
feat_m = np.stack([np.nanmean(f) for f in np.stack(features, axis=-1)])
feat_std = np.stack([np.nanstd(f) for f in np.stack(features, axis=-1)])
features = [(f - feat_m[:,None,None])/feat_std[:,None,None] for f in features]
hyp_m = np.stack([np.nanmean(h) for h in np.stack(hypotheses, axis=-1)])
hyp_std = np.stack([np.nanstd(h) for h in np.stack(hypotheses, axis=-1)])
hypotheses = [(h - hyp_m[:,None,None])/hyp_std[:,None,None] for h in hypotheses]

# Get names of hypotheses: different coarse land coverage classes
names = [k for k in du.create_cmap_dynamic_world().keys()]

# Extract relevant dimensions
radius = 5 # pixels, excluding center pixel (so diameter = 2 * radius + 1)
N_features = features[0].shape[0]
N_pixels = features[0].shape[1]
N_bands = hypotheses[0].shape[0]

# Precalculate for each pixel which values need to be read and written
read = np.zeros([N_pixels, N_pixels, 2, 2], dtype=int)
write = np.zeros([N_pixels, N_pixels, 2, 2], dtype=int)
for row in range(N_pixels):
    for col in range(N_pixels):
        # Choose region of this band to read out
        read[row, col, 0, 0] = max(0, row - radius) # row start
        read[row, col, 0, 1] = min(row + radius + 1, N_pixels) # row stop
        read[row, col, 1, 0] = max(0, col - radius) # col start
        read[row, col, 1, 1] = min(col + radius + 1, N_pixels) # col stop
        # Choose where in the roi to write this to
        write[row, col, 0, 0] = read[row, col, 0, 0] - (row-radius)
        write[row, col, 0, 1] = write[row, col, 0, 0] + (read[row, col, 0, 1] - read[row, col, 0, 0])
        write[row, col, 1, 0] = read[row, col, 1, 0] - (col - radius)
        write[row, col, 1, 1] = write[row, col, 1, 0] + (read[row, col, 1, 1] - read[row, col, 1, 0])

# Run through all patches, collecting spike triggered averages for each feature for each hypothesis
patch_stas = []
for p, (hypothesis, feature) in enumerate(zip(hypotheses, features)):
    print(f'Analysing patch {p+1} / {len(hypotheses)}')
    # Create empty region of interest maps: area around each pixel for each band
    rois = np.full([N_bands, N_pixels, N_pixels, radius * 2 + 1, radius * 2 + 1], np.nan)
    
    # Collect searchlight data for each pixel from all bands of current modality
    for b, band in enumerate(hypothesis):
        print(f'Copying band {b} / {len(hypothesis)}')
        for row in range(N_pixels):
            for col in range(N_pixels):
                # Grab the relevant pixels from the band
                rois[b, row, col, write[row, col, 0, 0]:write[row, col, 0, 1], write[row, col, 1, 0]:write[row, col, 1, 1]] = \
                    band[read[row, col, 0, 0]:read[row, col, 0, 1], read[row, col, 1, 0]:read[row, col, 1, 1]]
                    
    # Collect spike time averages for each band
    stas = []
    for b, band_rois in enumerate(rois):
        print(f'Calculating spike triggered averages for band {b} / {len(rois)}')
        # Then create the spike time average: multiply each roi by the pixel value of the feature pixel
        band_stas = [band_rois * d[:,:,None,None] for d in feature]
        # Then average across all pixels and stack to get big output array
        band_stas = np.stack([np.nansum(sta.reshape([-1, radius*2+1, radius*2+1]), axis=0) for sta in band_stas])
        # And append to output stas
        stas.append(band_stas)
    patch_stas.append(np.stack(stas))
# Average across patches to get final stas
all_stas = np.nanmean(np.stack(patch_stas), axis=0)
    
# Plot a selection
b_to_plot = len(all_stas)
f_to_plot = 32
common_scale = False
plt.figure(figsize=(f_to_plot, b_to_plot))
lim = np.nanmax(np.abs(all_stas[:b_to_plot, :f_to_plot]))
for row, band_stas in enumerate(all_stas[:b_to_plot]):
    for col, sta in enumerate(band_stas[:f_to_plot]):
        plt.subplot(b_to_plot, f_to_plot, row * f_to_plot + col + 1)
        plt.imshow(sta, cmap='RdBu_r' if common_scale else 'viridis', 
                   vmin=-lim if common_scale else np.nanmin(sta), 
                   vmax=lim if common_scale else np.nanmax(sta))
        plt.xticks([])
        plt.yticks([])
        if col == 0:
            plt.ylabel(names[row], rotation=0, labelpad=20)
        if row == 0:
            plt.title(f'F{col}')
            
# Plot the map for a particular feature that you might like across patches
cols=4
rows=int(np.ceil(patches/cols))
curr_f = 9
lim = np.nanmax(np.abs(np.stack(features, axis=-1)[curr_f]))
plt.figure(figsize=(2*cols, 2*rows))
for p, feature in enumerate(features):
    plt.subplot(rows, cols, p + 1)
    plt.imshow(feature[curr_f], vmin=-lim, vmax=lim, cmap="RdBu_r")
    plt.axis('off')
