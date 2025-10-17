#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 12:20:07 2025

@author: jbakermans
"""

import data_utils as du
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colormaps
from matplotlib.patches import Rectangle
import scipy.optimize as opt

# Quick utility function for fitting 2d gaussians
# See https://stackoverflow.com/a/77432576/8919448
def gauss_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    return g.ravel()

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
N_hypotheses = hypotheses[0].shape[0]

# Run through all patches, collecting spike triggered averages for each feature for each hypothesis
patch_stas = []
for p, (hypothesis, feature) in enumerate(zip(hypotheses, features)):
    print(f'Analysing patch {p+1} / {len(hypotheses)}')
    
    # Create empty region of interest maps: area around each pixel for each band
    rois = np.full([N_hypotheses, N_pixels - 2 * radius, N_pixels - 2 * radius,
                    radius * 2 + 1, radius * 2 + 1], np.nan)
    
    # Collect searchlight data for each pixel from all bands of current modality
    for h, hyp in enumerate(hypothesis):
        print(f'Copying hyp {h} / {len(hypotheses)}')
        for row in range(radius, N_pixels - radius):
            for col in range(radius, N_pixels - radius):
                # Grab the relevant pixels from the band
                rois[h, row - radius, col - radius, :, :] = \
                    hyp[(row - radius):(row + radius + 1), (col - radius):(col + radius + 1)]
                    
    # Collect spike time averages for each band
    stas = []
    for h, hyp_rois in enumerate(rois):
        print(f'Calculating spike triggered averages for hyp {h} / {len(rois)}')
        # Then create the spike time average: multiply each roi by the pixel value of the feature pixel
        hyp_stas = [hyp_rois * d[radius:(N_pixels-radius),radius:(N_pixels-radius),None,None] for d in feature]
        # Then average across all pixels and stack to get big output array
        hyp_stas = np.stack([np.nansum(sta.reshape([-1, radius*2+1, radius*2+1]), axis=0) for sta in hyp_stas])
        # Also average the hypothesis itself across all pixels
        hyp_norm = np.nansum(np.reshape(hyp_rois, [-1, radius*2+1, radius*2+1]),axis=0)
        # Regress out the contribution of simple hypothesis geometry from responses
        X = np.stack([np.ones(hyp_norm.size), hyp_norm.reshape(-1)], axis=-1)
        Y = hyp_stas.reshape([N_features,-1]).T
        b = np.linalg.pinv(X) @ Y
        e = Y - X@b
        corr_stas = e.T.reshape([N_features, radius * 2 + 1, radius * 2 + 1])
        # Plot for debugging purposes
        if False:
            plt.figure();
            for r in range(8):
                for c in range(8):
                    plt.subplot(8,8,r*8+c+1)
                    plt.imshow(feature[r*8+c])
            plt.figure();
            for r in range(8):
                for c in range(8):
                    plt.subplot(8,8,r*8+c+1)
                    plt.imshow(hyp_stas[r*8+c])
            plt.figure();
            for r in range(8):
                for c in range(8):
                    plt.subplot(8,8,r*8+c+1)
                    plt.imshow(corr_stas[r*8+c])
            plt.figure(); plt.imshow(hypothesis[h])
            plt.figure(); plt.imshow(hyp_norm)       
            import pdb; pdb.set_trace()             
        # And append to output stas
        stas.append(hyp_stas)
    patch_stas.append(np.stack(stas))
    
# Average across patches to get final stas
all_stas = np.nanmean(np.stack(patch_stas), axis=0)

# Plot a selection
h_to_plot = len(all_stas)
f_to_plot = 32
plt.figure(figsize=(f_to_plot, h_to_plot))
# Color plot borders by average value
lim = np.nanmax(np.abs(all_stas[:h_to_plot, :f_to_plot]))
cm = colormaps.get_cmap('RdBu_r')
# Plot one tuning curve per hypothesis per feature
for row, hyp_stas in enumerate(all_stas[:h_to_plot]):
    for col, sta in enumerate(hyp_stas[:f_to_plot]):
        ax = plt.subplot(h_to_plot, f_to_plot, row * f_to_plot + col + 1)
        ax.imshow(sta,cmap='Greys')
        ax.add_patch(Rectangle([0, 0,], sta.shape[1]-1, sta.shape[0]-1,
            facecolor=[0,0,0,0], 
            edgecolor=cm((np.nanmean(sta)+lim)/(2*lim)),
            linewidth=4))
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(names[row].replace('_','\n'), rotation=0, labelpad=20)
        if row == 0:
            ax.set_title(f'F{col}')
            
# Plot the map for a particular feature that you might like across patches
cols=4
rows=int(np.ceil(patches/cols))
curr_f = 12
lim = np.nanmax(np.abs(np.stack(features, axis=-1)[curr_f]))
plt.figure(figsize=(2*cols, 2*rows))
for p, feature in enumerate(features):
    plt.subplot(rows, cols, p + 1)
    plt.imshow(feature[curr_f], vmin=-lim, vmax=lim, cmap="RdBu_r")
    plt.axis('off')

# Fit a gaussian to each sta
f_to_plot=10
h_to_plot=5

x, y = np.meshgrid(np.arange(2*radius+1), np.arange(2*radius+1))
# Collect fitted parameters and resulting images
all_params = np.full(list(all_stas.shape[:-2]) + [7], np.nan)
all_fits = np.full(all_stas.shape, np.nan)
for row, hyp_stas in enumerate(all_stas[:h_to_plot]):
    print(f'Fitting Gaussians to hyp {row} / {N_hypotheses}')
    for col, sta in enumerate(hyp_stas[:f_to_plot]):
        curr_fits = []
        curr_pars = []
        for fit_type in ['pos', 'neg']:
            # Set fit type dependent initial guesses: amplitude, offset, center
            if fit_type == 'pos':       
                a_0 = np.max(sta) - np.min(sta)
                (y_0, x_0) = np.unravel_index(sta.argmax(), sta.shape)
                o_0 = np.min(sta)
            else:
                a_0 = np.min(sta) - np.max(sta)
                (y_0, x_0) = np.unravel_index(sta.argmin(), sta.shape)
                o_0 = np.max(sta)
            # Estimate the std as the pixel distance between the peak and the inflexion point:
            # The location where the second derivative is nearest to 0
            curve = np.mean(sta, axis=0)
            curve = np.diff(np.diff(np.mean(sta, axis=0)))
            sx_0 = np.abs(x_0 - np.argmin(np.abs(np.diff(np.diff(np.mean(sta, axis=0))))) - 1) # -1 because diff loses dim
            sy_0 = np.abs(y_0 - np.argmin(np.abs(np.diff(np.diff(np.mean(sta, axis=1))))) - 1)
            theta_0 = 0.0
            # Do the actual fit
            try:
                # find the optimal Gaussian parameters
                popt, pcov = opt.curve_fit(gauss_2d, (x, y), sta.ravel(), 
                                           p0=(a_0, x_0, y_0, sx_0, sy_0, theta_0, o_0),
                                           maxfev=int(1e5))   
                # Store the resulting parameters and fit
                curr_fits.append(gauss_2d((x, y), *popt).reshape(2*radius+1,2*radius+1))
                curr_pars.append(popt)
            except RuntimeError as e:
                # This generally happens when we run out of iterations
                # That's usually caused by strongly non-gaussian stas. Let's just ignore those
                print(f'Gaussian fit failed for type {fit_type}, hyp {row}, feature {col}. \n Error message: {e}')
                # Store the resulting parameters and fit
                curr_fits.append(np.zeros_like(sta))
                curr_pars.append(np.zeros(7))
        # if row == 1 and col == 5:
        #     import pdb; pdb.set_trace()
        # Only keep the best fit, between the positive and negative peak
        best_fit = int(np.sum(np.square(sta - curr_fits[0])) > np.sum(np.square(sta - curr_fits[1])))
        # Keep the best fit between min and max
        all_params[row, col, :] = curr_pars[best_fit]
        all_fits[row, col, :, :] = curr_fits[best_fit]

# Plot the fits
plt.figure(figsize=(f_to_plot, h_to_plot))
# Plot one tuning curve per hypothesis per feature
for row, hyp_fits in enumerate(all_fits[:h_to_plot]):
    for col, fit in enumerate(hyp_fits[:f_to_plot]):
        ax = plt.subplot(h_to_plot, f_to_plot, row * f_to_plot + col + 1)
        ax.imshow(fit)
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(names[row].replace('_','\n'), rotation=0, labelpad=20)
        if row == 0:
            ax.set_title(f'F{col}')