#!/usr/bin/env python3
"""
Creates a synthetic image from the results of a NEMESIS run and compares it to a specific observerd fitscube

TODO:
* Make a movie of the results with wavelength along the time axis rather than just a median image,
  want to see if we are getting good results or not
* Make a plot of residual and residual^2 vs wavelength
* Make a plot of chisq vs wavelength
* Do the same (all plots) for apriori data rather than retrieved values so we can see what's changed
"""

import sys, os
import glob
import numpy as np
from astropy.io import fits
import nemesis.read
import fitscube.limb_darkening.minnaert
import matplotlib.pyplot as plt
import matplotlib as mpl

#%% set up inputs

test_dir = os.path.expanduser('~/scratch/nemesis_run_results/slurm/combined_cbh_v_hsv_000/03')
test_runs = glob.glob(f'{test_dir}/lat*/neptune.inp')
test_runs = [_x[:-4] for _x in test_runs]

#comparison_fits = os.path.expanduser("~/scratch/reduced_images/SINFO.2018-08-18T04:00:14/MOV_Neptune---H+K_0.1_2_tpl/analysis/obj_NEPTUNE_cal_HIP001115_renormed.fits")
comparison_fits = os.path.expanduser("~/scratch/reduced_images/SINFO.2018-09-26T04:35:04/MOV_Neptune---H+K_0.025_1_tpl/analysis/obj_NEPTUNE_cal_HIP104320_renormed.fits")
#comparison_fits = os.path.expanduser("~/scratch/reduced_images/SINFO.2018-09-30T01:02:39/MOV_Neptune---H+K_0.025_3_tpl/analysis/obj_NEPTUNE_cal_HD216009_renormed.fits")

print(test_runs)

# get latitudes from file path
lat_mins = [float(_x.split(os.sep)[-2].split('_')[1]) for _x in test_runs]
lat_maxs = [float(_x.split(os.sep)[-2].split('_')[3]) for _x in test_runs]
lat_lims = np.array(list(zip(lat_mins, lat_maxs)))
sort_idxs = np.argsort(lat_lims, axis=0)
#lat_lims = lat_lims[sort_idxs]

print(lat_mins)
print(lat_maxs)
print(lat_lims)
print(lat_lims.shape)

lat_mre_ds = []
lat_spx_ds = []
for afile in test_runs:
	lat_mre_ds.append(nemesis.read.mre(afile))
	lat_spx_ds.append(nemesis.read.spx(afile))

print(lat_mre_ds[-1].keys())

with fits.open(comparison_fits) as hdul:
	lats = hdul['LATITUDE'].data
	zens = hdul['ZENITH'].data

print(lat_mre_ds[-1]['waveln'])


#%% calculate synth image data
#lats_3d = np.broadcast_to(lats, synth_img.shape)

synth_img = np.zeros((lat_mre_ds[-1]['waveln'].shape[1], *lats.shape))
coverage = np.zeros_like(synth_img)
print(f'INFO: synth_img.shape {synth_img.shape}')
print(f'INFO: lat_lims.shape {lat_lims.shape}')

for i in range(0, lat_lims.shape[0]):
	print(f'INFO: Calculating for latitude section {i}/{lat_lims.shape[0]} from {lat_lims[i,0]} to {lat_lims[i,1]}')
	#lat_idxs = np.nonzero(np.logical_and(lat_lims[i,0] <= lats_3d, lats_3d <= lat_lims[i,1]))
	lat_idxs = np.nonzero(np.logical_and(lat_lims[i,0] <= lats, lats <= lat_lims[i,1]))
	
	#print(lat_idxs)
	
	ngeom = lat_spx_ds[i]['ngeom']
	ul = []
	u_0l = []
	for j in range(ngeom):
		# get I/F0 and k values vs wavelength for data from mre file
		#print(lat_spx_ds[i]['fov_averaging_record'][j][:,3])
		ul.append(np.cos(lat_spx_ds[i]['fov_averaging_record'][j][:,3]*np.pi/180))
		u_0l.append(np.cos(lat_spx_ds[i]['fov_averaging_record'][j][:,2]*np.pi/180))
	
	IperF = lat_mre_ds[i]['radiance_retr']*1E-6 # change from uW cm-2 sr-1 um-1 to W cm-2 sr-1 um-1
	u = np.broadcast_to(np.array(ul), IperF.shape)
	u_0 = np.broadcast_to(np.array(u_0l), IperF.shape)
	#print(u.shape, u_0.shape, IperFs.shape)
	IperF0s, ks, log_IperF0s_var, ks_var = fitscube.limb_darkening.minnaert.get_coefficients_v_wavelength(u_0.transpose(), u.transpose(), IperF.transpose())
	
	#print('INFO:')
	#print(IperF0s)
	#print(ks)
	# get u and u0 values for latitude range
	u_in = np.cos(zens*np.pi/180)
	u_0_in = np.cos(zens*np.pi/180)
	
	#print(IperF0s.shape, ks.shape, u_in.shape, u_0_in.shape)
	
	# get I/F values for I/F0, k, u, u0 we got from the mre file and latitude range
	IperF_in = fitscube.limb_darkening.minnaert.calculate(IperF0s[:,None,None], ks[:,None,None], u_0_in[None,:,:], u_in[None,:,:])
	
	# add I/F to the synth_img map
	synth_img[:,lat_idxs[0],lat_idxs[1]] += IperF_in[:,lat_idxs[0],lat_idxs[1]]
	
	# add 1 to the coverage map
	coverage[:,lat_idxs[0],lat_idxs[1]] += 1
	
print(f'INFO: synth_img.shape {synth_img.shape}')
print(f'INFO: coverage.shape {coverage.shape}')
# divide synth_img map by the coverage map to get average synth_img (as latitude bands can overlap)
#coverage[coverage==0] = 1 # set zeros to 1 so that we can divide by it
#synth_img /= coverage
covered = coverage != 0
synth_img[covered] /= coverage[covered]
synth_img[~covered] = np.nan


#%% display synth_img and comparison fits as a movie with wavelength on the time axis.

# TODO: Fix this plot, the synth_img plot should agree mostly with the hdul plot
f1 = plt.figure(figsize=(18,6))
a1 = f1.subplots(1,3, squeeze=False)
cax = f1.add_axes([0.1,0.05, 0.8, 0.05])

synth_img_data = np.nanmedian(synth_img, axis=0)
with fits.open(comparison_fits) as hdul:
	compar_data = np.nanmedian(hdul['PRIMARY'].data, axis=0)
compar_lims = (np.nanmin(compar_data), np.nanmax(compar_data))

norm = mpl.colors.SymLogNorm(1E-10, linscale=1, vmin=-compar_lims[1], vmax=compar_lims[1])
cmap = 'Spectral'

"""
# create a divergent colour map
# 256 colors in entire range, therefore join between colorbars should be at position 256*x_join_frac
r_cmap = lambda x_join_frac: mpl.colors.LinearSegmentedColormap.from_list('residual_cmap',
															np.vstack([
															mpl.cm.gist_heat_r(np.linspace(0,1,int(np.ceil(x_join_frac*256)))), 
															mpl.cm.viridis(np.linspace(0,1,int(np.floor((1.0-x_join_frac)*256))))
															]))
cmap = r_cmap(0.5)
"""

im1 = a1[0,0].imshow(synth_img_data, origin='lower', interpolation=None, norm=norm, cmap=cmap)#vmin = compar_lims[0], vmax=compar_lims[1])
im2 = a1[0,1].imshow(compar_data, origin='lower', interpolation=None, norm=norm, cmap=cmap)#vmin = compar_lims[0], vmax=compar_lims[1])
im3 = a1[0,2].imshow(compar_data - synth_img_data, origin='lower', interpolation=None, norm=norm, cmap=cmap)#vmin = compar_lims[0], vmax=compar_lims[1])

f1.colorbar(im1, cax=cax, orientation='horizontal')


plt.show()


