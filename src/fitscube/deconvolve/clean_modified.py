#!/usr/bin/env python3

import sys, os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.convolution
import matplotlib as mpl
import scipy as sp
import scipy.signal
import fitscube.header
import scipy.ndimage

target_cube = os.path.expanduser('~/scratch/reduced_images/SINFO.2018-09-19T02:21:05/MOV_Neptune---H+K_0.025_1_tpl/analysis/obj_NEPTUNE_cal_HD2811_renormed.fits')
#target_cube = os.path.expanduser('~/scratch/reduced_images/SINFO.2018-08-31T05:45:50/MOV_Neptune---H+K_0.1_tpl/analysis/obj_NEPTUNE_cal_HIP001115_renormed.fits')


window = True # region over which to perform the deconvolution
gain = 0.2 # fraction of brightest pixel removed on each iteration ("loop gain")
threshold = 0.6 # fraction of brightest pixel that is used as an extended component and CLEANed in one step
#rms_threshold = 1E-8 # stop cleaning when the rms value of the residual is less than this
#fabs_threshold = 5E-6 # stop cleaning when the largest residual pixel is less than this
rms_threshold_fac = 1E-2
fabs_threshold_fac = 1E-2
max_iter = int(2E2) # maximum number of iterations before cleaning stops

def rect_overlap_lims(r1_shape, r2_shape, offset):
	r1_overlap_min = offset - r2_shape/2
	r1_overlap_max = offset + r2_shape/2
	r2_overlap_min = r2_shape/2 - offset
	r2_overlap_max = r2_shape/2 + (r1_shape - offset)
	return(	np.array(np.clip(r1_overlap_min, np.zeros_like(r1_shape), r1_shape), dtype=int), 
			np.array(np.clip(r1_overlap_max, np.zeros_like(r1_shape), r1_shape), dtype=int),
			np.array(np.clip(r2_overlap_min, np.zeros_like(r2_shape), r2_shape), dtype=int),
			np.array(np.clip(r2_overlap_max, np.zeros_like(r2_shape), r2_shape), dtype=int))

def clean_modified(dirty_img, psf_img, window=True, loop_gain=1E-1, threshold=0.9, rms_threshold=1E-7, fabs_threshold=1E-7, 
				   max_iter=int(1E2), norm_psf=False):
	print('INFO: In "clean_modified()"')
	residual = np.array(dirty_img)
	#r_shape = np.array(residual.shape)
	#p_shape = np.array(psf_img.shape)
	components = np.zeros_like(residual)
	rms_record = np.full((max_iter,), fill_value=np.nan)
	fabs_record = np.full((max_iter,), fill_value=np.nan)
	selected = np.zeros_like(residual)
	above_t = np.zeros_like(residual, dtype=bool)
	if norm_psf:
		psf_img /= np.nansum(psf_img)
	
	if window is True:
		window = np.ones(dirty_img[j].shape, dtype=bool)
	
	# DEBUGGING PLOT SETUP
	plt.ioff()
	nr,nc = (2, 4)
	s = 4
	f1 = plt.figure(1,figsize=(s*nc,s*nr))
	plt.show()
	
	
	# perform CLEAN algorithm
	for i in range(max_iter):
		print(f'INFO: Iteration {i}/{max_iter}')
		#my, mx = np.unravel_index(np.nanargmax(np.fabs(residual[window])), residual.shape)
		#mval = residual[my,mx]*gain
		#components[my,mx] += mval
		#print(f'my {my} mx {mx} mval {mval}')
		selected *= 0 # set to all zeros for each step
		fabs_residual = np.fabs(residual)
		#fabs_residual = residual
		max_fabs_residual = np.nanmax(fabs_residual)
		mod_clean_threshold = threshold*max_fabs_residual
		above_t[:,:] = fabs_residual > mod_clean_threshold
		n_selected = np.sum(above_t)
		selected[above_t] = residual[above_t]
		print(f'INFO: Number of selected points {n_selected} max_fabs_residual {max_fabs_residual:07.2E} mod_clean_threshold {mod_clean_threshold:07.2E}')
		convolved = sp.signal.convolve2d(selected, psf_img, mode='same')
		#factor = np.nanmax(np.fabs(convolved))/np.nanmax(fabs_residual)
		factor = np.nanmax(fabs_residual)/np.nanmax(np.fabs(convolved))
		components += selected*loop_gain*factor
		current_cleaned = sp.signal.convolve2d(components, psf_img, mode='same')
		residual = dirty_img - current_cleaned
		
		print(f'INFO: factor {factor:07.2E} sum_components {np.nansum(components):07.2E} max_fabs_convolved {np.nanmax(np.fabs(convolved)):07.2E} max_fabs_residual {np.nanmax(fabs_residual):07.2E}')
		
		# get overlapping rectangles for residual cube and psf_cube
		#r_o_min, r_o_max, p_o_min, p_o_max = rect_overlap_lims(r_shape, p_shape, np.array([my, mx]))
		
		#residual[r_o_min[0]:r_o_max[0],r_o_min[1]:r_o_max[1]] -= psf_img[p_o_min[0]:p_o_max[0],p_o_min[1]:p_o_max[1]]*mval
		
		resid_fabs = np.nanmax(np.fabs(residual))
		resid_rms = np.sqrt(np.nansum((residual)**2)/residual.size)
		#print(f'INFO: residual maximum {resid_max} threshold {threshold}')
		print(f'INFO: residual rms {resid_rms:07.2E} threshold {rms_threshold:07.2E} fabs {resid_fabs:07.2E} threshold {fabs_threshold:07.2E}')
		rms_record[i] = resid_rms
		fabs_record[i] = resid_fabs
		if (resid_rms < rms_threshold)  or (resid_fabs < fabs_threshold):
			break
		# DEBUGGING PLOTS
		cmap = ('Spectra','twilight','coolwarm', 'bwr')[-1]
		plt.clf()
		a1 = f1.subplots(nr,nc)
		f1.suptitle(f'Loop iteration {i}')
		a1[0,0].imshow(dirty_img, origin='lower', cmap=cmap, vmin=-np.nanmax(np.fabs(dirty_img)), vmax=np.nanmax(np.fabs(dirty_img)))
		a1[0,0].set_title('dirty_img')
		a1[0,1].imshow(residual, origin='lower', cmap=cmap, vmin=-np.nanmax(np.fabs(residual)), vmax=np.nanmax(np.fabs(residual)))
		a1[0,1].set_title('residual')
		a1[0,2].imshow(components, origin='lower', cmap=cmap, vmin=-np.nanmax(np.fabs(components)),
						vmax=np.nanmax(np.fabs(components)))
		a1[0,2].set_title('components')
		# put NANs back in
		current_cleaned[np.isnan(dirty_img)] = np.nan
		a1[0,3].imshow(current_cleaned, origin='lower', cmap=cmap, vmin=-np.nanmax(np.fabs(current_cleaned)), 
						vmax=np.nanmax(np.fabs(current_cleaned)))
		a1[0,3].set_title('current CLEANed image')
		
		a1[1,0].imshow(above_t, origin='lower')
		a1[1,0].set_title('above_t')
		a1[1,1].imshow(selected, origin='lower', cmap=cmap, vmin=-np.nanmax(np.fabs(selected)), vmax=np.nanmax(np.fabs(selected)))
		a1[1,1].set_title('selected')
		# put back in the nans
		#convolved[np.isnan(dirty_img)] = np.nan
		a1[1,2].imshow(convolved, origin='lower', cmap=cmap, vmin=-np.nanmax(np.fabs(convolved)), vmax=np.nanmax(np.fabs(convolved)))
		a1[1,2].set_title('convolved')
		
		a1[1,3].plot(rms_record, color='tab:blue', label='rms')
		a1[1,3].set_title('rms and |brightest pixel|')
		a1132 = a1[1,3].twinx()
		a1132.plot(fabs_record, color='tab:orange', label='absolute brightest pixel')
		h1,l1 = a1[1,3].get_legend_handles_labels()
		h2,l2 = a1132.get_legend_handles_labels()
		#print(h1+h2)
		#print(l1+l2)
		a1[1,3].legend(h1+h2,l1+l2)
		plt.draw()
		plt.waitforbuttonpress(0.001)
		
		
	return(residual, components, rms_record, fabs_record, i)

# %%
wav_ranges = (#(1.5,1.6),
			  #(1.7,1.8),
			  (1.9,2.0),
			  )

with fits.open(target_cube) as hdul:
	dirty_cube = hdul['PRIMARY'].data # dirty cube (to be CLEANed)
	psf_cube = hdul['PSF'].data # cube to use as point spread function
	disk_mask = hdul['DISK_MASK'].data
	wavgrid = fitscube.header.get_wavelength_grid(hdul['PRIMARY'])
	
	nz,ny,nx = (len(wav_ranges), dirty_cube.shape[1], dirty_cube.shape[2])
	dirty_img = np.zeros((nz,ny,nx))
	psf_img = np.zeros((nz,psf_cube.shape[1], psf_cube.shape[2]))
	components = np.zeros((nz,ny,nx))
	residual = np.zeros((nz,ny,nx))
	rms_record = np.full((nz, max_iter), fill_value=np.nan)
	fabs_record = np.full((nz, max_iter), fill_value=np.nan)
	reconvolve = np.zeros_like(dirty_img)
	
	
	for j, wav_range in enumerate(wav_ranges):
		
		wav_start_idx = np.nanargmin(np.fabs(wavgrid-wav_range[0]))
		wav_end_idx = np.nanargmin(np.fabs(wavgrid-wav_range[1]))
		
		dirty_img[j] = np.nansum(dirty_cube[wav_start_idx:wav_end_idx,:,:], axis=0)
		# GET RID OF ANY AREAS OF THE IMAGE WE DON'T NEED TO FIT
		dirty_img[j][dirty_img[j]==0] = np.nan # set zeros to NAN, should remove holes in image
		#dirty_img[j][dirty_img[j] > 1.5E-3] = np.nan
		dirty_img[j][176:179,116:122] = np.nan # remove bright artifact from image
		dirty_img[j][disk_mask != 1] = np.nan
		#dirty_img[j,148:205,117:174] *= 0.8
		#dirty_img[j,205:206,169:174] = np.nan
		#sobel_filter = np.fabs(sp.ndimage.sobel(dirty_img[j], axis=0, mode='constant') 
		#						+ sp.ndimage.sobel(dirty_img[j], axis=1, mode='constant'))
		gl_filter = sp.ndimage.gaussian_laplace(dirty_img[j], 0.75)
		dirty_img[j][np.fabs(gl_filter)>1E-4] = np.nan
		plt.imshow(dirty_img[j], origin='lower')
		#plt.imshow(disk_mask, origin='lower')
		#plt.imshow(gl_filter, origin='lower')
		plt.show()
		sys.exit()
		
		
		#psf_img = np.nansum(psf_cube, axis=0)
		psf_img[j] = np.nansum(psf_cube[wav_start_idx:wav_end_idx,:,:], axis=0)
		residual[j] = dirty_img[j]
		
		# THIS BIT COULD BE TRICKY
		print(f'INFO: psf_img sum before norm {np.nansum(psf_img)}')
		# normalise psf cube
		psf_img[j] /= (np.nansum(psf_img[j])/1)
		#psf_img /= np.nanmax(psf_img)
		print(f'INFO: psf_img sum after norm {np.nansum(psf_img)}') # should have a value of 1 for each wavelength bin?
		
		rms_threshold = np.sqrt(np.nansum((residual[j])**2)/residual[j].size)*rms_threshold_fac
		fabs_threshold = np.nanmax(np.fabs(residual[j]))*fabs_threshold_fac
		
	
		resid_j, component_j, rms_record_j, fabs_record_j, i_j = clean_modified(dirty_img[j], 
																				psf_img[j], 
																				loop_gain=gain,
																				threshold=threshold, 
																				max_iter=int(max_iter),
																				rms_threshold=rms_threshold, 
																				fabs_threshold=fabs_threshold
																				)
		residual[j,:,:] = resid_j
		components[j,:,:] = component_j
		rms_record[j,:] = rms_record_j
		fabs_record[j,:] = fabs_record_j
		reconvolve[j] = sp.signal.convolve2d(components[j], psf_img[j], mode='same')
		
#%% plot result
j = 0
#gaussian_kernel_std = 0.75 
gaussian_kernel_std = 2
nr,nc=(2,5)
s = 4
f1 = plt.figure(figsize=(nc*s,nr*s))
a1 = f1.subplots(nr,nc,squeeze=False)

cmin = np.nanmin(dirty_img[j])
cmin = cmin if cmin > 1E-6 else 1E-6
cmax = np.nanmax(dirty_img[j])

norm = mpl.colors.SymLogNorm(1E-7, linscale=1, vmin=0, vmax=cmax)
#norm = mpl.colors.LogNorm(vmin=cmin, vmax=cmax)
#norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)


im100 = a1[0,0].imshow(components[j], origin='lower', norm=norm)
a1[0,0].set_title(f'sum of components {np.nansum(components[j]):07.2E}')
f1.colorbar(im100, ax=a1[0,0])


im101 = a1[0,1].imshow(residual[j], origin='lower', norm=norm)#, norm=mpl.colors.SymLogNorm(1E-11, linscale=1, vmin=np.nanmin(residual), vmax=np.nanmax(residual)))
a1[0,1].set_title(f'sum of residual {np.nansum(residual[j]):07.2E}')
f1.colorbar(im101,ax=a1[0,1])

im110 = a1[1,0].imshow(dirty_img[j], origin='lower', norm=norm)
a1[1,0].set_title(f'sum of dirty img {np.nansum(dirty_img[j]):07.2E}')
f1.colorbar(im110, ax=a1[1,0])

im111 = a1[1,1].imshow(psf_img[j], origin='lower')
a1[1,1].set_title(f'sum of psf {np.nansum(psf_img[j]):07.2E}')
f1.colorbar(im111, ax=a1[1,1])

a1[0,2].plot(rms_record[j])
a1[0,2].set_title('Residual RMS vs iteration')
a1[0,2].set_yscale('log')

a1[1,2].plot(fabs_record[j])
a1[1,2].set_title('Residual fabs vs iteration')
a1[1,2].set_yscale('log')

a1[0,3].imshow(reconvolve[j], origin='lower', norm=norm)
a1[0,3].set_title('components convolved with psf')

a1[1,3].imshow(reconvolve[j] + residual[j], origin='lower', norm=norm)
a1[1,3].set_title('components convolved with psf and residual added')

gaussian_kernel = astropy.convolution.Gaussian2DKernel(gaussian_kernel_std)
a1[0,4].imshow(astropy.convolution.convolve(components[j], gaussian_kernel), origin='lower', norm=norm)
a1[0,4].set_title('components convolved with gaussian kernel')

gk_array = np.zeros_like(psf_img[j])
gk_array[psf_img[j].shape[0]//2, psf_img[j].shape[1]//2] = 1
gk_array = astropy.convolution.convolve(gk_array, gaussian_kernel)
a1[1,4].imshow(gk_array, origin='lower')
a1[1,4].set_title('gaussian kernel')

plt.show()