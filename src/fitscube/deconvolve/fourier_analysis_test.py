#!/usr/bin/env python3

import sys, os
import numpy as np
from astropy.io import fits
import astropy.convolution
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy as sp
import scipy.signal
import fitscube.header
import scipy.ndimage

def clean_modified(dirty_img, psf_img, window=True, loop_gain=1E-1, threshold=0.6, rms_frac_threshold=0.05, fabs_frac_threshold=0.05, 
				   max_iter=int(1E2), norm_psf=False, plots=True):
	print('INFO: In "clean_modified()"')
	use_img = np.array(dirty_img)
	use_img[~window] = np.nan
	residual = np.array(use_img)
	#r_shape = np.array(residual.shape)
	#p_shape = np.array(psf_img.shape)
	components = np.zeros_like(residual)
	rms_record = np.full((max_iter,), fill_value=np.nan)
	fabs_record = np.full((max_iter,), fill_value=np.nan)
	selected = np.zeros_like(residual)
	above_t = np.zeros_like(residual, dtype=bool)
	accumulator = np.zeros_like(residual, dtype=int)
	if norm_psf:
		psf_img /= np.nansum(psf_img)
	
	if window is True:
		window = np.ones(use_img.shape, dtype=bool)
	
	if plots:
		# DEBUGGING PLOT SETUP
		plt.ioff()
		nr,nc = (2, 4)
		s = 4
		f1 = plt.figure(1,figsize=(s*nc,s*nr))
		plt.show()
	
	fabs_threshold = np.nanmax(np.fabs(residual))*fabs_frac_threshold
	rms_threshold = np.sqrt(np.nansum((residual)**2)/residual.size)*rms_frac_threshold
	
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
		accumulator[above_t] += 1
		print(f'INFO: Number of selected points {n_selected} max_fabs_residual {max_fabs_residual:07.2E} mod_clean_threshold {mod_clean_threshold:07.2E}')
		convolved = sp.signal.convolve2d(selected, psf_img, mode='same')
		#factor = np.nanmax(np.fabs(convolved))/np.nanmax(fabs_residual)
		factor = np.nanmax(fabs_residual)/np.nanmax(np.fabs(convolved))
		components += selected*loop_gain*factor
		current_cleaned = sp.signal.convolve2d(components, psf_img, mode='same')
		residual = use_img - current_cleaned
		
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
		
		if plots:
			# DEBUGGING PLOTS
			cmap = ('Spectra','twilight','coolwarm', 'bwr')[-1]
			plt.clf()
			a1 = f1.subplots(nr,nc)
			f1.suptitle(f'Loop iteration {i}')
			a1[0,0].imshow(use_img, origin='lower', cmap=cmap, vmin=-np.nanmax(np.fabs(use_img)), vmax=np.nanmax(np.fabs(use_img)))
			a1[0,0].set_title('dirty_img')
			a1[0,1].imshow(residual, origin='lower', cmap=cmap, vmin=-np.nanmax(np.fabs(residual)), vmax=np.nanmax(np.fabs(residual)))
			a1[0,1].set_title('residual')
			a1[0,2].imshow(components, origin='lower', cmap=cmap, vmin=-np.nanmax(np.fabs(components)),
							vmax=np.nanmax(np.fabs(components)))
			a1[0,2].set_title('components')
			# put NANs back in
			current_cleaned[np.isnan(use_img)] = np.nan
			a1[0,3].imshow(current_cleaned, origin='lower', cmap=cmap, vmin=-np.nanmax(np.fabs(current_cleaned)), 
							vmax=np.nanmax(np.fabs(current_cleaned)))
			a1[0,3].set_title('current CLEANed image')
			
			a1[1,0].imshow(above_t, origin='lower')
			a1[1,0].set_title('above_t')
			a1[1,1].imshow(accumulator, origin='lower')#, cmap=cmap, vmin=-np.nanmax(np.fabs(selected)), vmax=np.nanmax(np.fabs(selected)))
			a1[1,1].set_title('accumulator')
			# put back in the nans
			#convolved[np.isnan(use_img)] = np.nan
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
		
		
	return(residual, components, rms_record, fabs_record, i, accumulator)

def plot_clean_result(dirty_img, components, residual, psf_img, rms_record, fabs_record, accumulator, gaussian_kernel_std=2,
						fig=None, axes=None, show_plot=True):
	
	reconvolve = sp.ndimage.convolve(components, psf_img)
	
	nr,nc=(2,5)
	s = 4
	if fig is None:
		f1 = plt.figure(figsize=(nc*s,nr*s))
	else:
		f1 = fig
	if axes is None:
		a1 = f1.subplots(nr,nc,squeeze=False)
	else:
		a1 = axes
	
	cmin = np.nanmin(dirty_img)
	cmin = cmin if cmin > 1E-6 else 1E-6
	cmax = np.nanmax(dirty_img)
	
	norm = mpl.colors.SymLogNorm(1E-7, linscale=1, vmin=0, vmax=cmax)
	#norm = mpl.colors.LogNorm(vmin=cmin, vmax=cmax)
	#norm = mpl.colors.Normalize(vmin=cmin, vmax=cmax)
	
	
	im100 = a1[0,0].imshow(components, origin='lower', norm=norm)
	a1[0,0].set_title(f'sum of components {np.nansum(components):07.2E}')
	f1.colorbar(im100, ax=a1[0,0])
	
	
	im101 = a1[0,1].imshow(residual, origin='lower', norm=norm)#, norm=mpl.colors.SymLogNorm(1E-11, linscale=1, vmin=np.nanmin(residual), vmax=np.nanmax(residual)))
	a1[0,1].set_title(f'sum of residual {np.nansum(residual):07.2E}')
	f1.colorbar(im101,ax=a1[0,1])
	
	im110 = a1[1,0].imshow(dirty_img, origin='lower', norm=norm)
	a1[1,0].set_title(f'sum of dirty img {np.nansum(dirty_img):07.2E}')
	f1.colorbar(im110, ax=a1[1,0])
	
	im111 = a1[1,1].imshow(psf_img, origin='lower')
	a1[1,1].set_title(f'sum of psf {np.nansum(psf_img):07.2E}')
	f1.colorbar(im111, ax=a1[1,1])
	
	a1[0,2].imshow(accumulator, origin='lower')
	a1[0,2].set_title(f'Sum of accumulator {np.nansum(accumulator)}')
	
	a1[1,2].plot(fabs_record, label='fabs', color='tab:blue')
	a112x = a1[1,2].twinx()
	a112x.plot(rms_record, label='rms', color='tab:orange')
	h1,l1 = a1[1,2].get_legend_handles_labels()
	h2,l2 = a112x.get_legend_handles_labels()
	a1[1,2].set_title('Residual fabs and rms')
	a1[1,2].set_yscale('log')
	a1[1,2].legend(h1+h2,l1+l2)
	a1[1,2].set_xlabel('Iteration')
	
	
	a1[0,3].imshow(reconvolve, origin='lower', norm=norm)
	a1[0,3].set_title('components convolved with psf')
	
	a1[1,3].imshow(reconvolve + residual, origin='lower', norm=norm)
	a1[1,3].set_title('components convolved with psf and residual added')
	
	gaussian_kernel = astropy.convolution.Gaussian2DKernel(gaussian_kernel_std)
	a1[0,4].imshow(astropy.convolution.convolve(components, gaussian_kernel), origin='lower', norm=norm)
	a1[0,4].set_title('components convolved with gaussian kernel')
	
	gk_array = np.zeros_like(psf_img)
	gk_array[psf_img.shape[0]//2, psf_img.shape[1]//2] = 1
	gk_array = astropy.convolution.convolve(gk_array, gaussian_kernel)
	a1[1,4].imshow(gk_array, origin='lower')
	a1[1,4].set_title('gaussian kernel')

	if show_plot:
		plt.show()
	return

def grad_sobel(x):
	return(np.sqrt(sp.ndimage.sobel(x, axis=0, mode='constant')**2 + sp.ndimage.sobel(x, axis=1, mode='constant')**2))

def get_wavrange_img_from_fits(target_cube, wav_range=(1.9,2.0), function=np.nansum):
	with fits.open(target_cube) as hdul:
		psf_data = hdul['PSF'].data
		cube_data = hdul['PRIMARY'].data
		wavgrid = fitscube.header.get_wavelength_grid(hdul['PRIMARY'])
		disk_mask = hdul['DISK_MASK'].data
		
	wav_idxs = (np.nanargmin(np.fabs(wavgrid-wav_range[0])), np.nanargmin(np.fabs(wavgrid-wav_range[1])))
			
	psf_img = function(psf_data[wav_idxs[0]:wav_idxs[1]], axis=0)
	psf_img /= np.nansum(psf_img) # normalise psf so integral is one
	
	
	cube_img = np.nan_to_num(function(cube_data[wav_idxs[0]:wav_idxs[1]], axis=0))
	cube_nans = (cube_img==0) & np.isnan(cube_img)
	
	return(cube_img, psf_img, cube_nans, disk_mask)
	
def remove_img_artifacts_1(cube_img, psf_img):
	kernel = np.array(psf_img)
	kernel /= np.nansum(kernel)
	
	psf_img /= 1
	
	psf_kernel_conv = sp.ndimage.convolve(psf_img, kernel)
	psf_kernel_conv /= np.nanmax(psf_kernel_conv)
	
	#psf_test = np.fabs(sp.ndimage.convolve(psf_img, kernel) - psf_img)/np.fabs(sp.ndimage.convolve(psf_img, kernel))
	psf_test = np.fabs(sp.ndimage.convolve(psf_img, kernel) - psf_img)
	max_psf_test = np.nanmax(psf_test) 
	
	#cube_test= np.fabs(sp.ndimage.convolve(cube_img, kernel) - cube_img)/np.fabs(sp.ndimage.convolve(cube_img, kernel))
	cube_test= np.fabs(sp.ndimage.convolve(cube_img, kernel) - cube_img)
	#cube_img[cube_test/cube_img > max_psf_test] = np.nan
	
	cube_conv_psf = sp.ndimage.convolve(np.nan_to_num(cube_img), psf_img)*max_psf_test
	
	
	#max_cube_test = max_psf_test*sp.ndimage.convolve(cube_img,psf_img)
	#max_cube_test = 0.75
	max_cube_test = 4.12E-5
	test_idxs = cube_test > max_cube_test
	
	
	cube_img_2 = np.array(cube_img)
	cube_img_2[cube_nans] = np.nan
	#cube_img_2[test_idxs] = np.nan
	replace_idxs = np.isnan(cube_img_2) | test_idxs
	#cube_img_2[replace_idxs] = astropy.convolution.convolve(cube_img_2, psf_img[:-1,:-1])[replace_idxs]
	cube_img_2[replace_idxs] = np.nan
	
	print(f'max_psf_test {max_psf_test} max_cube_test {max_cube_test}')
	nr, nc = (2,4)
	s = 5
	f1 = plt.figure(figsize=(nc*s, nr*s))
	a1 = f1.subplots(nr,nc,squeeze=False)
	
	a1[0,0].imshow(psf_img, origin='lower')
	a1[0,1].imshow(psf_test, origin='lower', vmin=0, vmax=1)
	a1[0,2].imshow(psf_test>max_psf_test, origin='lower')
	#a1[0,3].imshow(psf_test, origin='lower')
	
	a1[1,0].imshow(cube_img, origin='lower')
	a1[1,1].imshow(cube_test, origin='lower', vmin=0, vmax=1)
	a1[1,2].imshow(cube_test>max_cube_test, origin='lower')
	a1[1,3].imshow(cube_img_2, origin='lower')
	
	plt.show()
	return(cube_img_2)

def remove_img_artifacts_2(cube_img, psf_img):
	kernel = np.array(psf_img)
	psf_img_sobel = np.sqrt(sp.ndimage.sobel(psf_img, axis=0, mode='constant')**2 
							+ sp.ndimage.sobel(psf_img, axis=1, mode='constant')**2)
	
	cube_img_sobel = np.sqrt(sp.ndimage.sobel(cube_img, axis=0, mode='constant')**2 
							+ sp.ndimage.sobel(cube_img, axis=1, mode='constant')**2)


	psf_test = psf_img_sobel/sp.ndimage.convolve(psf_img_sobel, kernel)
	cube_test = cube_img_sobel/sp.ndimage.convolve(cube_img_sobel, kernel)
	
	max_cube_test = np.nanmax(psf_test)
	
	cube_selected = cube_test>max_cube_test
	
	#cube_selected = sp.ndimage.binary_closing(cube_selected, iterations=1)
	#cube_selected = sp.ndimage.binary_fill_holes(cube_selected)
	cube_selected = sp.ndimage.binary_dilation(cube_selected, iterations=2)
	
	cube_img_2 = np.array(cube_img)
	cube_img_2[cube_selected] = np.nan
	
	# PLOTTING
	nr, nc = (2,5)
	s = 5
	f1 = plt.figure(figsize=(nc*s, nr*s))
	a1 = f1.subplots(nr,nc,squeeze=False)
	a1[0,0].imshow(psf_img, origin='lower')
	a1[0,1].imshow(psf_img_sobel, origin='lower')
	a1[0,2].imshow(psf_test, origin='lower')
	a1[0,3].imshow(psf_test>max_cube_test, origin='lower')
	
	a1[1,0].imshow(cube_img, origin='lower')
	a1[1,1].imshow(cube_img_sobel, origin='lower')
	a1[1,2].imshow(cube_test, origin='lower')
	a1[1,3].imshow(cube_selected, origin='lower')
	a1[1,4].imshow(cube_img_2, origin='lower')
	
	plt.show()
	
	return(cube_img_2)

def clean_modified_autopoint_detect(dirty_img, psf_img, window=True, loop_gain=1E-1, threshold=0.6,
										rms_frac_threshold=0.05, fabs_frac_threshold=0.05, max_iter=int(1E2), 
										norm_psf=False, plots=True, accumulator_reject_factor=1, autodetect_niter=10):
	plt.ioff()
	img_shape = (autodetect_niter, dirty_img.shape[0], dirty_img.shape[1])
	residual = np.full(img_shape, fill_value=np.nan)
	components = np.full(img_shape, fill_value=np.nan)
	rms_record = np.full((autodetect_niter, max_iter), fill_value=np.nan)
	fabs_record = np.full((autodetect_niter, max_iter), fill_value=np.nan)
	n_iters = np.full((autodetect_niter, max_iter), fill_value=np.nan, dtype=int)
	accumulator = np.full(img_shape, fill_value=np.nan, dtype=int)
	n_accumulator_rejected = np.zeros((autodetect_niter), dtype=int)
	
	accumulator_mask = np.ones_like(dirty_img, dtype=bool)
	(residual[0], components[0], rms_record[0],
	fabs_record[0], n_iters[0], accumulator[0]) = clean_modified(dirty_img, psf_img, window, loop_gain,
																						threshold, rms_frac_threshold,
																						fabs_frac_threshold, max_iter,
																						norm_psf, plots)
	#plot_clean_result(dirty_img, components, residual, psf_img, rms_record, fabs_record, accumulator, gaussian_kernel_std=2)
	#plt.waitforbuttonpress(0.001)
	
	for i in range(1, autodetect_niter):
		accumulator_reject_n = accumulator_reject_factor*np.log(fabs_frac_threshold)/np.log(1-loop_gain)
		accumulator_mask_update = (accumulator[i-1]<accumulator_reject_n)
		n_accumulator_rejected[i] = np.nansum(~accumulator_mask_update)
		print(f'INFO: autodetect step {i} accumulator_reject_threshold {accumulator_reject_n} n_rejected {n_accumulator_rejected[i]}')
		
		if np.nansum(~accumulator_mask_update)==0: # if we don't have any rejects then we are finished
			break
		
		accumulator_mask = accumulator_mask & accumulator_mask_update
		
		if plots:
			plot_clean_result(dirty_img, 
								components[i-1], 
								residual[i-1], 
								psf_img, 
								rms_record[i-1], 
								fabs_record[i-1], 
								accumulator_mask, 
								gaussian_kernel_std=2)
			plt.waitforbuttonpress(0.001)

		residual[i], components[i], rms_record[i], fabs_record[i], n_iters[i], accumulator[i] = clean_modified(dirty_img, psf_img,
																						window & accumulator_mask, 
																						loop_gain,
																						threshold, rms_frac_threshold,
																						fabs_frac_threshold, max_iter,
																						norm_psf, plots)
	return(residual, components, rms_record, fabs_record, n_iters, accumulator, accumulator_mask, n_accumulator_rejected)
	
	

#%% run functions
target_cube = os.path.expanduser('~/scratch/reduced_images/SINFO.2018-09-19T02:21:05/MOV_Neptune---H+K_0.025_1_tpl/analysis/obj_NEPTUNE_cal_HD2811_renormed.fits')

wav_range = (#(1.5,1.6),
			  #(1.7,1.8),
			  (1.9,2.0),
			  )[-1]

cube_img, psf_img, cube_nans, disk_mask = get_wavrange_img_from_fits(target_cube, wav_range, function=np.nanmedian)

#cube_img_2 = remove_img_artifacts_1(cube_img, psf_img)

#cube_img_2 = remove_img_artifacts_2(cube_img, psf_img)

cube_img_2 = np.array(cube_img)

cube_img_2[cube_nans] = np.nan

# DEBUGGING see how a single point source is affected by our process
#cube_img[-psf_img.shape[0]:,-psf_img.shape[1]:] = psf_img*np.nanmax(cube_img)

#%%
autodetect_niter = 20
accumulator_reject_factor = 10
(residual, 
 components, 
 rms_record, 
 fabs_record, 
 n_iters, 
 accumulator, 
 accumulator_mask,
 n_accumulator_rejected) = clean_modified_autopoint_detect(cube_img_2,
													psf_img,
													max_iter=500,
													threshold=0.6,
													window = True,#disk_mask==1,
													plots=False,
													autodetect_niter=autodetect_niter,
													accumulator_reject_factor=accumulator_reject_factor)

#%%

cube_img_2_sobel = grad_sobel(cube_img_2)/cube_img_2
plt.imshow(cube_img_2_sobel, origin='lower', vmin=-10, vmax=10)
plt.show()
														   
#%%
imgs_to_show = np.zeros_like(components)
for i in range(imgs_to_show.shape[0]):
	imgs_to_show[i] = sp.ndimage.convolve(components[i], psf_img)
title='convolution'
#%%
imgs_to_show = accumulator
#%%
plt.imshow(imgs_to_show[8], origin='lower')
plt.show()
#%%
for i in range(imgs_to_show.shape[0]):
	plt.imshow(imgs_to_show[i], origin='lower')
	plt.title(f'{title} {i}')
	plt.draw()																										
	plt.waitforbuttonpress(0.5)

#%%
lines_to_show = fabs_record
plt.plot(lines_to_show[0])
plt.show()
for i in range(lines_to_show.shape[0]):
	plt.plot(lines_to_show[i])
	plt.yscale('log')
	plt.draw()
	plt.waitforbuttonpress(0.5)
	
#%% plot set of autopoint detect cleans
nr,nc=(2,5)
s = 4
f1 = plt.figure(figsize=(nc*s,nr*s))
plt.show()
for i in range(components.shape[0]):
	f1.clear()
	a1 = f1.subplots(nr,nc,squeeze=False)
	
	plot_clean_result(cube_img, components[i], residual[i], psf_img, rms_record[i], fabs_record[i], accumulator[i],
						fig=f1, axes=a1, show_plot=False)
	f1.suptitle(f'Iteration {i}/{autodetect_niter} accumulator_reject_factor {accumulator_reject_factor}')
	plt.draw()
	plt.waitforbuttonpress(0.5)

#%%
plt.imshow(np.fabs(cube_img), origin='lower')
plt.show()


#%%

residual, components, rms_record, fabs_record, i, accumulator = clean_modified(cube_img_2, psf_img, max_iter=100, threshold=0.6,
																	window=disk_mask==1, plots=False)

#%%
plot_clean_result(cube_img, components, residual, psf_img, rms_record, fabs_record, accumulator)

#%%
n = 1*np.log(1E-2)/np.log(1-0.1)
plt.imshow(accumulator>n, origin='lower')
plt.show()

#%%
plt.imshow(accumulator/components, origin='lower')
plt.show()

#%%
residual, components, rms_record, fabs_record, i, accumulator = clean_modified(cube_img_2, psf_img, max_iter=100, threshold=0.6,
																	window=((disk_mask==1)&(accumulator<n)))
#%%
plot_clean_result(cube_img, components, residual, psf_img, rms_record, fabs_record, accumulator)

#%%
plt.hist(accumulator, bins=30)
ax = plt.gca()
ax.set_yscale('log')
plt.axvline(n)
plt.show()
