#!/usr/bin/env python3

import sys, os
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation
import fitscube.header
import fitscube.process.sinfoni
import const
import scipy as sp
import scipy.signal

import geometry
import fitscube.deconvolve.clean
import fitscube.deconvolve.multiresolution
import fitscube.deconvolve.lucy_richardson
import fitscube.stack
import numpy_helper

# assume that most normalising has been done

#file_associations = [	(	os.path.expanduser('~/scratch/transfer/Neptune_SINFONI_20131009_1_H_renorm.fits'), # observation
#							os.path.expanduser('~/scratch/transfer/09.STD.H.T0144.T0145.fits') # standard star
#						)
#					]

file_associations = [(os.path.expanduser("~/scratch/transfer/MUSE_OBS/V_BAND/DATACUBE_FINAL.fits"), os.path.expanduser("~/scratch/transfer/MUSE_OBS/V_BAND/DATACUBE_STD_0001.fits"))]

reformat_tag = '_reformat'
force_reformat = False

show_plots=True
find_psf_flag = True

remove_axes = lambda x: (x.xaxis.set_visible(False),x.yaxis.set_visible(False))

w_bins, w_bin_params = geometry.get_bin_edges(4750, 9350, 200,
									None, None)
for obsf, stdf in file_associations:
	reformat_file_name = reformat_tag.join(os.path.splitext(obsf))
	
	# Ensure that we have the Observation and PSF set up correctly
	if (not os.path.exists(reformat_file_name)) or (force_reformat):
		# First, center the PSF, then ensure the Observation is correct
		
		# open the PSF
		with fits.open(stdf) as std_hdul:
			print(std_hdul.info())
			# stack the PSF along the spectral axis
			std_hdu = fitscube.stack.stack_hdu_along_spectral_axis(std_hdul[1], w_bins, stack_func=np.nanmedian, w_idx=0)
		
		# remove any NANs
		std_hdu.data = np.nan_to_num(std_hdu.data)
		
		# find the centroid of the PSF in each spectral slice
		axis = (1,2)
		centroids = geometry.centroid(std_hdu.data, axis=axis)
		#print(centroids)
		
		# find the brightest pixel of the PSF in each spectral slice
		brightest_pixels = np.array([np.unravel_index(np.nanargmax(std_hdu.data[i]), std_hdu.data[i].shape) for i in range(std_hdu.data.shape[0])]).T
		
		# Find the shifts needed to center the PSF (brightest pixel or centroids?)
		shifts = np.array([std_hdu.data.shape[axis[i]]/2 - (brightest_pixels[i]+1) for i in range(len(axis))]).T
		#shifts = np.array([std_hdu.data.shape[axis[i]]/2 - (centroids[i]+1) for i in range(len(axis))]).T
		#print(shifts)
		
		# create a holder for the shifted PSF
		shifted_data = np.zeros_like(std_hdu.data)
		
		# actually shift the PSF data
		for i, s in enumerate(shifts):
			shifted_data[i,:,:] = sp.ndimage.shift(std_hdu.data[i,:,:], s)
		
		# Plot the result if we want to inspect it
		if False:
			centroids_new = np.array(geometry.centroid(shifted_data, axis=axis))
			#print(centroids_new)
			
			idx = 0
			s, nr, nc = (6, 1, 2)
			f1 = plt.figure(figsize=[s*nc, s*nr])
			a1 = f1.subplots(nr, nc, squeeze=False)
			a1[0,0].imshow(std_hdu.data[idx], origin='lower')
			a1[0,0].scatter([centroids[1][idx]], [centroids[0][idx]], s=1, c='cyan')
			a1[0,0].scatter([std_hdu.data.shape[2]/2], [std_hdu.data.shape[1]/2], s=1, c='red')
			
			a1[0,1].imshow(shifted_data[idx], origin='lower')
			a1[0,1].scatter([centroids_new[1][idx]], [centroids_new[0][idx]], s=1, c='cyan')
			a1[0,1].scatter([shifted_data.shape[2]/2], [shifted_data.shape[1]/2], s=1, c='red')
			plt.show()
		
		# open the Observatoin
		with fits.open(obsf) as obs_hdul:
			# stack the Observation along the spectral axis
			obs_hdu = fitscube.stack.stack_hdu_along_spectral_axis(obs_hdul[1], w_bins, stack_func=np.nanmedian, w_idx=0)
		
		# create a HDUList using the stacked and centered PSF and Observation
		reformat_hdul = fits.HDUList([	fits.PrimaryHDU(data=obs_hdu.data, header=obs_hdu.header),
										fits.ImageHDU(data=shifted_data, header=std_hdu.header, name='PSF')
										])
		# write the new HDUL to file
		reformat_hdul.writeto(reformat_file_name, overwrite=force_reformat)
		
	# Now, CLEAN the reformatted data	
	with fits.open(reformat_file_name) as reformat_hdul:
		obs_hdu = reformat_hdul['PRIMARY']
		psf_hdu = reformat_hdul['PSF']
			
		clean_data = np.zeros_like(obs_hdu.data)
		residual_data = np.zeros_like(clean_data)

		idxs_to_clean = np.array(range(obs_hdu.data.shape[0]))
		#idxs_to_clean = np.array([137]) # DEBUGGING

		deconvolve_algorithms = ['clean_modified', 'lucy_richardson', 'multiresolution_clean_modified', 'multiresolution_lucy_richardson', 'clean_hogbom']
		#deconvolve_algorithms = deconvolve_algorithms[-1:] # DEBUGGING
		sigmas = (10,5,2,1)

		for algo in deconvolve_algorithms:
			print(f'INFO: Using algorithm "{algo}"')
			# PERFORM CLEAN LOOP
			for i in idxs_to_clean: #range(obs_hdu.data.shape[0]):
				print(f'INFO: CLEANing slice {i}/{len(idxs_to_clean)}')
				#print(f'INFO: number of NANs {np.sum(np.isnan(psf_hdu.data[i]))} out of {np.prod(psf_hdu.data[i].shape)}')
				if (np.all(np.isnan(psf_hdu.data[i])) 
						or np.all(np.isnan(obs_hdu.data[i])) 
						or (np.nansum(psf_hdu.data[i])==0) 
						or (np.nansum(obs_hdu.data[i])==0)):
					print(f'INFO: slice {i} is all NAN or sums to zero, skipping...')
					continue
				
				
				obs_decomp, obs_filt, obs_fac = fitscube.deconvolve.multiresolution.decompose_image(np.nan_to_num(obs_hdu.data[i]),
																							sigmas)
				psf_decomp, psf_filt, psf_fac = fitscube.deconvolve.multiresolution.decompose_image(np.nan_to_num(psf_hdu.data[i]),
																							sigmas)
				if False: # PLOT DECOMPISTION OF IMG AND PSF
					(sz, nr, nc) = (6, len(sigmas)+1, 4)
					f2 = plt.figure(figsize=[sz*x for x in (nc, nr)])
					a2 = f2.subplots(nr, nc, squeeze=False)
					sigma_labels = list(sigmas)+['smallest']
					for j, (s_o, s_p, h, f_o, f_p) in enumerate(zip(obs_decomp, psf_decomp, obs_filt, obs_fac, psf_fac)):
						#s = np.fabs(s)
						#brightest_pixels = np.nonzero(np.fabs(s) > 0.1*np.nanmax(s))
						
						a2[j,0].set_title(f's_obs[{j}] sigma={sigma_labels[j]}\nsum={np.nansum(s_o):07.2E}')
						a2[j,0].imshow(s_o, origin='lower')
						remove_axes(a2[j,0])
						
						a2[j,1].set_title(f's_psf[{j}] sigma={sigma_labels[j]}\nsum={np.nansum(s_p):07.2E}')
						a2[j,1].imshow(s_p, origin='lower')
						remove_axes(a2[j,1])

						a2[j,2].set_title(f'h[{j}]\nsum {np.nansum(h[j]):07.2E}')
						a2[j,2].imshow(h, origin='lower')
						remove_axes(a2[j,2])

						a2[j,3].set_title(f'histograms')
						#a2[j,3].hist(obs_hdu.data[i].ravel(), bins=100, color='tab:green', alpha=0.3, label='original', density=True)
						a2[j,3].hist(s_o.ravel(), bins=100, color='tab:blue', alpha=0.3, label=f's_obs[{j}]', density=True)
						a2[j,3].hist(s_p.ravel(), bins=100, color='tab:orange', alpha=0.3, label=f's_psf[{j}]', density=True)
						a2[j,3].set_yscale('log')
						#a2[j,3].set_xscale('symlog')
						a2[j,3].legend(loc='upper right')

						#a2[j,1].imshow(obs_hdu.data[i], origin='lower')
						#a2[j,1].scatter(*brightest_pixels[::-1], c='red', s=0.1)
						
						#a2[j,2].imshow(h, origin='lower')
						#a2[j,2].imshow(window, origin='lower')	
					plt.show()
					#sys.exit() # DEBUGGING
				
				window=True
				"""
				# define window to CLEAN in
				window = np.zeros_like(obs_hdu.data[i], dtype=bool)
				pixels_to_clean = np.nonzero(np.fabs(obs_decomp[0]) > 1E-3*np.nanmax(np.fabs(obs_decomp[0])))
				pixels_to_ignore = np.nonzero(np.fabs(obs_decomp[2]) > 0.1*np.nanmax(np.fabs(obs_decomp[2])))
				window[pixels_to_clean] = True
				window[pixels_to_ignore] = False
				"""
				dirty_img = obs_hdu.data[i]
				dirty_img[~window]=np.nan
			
				if algo is 'multiresolution_clean_modified':
					base_algorithm = fitscube.deconvolve.clean.clean_modified
					base_algorithm_kwargs = dict(	norm_psf='sum', 
													window=window,
													quiet=False,
													show_plots=False,
													max_iter=int(1E4),
													n_positive_iter=1E3,
													loop_gain=0.05,
													threshold=0.4,
													rms_frac_threshold=1E-2,
													fabs_frac_threshold=1E-2
													)
				elif algo is 'multiresolution_lucy_richardson':				
					base_algorithm = fitscube.deconvolve.lucy_richardson.lucy_richardson
					kwargs = dict(	n_iter=500,
									nudge=1E-2,
									strength=1E-0,
									correction_factor_negative_fix=True,
									correction_factor_limit=np.inf,
									correction_factor_uclip=np.inf,
									correction_factor_lclip=-np.inf,
									show_plots=False,
									verbose=1
									)
					#base_algorithm_kwargs = kwargs
					base_algorithm_kwargs = [	{**kwargs, **dict(n_iter=20)}, 
												{**kwargs, **dict(nudge=1E+4, strength=1E-0, n_iter=100)}, 
												{**kwargs, **dict(nudge=1E+3, strength=5E-2)}, 
												{**kwargs, **dict(nudge=5E+4, strength=1E-0, offset_image=False)},
												{**kwargs, **dict(n_iter=200, nudge=1E5, strength=1E-0)},
											]
					
				
				if 'multiresolution' in algo:		
					clean_result = fitscube.deconvolve.multiresolution.multiresolution(	dirty_img, 
																						psf_hdu.data[i],
																						sigmas=sigmas, 
																						reject_decomposition_idxs=[],
																						calculation_mode='convolution_fft',
																						base_algorithm = base_algorithm,
																						base_algorithm_kwargs = base_algorithm_kwargs,
																						show_plots=False,
																						plot_dir = os.path.dirname(obsf),
																						plot_suffix = f'_{algo}_{i}',
																						verbose=1
																						)
				
				
				if False: # should we interpolate some of the data?
					# come up with some way to use the residual to decide which pixels should be interpolated
					noise_map = clean_result[0]
					noise_map[np.isnan(dirty_img)] = np.nan
					#noise_map = np.sqrt(sp.ndimage.sobel(noise_map, 0)**2 + sp.ndimage.sobel(noise_map, 1)**2)
					#noise_map = sp.ndimage.gaussian_gradient_magnitude(noise_map, 1)
					#noise_map = np.fabs(noise_map)/sp.ndimage.gaussian_filter(np.fabs(np.nan_to_num(noise_map)), 5)
					#noise_map -= sp.ndimage.gaussian_filter(np.nan_to_num(noise_map), 1)
					noise_map -= sp.ndimage.median_filter(np.nan_to_num(noise_map), 100)
					noise_map[np.isinf(noise_map)]=np.nan

					extrema_factor = 0.3
					r_extrema = (np.fabs(noise_map) > np.nanmax(np.fabs(noise_map))*extrema_factor) | np.isnan(noise_map)
				
					# interpolate pixels that are artifacts of the observation
					points = np.array(numpy_helper.mgrid_from_array(dirty_img, gridder=np.mgrid))
				
					p_known = points[:,~r_extrema].T
					p_unknown = points[:,r_extrema].T
					known_values = dirty_img[~r_extrema]
				
					print(f'DEBUG: {p_known.shape} {known_values.shape} {p_unknown.shape}')
				
					interp_values = sp.interpolate.griddata(p_known, known_values, p_unknown, method='linear')
				
					data_interp = np.array(dirty_img)
					data_interp[r_extrema] = interp_values

					# show a debugging plot
					print('DEBUG: Plotting interpolated data')
					plt.close('all')
					(nr, nc, ss) = (2,3,6)
					f3 = plt.figure(figsize=[x*ss for x in (nc,nr)])
					a3 = f3.subplots(nr,nc,squeeze=False)
					
					a3[0,0].set_title(f'noise map\nsum {np.nansum(noise_map):07.2E}')
					a3[0,0].imshow(noise_map, origin='lower')
					remove_axes(a3[0,0])

					print(f'DEBUG: {np.sum(r_extrema)}')
					a3[0,1].set_title(f'noisemap extrema\nsum {np.sum(r_extrema):07.2E}')
					a3[0,1].imshow(r_extrema, origin='lower')
					remove_axes(a3[0,1])

					a3[0,2].set_title(f'interpolated data\nsum {np.nansum(data_interp):07.2E}')
					a3[0,2].imshow(data_interp, origin='lower')
					remove_axes(a3[0,2])

					a3[1,0].set_title(f'original data\nsum {np.nansum(dirty_img):07.2E}')
					a3[1,0].imshow(dirty_img, origin='lower')
					remove_axes(a3[1,0])

					#a3[1,1].set_title('original - interpolated')
					#a3[1,1].imshow(dirty_img - data_interp, origin='lower')

					a3[1,1].set_title(f'histogram of |original data|,\n|interpolated data|, and |noise map|')
					a3[1,1].hist(np.fabs(dirty_img).ravel(), bins=100, density=False, color='tab:green', alpha=0.3,
									label='original')
					a3[1,1].hist(np.fabs(data_interp).ravel(), bins=100, density=False, color='tab:blue', alpha=0.3,
									label='interp')
					#a3112 = a3[1,1].twinx()
					a3[1,1].hist(np.fabs(noise_map).ravel(), bins=100, density=False, color='tab:orange', alpha=0.3,
									label='noise')
					a3[1,1].axvline(np.nanmax(np.fabs(noise_map))*extrema_factor, color='red', ls='--')
					a3[1,1].legend(loc='upper right')
					a3[1,1].set_yscale('log')
					#a3[1,1].set_xscale('log')
					#a3[1,1].set_xlim(np.nanmin(np.fabs(data_interp)), np.nanmax(np.fabs(data_interp)))

					a3[1,2].set_title(f'histogram of original data,\ninterpolated data, and noise map')
					a3[1,2].hist(dirty_img.ravel(), bins=100, density=False, color='tab:green', alpha=0.3, label='original')
					a3[1,2].hist(data_interp.ravel(), bins=100, density=False, color='tab:blue', alpha=0.3, label='interp')
					a3[1,2].hist(noise_map.ravel(), bins=100, density=False, color='tab:orange', alpha=0.3, label='noise')
					a3[1,2].axvline(np.nanmax(np.fabs(noise_map))*extrema_factor, color='red', ls='--')
					a3[1,2].set_yscale('log')
					a3[1,2].legend(loc='upper right')
					

					plt.show()
					

					# our interpolated data is what we are actually going to CLEAN
					obs_hdu.data[i] = data_interp
				# end data interpolation if statement	
		
				if algo is 'lucy_richardson':	
					base_algorithm = fitscube.deconvolve.lucy_richardson.lucy_richardson
					kwargs = dict(	n_iter=200,
									nudge=5E+0,
									strength=1E-0,
									correction_factor_negative_fix=True,
									correction_factor_limit=np.inf,
									correction_factor_uclip=np.inf,
									correction_factor_lclip=-np.inf,
									show_plots=False,
									verbose=1,
									)
					base_algorithm_kwargs = kwargs
					#base_algorithm_kwargs = [{**kwargs, **dict(n_iter=20)}, {**kwargs, **dict(nudge=1E-0, strength=1E0, n_iter=200)}, {**kwargs, **dict(nudge=10)}, {**kwargs, **dict(nudge=10)}]

				elif algo is 'clean_modified':
					base_algorithm = fitscube.deconvolve.clean.clean_modified
					base_algorithm_kwargs = dict(	norm_psf='sum', 
													window=window,
													quiet=False,
													show_plots=False,
													max_iter=int(1E4),
													n_positive_iter=1E3,
													loop_gain=0.1,
													threshold=0.6,
													rms_frac_threshold=1E-2,
													fabs_frac_threshold=1E-2
													)
				elif algo is 'clean_hogbom':
					base_algorithm = fitscube.deconvolve.clean.clean_hogbom
					base_algorithm_kwargs = dict(	norm_psf='sum', 
													window=window,
													quiet=False,
													show_plots=False,
													max_iter=int(1E6),
													n_positive_iter=0,
													loop_gain=0.1,
													rms_frac_threshold=1E-2,
													fabs_frac_threshold=1E-2,
													sum_limit_factor=10
													)
				if 'multiresolution' not in algo:	
					clean_result = base_algorithm(obs_hdu.data[i],  psf_hdu.data[i], **base_algorithm_kwargs)
		
		
				clean_data[i,:,:] = clean_result[1]
				residual_data[i,:,:] = clean_result[0]
			
				# NAN pixels will make this check not match properly	
				print(f'sum(obs_hdu.data[{i}]) {np.nansum(obs_hdu.data[i])}')
				print(f'sum(clean_data[{i}]) {np.nansum(clean_data[i])}')
				print(f'sum(residual{i}) {np.nansum(residual_data[i])}')
				print(f'sum(obs_hdu.data[{i}]) - sum(residual_data[{i}]) {np.nansum(obs_hdu.data[i]) - np.nansum(residual_data[i])}')
				print(f'sum(clean_data[{i}])+sum(residual_data[{i}]) {np.nansum(clean_data[i])+np.nansum(residual_data[i])}')
				print(f'sum(clean_data[{i}])+sum(residual_data[{i}])-sum(obs_hdu.data[{i}]) {np.nansum(clean_data[i])+np.nansum(residual_data[i])-np.nansum(obs_hdu.data[i])}')		

				# END OF LOOP
		
			output_hdul = fits.HDUList([fits.PrimaryHDU(data=obs_hdu.data, header=obs_hdu.header),
										fits.ImageHDU(data=psf_hdu.data, header=psf_hdu.header, name='PSF'), 
										fits.ImageHDU(data=clean_data, header=obs_hdu.header ,name='COMPONENTS'),	
										fits.ImageHDU(data=residual_data, header=obs_hdu.header, name='RESIDUAL'),
										fits.ImageHDU(data=sp.ndimage.gaussian_filter(clean_data, (0,3,3)), 
														header=obs_hdu.header, name='SMOOTHED_3_COMPONENTS'),
										fits.ImageHDU(data=obs_hdu.data-residual_data, header=obs_hdu.header, 
														name='DIRTY-RESIDUAL')
										])
			
			output_hdul.writeto(os.path.expanduser(f"~/scratch/transfer/MUSE_OBS/V_BAND/DATACUBE_CLEAN_{algo}.fits"), 
								overwrite=True)
	
	
