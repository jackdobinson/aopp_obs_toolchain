#!/usr/bin/env python3
"""
Routines that implement the Lucy-Richardson deconvolution algorithm

TODO:
	* Put updating plots into a class?
	* Integrate with clean_multiresolution(), make it one of the available base
	  algorithms.
"""




import numpy as np
import scipy as sp
import scipy.ndimage
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors
import copy
import plotutils
import utilities as ut
import utilities.sp
import utilities.np



def lucy_richardson(	obs,
						psf,
						n_iter=200,
						nudge=1E-2,
						strength=1.0,
						correction_factor_negative_fix=True,
						correction_factor_limit=np.inf,
						correction_factor_uclip=np.inf,
						correction_factor_lclip=-np.inf,
						show_plots=True,
						verbose=2,
						offset_image=False
						):
	"""
	Performs Lucy-Richardson deconvolution on an image
	
	ARGUMENTS
		obs [n,m]
			image to deconvolve
		psf [np,mp]
			point spread function to deconvolve with
		n_iter
			number of iterations
		nudge
			A value added to both elements in part of the algorithm that 
			involves a division, used to avoid divide by zero errors, but also
			makes the algorithm a lot more stable. Range of (0, inf) 1E-3 to 
			1.0 are decent values, reduce strength if there is still 
			instability.
		strength
			A factor that multiples the correction factors when they are
			applied to the current image guess. Technically it multiplies
			(correction_factors-1), as we always want correction factors to
			be centered around 1. Range of [1,0), usually stays at 1 but can go
			lower, try 0.5, 1E-1, 1E-2.
		correction_factor_negative_fix
			If we assume we only want positive sources in our deconvolved
			image, (and any negative regions are due to noise) we should not 
			have any negative correction factors. This sets all negative
			correction factors to a small but positive value.
		correction_factor_limit
			Often if the correction factors are getting large, it is an
			indication that the algorithm has stopped converging. Therefore we
			should give the user the result we have so far.
		correction_factor_uclip
			If the correction factor is getting overly large, we can clip it
			to some upper value. This can reduce instability of the solution.
		correction_factor_lclip
			If the correction factor is lower that this value, just clip it
			to zero. This will have the effect of ignoring smaller sources.
		show_plots
			If True, will show plots of the algorithm in action.
		
		RETURNS
			residual [n,m]
				The dirty_image minus the convolution of the psf and the 
				components
			components [n,m]
				Deconvolved image
			rms_record [i]
				Root mean squared of the residual for each iteration
			fabs_record [i]
				The absolute brightest pixel of the residual for each iteration
			i
				number of iterations
			correction_factor_stats [i,3]
				An array holding the min, max, and median of the correction
				factors for each iteration
	"""
	if verbose>0:
		print('\n'.join((	f'INFO: obs.shape {obs.shape}',
							f'----: psf.shape {psf.shape}',
							f'----: n_iter {n_iter}',
							f'----: nudge {nudge}',
							f'----: strength {strength}',
							f'----: correction_factor_negative_fix {correction_factor_negative_fix}',
							f'----: correction_factor_limit {correction_factor_limit}',
							f'----: correction_factor_uclip {correction_factor_uclip}',
							f'----: correction_factor_lclip {correction_factor_lclip}',
							f'----: show_plots {show_plots}',
							f'----: verbose {verbose}'
						))
					)
	
	# ensure psf is normalised so it sums to 1
	psf /= np.nansum(psf)
	
	# remove all nans
	#offset = 1 - np.nanmin(obs)
	nan_map = np.isnan(obs)
	nanfix_obs = np.nan_to_num(obs)
	#nanfix_obs = obs
	
	
	dirty_img = sp.signal.fftconvolve(nanfix_obs, psf, mode='full')
	put_in_center_of_image(dirty_img, nanfix_obs)

	dirty_img[ut.np.slice_center(dirty_img, nan_map)][nan_map] = np.nan
	dirty_img = ut.sp.interpolate_at_mask(dirty_img, np.isnan(dirty_img))

	offset = 1 - np.nanmin(dirty_img) if offset_image else 0
	dirty_img += offset
	
	#cf_taper = np.ones_like(dirty_img)
	#cf_taper = np.pad(np.ones_like(obs), pad_width=[(_x//2, (_x-1)//2) for _x in psf.shape], mode='linear_ramp', end_values=0)
	#cf_taper *= cf_taper
	
	cf_taper = np.ones_like(dirty_img, dtype=bool)
	cf_taper[tuple([slice(_x//2,-_x//2) for _x in psf.shape])] = False
	cf_taper = ~cf_taper
	
	#plt.imshow(cf_taper)
	#plt.show()
	
	# initialise all arrays
	#components = np.array(dirty_img)
	#components = np.array(dirty_img)*np.nanmean(dirty_img)/np.nanmax(dirty_img)
	#components = np.ones_like(dirty_img)
	components = np.ones_like(dirty_img)*np.nanmean(dirty_img)
	#components = np.fabs(dirty_img)
	
	components *= cf_taper

	residual = np.zeros_like(dirty_img)
	psf_reversed = np.array(psf[::-1,::-1])
	rms_record = np.full((n_iter,), fill_value=np.nan)
	fabs_record = np.full((n_iter,), fill_value=np.nan)
	iter_stat_record = np.full((n_iter,5), fill_value=np.nan)
	
	if show_plots:
		f1, a1 = plotutils.create_figure_with_subplots(2,4)
		
		# create initial plots
		cmap = copy.copy(plt.get_cmap(('Spectra', 'twilight', 'coolwarm', 'bwr')[3]))
		cmap.set_under('green')
		cmap.set_over('magenta')
		cmap.set_bad('black')
		
		
		a1[0,0].set_title('deconvolved image')
		im11 = a1[0,0].imshow(components, origin='lower', cmap=cmap)
		plotutils.remove_axes_ticks_and_labels(a1[0,0])
		
		a1[0,1].set_title('blurred_estimate')
		im12 = a1[0,1].imshow(components, origin='lower', cmap=cmap)
		plotutils.remove_axes_ticks_and_labels(a1[0,1])
		
		a1[1,0].set_title('observation divided by blurred_estimate')
		im13 = a1[1,0].imshow(components, origin='lower', cmap=cmap)
		plotutils.remove_axes_ticks_and_labels(a1[1,0])
		
		a1[1,1].set_title('correction_factors')
		im14 = a1[1,1].imshow(components, origin='lower', cmap=cmap)
		sca11 = a1[1,1].scatter([],[], c='g', s=0.1, marker='.')
		f1.colorbar(im14, ax=a1[1,1])
		plotutils.remove_axes_ticks_and_labels(a1[1,1])
		
		a1[0,2].set_title(f'psf\nsum {np.nansum(psf):07.2E}')
		psf_lims = plotutils.lim_sym_around_value(psf)
		im15 = a1[0,2].imshow(psf, origin='lower', cmap=cmap)
		im15.set_clim(psf_lims)
		f1.colorbar(im15, ax=a1[0,2])
		plotutils.remove_axes_ticks_and_labels(a1[0,2])
		
		#dirty_img_lims = plotutils.lim_sym_around_value(dirty_img)
		dirty_img_lims = plotutils.lim_sym_around_value(dirty_img, value=offset)
		a1[1,2].set_title(f'dirty_img\nsum {np.nansum(dirty_img):07.2E} lims [{dirty_img_lims[0]:07.2E}, {dirty_img_lims[1]:07.2E}]')
		im16 = a1[1,2].imshow(dirty_img, origin='lower', cmap=cmap)
		im16.set_clim(dirty_img_lims)
		plotutils.remove_axes_ticks_and_labels(a1[1,2])
		
		residual = dirty_img - components
		#ri_lims = dirty_img_lims
		ri_lims = [_x-offset for _x in dirty_img_lims]
		a1[1,3].set_title(f'residual_img\nsum {np.nansum(residual):07.2E} lims [{ri_lims[0]:07.2E}, {ri_lims[1]:07.2E}]')
		im17 = a1[1,3].imshow(residual, origin='lower', cmap=cmap)
		im17.set_clim(ri_lims)
		plotutils.remove_axes_ticks_and_labels(a1[1,3])
		
		lines11 = a1[0,3].plot(range(n_iter), iter_stat_record[:,:3])
		a1[0,3].set_xlim((0,n_iter))
		a1[0,3].set_ylim((0,2))
		
		# end show_plots
	
	for i in range(n_iter):
		if verbose>1: print(f'INFO: {i}/{n_iter}')
		# get an estimate of the dirty image by blurring the current clean image
		blurred_est = sp.signal.fftconvolve(components, psf, mode='same')
		blurred_est[blurred_est==0] = np.min(np.fabs(blurred_est)>0)
		
		# find the relative blur of the dirty image, the nudges try to ensure 
		# that the division will converge even if arrays are small
		obs_per_est = (dirty_img + nudge)/(blurred_est + nudge)
		
		# find correction factors to the current estimate
		correction_factors = sp.signal.fftconvolve(obs_per_est, psf_reversed, mode='same')
		
		# apply correction factors to the current estimate
		# first, multiply the factors' distance from 1 by the strength
		correction_factors = strength*(correction_factors - 1) + 1

		# once these get large the result becomes unstable, clip the upper bound if desired
		if (np.max(correction_factors) > correction_factor_uclip): 
			correction_factors *= correction_factor_uclip/np.max(correction_factors)
		
		# anything close to zero can just be zero, clip the lower bound if desired
		if (correction_factor_lclip != -np.inf) and np.any(correction_factors < correction_factor_lclip):
			correction_factors[correction_factors<correction_factor_lclip] = 0
			
		# we probably shouldn't even have -ve correction factors, turn them into a close-to-zero factor instead
		if (correction_factor_negative_fix and np.any(correction_factors < 0)):
			cf_negative = correction_factors < 0
			cf_positive = correction_factors > 0
			if not np.any(cf_positive):
				print('ERROR: All correction factors in Lucy-Richardson deconvolution have become negative, exiting...')
				sys.exit()
			correction_factors[cf_negative] = np.min(correction_factors[cf_positive])*np.exp(correction_factors[cf_negative])
		
		components[...] = components*correction_factors
		# TRY FIXING COMPONENTS OUTSIDE OF IMAGE FIELD
		#components[cf_taper] = 0
		
		# technically this is the residual for the last iteration but it saves 
		# us a convolution operation and is close enough, change if needed
		residual[...] = dirty_img - blurred_est
		
		# update statistics arrays
		iter_stat_record[i] = (	np.min(correction_factors), 
								np.max(correction_factors), 
								np.median(correction_factors), 
								np.sqrt(np.nansum(residual**2))/residual.size,
								np.nanmax(np.fabs(residual))
								)
		rms_record[i] = np.sqrt(np.nansum(residual**2))/residual.size
		fabs_record[i] = np.nanmax(np.fabs(residual))
		
		# update plots if we are displaying them
		if show_plots:
			f1.suptitle(f'Iteration {i}/{n_iter}')
			
			im11.set_data(components)
			#im11_lims = plotutils.lim_sym_around_value(components)
			im11_lims = plotutils.lim_sym_around_value(components, value=offset)
			a1[0,0].set_title(f'deconvolved image components\nlims [{im11_lims[0]:07.2E}, {im11_lims[1]:07.2E}]')
			im11.set_clim(im11_lims)
			
			im12.set_data(blurred_est)
			im12_lims = dirty_img_lims#plotutils.lim_sym_around_value(blurred_est)
			a1[0,1].set_title(f'blurred_estimate\nlims [{im12_lims[0]:07.2E}, {im12_lims[1]:07.2E}]')
			im12.set_clim(im12_lims)
			
			im13.set_data(obs_per_est)
			im13_lims = plotutils.lim_sym_around_value(obs_per_est, value=1)
			a1[1,0].set_title(f'observation divided by blurred estimate\nlims [{im13_lims[0]:07.2E}, {im13_lims[1]:07.2E}]')
			im13.set_clim(im13_lims)
			
			l_cf = np.log10(correction_factors)
			im14.set_data(l_cf)
			im14_lims = plotutils.lim_sym_around_value(l_cf, value=0)
			a1[1,1].set_title(f'correction_factors\nlims [{im14_lims[0]:07.2E}, {im14_lims[1]:07.2E}]')
			im14.set_clim(im14_lims)
			#im14.set_norm(mpl.colors.SymLogNorm(linthresh=0.1, vmin=im14_lims[0], vmax=im14_lims[1]))
			#sca11.set_offsets(np.array(np.nonzero(correction_factors<0)).T[:,::-1])
			
			a1[1,3].set_title(f'residual_img\nsum {np.nansum(residual):07.2E} lims [{ri_lims[0]:07.2E}, {ri_lims[1]:07.2E}]')
			im17.set_data(residual)
			
			for j in range(iter_stat_record[:,:3].shape[1]):
				lines11[j].set_ydata(iter_stat_record[:,:3][:,j])
			lines11_lims = plotutils.lim_around_extrema(iter_stat_record[:,:3])
			a1[0,3].set_ylim(lines11_lims)
			a1[0,3].set_title(' '.join((	f'min {iter_stat_record[:,:3][i,0]:07.2E}',
											f'max {iter_stat_record[:,:3][i,1]:07.2E}',
											f'median {iter_stat_record[:,:3][i,2]:07.1E}'
											)
										)
									)
			
			plt.pause(0.001)
			# end show_plots
		
		# If any of our exit conditions trip, exit the loop
		if np.nanmax(np.fabs(correction_factors))>correction_factor_limit:
			print('WARNING: Correction factors getting large, stopping iteration')
			break
		if np.all(np.isnan(correction_factors)):
			print('ERROR: Correction factors have all become NAN, stopping iteration')
			break
		
	# get the correct residual to return to the user
	residual = dirty_img - sp.signal.fftconvolve(components, psf, mode='same')
	
	if show_plots:
		plt.close(f1) # close the plots we created to track progress
	
	residual = take_center_of_image(residual, obs.shape)
	components = take_center_of_image(components, obs.shape)
	
	return(residual, components, rms_record, fabs_record, i, iter_stat_record)


def take_center_of_image(a, sb):
	"""
	Returns the pixels of a in the shape sb centered around the middle.
	"""
	s_diff = np.array(a.shape)-np.array(sb)
	return(a[tuple([slice(d//2, s-d//2) for s, d in zip(a.shape, s_diff)])])	

def put_in_center_of_image(a,b):
	"""
	Puts the data 'b' in the center of image 'a'
	"""
	s_diff = np.array(a.shape)-np.array(b.shape)
	a[tuple([slice(d//2, s-d//2) for s, d in zip(a.shape, s_diff)])] = b
	return

if __name__=='__main__':
	# define test psf
	tps=(301,301)
	x,y=(tps[0]//2, tps[1]//2) # center point of PSF
	test_psf = np.zeros(tps)
	test_psf[x,y]=1
	#test_psf[x-4:x+5,y-4:y+5]=0.5
	#test_psf[x-9:x+10,y-9:y+10]=0.2
	#test_psf[x-20,y-20:y+20]=0.4
	test_psf = sp.ndimage.gaussian_filter(test_psf, 20)
	
	# normalise PSF
	test_psf/=np.nansum(test_psf) 
	
	# define test dirty image
	test_image = 'squares'
	if (type(test_image) is np.ndarray) and (a.ndim == 2):
		# we have been given a numpy array directly, therefore use it
		real_img = sp.signal.fftconvolve(test_image, test_psf, mode='same')
	elif test_image is 'squares':
		# create a set of squares
		real_img = np.zeros((400,300))
		real_img[150:170,40:50] += 1
		real_img[270:290,70:250] += 0.5
		real_img[130:180,130:180] += 1
		#real_img = sp.ndimage.gaussian_filter(dirty_img, 10)
	elif test_image is 'points':
		# create a set of points
		real_img = np.zeros((400,300))
		real_img = np.random.random((400,300))
		threshold = 0.9999
		intensity = 1000
		real_img[real_img>threshold] *= intensity
		real_img[real_img<=threshold] = 0
	else:
		print(f'ERROR: test_image "{test_image}" not recognised, should be a 2D numpy array or one of ("squares", "points")')
	
	# add noise to dirty image
	real_img_min, real_img_max = (np.nanmin(real_img), np.nanmax(real_img))
	dirty_img = sp.signal.fftconvolve(real_img, test_psf, mode='same')/np.nansum(test_psf)
	shape_diff = np.array(dirty_img.shape) - np.array(real_img.shape)
	np.random.seed(1000) # seed generator for repeatability
	noise_power = 0.1
	if True:
		for i, s in enumerate(np.logspace(np.log(np.max(real_img.shape)),0,10,base=np.e)):
			print(f'INFO: Adding noise set {i} at scale {s} to dirty image')
			dirty_img[[slice(diff//2,shape-diff//2) for diff, shape in zip(shape_diff, dirty_img.shape)]] \
				+= sp.ndimage.gaussian_filter(np.random.normal(0,noise_power,real_img.shape), s)
	
	if False:
		(nr, nc, s) = (1,2,6)
		f1 = plt.figure(figsize=[x*s for x in (nc,nr)])
		a1 = f1.subplots(nr, nc, squeeze=False)
		a1[0,0].imshow(test_psf, origin='lower')
		a1[0,1].imshow(dirty_img, origin='lower')
		plt.show()
	
	results = []
	
	dirty_img[10:20,:] = np.nan
	"""
	result = lucy_richardson(	dirty_img, 
								test_psf,
								n_iter=200,
								nudge=1E-2,
								strength=1.0,
								correction_factor_negative_fix=True,
								correction_factor_limit=np.inf,
								correction_factor_uclip=np.inf,
								correction_factor_lclip=-np.inf,
								show_plots=True
								)
	"""
	
	results = []
	import fitscube.deconvolve.algorithms
	result = fitscube.deconvolve.algorithms.LucyRichardson(	dirty_img, 
															test_psf,
															n_iter=200,
															nudge=1E-3,
															strength=1.0,
															cf_negative_fix=True,
															cf_limit=np.inf,
															cf_uclip=np.inf,
															cf_lclip=-np.inf,
															show_plots=True,
															offset_obs=False
															)()
	results.append([result[1],result[0]])
	
	import fitscube.deconvolve.multiresolution
	import fitscube.deconvolve.clean
	
	"""
	sigmas = (20,)#10,5)
	
	base_algorithm = lucy_richardson
	kwargs = dict(	n_iter=200,
					nudge=1E-1,
					strength=1E-1,
					correction_factor_negative_fix=True,
					correction_factor_limit=np.inf,
					correction_factor_uclip=np.inf,
					correction_factor_lclip=-np.inf,
					show_plots=False,
					verbose=1
					)
	base_algorithm_kwargs = kwargs#[{**kwargs, **dict(n_iter=20)}, {**kwargs, **dict(nudge=1E-0, strength=1E0, n_iter=200)}, {**kwargs, **dict(nudge=10)}, {**kwargs, **dict(nudge=10)}]
	result = fitscube.deconvolve.multiresolution.multiresolution(	dirty_img, 
																	test_psf,
																	base_algorithm = base_algorithm,
																	base_algorithm_kwargs = base_algorithm_kwargs,
																	show_plots=False,
																	sigmas = sigmas
																	)
	results.append(result)
	"""
	"""
	base_algorithm = fitscube.deconvolve.clean.clean_modified
	base_algorithm_kwargs = dict(	window=True,
									max_iter=int(1E3),
									n_positive_iter=1E3,
									threshold=0.6,
									loop_gain=0.1,
									rms_frac_threshold=1E-2,
									fabs_frac_threshold=1E-2,
									norm_psf=True,
									show_plots=False
									)
	
	result = fitscube.deconvolve.multiresolution.multiresolution(	dirty_img, 
																	test_psf,
																	base_algorithm = base_algorithm,
																	base_algorithm_kwargs = base_algorithm_kwargs,
																	show_plots=False,
																	sigmas = sigmas
																	)
	results.append(result)
	"""
	
	cmap = copy.copy(plt.get_cmap('bwr'))
	cmap.set_over('magenta')
	cmap.set_under('green')
	cmap.set_bad('black')
	f2, a2 = plotutils.create_figure_with_subplots(len(results),4)
	for i, result in enumerate(results):
		a2[i,0].set_title('dirty image')
		im20 = a2[i,0].imshow(dirty_img, origin='lower', cmap=cmap)
		im20_lims = plotutils.lim_sym_around_value(dirty_img)
		im20.set_clim(im20_lims)
		
		a2[i,1].set_title('PSF')
		im21 = a2[i,1].imshow(test_psf, origin='lower', cmap=cmap)
		im21_lims = plotutils.lim_sym_around_value(test_psf)
		im21.set_clim(im21_lims)
		
		a2[i,2].set_title('Components')
		im22 = a2[i,2].imshow(result[1], origin='lower', cmap=cmap)
		im22_lims = plotutils.lim_sym_around_value(dirty_img)
		im22.set_clim(im22_lims)
		
		a2[i,3].set_title('Residual')
		im23 = a2[i,3].imshow(result[0], origin='lower', cmap=cmap)
		im23_lims = plotutils.lim_sym_around_value(dirty_img)
		im23.set_clim(im23_lims)
	
	plt.show()
	
