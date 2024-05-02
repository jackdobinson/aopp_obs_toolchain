
from typing import Literal

import numpy as np
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt

import aopp_deconv_tool.context as context
from aopp_deconv_tool.context.next import Next

from aopp_deconv_tool.stats.empirical import EmpiricalDistribution
from aopp_deconv_tool.scipy_helper.interp import interpolate_at_mask

import aopp_deconv_tool.plot_helper as plot_helper


import aopp_deconv_tool.cfg.logs
_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'DEBUG')


def ssa2d_sum_prob_map(
		ssa, 
		start=5, 
		stop=None, 
		value=0.99, 
		strategy : Literal['prob_frac_from_median'] | Literal['n_std_dev_from_median'] = 'n_std_dev_from_median',
		show_plots=0, 
		transform_value_as : list[Literal['median_prob', 'ppf'],] = ['ppf'], 
		weight_by_evals=False,
		smooth_sigma=None,
	):
	"""
	Computes a bad pixel map from an SSA2D object. Concentrates on hot/cold pixels w.r.t background.
	
	Idea is that for a hot/cold pixel, the high frequency components of the SSA decomposition must 
	also be hot/cold. Therefore, the hot/cold pixels will consistently be at the *edges* of the
	pixel probability distribution for the higher frequency SSA components. As high frequency
	SSA components outnumber low frequency SSA components, we can just use all components as 
	long as we don't weight them by their eigenvalues (if we do, we should try and be selective
	to only the components where the scale of changes matches the scale of the hot/cold pixels).
	
	For each component of the SSA decomposition, a map is made of how far a given pixel deviates 
	from the median (range [-1,1]). The median differences for each component are combined together
	(by mean or weighted mean) to give a 'score' for each pixel. Bad pixels should have scores at the
	extreme ends of the [-1,1] range. A bad pixel map is chosen by taking the absolute value of the score,
	and selecting all pixels whose |score| is larger than a give value.

	# ARGUMENTS #
		ssa
			An SSA2D object
		start
			The first SSA component to include in the probability map
		stop
			The last SSA component to include in the probability map.
			If None, then will include as many SSA components as are present
		value
			A float that describes the quantile that denotes a 'bad pixel'
		strategy : 'prob_frac_from_median' | 'n_std_dev_from_median'
			How should we define an 'improbable' pixel?
				'prob_frac_from_median'
					Fraction of probability space a pixel is away from 
					the median value
				'n_std_dev_from_median'
					Number of standard deviations a pixel is away from
					the median value
		show_plots
			0 = do not show plots
			1 = show some plots
			2 = show interim plots
		transform_value_as
			List of literal strings that tells us how to tranform the "value"
			argument after which it is used as the cutoff for a bad pixel.
			If no options are given, "value" is used as the cutoff directly,
			i.e. if "value" is 0.99, then only pixels which have a median
			probability > 0.99 are classified as "bad pixels".

			Options are:
				"identity"
					Use the value as-is, do not transform it
				"median_prob"
					Transform "value" using the same function that we use to
					change cumulative probabilities of pixels into probability
					a pixel is a certain distance from the median.
				"ppf"
					Transform "value" so that the cutoff is the percentage
					point function of the pixel median probabilities. I.e.
					if "value" is 0.9, then the cutoff will be set such that
					10% of the pixels will be classified as "bad pixels"
		weight_by_evals
			If True, will weight each SSA component pixel probability by
			the component's eigenvalue when averaging into a combined score.
		perform_binary_opening : bool = True
			If True will perform binary opening on the result to remove single pixels
	# RETURNS #
		bp_mask
			A bad pixel mask. Pixels to reject are coded as True.
	"""
	# set up arrays and defaults
	stop = ssa.X_ssa.shape[0] if stop is None else stop
	start = 0 if start is None else start
	distrib = sp.stats.cauchy
	data_probs = np.zeros((stop-start, *ssa.a.shape),dtype=float)
	
	# choose a pixel->probability function
	# we get a range of (0,1) from "ut.sp.construct_cdf_from()"
	# it gives us the cumulative probability for each pixel
	# This function changes it to 'distance away from median' in the range [-1, 1]
	prob_median_transform_func = lambda x: (2*(x-0.5))
	
	# apply pixel->probability function to each SSA component
	for i in range(0, stop-start):
		match strategy:
			case 'prob_frac_from_median':
				data_distribution = EmpiricalDistribution(ssa.X_ssa[i+start].ravel())
				data_probs[i,...] = prob_median_transform_func(data_distribution.cdf(ssa.X_ssa[i+start].ravel()).reshape(ssa.a.shape))
			case 'n_std_dev_from_median':
				data_probs[i,...] = (ssa.X_ssa[i+start] - np.nanmean(ssa.X_ssa[i+start]))/np.nanstd(ssa.X_ssa[i+start])
			case _:
				raise RuntimeError(f'Unknown `strategy` for getting probability of a pixel {strategy=}')

	# combine pixel probabilites together, pixels that are "strange" will have correlated distances away from
	# the median for each SSA component. Therefore, "normal" pixels will tend towards zero when taking the mean of `data_probs`,
	# but "strange" (hot/cold) pixels will stay close to +/- 1. Finally, then take absolute value 
	# as we want a range of [0,1] where 0 is data is "normal" and 1 is data is "strange"
	# NOTE: Instead of using `np.fabs` here, I could square the results. It shouldn't make much of a difference apart from adjusting
	#       the exact values of the `data_probs_sum`.
	data_probs_sum_func = lambda x, _start, _stop: np.fabs(
		np.mean(x[:_stop-(_start-1)], axis=0)
	)
	if weight_by_evals:
		# try weighted by eigenvalues, generally do not want this one but the option is there
		data_probs_sum_func = lambda x, _start, _stop: np.fabs(
			(1/(np.sum((np.diag(ssa.s)**2)[_start:_stop+1]))) * np.sum(((np.diag(ssa.s)**2)[_start:_stop+1])[:,None,None] * x[:_stop-(_start-1)],axis=0)
		)

	if smooth_sigma is not None:
		old_data_probs_sum_func = data_probs_sum_func
		data_probs_sum_func = lambda x, _start, _stop: sp.ndimage.gaussian_filter(old_data_probs_sum_func(x, _start, _stop), smooth_sigma, order=1)


	# apply the pixel->probability function to 'value' argument if desired
	if 'median_prob' in transform_value_as:
		value = prob_median_transform_func(value)
	if 'ppf' in transform_value_as:
		cutoff_func = lambda _test, _value: EmpiricalDistribution(_test.ravel()).ppf(_value)
	else:
		cutoff_func = lambda _test, _value: _value
		
	bp_mask = pixel_map(
		ssa.a, 
		data_probs_sum_func(data_probs,start,stop), 
		value, 
		show_plots=show_plots, 
		cutoff_func=cutoff_func, 
		plot_kw = {
				'suptitle':f'Sum ssa2d probability maps\n{value=}'
			}
		)

	# plots for debugging and progress
	if show_plots > 1:
		nplots = ssa.X_ssa.shape[0]#(stop-start)
		
		#for i in range(nplots):
		for i in range(start, stop):
			f1, a1 = plot_helper.figure_n_subplots(4)
			f1.suptitle(f'Sum ssa.X_ssa[{i}/{nplots}] probability maps')
			ax_iter=iter(a1.flatten())
			
			with Next(ax_iter) as ax:
				ax.set_title(f'ssa.X_ssa[{i}]\nsum {np.nansum(ssa.X_ssa[i])}')
				ax.imshow(ssa.X_ssa[i])
				plot_helper.remove_axes_ticks_and_labels(ax)
			with Next(ax_iter) as ax:
				ax.set_title(f'histogram of ssa.X_ssa[{i}] range [{np.min(ssa.X_ssa[i])}, {np.max(ssa.X_ssa[i])}] sigma {np.std(ssa.X_ssa[i])}')
				hvals, hbins, hpatches = ax.hist(ssa.X_ssa[i].ravel(), bins=100, density=True)
				x = np.linspace(np.min(ssa.X_ssa[i]),np.max(ssa.X_ssa[i]), 100)
				ax.twinx().plot(x, EmpiricalDistribution(ssa.X_ssa[i].ravel()).cdf(x), color='tab:orange', ls='-')
			if i < stop and i >= start:
				with Next(ax_iter) as ax:
					ax.set_title(f'probabilities of ssa.X_ssa[{i}]')
					#ax.imshow(data_probs[i-start], vmin=-1, vmax=1)
					ax.imshow(data_probs[i-start])
					plot_helper.remove_axes_ticks_and_labels(ax)
				with Next(ax_iter) as ax:
					ax.set_title(f'|sum of probabilities of ssa.X_ssa[{start}:{i+1}]|')
					#ax.imshow(data_probs_sum_func(data_probs, start, i), vmin=0, vmax=1)
					ax.imshow(data_probs_sum_func(data_probs, start, i))
					plot_helper.remove_axes_ticks_and_labels(ax)
				
			plot_helper.output(
				True, 
				None,
				#f'ssa_{i}_bad_pixel_prob_maps.png',
			)
			
	
		
	return(bp_mask)



def pixel_map(
		img, 
		test, 
		value, 
		show_plots=0, 
		plot_kw={}, 
		cutoff_func = lambda _test, _value: EmpiricalDistribution(_test.ravel()).ppf(_value), 
		bp_mask_func = lambda _test, _cutoff: _test > _cutoff
	) -> np.ndarray:
	"""
	Finds a mask for an image based on values in 'test'.

	# ARGUMENTS #
		img : np.ndarray([:,:], type.T1)
			image to construct mask for (not actually used in calculation, but used in plotting)
		test : np.ndarray([:,:], type.T2)
			Array that cutoff will be applied to to find mask
		value : type.T3
			Value to calculate cutoff from using 'cutoff_func'
		show_plots : bool
			If True, will plot results of mapping.
		plot_kw : dict
			Dictionary to pass optional arguments to plots.
			{	'suptitle' : str
					A string to display as the super-title of the plotted figure
			}
		cutoff_func : Callable[np.ndarray([:,:],type.T2), type.T3] -> type.T4
				= lambda _t, _v: ut.sp.construct_ppf_from(_t.ravel())(_v)
			A function that calculates the cutoff from 'value'. The cutoff is used
			to detemine the mask.
		bp_mask_func : Callable[np.ndarray([:,:],type.T2), type.T4] -> np.ndarray([:,:],bool)
				= lambda _t, _c: _t > _c
			A function that calculates the mask from the test and cutoff (cutoff is calculated
			from 'cutoff_func')
			
	# RETURNS #
		mask : np.ndarray([:,:],bool)
			A boolean array that can be used to mask 'img'
	"""
	cutoff = cutoff_func(test, value)
	bp_mask = bp_mask_func(test, cutoff)

	#breakpoint() # DEBUGGING
	if show_plots > 0:
		interpolated = interpolate_at_mask(img, bp_mask, edges='convolution', method='cubic')
		plot_pixel_map_test(img, test, cutoff, bp_mask, interpolated, plot_kw=plot_kw)
		plot_helper.output(
			True,
			None,
			#f'pixel_map_plot.png'
		)
	return(bp_mask)


def plot_pixel_map_test(img, test, cutoff, mask, interp, plot_kw={}):
	f2, a2 = plot_helper.figure_n_subplots(6)
	a2_iter = iter(a2.ravel())
	f2.suptitle(plot_kw.get('suptitle', 'pixel map function'))
	
	with Next(a2_iter) as ax:
		ax.set_title(f'Original image\nsum {np.nansum(img):08.2E}\nsqrt(sum^2) {np.sqrt(np.nansum(img**2)):08.2E}')
		ax.imshow(img)
		plot_helper.remove_axes_ticks_and_labels(ax)
		
	with Next(a2_iter) as ax:
		ax.set_title(f'pixel choice test function\nsum {np.nansum(test):08.2E} sqrt(sum^2) {np.sqrt(np.nansum(test**2)):08.2E}')
		#ax.imshow(test)
		ax.imshow(np.log(test))
		plot_helper.remove_axes_ticks_and_labels(ax)
		
	with Next(a2_iter) as ax:
		ax.set_title(f'histogram of choice function\ncutoff {cutoff:06.4f}')
		ax.hist(test.ravel(), bins=100)
		ax.axvline(cutoff, color='red', ls='--')
	
	with Next(a2_iter) as ax:
		ax.set_title(f'mask from cut of choice function\nn_masked {np.nansum(mask)} frac_masked {np.nansum(mask)/mask.size:08.2E}')
		ax.imshow(mask)
		plot_helper.remove_axes_ticks_and_labels(ax)
		
	with Next(a2_iter) as ax:
		ax.set_title('\n'.join((f'original interpolated at mask',
								f'sum {np.nansum(interp):08.2E} frac {np.nansum(interp)/np.nansum(img):08.2E}',
								f'sqrt(sum^2) {np.sqrt(np.nansum(interp**2)):08.2E} frac {np.sqrt(np.nansum(interp**2))/np.sqrt(np.nansum(img**2)):08.2E}',
		)))
		ax.imshow(interp)
		plot_helper.remove_axes_ticks_and_labels(ax)
		
	with Next(a2_iter) as ax:
		residual = interp - img
		ax.set_title('\n'.join((f'residual of interpolation - original',
								f'sum {np.nansum(residual):08.2E} frac {np.nansum(residual)/np.nansum(img):08.2E}',
								f'sqrt(sum^2) {np.sqrt(np.nansum(residual**2)):08.2E} frac {np.sqrt(np.nansum(residual**2))/np.sqrt(np.nansum(img**2)):08.2E}',
		)))
		ax.imshow(residual)
		plot_helper.remove_axes_ticks_and_labels(ax)
