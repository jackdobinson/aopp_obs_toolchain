#!/usr/bin/env python3
"""
Provides routines that will flag bad pixels of an image according to various
criteria. Useful for ignoring regions that are obviously bad when deconvolving
"""
import utilities.logging_setup
import importlib
logging, _lgr = utilities.logging_setup.getLoggers(__name__, 'INFO')

import numpy as np
import scipy as sp
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors
import typing
import types
import utilities as ut
from utilities import if_none
import utilities.plt
import py_ssa
from utilities.classes import Next, Alias
import utilities.mfunc
import utilities.np
import utilities.sp
import utilities.plt


def set_bp(a, bp_map, fill_value=np.nan, copy=True):
	"""
	Applys the bad pixel map "bp_map" to the array "a"
	"""
	a_dash = np.array(a) if copy else a
	a_dash[bp_map] = fill_value
	return(a_dash)

def ssa2d_cumulative_histograms(ssa, 
		bins=100, 
		cutoff_type='>cauchy_cdf<', 
		cutoff_value=0.95, 
		start=0, 
		stop=None, 
		sum_stop=None, 
		show_plots=True
	):
	"""
	Calculate a set of bad pixel maps based on histograms of cumulative sums of SSA components
	"""
	stop = ssa.X_ssa.shape[0] if stop is None else stop
	sum_stop = ssa.X_ssa.shape[0] if sum_stop is None else sum_stop
	n = stop - start
	
	if show_plots:
		f1, a1 = ut.plt.figure_n_subplots(2*n)
		a1=a1.flatten()
		f1.suptitle(f'Cumulative ssa2d histograms')
		ax_iter=iter(a1)
	
		hist_data, bin_edges = np.histogram(ssa.a, bins=bins, density=True)
		bin_mids = 0.5*(bin_edges[:-1]+bin_edges[1:])
	
	oclim = (np.min(ssa.X_ssa),np.max(ssa.a))
	bp_maps = np.zeros((n, *ssa.X_ssa.shape[1:]), dtype=bool)
	cutoffs = [None]*(n)
	determining_funcs = [None]*(n)
	
	for i in range(start,stop):
		sum_X_ssa = np.sum(ssa.X_ssa[i:sum_stop], axis=0)
		#print(i-start)
		#cutoffs[i-(n+start)], bp_maps[i], determining_funcs[i-(n+start)] = apply_cutoff(sum_X_ssa, '>std<', std_factor)
		cutoffs[i-start], bp_maps[i-start,:,:], determining_funcs[i-start] = apply_cutoff(sum_X_ssa, cutoff_type, cutoff_value)		
		
		if show_plots:
			sum_X_ssa_cutoff = set_bp(sum_X_ssa, bp_maps[i-start])
			idxs = np.nonzero(bp_maps[i-start])
			with Next(ax_iter) as ax:
				cutoffs[i-start] = list(cutoffs[i-start])
				cutoffs[i-start][0] = np.min(sum_X_ssa) if cutoffs[i-start][0] < np.min(sum_X_ssa) else cutoffs[i-start][0]
				cutoffs[i-start][1] = np.max(sum_X_ssa) if cutoffs[i-start][1] > np.max(sum_X_ssa) else cutoffs[i-start][1]
				ax.set_title(f'sum(X_ssa[{i}:])')
				im = ax.imshow(np.fabs(sum_X_ssa), vmin=cutoffs[i-start][0], vmax=cutoffs[i-start][1])
				#im.set_clim(oclim)
				ax.plot(*idxs[::-1], ls='none', marker='o', markersize=1, mfc='none', mec='red', mew=0.5)
				
			with Next(ax_iter) as ax:
				ax.set_title(f'Histogram of sum(X_ssa[{i}:])')
				hvals, hbins, hpatches = ax.hist(sum_X_ssa.ravel(), bins=bins, density=True)
				ax.set_yscale('log')
				ax.step(bin_mids, hist_data, where='mid')
				[ax.axvline(cutoff, color='tab:red', ls='--') for cutoff in cutoffs[i-start]]
				#ax.axvline(cutoff_max, color='tab:red', ls='--')
				ax.axvline(np.mean(sum_X_ssa), color='tab:red', ls=':')
				ylims = ax.get_ylim()
				x = ut.np.get_bin_mids(hbins)
				ax.plot(x, determining_funcs[i-start](x), label='determining func')
				ax.set_ylim(ylims)
	
	return(bp_maps)

def ssa2d_cumulative_bp_map(ssa, /,
		bins=100, cutoff_type='>cauchy_cdf<', cutoff_value=0.95, start=0, 
		stop=12, sum_stop=25, show_plots=0, cumulative_cutoff=0.99
	):
	"""
	Calculates a bad pixel map by applying cutoffs to histograms of cumulative sums of SSA components.

	# ARGUMENTS #
		ssa
			SSA2D object to operate on
		bins
			The number of bins to use in histograms
		cutoff_type
			A string that describes how the cutoffs are applied, see 'apply_cutoff()' for details
		cutoff_value
			The quantile where cutoffs should be applied to each histogram
		start
			The index of the first SSA component that should be included
		stop
			The index of the last SSA component that should be included (if None, will use all available components)
		show_plots
			0 = no plots
			1 = summary plots
			2 = step-by-step plots
		cumulative_cutoff
			The quantile that determines which pixels are flagged as bad from the combined histrograms
	# RETURNS #
		bp_mask
			A boolean mask of bad pixels. Rejected pixels are coded as True.
	"""
	stop = ssa.X_ssa.shape[0] if stop is None else stop
	sum_stop = ssa.X_ssa.shape[0] if sum_stop is None else sum_stop
	bp_maps = ssa2d_cumulative_histograms(ssa, bins, cutoff_type, cutoff_value, start, stop, sum_stop, show_plots=False)
	
	cbp_map = np.sum(bp_maps, axis=0)
	cbp_bins = ut.np.get_bin_edges(np.linspace(0,bp_maps.shape[0]-1,bp_maps.shape[0]))
	cbp_distrib = sp.stats.rv_histogram(np.histogram(cbp_map.ravel(), bins=cbp_bins))
	cbp_cutoff = cbp_distrib.ppf(cumulative_cutoff)
	cbp_mask = cbp_map > cbp_cutoff
	
	
	if show_plots:
		# use convolution to set the edges correctly for interpolation
		ssa_interp = ut.sp.interpolate_at_mask(ssa.a, cbp_mask, edges='convolution', method='cubic')
		if show_plots == 1:
			f1, a1 = ut.plt.figure_n_subplots(6)
		else:
			f1, a1 = ut.plt.figure_n_subplots(2*(stop-start)+6)
		a1=a1.flatten()
		f1.suptitle(f'Cumulative ssa2d bad pixel maps')
		ax_iter=iter(a1)
		
		if show_plots > 1:
			for i, ii in enumerate(range(start, stop)):
				#with Next(ax_iter) as ax:
				#	ax.set_title('original')
				#	ax.imshow(ssa.a)
				with Next(ax_iter) as ax:
					ax.set_title('bad pixel map')
					ax.imshow(bp_maps[i])
				with Next(ax_iter) as ax:
					ax.set_title('interpolated at bp map')
					ax.imshow(ut.sp.interpolate_at_mask(ssa.a, bp_maps[i], method='cubic'))
		
		with Next(ax_iter) as ax:
			ax.set_title('cumulative bad pixel map')
			ax.imshow(cbp_map)
		with Next(ax_iter) as ax:
			ax.set_title('cumulative bp map histogram')
			ax.hist(cbp_map.ravel(), bins=cbp_bins, density=True)
			ax.step(cbp_bins, cbp_distrib.pdf(cbp_bins), where='post')
			ax.axvline(cbp_cutoff, ls='--', color='tab:red')
			ax.set_yscale('log')
		with Next(ax_iter) as ax:
			ax.set_title('cumulative bp map with cutoff')
			ax.imshow(cbp_map > cbp_cutoff)
		with Next(ax_iter) as ax:
			ax.set_title('original image')
			ax.imshow(ssa.a)
		with Next(ax_iter) as ax:
			ax.set_title('interpolated at cumulative bp map')
			ax.imshow(ssa_interp)
		with Next(ax_iter) as ax:
			ax.set_title('residual')
			ax.imshow(ssa.a - ssa_interp)
		
	return(cbp_mask)

def ssa2d_sum_prob_map(ssa, start=3, stop=12, value=0.995, show_plots=0, 
		transform_value_as : typing.List[typing.Literal['median_prob', 'ppf'],] = [], 
		weight_by_evals=True):
	"""
	Computes a bad pixel map from an SSA2D object. 
	
	For each component of the SSA decomposition, a probability map is made of how far a given
	pixel deviates from the median. The probabilities for each component are combined together
	(by mean or weighted mean) to give a 'score' for each pixel. A bad pixel map is chosen by
	selecting all pixels whose 'score' is larger than a passed value.

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
	# RETURNS #
		bp_mask
			A bad pixel mask. Pixels to reject are coded as True.
	"""
	# set up arrays and defaults
	stop = if_none(stop, ssa.X_ssa.shape[0])
	distrib = sp.stats.cauchy
	data_probs = np.zeros((stop-start, *ssa.a.shape),dtype=float)
	#data = np.zeros((stop-start, *ssa.a.shape))
	
	# choose a pixel->probability function
	# we get a range of (0,1) from "ut.sp.construct_cdf_from()"
	# it gives us the cumulative probability for each pixel
	# we want to change this to a 'probability away from median' value
	prob_median_transform_func = lambda x: (2*(x-0.5))**2
	#prob_median_transform_func = lambda x: np.fabs(2*(x-0.5))
	#prob_median_transform_func = lambda x: (2*(x-0.5))
	
	# apply pixel->probability function to each SSA component
	for i in range(0, stop-start):
		data_cdf = ut.sp.construct_cdf_from(ssa.X_ssa[i+start].ravel())
		data_probs[i,...] = prob_median_transform_func(data_cdf(ssa.X_ssa[i+start].ravel()).reshape(ssa.a.shape))

	# combine pixel probabilites together
	if weight_by_evals:
		# try mean weighted by eigenvalues
		evals_trimmed = (np.diag(ssa.s)**2)[start:stop]
		data_probs_sum = (1/np.sum(evals_trimmed))*np.sum(evals_trimmed[:,None,None]*data_probs, axis=0)
	else:
		# try using mean instead of sum
		data_probs_sum = (1/(stop-start))*np.sum(data_probs, axis=0)

	# apply the pixel->probability function to 'value' argument if desired
	if 'median_prob' in transform_value_as:
		value = prob_median_transform_func(value)
	if 'ppf' in transform_value_as:
		cutoff_func = lambda _test, _value: ut.sp.construct_ppf_from(_test.ravel())(_value)
	else:
		cutoff_func = lambda _test, _value: _value
	bp_mask = pixel_map(ssa.a, data_probs_sum, value, show_plots=show_plots, cutoff_func=cutoff_func, plot_kw={'suptitle':f'Sum ssa2d probability maps\n{value=}'})

	# plots for debugging and progress
	if show_plots > 1:
		nplots = 4*(stop-start)
		f1, a1 = ut.plt.figure_n_subplots(nplots)
		f1.suptitle(f'Sum ssa2d probability maps')
		ax_iter=iter(a1.flatten())
		
		for i in range(stop-start):
			with Next(ax_iter) as ax:
				ax.set_title(f'ssa.X_ssa[{i}]\nsum {np.nansum(ssa.X_ssa[i])}')
				ax.imshow(ssa.X_ssa[i])
				ut.plt.remove_axes_ticks_and_labels(ax)
			with Next(ax_iter) as ax:
				ax.set_title(f'histogram of ssa.X_ssa[{i}]')
				hvals, hbins, hpatches = ax.hist(ssa.X_ssa[i].ravel(), bins=100, density=True)
				x = np.linspace(np.min(ssa.X_ssa[i]),np.max(ssa.X_ssa[i]), 100)
				ax.twinx().plot(x, ut.sp.construct_cdf_from(ssa.X_ssa[i].ravel())(x), color='tab:orange', ls='-')
			with Next(ax_iter) as ax:
				ax.set_title(f'probabilities of ssa.X_ssa[i]')
				ax.imshow(data_probs[i], vmin=0, vmax=1)
				ut.plt.remove_axes_ticks_and_labels(ax)
			with Next(ax_iter) as ax:
				ax.set_title(f'histogram of prob. of ssa.X_ssa[{i}]')
				hvals, hbins, hpatches = ax.hist(data_probs[i].ravel(), bins=100, density=False)
	return(bp_mask)

def ssa2d_chisq_bp_map(ssa, value=0.95, show_plots=1):
	"""
	Find a bad pixel map by computing the chi-squared statistic between the original image
	and the 0th SSA component.

	# ARGUMENTS #
		ssa
			An SSA2D instance. This class contains the single spectrum analysis of an image
		value
			A float between 0 and 1 that describes the quantile above which pixels will 
			be flagged as bad.
	# RETURNS #
		bp_map
			A boolean map of the bad pixels. Bad pixels are coded as True.
	"""
	small_adj = np.std(ssa.a)
	chisq = ((ssa.X_ssa[0] - ssa.a)**2 + small_adj)/(small_adj + np.fabs(img))

	bp_map = pixel_map(ssa.a, chisq, value, show_plots=show_plots, plot_kw={'suptitle':f'ssa2d chi squared maps\n{value=}'})
	return(bp_map)

def apply_cutoff(
		data : np.ndarray, 
		mode : str, 
		value : float
	) -> typing.Tuple[typing.Tuple[typing.Union[float, int]], np.ndarray, typing.Callable]:
	"""
	Takes in some data, a mode, and a value. Returns the cutoffs for that data.
	
	Applies different cutoffs (">" greater than, "<" less than, "><" out of interval, "<>" in interval)
	to distributions (generally probability distributions, but could be anything in principle) fitted
	to the input data. "value" tells us where the cutoffs should be.
	
	# EXAMPLE #
	>>> cutoffs, cutoff_map, det_func = apply_cutoff(some_array, '>gauss', 0.9)
	
	This would try to fit a gaussian distribution to 'some_array' (store the 
	fitted pdf in 'det_func'), find the cutoff where the cumulative density 
	function is 0.9 (pack the cutoff value into the tuple 'cutoffs'), and 
	apply the found cutoff to some_array in the sense described by the ">,<,><,<>"
	characters in 'mode' (store the boolean result in 'cutoff_map')
	"""
	_lgr.INFO(f'{data=} {mode=} {value=}')
	c_map = np.zeros_like(data, dtype=bool)
	determining_function = None # if applicable, this should be a function that we used to determine the cutoff for illustrative purposes
	
	value_sym_about_zero = lambda value: tuple(sorted((-value, value)))
	value_sym_about_z = lambda value,z: tuple(sorted((z-value, z+value)))
	value_sym_in_range = lambda value, r: tuple(sorted((r[0]+value, r[1]-value)))
	
	gt_idx = mode.find('>')
	lt_idx = mode.find('<')
	
	_lgr.INFO(f'{gt_idx=} {lt_idx=}')
	
	interval, in_interval, below_value, above_value = False, False, False, False
	if (gt_idx != -1) and (lt_idx!=-1):
		interval = True
		in_interval = lt_idx < gt_idx
		cutoff_method = lambda x, v: x.interval(v)
	elif gt_idx == 0:
		below_value = True
		cutoff_method = lambda x, v: x.ppf(v)
	elif lt_idx == len(mode)-1:
		above_value = True
		cutoff_method = lambda x, v: x.ppf(v)
	else:
		raise ValueError(f'Unknown mode modifier, expected (">mode<" | ">mode" | "mode<")')
	_lgr.INFO(f'{interval=} {in_interval=} {below_value=} {above_value=}')
	
	# strip out modifier characters
	sd = ''.join([s for s in mode if s not in '><'])
	_lgr.INFO(f'{sd=}')
	if sd == '':
		distrib = types.SimpleNamespace()
		distrib.__call__ = lambda *a, **k: distrib
		distrib.fit = lambda *a, **k: tuple()
		distrib.interval = lambda v: (v,v)
		distrib.ppf = lambda v: v
		distrib.pdf = lambda x: (x==value).astype(float)
	elif sd in ('std', 'normal', 'norm', 'gauss'):
		distrib = sp.stats.norm
	elif sd in ('gamma', 'gamma_cdf'):
		distrib = sp.stats.gamma
	elif sd in ('cauchy', 'cauchy_cdf'):
		distrib = sp.stats.cauchy 
	elif sd in ('betabinom', 'betabinom_cdf', 'betabinomial_cdf'):
		distrib = sp.stats.betabinom
		def distrib_fit(data, *args, **kwargs):
			distrib_fitted, distrib_params, distrib_err = ut.sp.fit_discrete_distribution_to(
														data, 
														distrib, 
														fixed_args=args, 
														p0=[1 for a in distrib.shapes.split(', ')]
														)

			return(*distrib_fitted.args,)
		distrib.fit = distrib_fit
	elif sd in ('cmp', 'cmp_cdf'):
		distrib = types.SimpleNamespace()
		def cmp_call(a, b):
			distrib.a = a
			distrib.b = b
			return(lambda x: ut.mfunc.cmp_pmf(x, a, b))
		distrib.__call__ = lambda a, b: (lambda x: ut.mfunc.cmp_pmf(x,a,b))	
		distrib.fit = lambda data: ut.mfunc.cmp_mle(data)
		distrib.interval = lambda v: (ut.mfunc.cmp_qf(0.5-v, distrib.a, distrib.b), ut.mfunc.cmp_qf(0.5+v, distrib.a, distrib.b))
		distrib.ppf = lambda v: ut.mfunc.cmp_qf(v, distrib.a, distrib.b)
		distrib.pdf = lambda x: ut.mfunc.cmp_pmf(x, distrib.a, distrib.b)
	else:
		raise ValueError(f'Unknown statistical distribution from {mode=}')
	
	
	distrib_params = distrib.fit(data.ravel())
	#print(distrib_params, type(distrib_params))
	#print(distrib)
	distrib_fitted = distrib(*distrib_params)
	
	# 'fake' a probability density function for discrete distributions
	# it won't have the correct properties, but should be good enough for
	# plotting to see what happens.
	if isinstance(distrib, sp.stats.rv_discrete):
		_lgr.INFO(f'Adding .pdf() method to discrete distribution {distrib_fitted}')
		distrib_fitted.pdf = lambda x: ut.np.with_continuous_domain(x, distrib_fitted.pmf)
		
	determining_function = distrib_fitted.pdf
	if interval:
		cutoffs = distrib_fitted.interval(value)
	elif below_value or above_value:
		cutoffs = (distrib_fitted.ppf(value),)
	_lgr.INFO(f'{cutoffs=}')
	
	
	if interval:
		if in_interval:
			c_map = (data > cutoffs[0]) & (data < cutoffs[1])
		else:
			c_map = (data < cutoffs[0]) | (data > cutoffs[1])
	elif below_value:
		c_map = data < cutoffs[0]
	elif above_value:
		c_map = data > cutoffs[0]
	else:
		raise RuntimeError('Unknown cutoff treatment')
	
	
	return(cutoffs, c_map, determining_function)

def ssa2d_ratio_bp_maps(ssa, 
		bins=100, start=0, stop=None, 
		cutoff_mode='>cauchy_cdf<', cutoff_value=0.90, 
		combine_what='bp_maps', combine_mode='sum', combined_cutoff_mode=None, combined_cutoff_value=0.8, 
		make_plots=0
	):
	"""
	Computes the ratio between the input image ssa.a and the np.sum(ssa.X_ssa[start:z], axis=0), 
	where z = range(start,stop)+1, representation of an image. I.e. the partial-sum of SSA components.
	
	# ARGUMENTS #
		ssa : py_ssa.SSA2D
			An instance of the SSA2D class, holds the single spectrum analysis of data passed to it.
		bins : int
			Number of bins to use in histograms
		start : int
			Where to start partial-summing SSA components (usually 0)
		stop : int | None
			Where to stop partial-summing the SSA components (None means create all partial sums)
		cutoff_mode : str
			How we should determine if the ratio of original image to SSA component
			partial sum is not consistent. Usually use ">cauchy_cdf<" which means
			reject a pixel if it is outside the (cutoff_value -> 1 - cutoff_value)
			range on the cumulative distribution function of a fitted cauchy distribution
		cutoff_value : Number
			Value passed to determine pixel cutoff in function described by "cutoff_mode"
		combine_what : "fracs" | "bp_map"
			What should we combine to generate a combined bad-pixel map?
			"fracs"
				combine SSA component partial sum ratios
			"bp_maps"
				combine bad-pixel maps found from applying "cutoff_mode" and 
				"cutoff_value" to SSA component partial sum ratios
		combine_mode : "sum" | "mean"
			How should we combine the bad-pixel maps found from SSA component partial sums
			to create a combined bad-pixel map?
		combined_cutoff_mode : str
			How should we determine if a pixel in the combined bad-pixel map is 
			inconsistent with out image? Same interpretation as "cutoff_mode", but 
			default is ">cauchy_cdf<" if "combine_mode" is "mean", or 
			">betabinomial_cdf" if "combine_mode" is "sum"
		combined_cutoff_value :  Number
			value passed to determine pixel cutoff in function described by "combined_cutoff_mode"
		make_plots : int
			Should we make plots of the results? 
			0 = No plots
			1 = Only combined plots
			2 = combined plots and ratio SSA component partial sum plots
	"""
	stop = if_none(stop, ssa.X_ssa.shape[0])
	if combine_what == 'fracs':
		combined_cutoff_mode = if_none(combined_cutoff_mode, '>cauchy_cdf<')
		combined_cutoff_value = if_none(combined_cutoff_value, 0.05)
	if combine_what == 'bp_maps':
		combined_cutoff_mode = if_none(combined_cutoff_mode, '>betabinomial_cdf')
		combined_cutoff_value = if_none(combined_cutoff_value, 0.766)
		
	# print arguments for debugging
	"""
	print(f'{ssa=}')
	print(f'{bins=}')
	print(f'{start=}')
	print(f'{stop=}')
	print(f'{cutoff_mode=}')
	print(f'{cutoff_value=}')
	print(f'{combine_what=}')
	print(f'{combine_mode=}')
	print(f'{combined_cutoff_mode=}')
	print(f'{combined_cutoff_value=}')
	print(f'{make_plots=}')
	"""
	
	fracs = np.zeros_like(ssa.X_ssa[start:stop])
	f_bp_maps = np.zeros_like(fracs)
	f_cutoffs = np.zeros((fracs.shape[0], 2))
	f_detfs = []
	
	# calculate ratio of original data to successive sums of ssa.X_ssa
	for i in range(stop-start):
		ii = i + start
		fracs[i] = ssa.a/np.sum(ssa.X_ssa[start:ii+1], axis=0)
		f_cutoffs[i], f_bp_maps[i], f_detf = apply_cutoff(fracs[i], cutoff_mode, cutoff_value)
		f_detfs.append(f_detf)
	
	if combine_what == 'fracs':
		combine_data = fracs
	elif combine_what == 'bp_maps':
		combine_data = f_bp_maps
	else:
		raise ValueError(f'Unknown dataset to combine {combine_what=}')
	
	if combine_mode=='sum':
		combine_function = lambda x: np.nansum(x, axis=0)
	elif combine_mode=='mean':
		combine_function = lambda x: np.nanmean(x, axis=0)
	else:
		raise ValueError(f'Unknow mode to combine dataset with {combine_mode=}')
		
	combined_map = combine_function(combine_data)
	combined_map_cutoffs, combined_map_bp_map, combined_detf = apply_cutoff(combined_map, combined_cutoff_mode, combined_cutoff_value)
	
	
	if make_plots:
		n_extra = 5
		n_axes = (stop-start)*2 + n_extra if make_plots > 1 else n_extra
		f1, a1 = ut.plt.figure_n_subplots(n_axes)
		a1=a1.flatten()
		f1.suptitle(f'ssa2d bad pixel identification via ratio\n{cutoff_mode=} {cutoff_value=}\n{combine_what=} {combine_mode=} {combined_cutoff_mode=} {combined_cutoff_value=}')
		ax_iter=iter(a1)
		
		with Next(ax_iter) as ax:
			im = ax.imshow(combined_map, norm=mpl.colors.SymLogNorm(1))
			#if mode=='mean':
			#	im.set_clim(ut.plt.lim_sym_around_value(combined_map, 1))
			#	im.set_cmap('bwr')
			clim = im.get_clim()
			ax.set_title(f'Combined map with flagged pixels (red)\nclim [{clim[0]:07.2E}, {clim[1]:07.2E}]')
			ax.plot(*np.nonzero(combined_map_bp_map)[::-1], ls='none', marker='o', markersize=1, mfc='none', mec='red', mew=0.5)
			
		with Next(ax_iter) as ax:
			ax.set_title(f'histogram of combined map ({combined_cutoff_mode=} cutoffs=[' + ' '.join([f'{_c:07.2E}' for _c in combined_map_cutoffs]) + '])')
			if combine_mode=='sum':
				these_bins=np.linspace(-0.5,f_bp_maps.shape[0]+0.5,f_bp_maps.shape[0]+2)
				#these_bins=np.linspace(+0.5,f_bp_maps.shape[0]+0.5,f_bp_maps.shape[0]+1)
			else:
				these_bins = bins
			#print(f'{these_bins=}')
			hvals, hbin_edges, hpatches = ax.hist(combined_map.ravel(), bins=these_bins, density=True)
			#if mode=='mean':
			#	ax.set_yscale('log')

			if combined_detf is not None:
				x = np.linspace(combined_map.min(), combined_map.max()+1, 100)
				#x = np.linspace(-combined_map.max(), combined_map.max(), 100)
				print(f'{x=}')
				print(f'{combined_detf(x)=}')
				ax.plot(x, combined_detf(x), color='tab:red', alpha=1, label='cutoff determining function')
			
			combined_cutoff_label = 'cutoffs'
			for combined_cutoff in combined_map_cutoffs:
				ax.axvline(combined_cutoff, color='tab:red', ls=':', label=combined_cutoff_label)
				combined_cutoff_label = None # only have one cutoff label
			#ax.set_yscale('log')
			ax.legend()
			
		with Next(ax_iter) as ax:
			ax.set_title('original flagged pixels removed')
			ax.imshow(set_bp(ssa.a, combined_map_bp_map))
			
		with Next(ax_iter) as ax:
			ax.set_title('Original image')
			ax.imshow(ssa.a)
			
		with Next(ax_iter) as ax:
			ax.set_title('Original image interpolated')
			#combined_map_bp_map[((0,0,-1,-1),(0,-1,0,-1))] = False # don't mask corners
			edge_mask = np.ones_like(combined_map_bp_map, dtype=bool)
			edge_mask[1:-1,1:-1] = False
			print(edge_mask)
			change_edge_mask = edge_mask & combined_map_bp_map
			
			
			data_altered = np.array(ssa.a)
			data_altered[change_edge_mask] = np.std(data_altered[combined_map_bp_map])
			combined_map_bp_map_unmasked_corners = np.array(combined_map_bp_map)
			combined_map_bp_map_unmasked_corners[edge_mask] = False
			
			ax.imshow(ut.sp.interpolate_at_mask(data_altered, combined_map_bp_map_unmasked_corners, method='cubic'))
			
		if make_plots > 1:
			for i in range(start,stop):
				j = i-start
				with Next(ax_iter) as ax:
					ax.set_title(f'frac[{j}]')
					ax.imshow(fracs[j])
					ax.plot(*np.nonzero(f_bp_maps[i])[::-1], ls='none', marker='o', markersize=1, mfc='none', mec='red', mew=0.5)
				
				with Next(ax_iter) as ax:
					ax.set_title(f'frac[{j}] histogram cutoffs=[' + ' '.join([f'{_c:07.2E}' for _c in f_cutoffs[j]]) + '])')
					hv, hb, hp = ax.hist(fracs[j].ravel(), bins=bins, density=True)
					x = np.linspace(np.min(hb), np.max(hb), 100)
					ax.set_yscale('log')
					ylims = ax.get_ylim()
					#ax_n_std_lines(ax.axvline, fracs[j], n=value)
					[ax.axvline(_v, color='tab:red', ls='--') for _v in f_cutoffs[j]]
					ax.plot(x, f_detfs[j](x), label='cutoff detf')
					ax.set_ylim(ylims)
			
	return(combined_map_bp_map)
	

def interp_px_where(img, mask, show_plots=0):
	img_dp_removed = ut.sp.interpolate_at_mask(img, mask, edges='convolution', method='cubic')
	if show_plots > 0:
		f1, a1 = ut.plt.figure_n_subplots(4)
		a1=a1.flatten()
		f1.suptitle(f'interpolating pixels at mask')
		ax_iter=iter(a1)
		
		with Next(ax_iter) as ax:
			ax.set_title('original')
			ax.imshow(img)
			
		with Next(ax_iter) as ax:
			ax.set_title('mask')
			ax.imshow(mask)
		
		with Next(ax_iter) as ax:
			ax.set_title('interpolated')
			ax.imshow(img_dp_removed)
			
		with Next(ax_iter) as ax:
			ax.set_title('residual')
			ax.imshow(img_dp_removed - img)
		
	return(img_dp_removed)

def plot_pixel_map_test(img, test, cutoff, mask, interp, plot_kw={}):
	f2, a2 = ut.plt.figure_n_subplots(6)
	a2_iter = iter(a2.ravel())
	f2.suptitle(plot_kw.get('suptitle', 'pixel map function'))
	
	with Next(a2_iter) as ax:
		ax.set_title(f'Original image\nsum {np.nansum(img):08.2E}\nsqrt(sum^2) {np.sqrt(np.nansum(img**2)):08.2E}')
		ax.imshow(img)
		ut.plt.remove_axes_ticks_and_labels(ax)
		
	with Next(a2_iter) as ax:
		ax.set_title(f'pixel choice test function\nsum {np.nansum(test):08.2E} sqrt(sum^2) {np.sqrt(np.nansum(test**2)):08.2E}')
		ax.imshow(test)
		ut.plt.remove_axes_ticks_and_labels(ax)
		
	with Next(a2_iter) as ax:
		ax.set_title(f'histogram of choice function\ncutoff {cutoff:06.4f}')
		ax.hist(test.ravel(), bins=100)
		ax.axvline(cutoff, color='red', ls='--')
	
	with Next(a2_iter) as ax:
		ax.set_title(f'mask from cut of choice function\nn_masked {np.nansum(mask)} frac_masked {np.nansum(mask)/mask.size:08.2E}')
		ax.imshow(mask)
		ut.plt.remove_axes_ticks_and_labels(ax)
		
	with Next(a2_iter) as ax:
		ax.set_title('\n'.join((f'original interpolated at mask',
								f'sum {np.nansum(interp):08.2E} frac {np.nansum(interp)/np.nansum(img):08.2E}',
								f'sqrt(sum^2) {np.sqrt(np.nansum(interp**2)):08.2E} frac {np.sqrt(np.nansum(interp**2))/np.sqrt(np.nansum(img**2)):08.2E}',
		)))
		ax.imshow(interp)
		ut.plt.remove_axes_ticks_and_labels(ax)
		
	with Next(a2_iter) as ax:
		residual = interp - img
		ax.set_title('\n'.join((f'residual of interpolation - original',
								f'sum {np.nansum(residual):08.2E} frac {np.nansum(residual)/np.nansum(img):08.2E}',
								f'sqrt(sum^2) {np.sqrt(np.nansum(residual**2)):08.2E} frac {np.sqrt(np.nansum(residual**2))/np.sqrt(np.nansum(img**2)):08.2E}',
		)))
		ax.imshow(residual)
		ut.plt.remove_axes_ticks_and_labels(ax)

def pixel_map(
		img, 
		test, 
		value, 
		show_plots=0, 
		plot_kw={}, 
		cutoff_func = lambda _test, _value: ut.sp.construct_ppf_from(_test.ravel())(_value), 
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
		interpolated = ut.sp.interpolate_at_mask(img, bp_mask, edges='convolution', method='cubic')
		plot_pixel_map_test(img, test, cutoff, bp_mask, interpolated, plot_kw=plot_kw)
	return(bp_mask)





if __name__=='__main__':
	import fitscube.deconvolve.helpers
	#obs, psf = fitscube.deconvolve.helpers.get_test_data()
	#img_list = [obs, psf]

	args=dict(
		bp_mask_function = ('cumulative', 'probability_sum', 'ratio', 'chi_squared')[1]
	)
	
	def path_filter_f(apath, exclude_strs = ('reformat', 'CLEAN', 'clean', 'reformat', 'deconvolve', 'STD', 'GEMINI')):
		for x in exclude_strs:
			if x in apath: return(False)
		return(True)
	
	img_paths = fitscube.deconvolve.helpers.get_test_data_set_list(filter_function = path_filter_f)
	for img_path in img_paths:
		print(img_path)
	#sys.exit()
	
	img_list = fitscube.deconvolve.helpers.get_test_data_set(filter_function = path_filter_f, aslice=slice(None,3), verbose=1)
	
	
	for i, img in enumerate(img_list):
		#if img.size < 10000: continue
		#img[img==0] = np.nan
		print(f'{img.shape=}')
		
		#img = np.nan_to_num(img)
		#img = ut.np.remove_value_frame(img, value=0)
		
		print(f'after remove frame {img.shape=}')
		img, trim_slices = ut.np.trim_array(img, signal_loss=0.05)
		print(f'after trim {img.shape=}')
		
		# we need to remove all NAN and INF pixels for SSA to work
		img = interp_px_where(img, np.isnan(img))
		img = interp_px_where(img, np.isinf(img))
		
		#w_shape = tuple([s//10 for s in img.shape])
		w_shape = tuple([10 for s in img.shape])
		print(f'Window size {w_shape}')
		
		img_ssa = py_ssa.SSA2D(img, w_shape)
		
	
		if args['bp_mask_function'] == 'cumulative':
			_lgr.INFO('Finding bad pixel mask from cumulative histograms of SSA components')
			cbp_mask = ssa2d_cumulative_bp_map(	img_ssa, stop=img_ssa.m//8, sum_stop=img_ssa.m//4, 
												show_plots=1, cutoff_value=0.90, cumulative_cutoff=0.95
												)	
		elif args['bp_mask_function'] == 'probability_sum':
			_lgr.INFO('Finding bad pixel mask from sum of probability maps of SSA components')
			cbp_mask = ssa2d_sum_prob_map(img_ssa, start=3, stop=img_ssa.m//4)#, show_plots=1) # best so far
		
		elif args['bp_mask_function'] == 'ratio':
			_lgr.INFO('Finding bad pixel mask from ratio of original image to partial sums of SSA components')
			ssa2d_ratio_bp_maps(img_ssa, start=0, stop=None, make_plots=1, cutoff_value=0.90, combined_cutoff_value=0.8)

		elif args['bp_mask_function'] == 'chi_squared':	
			_lgr.INFO('Finding bad pixel mask from chi-squared statistic between original image and 0th SSA component')
			cbp_mask = ssa2d_chisq_bp_map(img_ssa, value=0.95) # pretty good

		else:
			_lgr.ERROR('Unknown value {repr(args["bp_mask_function"])} to argument "bp_mask_function"')
			raise ValueError




	plt.show()
	
	
	
	
	
	
	
	
