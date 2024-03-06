"""
Uses SSA to get values for interpolation of data
"""

import numpy as np
import scipy as sp
import scipy.stats
import scipy.ndimage

import context
from context.next import Next

from stats.empirical import EmpiricalDistribution
from scipy_helper.interp import interpolate_at_mask

import plot_helper
import matplotlib.pyplot as plt

import cfg.logs
_lgr = cfg.logs.get_logger_at_level(__name__, 'DEBUG')


def ssa_intepolate_at_mask(
		ssa,
		mask,
		start=0, 
		stop=None, 
		value=0.5, 
		show_plots=0,
		median_size = 5,
	):
	"""
	Using SSA for interpolation. E.g. 
		1) pass in a set of pixels
		2) Calculate the 'difference from median of SSA component' score for each pixel
		3) For each pixel, only combine SSA components when the |score| < 'some value'
		This should ensure that 'extreme' values for that pixel are ignored and the 
		reconstructed pixel value is more similar to the surrounding pixels.
	"""
	# set up arrays and defaults
	stop = ssa.X_ssa.shape[0] if stop is None else stop
	start = 0 if start is None else start
	data_probs = np.zeros((stop-start, *ssa.a.shape),dtype=float)
	
	# choose a pixel->probability function
	# we get a range of (0,1) from "ut.sp.construct_cdf_from()"
	# it gives us the cumulative probability for each pixel
	# This function changes it to 'distance away from median' in the range [-1, 1]
	prob_median_transform_func = lambda x: (2*(x-0.5))
	
	# use the mask to 
	interp_mask_accumulator = np.zeros((np.count_nonzero(mask),), dtype=float)
	px_contrib_mask = np.zeros_like(interp_mask_accumulator, dtype=bool)
	px_contrib_temp = np.zeros_like(interp_mask_accumulator, dtype=float)
	
	# apply pixel->probability function to each SSA component
	for i in range(ssa.X_ssa.shape[0]):
		data_distribution = EmpiricalDistribution(ssa.X_ssa[i].ravel())
		
		px_contrib_temp[:] = ssa.X_ssa[i][mask]
		
		_lgr.debug(f'{px_contrib_temp=}')

		if i >= start and i < stop:
			j = i - start
			#median_value = data_distribution.ppf(0.5)
			median_filtered_components = sp.ndimage.median_filter(ssa.X_ssa[i], size=median_size)
			data_probs[j,...] = prob_median_transform_func(data_distribution.cdf(ssa.X_ssa[j].ravel()).reshape(ssa.a.shape))

			px_contrib_mask[...] = np.fabs(data_probs[j, mask]) >= value
			_lgr.debug(f'{px_contrib_mask=}')
			#px_contrib_temp[px_contrib_mask] = median_value
			px_contrib_temp[px_contrib_mask] = median_filtered_components[mask][px_contrib_mask]
		
		_lgr.debug(f'{px_contrib_temp=}')
		interp_mask_accumulator += px_contrib_temp
		_lgr.debug(f'{interp_mask_accumulator=}')

	result = np.sum(ssa.X_ssa, axis=0)
	result[mask] = interp_mask_accumulator
	
	return result