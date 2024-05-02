"""
Uses SSA to get values for interpolation of data
"""

import numpy as np
import scipy as sp
import scipy.stats
import scipy.ndimage
import matplotlib.pyplot as plt


import aopp_deconv_tool.context as context
from aopp_deconv_tool.context.next import Next
from aopp_deconv_tool.stats.empirical import EmpiricalDistribution
from aopp_deconv_tool.scipy_helper.interp import interpolate_at_mask
import aopp_deconv_tool.plot_helper as plot_helper



import aopp_deconv_tool.cfg.logs
_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'DEBUG')


def ssa_intepolate_at_mask(
		ssa,
		mask,
		start=0, 
		stop=None, 
		value=0.5, # 0.5 seems to be the best value
		show_plots=0,
	):
	"""
	Using SSA for interpolation. E.g. 
		1) pass in a set of pixels
		2) Calculate the 'difference from median of SSA component' score for each pixel
		3) For each pixel, only combine SSA components when the |score| < 'some value'
		This should ensure that 'extreme' values for that pixel are ignored and the 
		reconstructed pixel value is more similar to the surrounding pixels.
	
	This should help when interpolating across a e.g. "hot" pixel as the components
	that contribute to the "hotness" will be ignored and the components that are more
	similar to the surroundings will contribute.
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
		_lgr.debug(f'i={i}/{ssa.X_ssa.shape[0]}')
		data_distribution = EmpiricalDistribution(ssa.X_ssa[i].ravel())
		
		px_contrib_temp[:] = ssa.X_ssa[i][mask]
		
		_lgr.debug(f'{px_contrib_temp=}')

		if i >= start and i < stop:
			j = i - start

			# Interpolate the SSA components
			#points = np.indices(ssa.X_ssa[i].shape)
			#p_known = points[:,~mask].T
			#p_unknown = points[:,mask].T
			#known_values = ssa.X_ssa[i][~mask]
			#interp_values = sp.interpolate.griddata(p_known, known_values, p_unknown)
			interp_values = interpolate_at_mask(ssa.X_ssa[i], mask, edges='convolution')[mask]
			
			# Get the distance from the median for each pixel
			data_probs[j,...] = prob_median_transform_func(data_distribution.cdf(ssa.X_ssa[j].ravel()).reshape(ssa.a.shape))
			
			# If masked pixel is close to the median, don't replace it. Otherwise, replace with interpolated value
			px_contrib_mask[...] = np.fabs(data_probs[j, mask]) >= value
			_lgr.debug(f'{px_contrib_mask=}')
			
			# Only use the interpolated components when we have extreme values
			px_contrib_temp[px_contrib_mask] = interp_values[px_contrib_mask]
		
		_lgr.debug(f'{px_contrib_temp=}')
		interp_mask_accumulator += px_contrib_temp
		_lgr.debug(f'{interp_mask_accumulator=}')

	result = np.sum(ssa.X_ssa, axis=0)
	result[mask] = interp_mask_accumulator
	
	return result