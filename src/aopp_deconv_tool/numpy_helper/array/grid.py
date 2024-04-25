"""
Regrids a numpy array.


Example of grid definition:

	#############################################################

	input_grid    : |---^---|   |---^---|   |---^---|   |---^---|
	              :       |---^---|   |---^---|   |---^---|   	
	
	bin_width     : |-------|

	bin_step      : |-----|

	#############################################################

	output_grid   :     |--------^-------|    |--------^-------|
	              :                |--------^-------|
	
	offset        : |---|

	new_bin_width :     |----------------|
	
	new_bin_step  :     |----------|


	#############################################################

	"|" = bin edge

	"^" = bin center

	width
		The distance between the start and end of a bin
	
	step
		The distance between the start of two consencutive bins

	offset
		The distance between the start/end of the new grid and the start/end of the old grid



"""

import numpy as np

import aopp_deconv_tool.cfg.logs
_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'DEBUG')


def edges_from_midpoints(
		midpoints : np.ndarray,
		fix_edges : bool = True
	) -> np.ndarray:
	"""
	Get bin edges as a 2d-numpy array of shape (2,N), where the
	0th sub-array is the starting point of each bin, and the 1st sub-array is the ending points.

	# ARGUMENTS #
		midpoints : np.ndarray[N]
			Midpoints of bins to find the edges of.
		fix_edges : bool = True
			If we should account for bin size when working out the 0th and Nth bin edges.

	# RETURNS #
		edges : np.ndarray[2,N]
			A 2D numpy array, "edges[0]" holds the starting edges of each bin, 
			"edges[1]" holds the ending edges of each bin.
			
	"""
	if fix_edges:
		x0 = (3*midpoints[0] - midpoints[1])/2
		x1 = (3*midpoints[[-1]] - midpoints[-2])/2
	else:
		x0 = midpoints[0]
		x1 = midpoints[-1]
	edges = np.zeros((2, midpoints.size))
	edges[0,0] = x0
	edges[1,-1] = x1
	_x = (midpoints[:-1] + midpoints[1:])/2
	edges[0,1:] = _x
	edges[1,:-1] = _x

	return edges

def edges_from_bounds(
		lbound : float, 
		ubound : float, 
		steps : float, 
		widths : float, 
		bins_extend_past_ubound=True
	) -> np.ndarray:
	"""
	Get the bin edges of bins defined by a lower bound, upper bound, steps between bins, and widths of each bin.

	# ARGUMENTS #
		lbound : float
			Lower bound of the range to create bins on.
		ubound : float
			Upper bound of the range to create bins on.
		steps : float
			Size of the steps between the starting edge of each bin.
		widths : float
			Size of the bins, i.e. distance between the starting edge and ending edge of each bin.
		bins_extend_past_ubound : bool = True
			Should the bins be allowed to extend past "ubound" if they could contain values < "ubound".
			I.e. Could a bin of width = 5, start at ubound-2 and end at ubound+3?

	# RETURNS #
		edges : np.ndarray[2,N]
			A 2D numpy array, "edges[0]" holds the starting edges of each bin, 
			"edges[1]" holds the ending edges of each bin.
	
	"""
	if not bins_extend_past_ubound:
		ubound = ubound - widths
	_lgr.debug(f'{lbound=} {ubound=} {steps=} {widths=} {bins_extend_past_ubound=}')
	n_bins = int(np.ceil((ubound-lbound)/steps))
	edges = np.full((2,n_bins), fill_value=np.nan)
	edges[0] = np.arange(lbound, ubound, steps)
	edges[1] = edges[0] + widths
	return edges


def regrid(
		data : np.ndarray, 
		input_bin_edges : np.ndarray, 
		output_bin_edges : np.ndarray, 
		axis : int = 0
	) -> np.ndarray:
	"""
	Rebin data from the old bins to a new set of bins, old data is summed into new data bins.
	Set NANs to zero before summing if you want to treat them that way.

	# ARGUMENTS #
		data : np.array[...,N,...]
			The data to be re-gridded. Should make sense to add the data together,
			N is the size of the axis over which data is to be rebinned. Choice of axis
			is controlled by "axis" argument.
		input_bin_edges : np.array[2,N]
			Edges of the old bins, input_bin_edges[0] are the starts of each bin
			input_bin_edges[1] are the ends of each bin.
		output_bin_edges : np.array[2,M]
			Edges of the new bins, output_bin_edges[0] are the starts of each bin,
			output_bin_edges[1] are the ends of each bin.
		axis : int = 0
			Axis to rebin "data" across.

	# RETURNS #
		output_bin_values : np.array[...,M,...]
			Rebinned data values, same shape as "data" except along the rebinned axis.
		n_bins_combined : np.array[M]
			Number of original bins combined into each new bin, fractional bins accounted for.
	"""
	# Move the axis we are working on to the 0th position, that way we
	# don't have to keep track of which axis we are using everywhere. Just remember to
	# move it back at the end.
	np.moveaxis(data, axis, 0)

	i_array= np.arange(input_bin_edges[0].size)
	i_dash_array = np.arange(input_bin_edges[1].size)

	# pos of new bin starts in index of old bin starts
	# floor of this tells us the index of the largest fractional bin at the starting edge
	i_j_array = np.interp(output_bin_edges[0], input_bin_edges[0], i_array)

	# pos of new bin starts in index of old bin ends
	# floor of this tells us the largest index that ends before this new bin starts
	i_dash_j_array = np.interp(output_bin_edges[0], input_bin_edges[1], i_dash_array)
	
	# pos of new bin ends in index of old bin starts
	# floor of this tells us index of the last bin to include in the new bin
	i_j_dash_array = np.interp(output_bin_edges[1], input_bin_edges[0], i_array)
	
	# pos of new bin ends in index of old bin ends
	# floor of this tells us the index of the last full bin included in the new bin
	i_dash_j_dash_array = np.interp(output_bin_edges[1], input_bin_edges[1], i_dash_array)

	# pos of old bin ends in terms of old bin starts
	i_i_dash_array = np.interp(input_bin_edges[1], input_bin_edges[0], i_array)

	output_bin_values = np.zeros((output_bin_edges[0].size, *data.shape[1:]))

	old_bin_idxs = np.arange(input_bin_edges[0].size)
	n_bins_combined = np.zeros((output_bin_edges[0].size,))

	# I feel like there should be a more elegant way to do this.
	for k, (i_j, i_dash_j_dash, i_dash_j, i_j_dash) in enumerate(zip(i_j_array, i_dash_j_dash_array, i_dash_j_array, i_j_dash_array)):
		include_bins = (old_bin_idxs >= i_dash_j) & (old_bin_idxs <= i_j_dash)
		lower_fractional_bins = (old_bin_idxs >= i_dash_j) & (old_bin_idxs < i_j)
		upper_fractional_bins = (old_bin_idxs < i_j_dash) & (old_bin_idxs > i_dash_j_dash)
		whole_bins = (old_bin_idxs >= i_j) & (old_bin_idxs <= i_dash_j_dash)
		whole_bin_idxs = old_bin_idxs[whole_bins]
		lower_fractional_bin_idxs = old_bin_idxs[lower_fractional_bins]
		upper_fractional_bin_idxs = old_bin_idxs[upper_fractional_bins]

		output_bin_values[k] += np.sum(data[whole_bins], axis=0)
		n_bins_combined[k] += np.sum(whole_bins)

		for l, a,b in zip(lower_fractional_bin_idxs, input_bin_edges[0][lower_fractional_bin_idxs], input_bin_edges[1][lower_fractional_bin_idxs]):
			frac_of_bin = (b - output_bin_edges[0][k])/(b-a)
			output_bin_values[k] +=  frac_of_bin* data[l]
			n_bins_combined[k] += frac_of_bin

		for l, a,b in zip(upper_fractional_bin_idxs, input_bin_edges[0][upper_fractional_bin_idxs], input_bin_edges[1][upper_fractional_bin_idxs]):
			frac_of_bin = (output_bin_edges[1][k] - a)/(b-a)
			output_bin_values[k] += frac_of_bin * data[l]
			n_bins_combined[k] += frac_of_bin


	# Once all calculation is complete, move the result back to the
	# axis it is supposed to be in.
	np.moveaxis(output_bin_values, 0, axis)

	return(output_bin_values, n_bins_combined)