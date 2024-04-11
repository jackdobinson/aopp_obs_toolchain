"""
Contains routines that are mostly wrappers around scipy interpolation functions
"""

import numpy as np
import scipy as sp
import scipy.interpolate

def interpolate_at_mask(data, mask, edges=None, **kwargs):
	"""
	Interpolates an array 'data' at the True points given in 'mask'
	**kwargs is passed through to "sp.interpolate.griddata()"
	"""
	if edges is None:
		interp_data = np.array(data)
	elif edges == 'convolution':
		# build convolution kernel
		kernel_shape = tuple(3 for s in data.shape)
		kernel_edge_shape = tuple(s//2 for s in kernel_shape)
		kernel = np.ones(kernel_shape)
		kernel /= np.sum(kernel)
		embed_shape = tuple(s+2*kes for kes, s in zip(kernel_edge_shape, data.shape))
		embed_slice = tuple(slice(kes,-kes) for kes in kernel_edge_shape)
		
		# build data and mask with a frame of zeros
		interp_data = np.zeros(embed_shape)
		interp_data[embed_slice] = data
		interp_mask = np.zeros(embed_shape, dtype=mask.dtype)
		interp_mask[embed_slice] = mask
		mask = interp_mask
		
		# interpolate the framed data, this should remove the effects of any masked out points
		interp_data = interpolate_at_mask(interp_data, interp_mask, **kwargs)
		# convolve the interpolated framed data, the frame should now be non-zero but tending towards zero
		interp_data = sp.signal.fftconvolve(interp_data, kernel, mode='same')
		# put the original data back into the frame, we are just using the frame to anchor the interpolation.
		interp_data[embed_slice] = data
	else:
		raise ValueError(f'Unknown edge interpolation strategy "{edges=}"')

	#points = np.array(mgrid_from_array(interp_data, gridder=np.mgrid))
	points = np.indices(interp_data.shape)

	p_known = points[:,~mask].T
	p_unknown = points[:,mask].T
	known_values = interp_data[~mask]

	interp_values = sp.interpolate.griddata(p_known, known_values, p_unknown, **kwargs)

	interp_data[mask] = interp_values
	
	if edges is None:
		return(interp_data)
	elif edges == 'convolution':
		return(interp_data[embed_slice]) # only want the embedded interpolated data, not the frame
	raise ValueError(f'Unknown edge interpolation strategy "{edges=}", this should never happen.')