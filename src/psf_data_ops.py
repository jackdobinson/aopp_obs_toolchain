"""
Module containing routines that operate on point spread function data
"""
import numpy as np
import numpy_helper as nph
import numpy_helper.array
import numpy_helper.slice
import scipy as sp
import scipy.ndimage

import cfg.logs
_lgr = cfg.logs.get_logger_at_level(__name__, 'DEBUG')

def normalise(
		data : np.ndarray, 
		axes : tuple[int,...] | None=None, 
		cutout_shape : tuple[int,...] | None = None,
	) -> np.ndarray:
	"""
	Ensure an array of data fufils the following conditions:
	
	* odd shape, to ensure a center pixel exists
	* center array on brightest pixel
	* ensure array sums to 1
	* cut out a region around the center to remove unneeded data.
	"""
	if axes is None:
		axes = tuple(range(data.ndim))
	
	data[np.isinf(data)] = np.nan # ignore infinities
	data = nph.array.ensure_odd_shape(data, axes)
	
	
	# center around brightest pixel
	for idx in nph.slice.iter_indices(data, group=axes):
		bp_offset = nph.array.get_center_offset_brightest_pixel(data[idx])
		data[idx] = nph.array.apply_offset(data[idx], bp_offset)
		data[idx] /= np.nansum(data[idx])
	
	
	# cutout region around the center of the image if desired,
	# this is pretty important when adjusting for center of mass, as long
	# as the COM should be close to the brightest pixel
	if cutout_shape is not None:
		_lgr.debug(f'{tuple(data.shape[x] for x in axes)=} {cutout_shape=}')
		center_slices = nph.slice.around_center(tuple(data.shape[x] for x in axes), cutout_shape)
		_lgr.debug(f'{center_slices=}')
		slices = [slice(None) for s in data.shape]
		for i, center_slice in zip(axes, center_slices):
			slices[i] = center_slice
		_lgr.debug(f'{slices=}')
		data = data[tuple(slices)]
	
	
	
	# move center of mass to middle of image
	# threshold
	threshold = 1E-3
	with nph.axes.to_start(data, axes) as (gdata, gaxes):
		t_mask = (gdata > threshold*np.nanmax(gdata, axis=gaxes))
		_lgr.debug(f'{t_mask.shape=}')
		indices = np.indices(gdata.shape)
		_lgr.debug(f'{indices.shape=}')
		com_idxs = (np.nansum(indices*gdata*t_mask, axis=tuple(a+1 for a in gaxes))/np.nansum(gdata*t_mask, axis=gaxes))[:len(gaxes)].T
		_lgr.debug(f'{com_idxs.shape=}')
	
	_lgr.debug(f'{data.shape=}')
	
	for _i, (idx, gdata) in enumerate(nph.axes.iter_axes_group(data, axes)):
		_lgr.debug(f'{_i=}')
		_lgr.debug(f'{idx=}')
		_lgr.debug(f'{gdata[idx].shape=}')
		
		
		# calculate center of mass
		#com_idxs = tuple(np.nansum(data[idx]*indices)/np.nansum(data[idx]) for indices in np.indices(data[idx].shape))
		center_to_com_offset = np.array([com_i - s/2 for s, com_i in zip(gdata[idx].shape, com_idxs[idx][::-1])])
		_lgr.debug(f'{idx=} {com_idxs[idx]=} {center_to_com_offset=}')
		_lgr.debug(f'{sp.ndimage.center_of_mass(np.nan_to_num(gdata[idx]*(gdata[idx] > threshold*np.nanmax(gdata[idx]))))=}')
		
		# regrid so that center of mass lies on an exact pixel
		old_points = tuple(np.linspace(0,s-1,s) for s in gdata[idx].shape)
		interp = sp.interpolate.RegularGridInterpolator(
			old_points, 
			gdata[idx], 
			method='linear', 
			bounds_error=False, 
			fill_value=0
		)
	
		# have to reverse center_to_com_offset here
		new_points = tuple(p-center_to_com_offset[i] for i,p in enumerate(old_points))
		_lgr.debug(f'{[s.size for s in new_points]=}')
		new_points = np.array(np.meshgrid(*new_points)).T
		_lgr.debug(f'{[s.size for s in old_points]=} {gdata[idx].shape=} {new_points.shape=}')
		gdata[idx] = interp(new_points)
		
	return data
