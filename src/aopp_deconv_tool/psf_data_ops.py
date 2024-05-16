"""
Module containing routines that operate on point spread function data
"""
from typing import Callable, TypeVar, Generic, ParamSpec, TypeVarTuple, Any
import functools
import numpy as np
import scipy as sp
import scipy.ndimage

import aopp_deconv_tool.numpy_helper as nph
import aopp_deconv_tool.numpy_helper.array
import aopp_deconv_tool.numpy_helper.slice
from aopp_deconv_tool.optimise_compat import PriorParamSet

from aopp_deconv_tool.stats.empirical import EmpiricalDistribution

import aopp_deconv_tool.cfg.logs
_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'INFO')


#DEBUGGING
import matplotlib.pyplot as plt


IntVar = TypeVar('IntVar', bound=int)
T = TypeVar('T')
Ts = TypeVarTuple('Ts')
P = ParamSpec('P')
Q = ParamSpec('Q')
class S(Generic[IntVar]): pass
class S1(Generic[IntVar]): pass
N = TypeVar('N',bound=int)
M = TypeVar('N',bound=int)


def normalise(
		data : np.ndarray, 
		axes : tuple[int,...] | None=None, 
		cutout_shape : tuple[int,...] | None = None,
		recenter_around_center_of_mass = False,
		remove_background = True
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
		data[idx] /= np.nansum(data[idx]) # normalise
	
	if remove_background:
		# assume that the bottom-left corner is all background
		bg_region_slice = tuple(slice(0,data.shape[a]//10) for a in axes)
		for idx in nph.slice.iter_indices(data, group=axes):
			data[idx] -= np.median(data[idx][*bg_region_slice])
			data[idx] /= np.nansum(data[idx]) # normalise
	
	
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
	
	
	if recenter_around_center_of_mass:
		# move center of mass to middle of image
		# threshold
		threshold = 1E-2
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
	
	# Normalise again
	#for idx in nph.slice.iter_indices(data, group=axes):
	#	data[idx] /= np.nansum(data[idx])
	
	return data


def remove_offset(data, axes, mask, model_name='norm'):
	
	is_bad = np.isnan(data) | np.isinf(data)
	_lgr.debug(f'{is_bad.shape=}')
	
	
	match model_name:
		case 'norm':
			model = sp.stats.norm
		case 'gennorm':
			model = sp.stats.gennorm
		case _:
			raise RuntimeError(f'Unknown {model_name=} when fitting background for offset removal')
	
	param_names = ([x.strip() for x in model.shapes.split(',')] if model.shapes is not None else []) + ['loc', 'scale']
	
	
	nm_j_to_i = []
	noise_model_offsets = []
	noise_model_parameters = []
	noise_model_cdf = []
	noise_model_cdf_residual = []
	
	for i, (idx, gdata) in enumerate(nph.axes.iter_axes_group(data, axes)):
		if i % 10 == 0:
			_lgr.info(f'{i=}')
		_lgr.debug(f'{gdata[idx].shape=}')
		_lgr.debug(f'{mask[idx].shape=}')
		good_data = gdata[idx][mask[idx] & ~is_bad[idx]]
		_lgr.debug(f'{good_data.shape=}')
		
		if np.all(np.isnan(good_data)):
			_lgr.info(f'Skipping {i}^th offset calculation as whole slice is NANs')
			noise_model_offsets.append(np.nan)
			noise_model_parameters.append(dict([(k,np.nan) for k in param_names]))
			
			continue
			
		gd_distrib = EmpiricalDistribution(good_data.flatten())
		v, cdf = gd_distrib.whole_cdf()
		_lgr.debug(f'{v.shape=} {cdf.shape=}')
		
		
		params = model.fit(good_data)
		_lgr.debug(f'{param_names=} {params=}')
		noise_model_parameters.append(dict(zip(param_names, params)))
		_lgr.debug(f'{noise_model_parameters[-1]=}')
		noise_model_offsets.append(noise_model_parameters[-1]['loc'])
		
		good_data -= noise_model_offsets[-1]
		#gd_distrib = EmpiricalDistribution(good_data.flatten())
		#v, cdf = gd_distrib.whole_cdf()
		params = model.fit(good_data)
		
		noise_model_parameters[-1].update(dict(zip(param_names, params)))
	
	noise_model_at_values = []#np.linspace(np.nanmin(data), np.nanmax(data), 500)
		
	for i, (idx, gdata) in enumerate(nph.axes.iter_axes_group(data, axes)):
		if i % 100 == 0:
			_lgr.info(f'{i=}')
		good_data = gdata[idx][mask[idx] & ~is_bad[idx]]
		if np.all(np.isnan(good_data)):
			noise_model_at_values.append(np.full((500,), fill_value=np.nan))
			noise_model_cdf.append(np.full((500,), fill_value=np.nan))
			noise_model_cdf_residual.append(np.full((500,), fill_value=np.nan))
			continue
		gd_distrib = EmpiricalDistribution(good_data.flatten())
		
		noise_model_at_values.append(np.linspace(np.nanmin(good_data), np.nanmax(good_data), 500))
		noise_model_cdf.append(model(*noise_model_parameters[i].values()).cdf(noise_model_at_values[i]))
		noise_model_cdf_residual.append(noise_model_cdf[i] - gd_distrib.cdf(noise_model_at_values[i]))	
	
	
	return noise_model_offsets, noise_model_parameters, noise_model_at_values, noise_model_cdf, noise_model_cdf_residual


def trim_around_center(
		data : np.ndarray,
		axes : tuple[int,...],
		output_shape : tuple[int,...]
	) -> np.ndarray:
	"""
	# ARGUMENTS #
	
	data : np.ndarray
		Data to trim around center pixel, will remove pixels that are greater than shape[i]/2 from the center pixel.
	axes : tuple[int,...]
		Axes over which to operate
	output_shape : tuple[int,...]
		Desired shape of `axes` after trimming
	"""
	_lgr.debug(f'{data.shape=}')
	_lgr.debug(f'{axes=}')
	_lgr.debug(f'{output_shape=}')
	old_shape = data.shape
	new_shape = list(old_shape)
	for i in range(data.ndim):
		if i in axes:
			j = axes.index(i)
			new_shape[i] = output_shape[j]
		else:
			new_shape[i] = old_shape[i]
	new_shape = tuple(new_shape)
	
	
	slices = [slice(None)]*data.ndim
	for i in range(data.ndim):
		if i in axes:
			center_idx = data.shape[i]//2
			lo_idx = center_idx - new_shape[i]//2
			hi_idx = center_idx + new_shape[i]//2 + 1
			slices[i] = slice(lo_idx,hi_idx)
		else:
			slices[i] = slice(None)
	
	slices = tuple(slices)
	_lgr.debug(f'{slices=}')
	
	
	return data[slices]
	


def get_outlier_mask(
		data : np.ndarray,
		axes : tuple[int,...],
		n_sigma : float = 5,
	) -> np.ndarray:
	"""
	# ARGUMENTS #
	
	data : np.ndarray
		Data to get outlier mask of.
	axes : tuple[int,...]
		Axes over which to get outlier mask
	n_sigma : float =  5
		Number of standard deviations away from the mean a member of `data` must be to be considered an outlier
	"""
	
	outlier_mask = np.zeros_like(data, dtype=bool)
	
	for _i, (idx, gdata) in enumerate(nph.axes.iter_axes_group(data, axes)):
		mean = np.nanmean(gdata[idx])
		std = np.nanstd(gdata[idx])
		
		outlier_mask[idx] = np.fabs(gdata[idx] - mean) > n_sigma*std
		
		label_map, n_labels = sp.ndimage.label(outlier_mask[idx])
		# order labels by number of pixels they contain
						
		ordered_labels = list(range(1,n_labels+1)) # label 0 is background
		ordered_labels.sort(key = lambda x: np.count_nonzero(label_map == x), reverse=True)
		
		# Only keep the largest region as we expect everything to have one (and only one) calibration source.
		outlier_mask[idx] *= False
		for i in range(1,n_labels):
			outlier_mask[idx] |= (label_map == ordered_labels[i])
		
		
	return outlier_mask


def get_roi_mask(
		data : np.ndarray, 
		axes : tuple[int,...], 
		threshold : float = 1E-2,
		n_largest_regions : None | int = 1,
	) -> np.ndarray:
	"""
	data : np.ndarray
		Array to get region of interest of
	axes : tuple[int,...]
		Axes to get region of interest along.
	threshold : float = 1E-2
		When finding region of interest, only values larger than this fraction of the maximum value are included.
	n_largest_regions : None | int = 1
		When finding region of interest, if using a threshold will only the n_largest_regions in the calculation.
		A region is defined as a contiguous area where value >= `threshold` along `axes`. I.e. in a 3D cube, if
		we recenter about the COM on the sky (CELESTIAL) axes the regions will be calculated on the sky, not in
		the spectral axis (for example).
	"""
	mask = np.zeros_like(data, dtype=bool)
	with nph.axes.to_start(data, axes) as (gdata, gaxes), nph.axes.to_start(mask, axes) as (t_mask, t_axes):
	#with nph.axes.to_end(data, axes) as (gdata, gaxes):
		t_mask[...] = (gdata > threshold*np.nanmax(gdata, axis=gaxes))
		
		if n_largest_regions is not None:
			for j, (idx, g_mask) in enumerate(nph.axes.iter_axes_group(t_mask, gaxes)):
				label_map, n_labels = sp.ndimage.label(g_mask[idx])
				# order labels by number of pixels they contain
								
				ordered_labels = list(range(1,n_labels+1)) # label 0 is background
				ordered_labels.sort(key = lambda x: np.count_nonzero(label_map == x), reverse=True)
				
				if n_labels > n_largest_regions:
					g_mask[idx] *= False
					for i in range(n_largest_regions):
						g_mask[idx] |= (label_map == ordered_labels[i])
				
	return mask

def get_center_of_mass_offsets(
	data : np.ndarray, 
		axes : tuple[int,...], 
		roi_mask : np.ndarray | None = None,
	) -> np.ndarray:
	"""
	data : np.ndarray
		Array to recenter
	axes : tuple[int,...]
		Axes to get center of mass and recenter along.
	roi_mask : np.ndarray | None = None
		Mask for the region of interest. If present will restrict calculations to this region.
	"""
	not_axes = tuple(i for i in range(data.ndim) if i not in axes)
	indices = np.indices(data.shape)
	com_offsets = np.moveaxis((np.nansum(indices*data[None,...]*roi_mask[None,...], axis=tuple(a+1 for a in axes))/np.nansum(data*roi_mask, axis=axes))[axes,...], 0, -1)
	
	return com_offsets

def apply_offsets(
		data : np.ndarray, 
		axes : tuple[int,...], 
		offsets : np.ndarray
	) -> np.ndarray:
	"""
	data : np.ndarray
		Array to recenter
	axes : tuple[int,...]
		Axes to get center of mass and recenter along.
	offsets : np.ndarray
		Offsets to apply to data, will shift data's grid by this amount.
	"""
	_lgr.debug(f'{data.shape=}')
	_lgr.debug(f'{axes=}')
	_lgr.debug(f'{offsets.shape=}')
	
	for _i, (idx, gdata) in enumerate(nph.axes.iter_axes_group(data, axes)):
		_lgr.debug(f'{_i=}')
		_lgr.debug(f'{idx=}')
		_lgr.debug(f'{gdata[idx].shape=}')
		
		
		# calculate center of mass
		#com_idxs = tuple(np.nansum(data[idx]*indices)/np.nansum(data[idx]) for indices in np.indices(data[idx].shape))
		#center_to_com_offset = np.array([com_i - s/2 for s, com_i in zip(gdata[idx].shape, com_idxs[idx][::-1])])
		center_to_com_offset = np.array([s/2 - com_i for s, com_i in zip(gdata[idx].shape, offsets[idx])])
		_lgr.debug(f'{idx=} {offsets[idx]=} {center_to_com_offset=}')
		
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



def objective_function_factory(model_flattened_callable, data, err, mode='minimise'):
	"""
	Given a model function, some data, and the error on that data; returns an objective function that
	for either 'minimise'-ing or 'maximise'-ing the difference/similarity of the model and data.
	"""
	match mode:
		case 'minimise':
			def model_badness_of_fit_callable(*args, **kwargs):
				residual = model_flattened_callable(*args, **kwargs) - data
				result = np.nansum((residual/err)**2)
				return np.log(result)
			return model_badness_of_fit_callable
		
		case 'maximise':
			def model_likelihood_callable(*args, **kwargs):
				residual = model_flattened_callable(*args, **kwargs) - data
				result = np.nansum((residual/err)**2)
				#_lgr.debug(f'{-np.log(result)=}')
				return -np.log(result)
			
			return model_likelihood_callable
		
		case _:
			raise NotImplementedError
	return

def scipy_fitting_function_factory(scipy_func):
	"""
	Given some function that implements the same protocol as [scipy minimise](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize)
	returns a callable that accepts a PriorParamSet, a scipy-compatible objective function, a list of variable parameters, a list of constant parameters, and
	returns the fitted variable parameters.
	"""
	def scipy_fitting_function(params, objective_function, var_param_name_order, const_param_name_order):
		result = scipy_func(
			objective_function,
			tuple(params[p_name].const_value for p_name in var_param_name_order),
			bounds = tuple(params[p_name].domain for p_name in var_param_name_order)
		)
		return result.x

	return scipy_fitting_function

def fit_to_data(
		params : PriorParamSet,
		flattened_psf_model_callable : Callable[P, np.ndarray[S[N],T]],
		psf_data : np.ndarray[S[N],T],
		psf_err : np.ndarray[S[N],T],
		fitting_function : Callable[[PriorParamSet, Callable[Q,float]], Q],
		objective_function_factory : Callable[[Callable[P,np.ndarray[S[N],T]], np.ndarray[S[N],T], np.ndarray[S[N],T]], Callable[Q,float]] = functools.partial(objective_function_factory, mode='minimise'),
		plot_mode : str | bool | None = None
	) -> tuple[np.ndarray[S[N],T], dict[str,Any], dict[str,Any]]:
	"""
	Fits a model to some data with some error on that data.
	"""
	
	model_scipyCompat_callable, var_param_name_order, const_param_name_order = params.wrap_callable_for_scipy_parameter_order(
		flattened_psf_model_callable, 
		arg_names=flattened_psf_model_callable.arg_names if hasattr(flattened_psf_model_callable,'arg_names') else None
	)
	
	# Possibly take this plotting code out
	if plot_mode is not None:
		import matplotlib.pyplot as plt
		plt.close()
		test_result = model_scipyCompat_callable(tuple(params[p_name].const_value for p_name in var_param_name_order))
		
		f, ax = plt.subplots(2,2,squeeze=False,figsize=(12,8))
		ax = ax.flatten()
		
		f.suptitle(f'Example Plot\nvariables {dict((p.name,p.const_value) for p in params.variable_params)}\n constants {dict((p.name,p.const_value) for p in params.constant_params)}')
		
		ax[0].imshow(test_result)
		ax[0].set_title('example psf')
		
		ax[1].plot( test_result[:,test_result.shape[1]//2], np.arange(test_result.shape[0]),)
		ax[1].set_xscale('log')
		ax[1].set_title('y marginalisation')
		
		ax[2].plot(np.arange(test_result.shape[1]), test_result[test_result.shape[0]//2,:])
		ax[2].set_yscale('log')
		ax[2].set_title('x marginalisation')
		
		ax[3].remove()
		
		plt.show()
	
	
	objective_function = objective_function_factory(model_scipyCompat_callable, psf_data, psf_err)
	
	
	fitted_params = fitting_function(params, objective_function, var_param_name_order, const_param_name_order)
	
	return (
		model_scipyCompat_callable(fitted_params), 
		dict((k,v) for k,v in zip(var_param_name_order, fitted_params)), 
		dict((p.name,p.const_value) for p in params.constant_params)
	)