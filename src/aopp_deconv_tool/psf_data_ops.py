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

import aopp_deconv_tool.cfg.logs
_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'DEBUG')


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