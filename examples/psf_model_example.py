
import os
from pathlib import Path
import math
from collections import namedtuple
import inspect
import dataclasses as dc
from typing import ParamSpec, TypeVar, TypeVarTuple, Concatenate, Callable, Any
import json

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import ultranest
import ultranest.plot


import scipy as sp
import scipy.stats
import scipy.interpolate
import scipy.ndimage

import numpy_helper as nph
import astropy_helper as aph
from astropy_helper.fits.specifier import FitsSpecifier
import numpy_helper.array
import numpy_helper.axes
import numpy_helper.slice
import example_data_loader
import psf_model
from optics.turbulence_model import phase_psd_von_karman_turbulence
from optics.adaptive_optics_model import phase_psd_fetick_2019_moffat_function
from instrument_model.vlt import VLT
import plot_helper
import mfunc
import psf_data_ops

from optimise_compat import PriorParam, PriorParamSet
from optimise_compat.ultranest import UltranestResultSet, model_likelihood_callable_factory, model_fractional_likelihood_callable_factory

import cfg.logs
_lgr = cfg.logs.get_logger_at_level(__name__, 'DEBUG')

T = TypeVar('T')
SpecificPSFModel = TypeVar('SpecificPSFModel', bound=psf_model.PSFModel) # An instance of PSFModel that has had it's parameters set by invoking it's PSFModel.__call__(...) method
JumbledParameters = ParamSpec('JumbledParameters') # Parameters that can be in nested lists/tuples etc. i.e. someFunc(p1, [p2,p3,[p4]], p5, p6)
FlattenedParameters = ParamSpec('FlattenedParameters') # Parameters that are not in nested lists/tuples. i.e. someFunc(p1, p2, p3, p4, p5, p6)
TupleParameters = Concatenate[TypeVarTuple('TupleParameters'),...] # Parameters that are all in one tuple as the first argument. i.e. someFunc( (p1,p2,p3,p4,p5,p6) ), this is how scipy likes it's objective functions to be formatted.
P1 = ParamSpec('P1')
P2 = ParamSpec('P2')


def psf_model_flattened_callable_factory(
		psf_model_obj : Callable[JumbledParameters,SpecificPSFModel], 
	) -> Callable[FlattenedParameters,SpecificPSFModel]:
	"""
	Take in an instance of PSFModel, return a callable where all the parameters we want to vary
	are passed directly.
	
	If we change how the PSFModel instance `psf_model_obj` is called, we will need to change this function.
	
	I.e. if the arguments of `psf_model_obj.__call__` are not in a form that an optimisation package likes,
	this function should return a wrapper that does take arguments nicely.
	
	Note:
		PSFModel has two calls that need to be made get a result from it.
		1) PSFModel.__call__(...) that sets the model parameters, no computation happens yet.
		2) PSFModel.at(wavelength) that computes the PSF at a specific wavelength for the model parameters, all the computation happens here.
		
		The reason for the split is that conceptually, much of the computation could be moved to (1), however that has not happened yet.
	
	# ARGUMENTS #
	
		psf_model_obj
			An instance of PSFModel that will have it's arguments to __call__(...) flattened.
	
	# RETURNS #
	
		psf_model_callable
			A function which, when called, sets the model parameters of `psf_model_obj`
			(i.e. does step (1) in Note), and returns`psf_model_obj` ready to have it's `.at(...)`
			method called (i.e. ready to have step (2) in Note done to it).
		
			Note: 
				see `optimise_compat.PriorParamSet.wrap_callable_for_scipy_parameter_order(...)` for
				how to change this (a flattened callable) into a callable that accepts a tuple of parameters 
				(i.e. one that scipy likes to use as objective functions).
	
	"""
	def psf_model_callable(
			r0, 
			turb_ndim, 
			L0, 
			alpha, 
			beta,
			f_ao,
			ao_correction_amplitude, 
			ao_correction_frac_offset, 
			s_factor,
			factor
		):
		#_lgr.debug(f'{r0=} {turb_ndim=} {L0=} {alpha=} {beta=} {f_ao=} {ao_correction_amplitude=} {ao_correction_frac_offset=} {s_factor=} {factor=}')
		specific_model = psf_model_obj(
			None,
			(r0, turb_ndim, L0),
			(alpha, beta),
			f_ao,
			ao_correction_amplitude,
			ao_correction_frac_offset,
			s_factor
		)
		specific_model.factor = factor
		
		return specific_model
	
	return psf_model_callable


def psf_model_result_callable_factory(
		psf_model_scipyCompat_callable : Callable[TupleParameters,SpecificPSFModel], 
		wavelength : float, 
		show_plots : bool = False
	):
	"""
	Take in a "callable that accepts a single tuple of parameters and returns an instance of PSFModel that has had it's model parameters set" (i.e. a specific_model),
	and return a "callable that returns the result of the specific model" (i.e. a result_callable).
	
	If we change how PSFModel.at(...) works, we need to change this function.
	
	Note:
		PSFModel has two calls that need to be made get a result from it.
		1) PSFModel.__call__(...) that sets the model parameters, no computation happens yet.
		2) PSFModel.at(wavelength) that computes the PSF at a specific wavelength for the model parameters, all the computation happens here.
		
		The reason for the split is that conceptually, much of the computation could be moved to (1), however that has not happened yet.
	
	# ARGUMENTS #
	
		psf_model_scipyCompat_callable
			A callable that accepts a single tuple of parameters and returns a specific model, 
			see `optimise_compat.PriorParamSet.wrap_callable_for_scipy_parameter_order(...)` for
			how to get a callable that accepts a tuple of parameters from a flattened callable.
		
		wavelength
			The wavelength to compute the result of the specific model at
		
		show_plots
			A flag to pass through to PSFModel.at(...), if True will show plots for the model.
	
	# RETURNS #
	
		model_result_callable
			A callable that accepts a single tuple of parameters and returns the result of the model with those parameters at `wavelength`.
	
	"""
	def model_result_scipyCompat_callable(params):
		specific_model = psf_model_scipyCompat_callable(params)
		result = specific_model.at(wavelength, plots=show_plots).data
		result /= np.nansum(result)
		result *= specific_model.factor
		
		return result

	return model_result_scipyCompat_callable



def get_error(data, frac, low_limit, hi_limit, sigma):
	"""
	Try to strike a balance between fractional error and absolute error
	"""
	v = np.abs(data)*frac
	vmin, vmax = np.nanmin(v), np.nanmax(v)
	vrange = vmax - vmin
	vrange_center = 0.5*(vmax+vmin)
	err = mfunc.logistic_function(
		v,
		low_limit if low_limit > 0 else (-low_limit)*vmin,
		hi_limit if hi_limit > 0 else (-hi_limit)*vmax,
		sigma if sigma > 0 else vrange/(-sigma),
		vrange_center
	)
	return err



if __name__=='__main__':
	
	
	psf = FitsSpecifier(example_data_loader.example_standard_star_file, 'DATA', (slice(None),slice(None),slice(None)), {'CELESTIAL':(1,2)}) 
	
	
	
	def plot_example_result(result, psf_data, suptitle=None, show=True):
		if not show: return
	
		residual = psf_data-result
		
		f, a = plot_helper.figure_n_subplots(7)
		
		a[0].set_title(f'data sum={np.nansum(psf_data)}')
		a[0].imshow(np.log(psf_data))
		a[0].plot([psf_data.shape[0]/2],[psf_data.shape[1]/2], 'r.')
		
		a[1].set_title(f'result sum={np.nansum(result)}')
		a[1].imshow(np.log(result))
		a[1].plot([result.shape[0]/2],[result.shape[1]/2], 'r.')
		
		a[2].set_title(f'residual sum={np.nansum(residual)}')
		a[2].imshow(np.log(residual))
		
		a[3].set_title(f'residual_squared sqrt(sum)={np.sqrt(np.nansum(residual**2))}')
		a[3].imshow(np.log((residual)**2))
		
		a[4].set_title('data and result slice (horizontal)')
		a[4].plot(np.log(psf_data[psf_data.shape[0]//2,:]).flatten())
		a[4].plot(np.log(result[result.shape[0]//2,:]).flatten())
		a[4].axvline(result.shape[0]//2, color='red', ls='--')

		
		a[5].set_title('data and result slice (vertical)')
		a[5].plot(np.log(psf_data[:,psf_data.shape[1]//2]).flatten())
		a[5].plot(np.log(result[:,result.shape[1]//2]).flatten())
		a[5].axvline(result.shape[1]//2, color='red', ls='--')
		
		offsets_from_center = nph.array.offsets_from_point(psf_data.shape)
		offsets_from_center = (offsets_from_center.T - np.array([0.0,0.5])).T
		r_idx1 = np.sqrt(np.sum(offsets_from_center**2, axis=0))
		r = np.linspace(0,np.max(r_idx1),30)
		psf_radial_data = np.array([np.nansum(psf_data[(r_min <= r_idx1) & (r_idx1 < r_max) ]) for r_min, r_max in zip(r[:-1], r[1:])])
		
		offsets_from_center = nph.array.offsets_from_point(psf_data.shape)
		offsets_from_center = (offsets_from_center.T - np.array([0.5,0.5])).T
		r_idx2 = np.sqrt(np.sum(offsets_from_center**2, axis=0))
		r = np.linspace(0,np.max(r_idx2),30)
		result_radial_data = np.array([np.nansum(result[(r_min <= r_idx2) & (r_idx2 < r_max) ]) for r_min, r_max in zip(r[:-1], r[1:])])
		
		a[6].set_title('radial data and result')
		a[6].plot(r[:-1], psf_radial_data)
		a[6].plot(r[:-1], result_radial_data)
		
		f.suptitle(suptitle)
		
		plot_helper.output(show)
	
	
	
	
	
	
	with fits.open(psf.path) as psf_hdul:	
		with nph.axes.to_end(psf_hdul[psf.ext].data, psf.axes['CELESTIAL']) as psf_data:
			
			
			#plt.title(psf.path)
			#plt.imshow(np.log(psf_data[psf_data.shape[0]//2]))
			#plt.show()
			
			psf_data = psf_data_ops.normalise(psf_data, axes=psf.axes['CELESTIAL'], cutout_shape=(101,101))
			psf_err = get_error(psf_data, 0.01, -1E1, -1E-1, -0.9)
			
			#plt.title(str(psf.path) + ' normalised')
			#plt.imshow(np.log(psf_data[psf_data.shape[0]//2]))
			#plt.show()

	
			_lgr.debug(f'{psf_data.shape[-2:]=}')
			instrument = VLT.muse(
				expansion_factor = 3,
				supersample_factor = 2,
				obs_shape=psf_data.shape[-2:]
			)
			
			# Define parameters that are used by the psf_model
			# ERROR: These are not getting sent in the correct order to the class!!
			params = PriorParamSet(
				PriorParam('r0', 
					(0,np.inf),
					True,
					0.15
				),
				PriorParam('turb_ndim', 
					(1,2),
					True,
					1.3
				),
				PriorParam('L0', 
					(0,10),
					False,
					1.5
				),
				PriorParam('alpha', 
					(0.1, 3),
					False,
					0.4#0.7
				),
				PriorParam('beta', 
					(1.01, 10),
					True,
					1.6
				),
				PriorParam('ao_correction_frac_offset', 
					(-1,1),
					False,
					0
				),
				PriorParam('ao_correction_amplitude', 
					(0,5),
					False,
					2.2
				),
				PriorParam('factor', 
					(0.7,1.3),
					False,
					1
				),
				PriorParam('s_factor', 
					(0,np.inf),
					True, 
					0
				),
				PriorParam('f_ao',
					(24.0/(2*instrument.obj_diameter),52.0/(2*instrument.obj_diameter)),
					True,#False,
					instrument.f_ao
				)
			)
	
	
	
			test_psf_model = psf_model.PSFModel(
				instrument.optical_transfer_function(),
				phase_psd_von_karman_turbulence,
				phase_psd_fetick_2019_moffat_function,
				instrument
			)
			
			# One wavelength takes about 1 hr to run.
			wavelength_idxs = (
				(4.903E-7, 16),
				(5E-7, 26),
				(5.24469E-7, 50),
				(5.644E-7, 90),
				(6.06E-7, 134),
				(6.244E-7, 150),
				(6.64E-7, 190),
				(7E-7,226),
				(7.244E-7, 250),
				(7.644E-7, 290),
				(8E-7, 326),
				(8.244E-7, 350),
				(8.64E-7, 390),
				(9E-7,426),
				(9.244E-7, 450),
			)
			nested_sampling_stop_fraction = 0.01
			nested_sampling_max_iterations = 500 #2000
			min_live_points = 20
			show_plots = False
			update_params_search_region = False
			result_set_directory = Path('ultranest_logs')
			
			
			psf_model_flattened_callable = psf_model_flattened_callable_factory(test_psf_model)
			
			
			model_scipyCompat_callable, var_param_name_order, const_var_param_name_order = params.wrap_callable_for_scipy_parameter_order(psf_model_flattened_callable)
			_lgr.debug(f'{model_scipyCompat_callable=}')
			_lgr.debug(f'{var_param_name_order=}')
			
			result_set = UltranestResultSet(Path(result_set_directory))
			result_set.metadata['wavelength_idxs'] = wavelength_idxs
			result_set.metadata['constant_parameters'] = [p.to_dict() for p in params.constant_params]
			result_set.save_metadata()
			
			
			final_result = None
			initial_median_noise_estimate = None
			initial_median_noise_estimate_idx = 0
			for wavelength, idx in wavelength_idxs:
				
				_lgr.info(f'{idx=} {wavelength=}')
				
				
				if update_params_search_region and (final_result is not None):
					raise NotImplementedError('Updating the parameter search region is not implemented yet')
				
				
				
				if initial_median_noise_estimate is None:
					initial_median_noise_estimate_idx = idx
					data_for_median_noise_estimate = psf_data[initial_median_noise_estimate_idx]
					if np.all(np.isnan(data_for_median_noise_estimate)):
						_lgr.error(f'When operating on file {psf.path} data_for_median_noise_estimate = psf_data[{initial_median_noise_estimate_idx}] is all NANs')
						continue
					data_for_median_noise_estimate_removed_nans = np.nan_to_num(data_for_median_noise_estimate)
					initial_median_noise_estimate = np.std(data_for_median_noise_estimate_removed_nans - sp.ndimage.median_filter(data_for_median_noise_estimate_removed_nans,size=5))
				
				_lgr.debug(f'{initial_median_noise_estimate=}')
				median_noise = np.std(np.nan_to_num(psf_data[idx]) - sp.ndimage.median_filter(np.nan_to_num(psf_data[idx]),size=5))
				
				# Want to adjust for the increased variance on long-wavelength results, otherwise a lot of effort is spent fitting long-wavelengths more exactly than required
				median_noise_correction_factor = np.sqrt(median_noise/ initial_median_noise_estimate)
				_lgr.debug(f'{median_noise=} {median_noise_correction_factor=}')
				
				model_result_scipyCompat_callable = psf_model_result_callable_factory(model_scipyCompat_callable, wavelength, show_plots=show_plots)
				
				psf_model_likelihood_scipyCompat_callable = model_likelihood_callable_factory(
				#psf_model_likelihood_scipyCompat_callable = model_fractional_likelihood_callable_factory(
					model_result_scipyCompat_callable,
					psf_data[idx],
					psf_err[idx]*median_noise_correction_factor
				)
				
				# Debugging
				plot_example_result(
					model_result_scipyCompat_callable(tuple(params[p_name].const_value for p_name in var_param_name_order)),
					psf_data[idx],
					suptitle=f'{wavelength=}',
					show=show_plots
				)
				#continue # DEBUGGING
				
				# If using dependency injection mechanism like "amateur_data_analysis.py"
				# Instead of "psf_data_ops.scipy_fitting_function_factory
				# would have optimise_compat.ultranest.fitting_function_factory
				sampler = ultranest.ReactiveNestedSampler(
					var_param_name_order, 
					psf_model_likelihood_scipyCompat_callable,
					params.get_linear_transform_to_domain(var_param_name_order, (0,1)),
					log_dir=result_set_directory,
					resume='subfolder',
					run_num=idx,
					#warmstart_max_tau=0.5,
				)
				
				
				
				
				
				final_result = None
				
				# Note: Reducing the number of points used dramatically speeds up the algorithm. However it will be less accurate.
				for result in sampler.run_iter(
						max_iters=nested_sampling_max_iterations,
						max_ncalls=5000,
						frac_remain=nested_sampling_stop_fraction,
						Lepsilon = 0.1,
						min_num_live_points=min_live_points, #80
						cluster_num_live_points=1, #40
						dlogz=100,
						min_ess=1, #40
						update_interval_volume_fraction=0.99, #0.8
						max_num_improvement_loops=1,
						widen_before_initial_plateau_num_warn = 1.5*min_live_points,
						widen_before_initial_plateau_num_max = 2*min_live_points
					):
					sampler.print_results()
					sampler.plot()
					final_result=result
	
				for k, v in final_result.items():
					_lgr.debug(f'{k} = {v}')
			
			
			result_set.plot_params_vs_wavelength(show=False, save=True)
			result_set.plot_results(
				lambda wav: psf_model_result_callable_factory(model_scipyCompat_callable, wav, show_plots=show_plots),
				psf_data,
				show=False,
				save=True
			)
	
			
