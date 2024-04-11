
import os, sys
from pathlib import Path
import math
from collections import namedtuple
import inspect
import dataclasses as dc
from typing import ParamSpec, TypeVar, TypeVarTuple, Concatenate, Callable, Any
import json
from functools import partial


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
import plot_helper
import mfunc
import psf_data_ops

from optimise_compat import PriorParam, PriorParamSet
from optimise_compat.ultranest_compat import UltranestResultSet, fitting_function_factory

import cfg.logs
_lgr = cfg.logs.get_logger_at_level(__name__, 'DEBUG')



from psf_model_dependency_injector import MUSEAdaptiveOpticsPSFModelDependencyInjector



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
	
	
	
	def plot_result(result, psf_data, suptitle=None, show=True):
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
			
			
			# One wavelength takes about 1 hr to run.
			wavelength_idxs = (
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
			
			# Set to `True` if we just want to re-plot an existing result set
			skip_calculation_flag = False 
			
			result_set_directory = Path('ultranest_logs')
			
			
			result_set = UltranestResultSet(Path(result_set_directory))
			result_set.metadata['wavelength_idxs'] = wavelength_idxs
			result_set.save_metadata()
			
			
			final_result = None
			initial_median_noise_estimate = None
			initial_median_noise_estimate_idx = 0
			
			
			result_callables = []
			
			for wavelength, idx in wavelength_idxs: 
				_lgr.info(f'{idx=} {wavelength=}')
				
				di = MUSEAdaptiveOpticsPSFModelDependencyInjector(
					psf_data,
					var_params=['alpha','factor', 'ao_correction_frac_offset', 'ao_correction_amplitude', ],
					const_params=['f_ao', 'r0'],
					initial_values={'wavelength':wavelength}
				)
				psf_model_name = di.get_psf_model_name()
				params = di.get_parameters()
				psf_model_callable = di.get_psf_model_flattened_callable()
				psf_result_postprocess = di.get_psf_result_postprocessor()
				result_callables.append(di.get_fitted_parameters_callable())
				
				result_set.metadata['constant_parameters'] = [p.to_dict() for p in params.constant_params]
				result_set.save_metadata()
				
				if skip_calculation_flag:
					continue
				
				
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
				
				
				# Paramters to pass to ultranest
				num_live_points = 200
				
				# fit PSF model
				fitted_psf, fitted_vars, consts = psf_data_ops.fit_to_data(
					params, 
					psf_model_callable, 
					psf_data[idx], 
					psf_err[idx]*median_noise_correction_factor, 
					fitting_function_factory(
						reactive_nested_sampler_kwargs = {
							'log_dir' : result_set.directory,
							'run_num' : idx
						},
						sampler_run_kwargs = {
							'max_iters' : 2000, #500, # 2000,
							'max_ncalls' : 10000, #5000
							'frac_remain' : 1E-2,
							'Lepsilon' : 1E-1,
							'min_num_live_points' : num_live_points, #20, #80
							'cluster_num_live_points' : num_live_points/5, #1, #40
							'min_ess' : num_live_points, #1, #40
							'widen_before_initial_plateau_num_warn' : 1.5*num_live_points, #*min_live_points,
							'widen_before_initial_plateau_num_max' : 2*num_live_points #*min_live_points
						}
					), # optimise_compat.ultranest.fitting_function_factory
					partial(psf_data_ops.objective_function_factory, mode='maximise'),
					plot_mode=None
				)
				_lgr.info(f'{fitted_vars=}')
				_lgr.info(f'{consts=}')
				
				# Do any postprocessing if we need to
				if psf_result_postprocess is not None:
					result = psf_result_postprocess(params, psf_model_callable, fitted_vars, consts)
					plot_result(result, psf_data[idx], suptitle=f'{idx=}\n{fitted_vars=}', show=False)
				
				
			
			
			result_set.plot_params_vs_run_index(show=False, save=True)
			result_set.plot_results(
				result_callables,
				psf_data,
				show=False,
				save=True
			)
	
			
