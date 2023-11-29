
import os
import math

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import ultranest
import ultranest.plot
import scipy as sp
import scipy.stats

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


import cfg.logs
_lgr = cfg.logs.get_logger_at_level(__name__, 'DEBUG')

def normalise_psf(
		data : np.ndarray, 
		axes : tuple[int,...] | None=None, 
		cutout_shape : tuple[int,...] | None = None
	) -> np.ndarray:
	"""
	Ensure an array of data fufils the following conditions:
	
	* odd shape, to ensure a center pixel exists
	* center array on brightest pixel
	* ensure array sums to 1
	* optionally, cut out a region around the center to remove unneeded data.
	"""
	if axes is None:
		axes = tuple(range(data.ndim))
		
	data = nph.array.ensure_odd_shape(data, axes)
	
	
	for idx in nph.slice.iter_indices(data, group=axes):
		bp_offset = nph.array.get_center_offset_brightest_pixel(data[idx])
		data[idx] = nph.array.apply_offset(data[idx], bp_offset)
		data[idx] /= np.nansum(data[idx])
			
	if cutout_shape is not None:
		_lgr.debug(f'{tuple(data.shape[x] for x in axes)=} {cutout_shape=}')
		center_slices = nph.slice.around_center(tuple(data.shape[x] for x in axes), cutout_shape)
		_lgr.debug(f'{center_slices=}')
		slices = [slice(None) for s in data.shape]
		for i, center_slice in zip(axes, center_slices):
			slices[i] = center_slice
		_lgr.debug(f'{slices=}')
		data = data[tuple(slices)]
		
	
	return data


def from_unit_range_to(unit_range : float, vmin : float, vmax : float):
	"""
	Transform a value between (0,1) to between (vmin, vmax)
	"""
	return unit_range*(vmax-vmin)+vmin

def psf_model_prior_transform(cube):
	"""
	Ultranest parameter space is defined by a transfrom from (0,1) to the physical range
	"""
	params = cube.copy()
	
	# r0
	params[0] = from_unit_range_to(cube[0], 0.1,0.2)#0.05, 0.2)
	
	# turb_ndim
	params[1] = from_unit_range_to(cube[1], 1.0,2.0)
	
	# L0
	params[2] = from_unit_range_to(cube[2],7.5, 8.5)
	
	# sigma
	params[3] = from_unit_range_to(cube[3], 0.1, 3)
	
	# beta
	params[4] = from_unit_range_to(cube[4], 1.01,10)#0.1, 10) # remember to remove values of 1
	if params[4]==1:
		params[4] == math.nextafter(params[4], math.inf)
	
	# ao_correction_frac_offset
	params[5] = from_unit_range_to(cube[5], -1, 1)
	
	# ao_correction_amplitude
	params[6] = from_unit_range_to(cube[6], 0, 25)
	
	# factor
	params[7] = from_unit_range_to(cube[7], 0.7, 1.3)

	#s_factor
	#params[8] = from_unit_range_to(cube[8], 0, 1)

	return params


def create_psf_model_callable(psf_model_obj, instrument):
	
	return lambda *args: psf_model_obj(
		instrument.obs_shape, 
		instrument.expansion_factor, 
		instrument.supersample_factor, 
		None,
		args[:3],
		args[3:5],
		instrument.f_ao,
		args[6],
		args[5],
		#s_factor=args[7],
		plots=False
	)


def logistic_function(x, left_limit=0, right_limit=1, transition_scale=1, center=0):
	return (right_limit-left_limit)/(1+np.exp(-(np.e/transition_scale)*(x-center))) + left_limit

def create_psf_model_likelihood_callable(psf_model_callable, data, wavelength_idxs):
	
	def likelihood_callable(params, give_result=False):
		(	r0, 
   			turb_ndim, 
   			L0, 
   			sigma, 
   			beta, 
   			ao_correction_frac_offset, 
   			ao_correction_amplitude, 
   			factor
		) = params
		
		specific_model = psf_model_callable(
			r0, 
			turb_ndim, 
			L0, 
			np.array([sigma, sigma]), 
			beta, 
			ao_correction_frac_offset, 
			ao_correction_amplitude
		)
		
		data_scale = tuple(s*x for s,x in zip(data.shape[1:], (0.025*(1/3600)*(3.142/180), 0.025*(1/3600)*(3.142/180))))#(0.5*1.212E-7, 0.5*1.212E-7)))
		#_lgr.debug(f'{data.shape=} {data_scale=}')
		likelihood_accumulator = 0
		for wavelength, idx in wavelength_idxs:
			nan_mask = np.isnan(data[idx])
			result = specific_model.at(data_scale, wavelength, plots=False).data
			result /= np.nansum(result) # normalise model result
			
			# multiply by factor
			result *= factor
			
			if give_result: return result
			residual = data[idx] - result
			
			# err can be pre-computed
			# assume residual is gaussian distributed, with a sigma on each pixel and a flat value
			vals = np.abs(data[idx][~nan_mask])*0.1
			v_min = np.min(vals)
			v_max = np.max(vals)
			v_range = v_max - v_min
			v_range_center = 0.5*(v_max + v_min)
			err = logistic_function(
				vals, 
				10*v_min, 
				0.01*v_max, 
				v_range/2, 
				v_range_center
			)  # want a floor and a ceiling to the error on a pixel
			z = residual[~nan_mask]/err
			#likelihood = np.exp(-(z*z)/2)/np.sqrt(2*np.pi)
			likelihood = -(z*z)/2 # want the log of the pdf
			#_lgr.debug(f'{np.isnan(result[~nan_mask]).sum()=} {np.isnan(residual[~nan_mask]).sum()=} {np.isnan(likelihood).sum()=}')
			#_lgr.debug(f'{wavelength=} {idx=} {likelihood=}')
			
			#plt.title(f'err frac')
			#err_map = np.zeros_like(data[idx], dtype=float)
			#err_map[~nan_mask] = err
			#plt.imshow(np.log(err_map/np.abs(data[idx])))
			#plt.show()
			
			#plt.title(f'err value')
			#err_map = np.zeros_like(data[idx], dtype=float)
			#err_map[~nan_mask] = err
			#plt.imshow(np.log(err_map))
			#plt.show()
			
			
			
			#plt.title(f'data[{idx}]')
			#plt.hist(data[idx][~nan_mask].flatten(), bins=100)
			#plt.gca().set_yscale('log')
			#plt.show()
			#
			#plt.title('residual')
			#plt.hist(residual[~nan_mask].flatten(), bins=100)
			#plt.gca().set_yscale('log')
			#plt.show()
			#
			#plt.title('likelihood')
			#plt.hist(likelihood, bins=100)
			#plt.gca().set_yscale('log')
			#plt.show()
			
			# Hopefully mean is a more stable measure when wavelengths have different number of valid pixels
			likelihood_accumulator += likelihood.mean() # -= np.nansum(residual*residual/data)
		
		#_lgr.debug(f'{likelihood_accumulator=}')
		#return -np.log(np.abs(likelihood_accumulator))
		return likelihood_accumulator
	
	return likelihood_callable
	

if __name__=='__main__':
	
	
	psf = FitsSpecifier(example_data_loader.example_standard_star_file, 'DATA', (slice(None),slice(None),slice(None)), {'CELESTIAL':(1,2)}) 
	
	
	param_names = ['r0','turb_ndim', 'L0', 'sigma', 'beta', 'ao_correction_frac_offset', 'ao_correction_amplitude', 'factor']
	
		
	
	
	with fits.open(psf.path) as psf_hdul:	
		with nph.axes.to_end(psf_hdul[psf.ext].data, psf.axes['CELESTIAL']) as psf_data:
			
			
			#plt.title(psf.path)
			#plt.imshow(np.log(psf_data[psf_data.shape[0]//2]))
			#plt.show()
			
			psf_data = normalise_psf(psf_data, axes=psf.axes['CELESTIAL'], cutout_shape=(101,101))
			
			#plt.title(str(psf.path) + ' normalised')
			#plt.imshow(np.log(psf_data[psf_data.shape[0]//2]))
			#plt.show()

	
			_lgr.debug(f'{psf_data.shape[-2:]=}')
			instrument = VLT.muse(obs_shape=psf_data.shape[-2:])
	
	
			test_psf_model = psf_model.PSFModel(
				instrument.optical_transfer_function(3,3),
				phase_psd_von_karman_turbulence,
				phase_psd_fetick_2019_moffat_function,
			)
			
			psf_model_callable = create_psf_model_callable(test_psf_model, instrument)
			
			psf_model_likelihood_callable = create_psf_model_likelihood_callable(
				psf_model_callable,
				psf_data,
				((5E-7, 26), (6.06E-7, 134), (7E-7,226), (8E-7, 326), (9E-7,426))
			)
			
			result = psf_model_likelihood_callable(
				(	0.15, # r0
					1.3, #turb_ndim
					8, #l0
					1.5, #sigma
					6, # beta
					-0.1, # ao_correction_frac_offset
					12, # ao_correction_amplitude
					1.0,#np.exp(25.85), # factor
					#0.0, #s_factor
				),
				True
			)
			
			# Debugging
			f, a = plot_helper.figure_n_subplots(6)
			a[0].set_title('data')
			a[0].imshow(np.log(psf_data[26]))
			a[0].plot([psf_data[26].shape[0]//2 + 0.5],[psf_data[26].shape[1]//2 + 0.0], 'r.')
			
			a[1].set_title('result')
			a[1].imshow(np.log(result))
			
			a[2].set_title('residual')
			a[2].imshow(np.log(psf_data[26]-result))
			
			
			a[3].set_title('residual_squared')
			a[3].imshow(np.log((psf_data[26]-result)**2))
			
			a[4].set_title('data and result slice')
			a[4].plot(np.log(psf_data[26][psf_data.shape[1]//2,:]).flatten())
			a[4].plot(np.log(result[result.shape[0]//2,:]).flatten())
			
			offsets_from_center = nph.array.offsets_from_point(psf_data[26].shape)
			offsets_from_center = (offsets_from_center.T - np.array([0.0,0.5])).T
			r_idx1 = np.sqrt(np.sum(offsets_from_center**2, axis=0))
			r = np.linspace(0,np.max(r_idx1),30)
			psf_radial_data = np.array([np.nansum(psf_data[26][(r_min <= r_idx1) & (r_idx1 < r_max) ]) for r_min, r_max in zip(r[:-1], r[1:])])
			
			offsets_from_center = nph.array.offsets_from_point(psf_data[26].shape)
			offsets_from_center = (offsets_from_center.T - np.array([0.5,0.5])).T
			r_idx2 = np.sqrt(np.sum(offsets_from_center**2, axis=0))
			r = np.linspace(0,np.max(r_idx2),30)
			result_radial_data = np.array([np.nansum(result[(r_min <= r_idx2) & (r_idx2 < r_max) ]) for r_min, r_max in zip(r[:-1], r[1:])])
			
			a[5].set_title('radial data and result')
			a[5].plot(r[:-1], psf_radial_data)
			a[5].plot(r[:-1], result_radial_data)
			
			plt.show()
			
			
			
			sampler = ultranest.ReactiveNestedSampler(
				param_names, 
				psf_model_likelihood_callable,
				psf_model_prior_transform,
				log_dir='ultranest_logs',
				resume='subfolder',
				run_num=1,
				#warmstart_max_tau=0.5,
			)
			
			final_result = None
			for result in sampler.run_iter(
					#max_iters=1,#2000,
					min_num_live_points=80,
					dlogz=100,
					min_ess=40,
					update_interval_volume_fraction=0.8,
					max_num_improvement_loops=3,
					frac_remain=0.01
				):
				sampler.print_results()
				sampler.plot()
				final_result=result

			for k, v in final_result.items():
				_lgr.debug(f'{k} = {v}')

			psf_model_likelihood_callable_final = create_psf_model_likelihood_callable(
				psf_model_callable,
				psf_data,
				((5E-7, 26),)
			)
			
			result = psf_model_likelihood_callable_final(
				tuple(final_result['posterior']['mean']),
				True
			)
			
			# Debugging
			os.makedirs('./plots', exist_ok=True)
			vmin, vmax = np.nanmin(np.log(psf_data[26])), np.nanmax(np.log(psf_data[26]))
			_lgr.debug(f'{vmin=} {vmax=}')
			f, a = plot_helper.figure_n_subplots(4)
			a[0].set_title(f'log data [{vmin}, {vmax}]')
			a[0].imshow(np.log(psf_data[26]), vmin=vmin, vmax=vmax)
			
			a[1].set_title(f'log result [{vmin}, {vmax}]')
			a[1].imshow(np.log(result), vmin=vmin, vmax=vmax)
			
			a[2].set_title(f'log residual [{vmin}, {vmax}]')
			a[2].imshow(np.log(psf_data[26]-result), vmin=vmin, vmax=vmax)
			
			log_abs_residual = np.log(np.abs(psf_data[26]-result))
			a[3].set_title(f'log abs residual [{np.min(log_abs_residual)}, {np.max(log_abs_residual)}]')
			a[3].imshow(log_abs_residual)
			plt.savefig('./plots/log_ultranest_result.png')
			plt.show()
			
			
			vmin, vmax = np.nanmin(psf_data[26]), np.nanmax(psf_data[26])
			f, a = plot_helper.figure_n_subplots(4)
			a[0].set_title(f'data [{vmin}, {vmax}]')
			a[0].imshow(psf_data[26], vmin=vmin, vmax=vmax)
			
			a[1].set_title(f'result [{vmin}, {vmax}]')
			a[1].imshow(result, vmin=vmin, vmax=vmax)
			
			a[2].set_title(f'residual [{vmin}, {vmax}]')
			a[2].imshow(psf_data[26]-result, vmin=vmin, vmax=vmax)
			
			frac_residual = np.abs(psf_data[26]-result)/psf_data[26]
			fr_sorted = np.sort(frac_residual.flatten())
			vmin=fr_sorted[fr_sorted.size//4]
			vmax = fr_sorted[3*fr_sorted.size//4]
			a[3].set_title(f'frac residual [{vmin}, {vmax}]')
			a[3].imshow(frac_residual, vmin=vmin, vmax=vmax)
			plt.savefig('./plots/ultranest_result.png')
			plt.show()
			
