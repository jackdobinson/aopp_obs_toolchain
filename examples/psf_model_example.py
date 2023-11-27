

import math

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import ultranest


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
	params[0] = from_unit_range_to(cube[0], 0.1, 1)
	
	# turb_ndim
	params[1] = from_unit_range_to(cube[1], 1,2)
	
	# L0
	params[2] = from_unit_range_to(cube[2], 8, 8)
	
	# sigma
	params[3] = from_unit_range_to(cube[3], 1E-2, 1E-0)
	
	# beta
	params[4] = from_unit_range_to(cube[4], 0.1, 3) # remember to remove values of 1
	if params[4]==1:
		params[4] == math.nextafter(params[4], math.inf)
	
	# C
	params[5] = from_unit_range_to(cube[5], 5E-3, 5E-2)
	
	# A
	params[6] = from_unit_range_to(cube[6], 5E-3, 5E-2)

	return params


def create_psf_model_callable(psf_model_obj, instrument):
	
	return lambda *args: psf_model_obj(
		instrument.obs_shape, 
		instrument.expansion_factor, 
		instrument.supersample_factor, 
		instrument.f_ao,
		None,
		args[:3],
		(instrument.f_ao, *args[3:]),
		plots=False
	)


def create_psf_model_likelihood_callable(psf_model_callable, data, wavelength_idxs):
	
	def likelihood_callable(params, give_result=False):
		r0, turb_ndim, L0, sigma, beta, C, A = params
		specific_model = psf_model_callable(r0, turb_ndim, L0, np.array([sigma, sigma]), beta, C, A)
		
		data_scale = tuple(s*x for s,x in zip(data.shape[1:], (0.025*(1/3600)*(3.142/180), 0.025*(1/3600)*(3.142/180))))#(0.5*1.212E-7, 0.5*1.212E-7)))
		#_lgr.debug(f'{data.shape=} {data_scale=}')
		likelihood_accumulator = 0
		for wavelength, idx in wavelength_idxs:
			result = specific_model.at(data_scale, wavelength, plots=False)
			if give_result: return result
			residual = data[idx] - (result/np.nansum(result))
			likelihood_accumulator -= np.nansum(residual*residual/data)
		
		#_lgr.debug(f'{likelihood_accumulator=}')
		return likelihood_accumulator
	
	return likelihood_callable
	

if __name__=='__main__':
	
	
	psf = FitsSpecifier(example_data_loader.example_standard_star_file, 'DATA', (slice(None),slice(None),slice(None)), {'CELESTIAL':(1,2)}) 
	
	
	param_names = ['r0','turb_ndim', 'L0', 'sigma', 'beta', 'C', 'A']
	
		
	
	
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
				instrument.optical_transfer_function(1,1),
				phase_psd_von_karman_turbulence,
				phase_psd_fetick_2019_moffat_function,
			)
			
			psf_model_callable = create_psf_model_callable(test_psf_model, instrument)
			
			psf_model_likelihood_callable = create_psf_model_likelihood_callable(
				psf_model_callable,
				psf_data,
				((5E-7, 26), (6E-7, 126), (7E-7,226), (8E-7, 326), (9E-7,426))
			)
			
			result = psf_model_likelihood_callable(
				(0.17,2, 8, 5E-2,1.6,2E-2,0.05),
				True
			)
			
			# Debugging
			f, a = plot_helper.figure_n_subplots(4)
			a[0].set_title('data')
			a[0].imshow(np.log(psf_data[26]))
			
			a[1].set_title('result')
			a[1].imshow(np.log(result))
			
			a[2].set_title('residual')
			a[2].imshow(np.log(psf_data[26]-result))
			
			a[3].set_title('residual_squared')
			a[3].imshow(np.log((psf_data[26]-result)**2))
			plt.show()
			
			
			
			sampler = ultranest.ReactiveNestedSampler(
				param_names, 
				psf_model_likelihood_callable,
				psf_model_prior_transform,
				log_dir='ultranest_logs'
			)
			
			result = sampler.run(
				max_iters=2000, # maximum number of integration interations
				min_num_live_points=400,
				dlogz=100, # desired accuracy on logz
				min_ess=400, # number of effective samples
				update_interval_volume_fraction=0.4, # how often to update region
				max_num_improvement_loops=3, # how many times to go back and improve
			)
			result.print_results()
			ultranest.cornerplot(result)

