
import os
from pathlib import Path
import math
from collections import namedtuple
import inspect
import dataclasses as dc
from typing import Callable, Any
import json

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import ultranest
import ultranest.plot

# TESTING
import emcee

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


import cfg.logs
_lgr = cfg.logs.get_logger_at_level(__name__, 'DEBUG')


def logistic_function(x, left_limit=0, right_limit=1, transition_scale=1, center=0):
	return (right_limit-left_limit)/(1+np.exp(-(np.e/transition_scale)*(x-center))) + left_limit


def unit_range_to(vmin, vmax):
	return lambda unit: unit*(vmax-vmin)+vmin


def normalise_psf(
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


@dc.dataclass(slots=True)
class PriorParam:
	"""
	Class that holds parameter information.
	
	name : str
		String used to identify the parameter
	transform : callable[[float],float] | float
		Callable that transforms the unit range [0,1] to the actual range of the parameter, OR a constant value
	example_value : float
		A value to use for example plots
	"""
	name : str # string used to identify parameter
	transform : Callable[[float],float] | float # callable that transforms from unit range [0,1] to actual range, OR a constant
	example_value : float # value to use for example plots


class PriorParamSet:
	"""
	A collection of PriorParam instances
	"""
	def __init__(self, prior_params : tuple[PriorParam]):
		self.prior_params = prior_params
		self._pp_map = {}
		for i, p in enumerate(self.prior_params):
			self._pp_map[p.name] = i
		
		self.recalc()
		
		return
	
	def recalc(self):
		# Split into model params and const params, model params should
		# vary, const params always have the same value.
		self.model_param_idxs = []
		self.model_param_names = []
		self.model_param_transforms = []
		self.model_param_examples = []
	
		self.const_param_idxs = []
		self.const_param_names = []
		self.const_param_transforms = []
		self.const_param_examples = []
	
		i=0
		j=0
		for p in self.prior_params:
			if callable(p.transform):
				self.model_param_idxs.append(i)
				self.model_param_names.append(p.name)
				self.model_param_transforms.append(p.transform)
				self.model_param_examples.append(p.example_value)
				i+=1
			else:
				self.const_param_idxs.append(j)
				self.const_param_names.append(p.name)
				self.const_param_transforms.append(p.transform)
				self.const_param_examples.append(p.example_value)
				j+=1
		return
	
	def get_const_params(self):
		return dict(zip(self.const_param_names, self.const_param_transforms))
	
	def __getitem__(self, k):
		return self.prior_params[self._pp_map[k]]
	
	def append(self, p : PriorParam):
		self._pp_map[p.name] = len(self.prior_params)
		self.prior_params.append(p)
	
	def get_ultranest_prior_param_transform(self):
		"""
		Generate a function that takes in intervals on [0,1], and applies the
		transforms in self.model_param_transforms
		"""
		def prior_transform(cube):
			params = cube.copy()
			for i, transform in zip(self.model_param_idxs, self.model_param_transforms):
				params[i] = transform(cube[i])
			return params
		return prior_transform
		
	
	def get_ultranest_model_callable(self,
			model_callable, # callable that computes the model, e.g. psf_model(r0, turb_ndim, L0, sigma)
			arg_from_param_name_mapping : dict[str,str] | None= None, # maps the name of the name of the arguments in `model_callable` to the names in self.prior_params
		):
		"""
		From a callable that looks like `model_callable(arg1, arg2, arg3,...)`, create an equivalent
		callable that takes parameters defined in this object's self.model_param_names attribute as a
		single argument. E.g. maps params = (p1, p2,...) in `ultranest_model_callable(params)` to
		arg1, arg2, etc. by name.
		"""
		if arg_from_param_name_mapping is None:
			ap_name_map = lambda x: x
		else:
			ap_name_map = lambda x: arg_from_param_name_mapping.get(x,x)
		
		sig = inspect.signature(model_callable)
		
		arg_names = list(sig.parameters.keys())
		
		# [0,1,3,4,6]
		param_to_arg_ordering = [arg_names.index(ap_name_map(model_param_name)) for model_param_name in self.model_param_names]
		
		# [2,5]
		const_to_arg_ordering = [arg_names.index(ap_name_map(const_param_name)) for const_param_name in self.const_param_names]
		
		# [None,None,None,None,None,None,None]
		args = [None]*len(arg_names)
		
		# [None,None,c1,None,None,c2,None]
		for i, j in enumerate(const_to_arg_ordering):
			args[j] = self.const_param_transforms[i]
		
		
		def ultranest_model_callable(params):
			#_lgr.debug(f'{params=}')
			# [p1,p2,c1,p3,p4,c2,p5]
			for i, j in enumerate(param_to_arg_ordering):
				args[j] = params[i]
			
			return model_callable(
				*args
			)
		return ultranest_model_callable


def create_psf_model_callable(psf_model_obj, show_plots=False) -> Callable[[Any,...], Callable[[Any,...],Any]]:
	"""
	Create a callable that only accepts the parameters we care about for optimisation purposes.
	
	I.e. if the arguments of `psf_model_obj` are not in a form that an optimisation package likes,
	this function should return a wrapper that does take arguments nicely.
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
			s_factor,
			plots=show_plots # DEBUGGING
		)
		specific_model.factor = factor
		
		return specific_model
	
	return psf_model_callable


def psf_model_result_factory(psf_model_callable, wavelength, show_plots=False):
	"""
	Factory function that creates a function that returns the result of a specific model
	"""
	
	def model_result_callable(params):
		specific_model = psf_model_callable(params)
		result = specific_model.at(wavelength, plots=show_plots).data
		result /= np.nansum(result)
		result *= specific_model.factor
		
		return result

	return model_result_callable


def create_psf_model_likelihood_callable(psf_model_callable, data, err, wavelength, show_plots=False):
	"""
	Use the callable to work out the likelihood. This is what's actually called
	by ultranest
	"""
	def likelihood_callable(params, give_result=False):
		#_lgr.debug(f'{params=}')
		specific_model = psf_model_callable(params)
		
		#_lgr.debug(f'{data.shape=} {data_scale=}')

		nan_mask = np.isnan(data)
		result = specific_model.at(wavelength, plots=show_plots).data
		result /= np.nansum(result) # normalise model result
		
		result *= specific_model.factor
		
		if give_result : return result
		residual = data - result
		
		# err can be pre-computed
		# assume residual is gaussian distributed, with a sigma on each pixel and a flat value
		z = residual[~nan_mask]/err[~nan_mask]
		#likelihood = np.exp(-(z*z)/2)/np.sqrt(2*np.pi)
		likelihood = -(z*z)/2 # want the log of the pdf
		
		return likelihood.mean()
	
	return likelihood_callable


def get_error(data, frac, low_limit, hi_limit, sigma):
	"""
	Try to strike a balance between fractional error and absolute error
	"""
	v = np.abs(data)*frac
	vmin, vmax = np.nanmin(v), np.nanmax(v)
	vrange = vmax - vmin
	vrange_center = 0.5*(vmax+vmin)
	err = logistic_function(
		v,
		low_limit if low_limit > 0 else (-low_limit)*vmin,
		hi_limit if hi_limit > 0 else (-hi_limit)*vmax,
		sigma if sigma > 0 else vrange/(-sigma),
		vrange_center
	)
	return err


class UltranestResultSet:
	
	metadata_file : str = 'result_set_metadata.json'
	
	def __init__(self, result_set_directory : Path | str):
		self.directory = Path(result_set_directory)
		self.metadata = dict()
		self.metadata_path = self.directory / self.metadata_file
		if self.metadata_path.exists():
			self.load_metadata()
	
	def __repr__(self):
		return f'UltransetResultSet({self.directory.absolute()})'
	
	def load_metadata(self):
		with open(self.metadata_path, 'r') as f:
			self.metadata.update(json.load(f))
	
	def save_metadata(self, make_parent_dirs=True):
		if make_parent_dirs:
			self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
		with open(self.metadata_path, 'w') as f:
			json.dump(self.metadata, f)

	def clear_metadata(self):
		self.metadata = dict()

	def get_result_data_path(self, idx):
		"""
		Ultranest has "result_set_directory/run[INT]" to hold data for each run
		"""
		return self.directory / f'run{idx}'
		
	def get_result_data_from_path(self, result_data_path : Path):
		fname = result_data_path / 'info' / 'results.json'
		
		with open(fname, 'r') as f:
			rdata = json.load(f)
		
		return {
			'param_names': rdata['paramnames'],
			'best_point' : rdata['maximum_likelihood']['point'],
			'stats' : rdata['posterior']
		}


	def get_params_vs_wavelength(self) -> tuple[np.array, dict[str,np.array]]:
		
		wavs = np.array([w for w,idx in self.metadata['wavelength_idxs']])
		idxs = np.array([idx for w,idx in self.metadata['wavelength_idxs']])
		
		param_values = {}
		for i, (wavelength, idx) in enumerate(self.metadata['wavelength_idxs']):
		
			result = self.get_result_data_from_path(self.get_result_data_path(idx))
			for j, pname in enumerate(result['param_names']):
				if pname not in param_values:
					param_values[pname] = np.full_like(wavs, np.nan)
				param_values[pname][i] = result['best_point'][j]
		
		sort_indices = np.argsort(wavs)
		wavs = wavs[sort_indices]
		idxs = idxs[sort_indices]
		param_values = dict((k,v[sort_indices]) for k,v in param_values.items())
		
		return wavs, idxs, param_values

	def plot_params_vs_wavelength(self, show=False, save=True):
		fname = 'params_vs_wavelength.png'
		wavs, _, param_values = self.get_params_vs_wavelength()
		
		f, a = plot_helper.figure_n_subplots(len(param_values))
		#f.tight_layout(pad=16, w_pad=8, h_pad=4)
		f.set_layout_engine('constrained')
		
		f.suptitle('Parameters vs Wavelength')
		for i, pname in enumerate(param_values):
			a[i].set_title(pname)
			a[i].set_xlabel('wavelength')
			a[i].set_ylabel(pname)
			a[i].plot(wavs, param_values[pname], 'bo-')
		
		
		plot_helper.output(
			show, 
			None if save is None else plt.savefig(self.directory / fname)
		)
	
	
	def plot_results(self, 
			model_callable_factory : Callable[[float],Callable[[float,...],np.ndarray]], 
			ref_data : np.ndarray, 
			show=False, 
			save=True
		):
		log_plot_fname_fmt = 'log_result_{idx}.png'
		linear_plot_fname_fmt = 'linear_result_{idx}.png'
		
		wavs, idxs, param_values = self.get_params_vs_wavelength()
		
		
		for i, (wav, idx) in enumerate(zip(wavs, idxs)):
			
			
			params = tuple(param_values[pname][i] for pname in param_values)
			result = model_callable_factory(wav)(params)
			
			
			data = ref_data[idx]
			log_data = np.log(data)
			log_data[np.isinf(log_data)] = np.nan
			
			
			
			
			# plot log of result vs reference data
			f, a = plot_helper.figure_n_subplots(4)
			f.set_layout_engine('constrained')
			f.suptitle(f'log results {wav=} {idx=}')
			vmin, vmax = np.nanmin(log_data), np.nanmax(log_data)
			
			a[0].set_title(f'log data [{vmin}, {vmax}]')
			a[0].imshow(log_data, vmin=vmin, vmax=vmax)
			
			a[1].set_title(f'log result [{vmin}, {vmax}]')
			a[1].imshow(np.log(result), vmin=vmin, vmax=vmax)
			
			a[2].set_title(f'log residual [{vmin}, {vmax}]')
			a[2].imshow(np.log(data-result), vmin=vmin, vmax=vmax)
			
			log_abs_residual = np.log(np.abs(data-result))
			a[3].set_title(f'log abs residual [{np.nanmin(log_abs_residual)}, {np.nanmax(log_abs_residual)}]')
			a[3].imshow(log_abs_residual)
			
			plot_helper.output(
				show, 
				None if save is None else plt.savefig(self.directory / log_plot_fname_fmt.format(idx=idx))
			)
			
			
			# plot result vs reference data
			f, a = plot_helper.figure_n_subplots(4)
			f.set_layout_engine('constrained')
			f.suptitle(f'linear results {wav=} {idx=}')
			
			vmin, vmax = np.nanmin(data), np.nanmax(data)
			
			a[0].set_title(f'data [{vmin}, {vmax}]')
			a[0].imshow(data, vmin=vmin, vmax=vmax)
			
			a[1].set_title(f'result [{vmin}, {vmax}]')
			a[1].imshow(result, vmin=vmin, vmax=vmax)
			
			a[2].set_title(f'residual [{vmin}, {vmax}]')
			a[2].imshow(data-result, vmin=vmin, vmax=vmax)
			
			frac_residual = np.abs(data-result)/data
			fr_sorted = np.sort(frac_residual.flatten())
			vmin=fr_sorted[fr_sorted.size//4]
			vmax = fr_sorted[3*fr_sorted.size//4]
			a[3].set_title(f'frac residual [{vmin}, {vmax}]')
			a[3].imshow(frac_residual, vmin=vmin, vmax=vmax)
			
			plot_helper.output(
				show, 
				None if save is None else plt.savefig(self.directory / linear_plot_fname_fmt.format(idx=idx))
			)


if __name__=='__main__':
	
	
	psf = FitsSpecifier(example_data_loader.example_standard_star_file, 'DATA', (slice(None),slice(None),slice(None)), {'CELESTIAL':(1,2)}) 
	
	
	
	def plot_example_result(result, psf_data, show=True):
		if not show: return
		f, a = plot_helper.figure_n_subplots(6)
		a[0].set_title('data')
		a[0].imshow(np.log(psf_data))
		a[0].plot([psf_data.shape[0]/2],[psf_data.shape[1]/2], 'r.')
		
		a[1].set_title('result')
		a[1].imshow(np.log(result))
		
		a[2].set_title('residual')
		a[2].imshow(np.log(psf_data-result))
		
		
		a[3].set_title('residual_squared')
		a[3].imshow(np.log((psf_data-result)**2))
		
		a[4].set_title('data and result slice')
		a[4].plot(np.log(psf_data[psf_data.shape[0]//2,:]).flatten())
		a[4].plot(np.log(result[result.shape[0]//2,:]).flatten())
		
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
		
		a[5].set_title('radial data and result')
		a[5].plot(r[:-1], psf_radial_data)
		a[5].plot(r[:-1], result_radial_data)
		
		plot_helper.output(True)
	
	
	
	
	
	
	with fits.open(psf.path) as psf_hdul:	
		with nph.axes.to_end(psf_hdul[psf.ext].data, psf.axes['CELESTIAL']) as psf_data:
			
			
			#plt.title(psf.path)
			#plt.imshow(np.log(psf_data[psf_data.shape[0]//2]))
			#plt.show()
			
			psf_data = normalise_psf(psf_data, axes=psf.axes['CELESTIAL'], cutout_shape=(101,101))
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
		
			params = PriorParamSet((
				PriorParam('r0', 
					0.15,#unit_range_to(0.1,0.2), 
					0.15
				),
				PriorParam('turb_ndim', 
					unit_range_to(1,2), 
					1.3
				),
				PriorParam('L0', 
					8,#unit_range_to(7.5, 8.5), 
					8
				),
				PriorParam('alpha', 
					unit_range_to(0.1, 3), 
					0.7
				),
				PriorParam('beta', 
					unit_range_to(1.01, 10), 
					1.6
				),
				PriorParam('ao_correction_frac_offset', 
					unit_range_to(-1,1), 
					0
				),
				PriorParam('ao_correction_amplitude', 
					unit_range_to(0,5), 
					2.2
				),
				PriorParam('factor', 
					unit_range_to(0.7,1.3), 
					1
				),
				PriorParam('s_factor', 
					0, 
					0
				),
				PriorParam('f_ao',
					unit_range_to(24.0/(2*instrument.obj_diameter),52.0/(2*instrument.obj_diameter)),#instrument.f_ao,
					instrument.f_ao
				)
			))
	
	
	
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
			nested_sampling_max_iterations = 2000
			show_plots = False
			update_params_search_region = False
			result_set_directory = Path('ultranest_logs')
			
			
			psf_model_callable = create_psf_model_callable(test_psf_model, show_plots=show_plots)
			
			
			model_callable = params.get_ultranest_model_callable(psf_model_callable)
			_lgr.debug(f'{model_callable=}')
			
			result_set = UltranestResultSet(Path(result_set_directory))
			result_set.metadata['wavelength_idxs'] = wavelength_idxs
			result_set.metadata['constant_parameters'] = params.get_const_params()
			result_set.save_metadata()
			
			
			final_result = None
			
			for wavelength, idx in wavelength_idxs:
				
				if update_params_search_region and (final_result is not None):
					# Assume next results will be similar to previous ones
					pn = final_result['paramnames']
					pm = final_result['posterior']['mean']
					ps = final_result['posterior']['stdev']
					for n, m, s in zip(pn, pm, ps):
						params[n].transform = unit_range_to(m-2*s, m+2*s)
					params.recalc()
				
				psf_model_likelihood_callable = create_psf_model_likelihood_callable(
					model_callable,
					psf_data[idx],
					psf_err[idx],
					wavelength,
					show_plots=show_plots
				)
				
				# Debugging
				plot_example_result(
					psf_model_likelihood_callable(
						params.model_param_examples,
						give_result=True
					),
					psf_data[idx], 
					show=show_plots
				)
				
				
				
				
				ndim, nwalkers = len(params.model_param_names), 2*len(params.model_param_names)
				
				
				sampler = emcee.EnsembleSampler(
					nwalkers, 
					ndim, 
					psf_model_likelihood_callable
				)
				
				p0 = np.random.randn(nwalkers,ndim)
				
				state = sampler.run_mcmc(p0, 100, progress=True)
				
				print(f'{state=}')
				
				"""
				sampler = ultranest.ReactiveNestedSampler(
					params.model_param_names, 
					psf_model_likelihood_callable,
					params.get_ultranest_prior_param_transform(),
					log_dir=result_set_directory,
					resume='subfolder',
					run_num=idx,
					#warmstart_max_tau=0.5,
				)
				
			
				final_result = None
				for result in sampler.run_iter(
						max_iters=nested_sampling_max_iterations,
						min_num_live_points=80,
						dlogz=100,
						min_ess=40,
						update_interval_volume_fraction=0.8,
						max_num_improvement_loops=3,
						frac_remain=nested_sampling_stop_fraction
					):
					sampler.print_results()
					sampler.plot()
					final_result=result
				"""
				
				
				
				for k, v in final_result.items():
					_lgr.debug(f'{k} = {v}')
	
			
			
			result_set.plot_params_vs_wavelength(show=False, save=True)
			result_set.plot_results(
				lambda wav: psf_model_result_factory(model_callable, wav),
				psf_data,
				show=False,
				save=True
			)
	
			