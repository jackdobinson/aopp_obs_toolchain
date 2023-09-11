#!/usr/bin/env python3
"""
This script shows a bare-bones implementation of modelling a PSF using the
techniques from 
https://www.aanda.org/articles/aa/pdf/2019/08/aa35830-19.pdf#cite.Fetick2019
"""


import sys, os
# Add folder that contains scripts to $PATH environment variable.
# need to do this so directory structure is easily transferrable
# between machines.
SCRIPTS_PATH = "../scripts"
sys.path.insert(0,SCRIPTS_PATH)
print(f'{sys.path=}')

import dataclasses as dc
import pickle

import numpy as np
import scipy as sp
import scipy.optimize
from astropy.io import fits

import fitscube.deconvolve.psf_model as psf_model
import utilities as ut
import utilities.sp
import utilities.path

# Global variables hold locations of data, I would not normally do this but for
# illustrative purposes it's good enough.
EXAMPLE_INPUT_PSF_DATA = "../data/test_standard_star.fits"


@dc.dataclass
class ArrayAxesToEnd:
	"""
	Context manager that re-orders the axes of an array, then puts them back
	again. I'm not sure if __exit__ actually does anything, but best to be 
	sure I guess.
	"""
	data : np.ndarray
	axes : tuple

	def __enter__(self):
		self.axes_dest = tuple(self.data.ndim - 1 -i for i in range(len(self.axes)))
		self.data = np.moveaxis(self.data, self.axes, self.axes_dest)
		return(self.data)

	def __exit__(self, type, value, traceback):
		self.data = np.moveaxis(self.data, self.axes_dest, self.axes)

@dc.dataclass
class EnumerateSliceOf:
	"""
	Iterator that returns sliced parts of an array and the "real" index of 
	those slices.
	"""
	data : np.ndarray
	slice_tuple : tuple = None # can also be an np.ndarray[int]

	def __iter__(self):
		if self.slice_tuple is None:
			self.slice_tuple = tuple([slice(None,None,None)]*data.ndim)

		self.sliced_idxs = np.moveaxis(np.indices(self.data.shape[:len(self.slice_tuple)]), 0,-1)[self.slice_tuple]
		self.sliced_data = self.data[self.slice_tuple]
		return(zip(self.sliced_idxs, self.sliced_data))
	
@dc.dataclass
class DictToFitsHeaderIter:
	"""
	Put a dictionary into a format that we can insert into a FITS header file.
	Make all uppercase, replace dots with spaces, turn everything into a string
	and hope it's not over 80 characters long. Will need to use HIERARCH keys
	to have long parameter names, see 
	https://fits.gsfc.nasa.gov/fits_standard.html

	Also we should combine dictionaries so that we have a flat structure, one
	way to do this is to concatenate the keys for member dictionaries with the
	parent dictionary, i.e:
		{"some.key" : {"child_key_1" : X1, "child_key_2": X2}} 
	becomes
		{"some.key.child_key_1" : X1, "some.key.child_key_2": X2}
	and is turned into a FITS header like
		HIERARCH SOME KEY CHILD_KEY_1 = "X1"
		HIERARCH SOME KEY CHILD_KEY_2 = "X2"

	If we need to we can split header entries into a "name" and "value" format,
	e.g. the above example would become:
		PKEY1   = some.key.child_key_1
		PVAL1   = "X1"
		PKEY2   = some.key.child_key_2
		PVAL1   = "X2"
	"""
	adict : dict
	mode : str = 'standard' # 'hierarch' | 'standard'

	def __iter__(self):
		self.fits_dict = {}
		self.key_count = 0
		if self.mode == 'hierarch':
			self._to_fits_hierarch_format(self.adict, prefix='HIERARCH')
		elif self.mode == 'standard':
			self._to_fits_format(self.adict, prefix=None)
		else:
			raise RuntimeError(f"Unknown mode \"{self.mode}\" for creating FITS header cards from a dictionary, known modes are ('hierarch', 'standard').")
		return(iter(self.fits_dict.items()))

	def _to_fits_hierarch_format(self, bdict, prefix=None):
		for key, value in bdict.items():
			fits_key = ('' if prefix is None else ' ').join([('' if prefix is None else prefix), key.replace('.', ' ').upper()])
			if type(value) is not dict:
				self.fits_dict[fits_key] = str(value)
			else:
				self._to_fits_hierarch_format(value, fits_key)
		return

	def _to_fits_format(self, bdict, prefix=None):
		prefix_str = '' if prefix is None else prefix
		prefix_join_str = '' if prefix is None else '.'

		for k, v in bdict.items():
			_k = prefix_join_str.join([prefix_str, k])
			_v = str(v)
			if type(v) is not dict:
				self._add_kv_to_fits_dict(_k,_v)
			else:
				self._to_fits_format(v, _k)
		return

	def _add_kv_to_fits_dict(self, k, v):
		k_key = f'PKEY{self.key_count}'
		v_key = f'PVAL{self.key_count}'
		key_fmt_str = "{: <8}"
		val_fmt_str = "{}" # astropy can normalise the value string if needed
		self.fits_dict[key_fmt_str.format(k_key)] = val_fmt_str.format(k)
		self.fits_dict[key_fmt_str.format(v_key)] = val_fmt_str.format(v)
		self.key_count += 1
		return

## END HELPER CLASSES ##	


def normalise_input_psf(data, shape=None):
	"""
	Takes some input PSF data and normalises it in the following way:
		* Sums to 1
		* Brightest pixel in in center of image
		* Has the prescribed shape
	
	# ARGUMENTS #
		data : np.ndarray
			Numpy array (2D) holding the PSF image
		shape : tuple(int,int) | None
			Desired output shape of the normalised PSF image, if None
			will not change shape of PSF image.
	
	# RETURNS #
		normalised_data : np.ndarray
			Normalised, centered, and trimmed PSF image
	"""
	if shape is None:
		shape = data.shape

	# get brightest pixel index
	bp_idxs = np.unravel_index(np.nanargmax(data), shape=data.shape)

	# move brightest pixel to center of image
	data = np.roll(
		data,
		np.array([s//2 for s in data.shape]) - bp_idxs, axis=(0,1)
	)

	# reshape to desired size
	for _i, _s in enumerate(shape):
		s0 = data.shape[_i]
		d = s0 - _s
		if d > 0:
			data = data[
				tuple(slice((d)//2, -(d)//2) if _j==_i else slice(None,None) for _j in range(data.ndim))
			]
		if d < 0:
			data = np.pad(
				data,
				[((d)//2, d-(d+1)//2) if _j==_i else (0,0) for _j in range(data.ndim)],
				constant_values = np.nan
		)
	
	# normalise value
	data /= np.nansum(data)

	return(data)


def fit_psf_model_to_data(args, model):
	def cost_function(
			model_params,
			wavelength,
			model,
			param_names,
			reference_data
		):
		model_param_dict = dict(zip(param_names, model_params))
		factor = model_param_dict.pop('Factor')
		model_data = factor*model.get_psf(wavelength, **model_param_dict)
		cost = np.nansum((model_data - reference_data)**2)
		return(cost)
	
	# Open FITS file for operations on data
	with fits.open(args['input.file']) as hdul:
		hdu = hdul[args['input.ext']]

		# get the wavelengths in the HDU we are working on
		wavelengths = np.arange(hdu.header['NAXIS3'])*hdu.header['CD3_3']
		wavelengths += hdu.header['CRVAL3'] - wavelengths[hdu.header['CRPIX3']]
			
		with ArrayAxesToEnd(hdu.data, args['input.image_axes']) as data:

			all_results = []
			for p0 in args['model.param_set']['p0_values']:
				results = {
					'p0' : p0,
					'result_set' : []
				}
				all_results.append(results)

				# loop over input data and fit to each frame
				slices = args['input.slice']
				for (idxs, data_slice), wavelength in zip(EnumerateSliceOf(data, slices), wavelengths[slices]):
					print(f'Fitting model PSF to example at index {idxs[0] if idxs.size==1 else idxs}')

					# Normalise data to have the same shape as our model will, and put the
					# brightest pixel in the center
					data_slice = normalise_input_psf(data_slice, args['model.shape'])


					result = sp.optimize.minimize(
						cost_function, 
						p0, 
						args = (
							wavelength,
							model,
							args['model.param_set']['p0_descriptions'].keys(),
							data_slice,
						),
						bounds = args['model.param_set']['p0_bounds'],
						method = ['Nelder-Mead','L-BFGS-B','TNC','SLSQP','Powell','trust-constr'][1], # for bounded problem
						#method = ['CG','BFGS','Newton-CG','COBYLA','dogleg','trust-ncg','trust-exact','trust-krylov'][1], # for unbounded problems
						options = dict(
							disp = True,
							maxiter = args['fit.maxiter'],
							#eps = 1E-6
							gtol = 1E-12,
							ftol = 1E-12
						)

					)

					# add data that we need to result object
					result.wavelength = wavelength # wavelength 
					result.idxs = idxs # index of wavelength

					# collate results data into a single place for fiddling with later
					all_results[-1]['result_set'].append(result)
				
	datafiles = []
	for results in all_results:
		datafile_path = ut.path.fpath_as(args['output.datafile'], overwrite=args['output.datafile.overwrite'])
		datafiles.append(datafile_path)
		with open(datafile_path, 'wb') as f:
			pickle.dump(results, f)
	
	return(datafiles)


def create_psf_from_datafile(args, datafile, model):
	def model_function(
			model_params,
			wavelength,
			model,
			param_names,
		):
		model_param_dict = dict(zip(param_names, model_params))
		factor = model_param_dict.pop('Factor')
		model_data = factor*model.get_psf(wavelength, **model_param_dict)
		return(model_data)
	

	# Open FITS file for operations on data
	with fits.open(args['input.file']) as hdul:
		hdu = hdul[args['input.ext']]
		psf_data_shape = hdu.data.shape
		psf_hdr = hdu.header

		# get the wavelengths in the HDU we are working on, this is hard-coded
		# at the moment, but I should really make this adaptive to the cube
		# by looking up which axis is the spectral one.
		wavelengths = np.arange(hdu.header['NAXIS3'])*hdu.header['CD3_3']
		wavelengths += hdu.header['CRVAL3'] - wavelengths[hdu.header['CRPIX3']]
	

	# Want to get the shape of modelled psf with the correct number of channels
	modelled_psf_data_shape = []
	_j = 0
	for _i, _s in enumerate(psf_data_shape):
		print(_i, _s, _j)
		print(_i in args['input.image_axes'])
		if _i in args['input.image_axes']:
			modelled_psf_data_shape.append(args['model.shape'][_j])
			_j += 1
		else:
			modelled_psf_data_shape.append(_s)
	modelled_psf_data_shape = tuple(modelled_psf_data_shape)
	
	print(modelled_psf_data_shape)

	# allocate memory for our model data
	modelled_psf_data_raw = np.full(modelled_psf_data_shape, fill_value=np.nan)
	print(f'{modelled_psf_data_raw.shape=}')
	
	
	with ArrayAxesToEnd(modelled_psf_data_raw, args['input.image_axes']) as modelled_psf_data:
		all_idxs = np.indices(modelled_psf_data.shape[:-2])[0]

		with open(datafile, 'rb') as f:
			data = pickle.load(f)

		p0 = np.array(data['p0'])
		result_set = data['result_set']

		print(f'{p0=}')
		fitted_idxs = np.array([_r.idxs for _r in result_set])[:,0]
		fitted_xs = np.array([_r.x for _r in result_set])
		fitted_wavelengths = np.array([_r.wavelength for _r in result_set])

		# Interpolate the fitted parameter values
		print(f'{fitted_idxs=}')
		print(f'{all_idxs=}')
		print(f'{fitted_xs=}')
		interp_xs = np.full(all_idxs.shape+p0.shape, fill_value=np.nan)
		for _i in range(len(p0)):
			interp_xs[:,_i] = np.interp(all_idxs, fitted_idxs, fitted_xs[:,_i])
		print(f'{interp_xs}')


		# Create a cube of fitted parameter values at each wavelength
		for idx in all_idxs:
			print(f'Modelling PSF at index {idx}')
			modelled_psf_data[idx] = model_function(interp_xs[idx], wavelengths[idx], model, args['model.param_set']['p0_descriptions'].keys())

	hdr = psf_hdr
	hdr.update(DictToFitsHeaderIter(args))
	hdr.update(DictToFitsHeaderIter(
		{	'p0' : p0,
			'fitted_idxs' : dict([(f'{_i}',_x) for _i,_x in enumerate(fitted_idxs)]),
			'fitted_xs' : dict([(f'{_i}',dict([(f'{_ii}',_xx) for _ii,_xx in enumerate(_x)])) for _i,_x in enumerate(fitted_xs)]),
			'fitted_wavelengths' : dict([(f'{_i}',_x) for _i,_x in enumerate(fitted_wavelengths)]),
			'final_cost' : dict([(f'{_i}',_r.fun) for _i,_r in enumerate(result_set)]),
			'n_iter'	: dict([(f'{_i}',_r.nit) for _i,_r in enumerate(result_set)]),
		}
	))
	modelled_psf_hdu = fits.PrimaryHDU(
		data = modelled_psf_data,
		header = hdr
	)
	hdul = fits.HDUList([
		modelled_psf_hdu,
	])

	fpath = ut.path.fpath_from(datafile, args['output.modelfile'])
	hdul.writeto(fpath, overwrite=args['output.modelfile.overwrite'])
	return
		

def main(args):
	"""
	High-level logic goes here
	"""
	# print out arguments for clarity
	for k,v in args.items():
		print(f'{k} = {v}')
	
	# Define the model we want to fit to our data
	model = psf_model.AdaptiveOpticsModel(
			psf_model.TurbulenceModel,
			psf_model.Instrument_VLT_MUSE,
			shape=args['model.shape'],
			supersample_factor=args['model.supersample_factor'],
		)


	# DEBUGGING
	#datafiles = ["./output/psf_model_000.pkl", "./output/psf_model_001.pkl"]

	datafiles = fit_psf_model_to_data(args, model)

	for datafile in datafiles:
		create_psf_from_datafile(args, datafile, model)

		
	return





if __name__=='__main__':
	args = {
		'input.file' : EXAMPLE_INPUT_PSF_DATA,
		'input.ext' : 1,
		'input.slice' : (slice(None,None,300),), # only fit every 5 wavelengths
		'input.image_axes' : tuple([1,2]),
		
		'model.shape' : (201,201),
		'model.supersample_factor' : 4,
		'model.param_set' : {
			# Descriptions and names of each parameter
			'p0_descriptions' : {
				'r0' : 'fried parameter, scale over which light becomes out of phase',
				'alpha' : 'angular size of the core of the PSF',
				'beta' : 'Controls overall shape of core, close to 1 from top gives a gradual decline, high values give a steep decline',
				'C' : 'Controls importance of edges of core to wings, higher -> high gradient between core and wings. Generally want to set it so PSF is approx. continuous. However, high values of C have a similar effect to low (~0.01) values of beta.',
				'A': 'Controls importance of core to wings, higher -> more defined core -> better resolution',
				'Factor': 'Multiplies the entire solution, useful for moving everything up/down in log-space',
			},
			
			# Values to test
			'p0_values' : [
				[0.17, 5E-2, 1.6, 2E-3, 0.05, 1], # 2D turbulence starting Values
				[0.2, 5E-2, 1.6, 2E-2, 0.05, 1],
				[0.2, 5E-2, 1.6, 2E-1, 0.05, 1],
				[0.1, 5E-2, 1.6, 5E-3, 0.05, 1],
				[0.01, 5E-2, 1.6, 5E-3, 0.05, 1],
				[0.01, 5E-2, 1.6, 1E-3, 0.05, 1],
				[0.01, 5E-2, 1.6, 1E-4, 0.05, 1],
				[0.2, 5E-2, 1.6, 1E-2, 2, 1],
				[0.2, 5E-2, 1.6, 1E-3, 2, 1],
			],

			# (lower_bound, upper_bound) for each parameter
			'p0_bounds' : [
				(np.finfo(float).eps, np.inf), # r0
				(np.finfo(float).eps, np.inf), # alpha
				(1+np.finfo(float).eps, np.inf), #  beta
				(0, np.inf), # C
				(0, np.inf), # A
				(0, 1), # factor
			],
		},

		'fit.maxiter' : 100,

		'output.datafile'				: './output/psf_model_{incrementXXX}.pkl', # str
			# File to output filtered data to, will put interpolated data in 
			# PRIMARY extension, and interpolated pixels map in NAN_PIXEL_MAP,
			# POS_INF_PIXEL_MAP, NEG_INF_PIXEL_MAP.

		'output.datafile.overwrite'		: True, # bool

		'output.modelfile' 				: '{fdir}/{fname}.fits',
		'output.modelfile.overwrite'	: True,
	}
	main(args)
