#!/usr/bin/env python3
"""
This example script shows how to interpolate over bad pixels found by routines 
in "flag_bad_pixels.py"
"""

import sys, os
# Add folder that contains scripts to $PATH environment variable.
# need to do this so directory structure is easily transferrable
# between machines.
SCRIPTS_PATH = "../scripts"
sys.path.insert(0,SCRIPTS_PATH)
print(f'{sys.path=}')

import inspect
import dataclasses as dc

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

import fitscube.deconvolve.flag_bad_pixels as flag_bad_pixels
import py_ssa as py_ssa
import utilities as ut
import utilities.sp
import utilities.fits
	
# Global variables hold locations of data, I would not normally do this but for
# illustrative purposes it's good enough.
EXAMPLE_INPUT_OBS_DATA = "../data/test_rebin.fits"
EXAMPLE_INPUT_PSF_DATA = "../data/fit_example_psf_000.fits"

## HELPER CLASSES ##
# I could move all of these into a shared module, but I want to try and keep
# these tutorials as self-contained as I can

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


def get_callable_from_module_by_name(module, name):
	"""
	We want to be able to pass the filtering routine as an input, so need to be
	able to grab the actual function from it's containing module by name.
	"""
	for _n, _m in inspect.getmembers(module):
		if (_n == name) and (inspect.isclass(_m) or inspect.ismethod(_m) or inspect.isfunction(_m)):
			#print("Found callable, showing documentation:")
			#print(inspect.getdoc(_m))
			return(_m)
	raise RuntimeError(f"Could not find callable with name \"{name}\" in module \"{module.__name__}\"")


def main(args):
	"""
	High-level logic goes here
	"""
	# print out arguments for clarity
	for k,v in args.items():
		print(f'{k} = {v}')

	filtering_callable = get_callable_from_module_by_name(flag_bad_pixels, args['filtering.routine'])

	# Open FITS file for operations on data
	with fits.open(args['input.file']) as hdul:
		hdu = hdul[args['input.ext']]

		# Ensure that input data has no invalid pixels
		hdu.data = np.nan_to_num(hdu.data)
		bp_mask_raw = np.zeros_like(hdu.data, dtype=bool)

		with (
				ArrayAxesToEnd(hdu.data, args['input.image_axes']) as data,
				ArrayAxesToEnd(bp_mask_raw, args['input.image_axes']) as bp_mask,
			):
			# create a holder for each slice's bad pixel mask, have to do this
			# as Python doesn't let you mutate an object via it's iterator,
			# which is kinda annoying if you ask me but hey...
			bp_mask_slice = np.zeros(bp_mask.shape[len(args['input.image_axes']):], dtype=bool)

			# loop over input data and filter each frame
			for idxs, data_slice in EnumerateSliceOf(data, args['input.slice']):
				print(f'Filtering frame with index {idxs[0] if idxs.size==1 else idxs}')

				# get singular spectrum analysis of image
				ssa2d = py_ssa.SSA2D(
					data_slice,
					args['ssa.window_shape']
				)
				if args['ssa.show_plots']:
					ssa2d.plot_ssa()

				# find bad pixels
				bp_mask_slice = filtering_callable(
					ssa2d,
					**args['filtering.arguments']
				)

				# if no plots, this should silently return
				plt.show()

				# interpolate at bad pixels
				data[idxs] = ut.sp.interpolate_at_mask(
					data_slice, 
					bp_mask_slice
				)

				# move data into complete mask
				bp_mask[idxs] = bp_mask_slice


		# Save arguments to FITS header file so we can reference them
		# easily later
		hdr = hdu.header
		hdr.update(DictToFitsHeaderIter(args))

		# Create FITS Header Data Units
		hdu_filtered = fits.PrimaryHDU(
			data = hdu.data,
			header = hdr
		)
		bool_hdr = hdr
		bool_hdr['BITPIX'] = 8
		hdu_bp_mask = fits.ImageHDU(
			data = bp_mask.astype('u1'),
			header = bool_hdr,
			name = 'BAD_PIXEL_MAP'
		)

	# Combine HDUs into a Header Data Unit List (astropy's representation of a
	# FITS file)
	hdul_output = fits.HDUList([
		hdu_filtered,
		hdu_bp_mask,
	])

	# Write out FITS file
	hdul_output.writeto(args['output.file'], overwrite=args['output.file.overwrite'])
	return



	

if __name__=='__main__':
	"""
	This code will only execute when the file is run as a script.
	"""

	# Unfortunately the only way to type-hint a dictionary is still
	# to use comments.
	args={
		'input.file' 					: EXAMPLE_INPUT_OBS_DATA, # str
		# Path to a FITS file
		'input.ext'						: 1, # int | str
			# FITS extension to filter
		
		'input.slice' 						: (np.array([229, ]),), 
			# np.ndarray | Slice
			# The indices (0-indexed) of the extension that should be
			# operated upon. Will be applied as "slice" in 
			# "hdul[extension].data[slice]".
			# DEFAULT : 
			# 	(slice(None,None,None),)
			# 		Deconvolves the whole cube.
			# EXAMPLES: 
			# 	(np.array([0,1,2,3,4,5]),)
			#		Only deconvolves the first 6 images.

		'input.image_axes'					: tuple([2,1]), # tuple(int,int)
			# The axes of the datacube that denote the image-plane, rather than
			# the wavelength axis. NOTE, this is in "astropy" ordering, i.e. if
			# NAXIS1 = 'wav', NAXIS2='ra', NAXIS3='dec', the image axes will be
			# (1,2). I.e. "astropy" ordering is zero-indexed, AND reversed from
			# the numbers in the FITS files (yes I know it's annoying).

		'output.file'				: './output/ssa_filtered.fits', # str
			# File to output filtered data to, will put data in PRIMARY 
			# extension

		'output.file.overwrite'		: True, # bool

		'ssa.window_shape'			: (6,6), # tuple(int,int) | None
			# Shape of the window for calculating SSA2D, larger values give
			# better discrimination between component parts, but take more 
			# memory

		'ssa.show_plots' 			: False, # bool
			# Should we show plots for singular spectrum analysis?

		'filtering.routine'			: 'ssa2d_sum_prob_map', # str
			# Name of the routine that we should use for filtering

		'filtering.arguments'		: {
				'start' : 3,
				'stop' : 12,
				'value' : 0.90,
				'show_plots' : 0,
			},
			# keyword arguments passed to the filtering routine
		}
	main(args)



