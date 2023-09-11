#!/usr/bin/env python3
"""
Example script of how to interpolate a datacube to remove NANs and store the
map of adjusted pixels.
"""

import sys, os
# Add folder that contains scripts to $PATH environment variable.
# need to do this so directory structure is easily transferrable
# between machines.
SCRIPTS_PATH = "../scripts"
sys.path.insert(0,SCRIPTS_PATH)
print(f'{sys.path=}')

import dataclasses as dc
import numpy as np
from astropy.io import fits

import utilities as ut
import utilities.sp

# DEBUGGING
import matplotlib.pyplot as plt

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

def main(args):
	"""
	All high-level steps pertaining to interpolating accross a datacube go in 
	here.
	"""

	# print out arguments for clarity
	for k,v in args.items():
		print(f'{k} = {v}')


	# Open FITS file for operations on data
	with fits.open(args['input.file']) as hdul:
		hdu = hdul[args['input.ext']]

		nan_mask = np.isnan(hdu.data)
		inf_pos_mask = np.isposinf(hdu.data)
		inf_neg_mask = np.isneginf(hdu.data)
		with (
				ArrayAxesToEnd(hdu.data, args['input.image_axes']) as data,
				ArrayAxesToEnd(nan_mask, args['input.image_axes']) as nans,
				ArrayAxesToEnd(inf_pos_mask, args['input.image_axes']) as pinfs,
				ArrayAxesToEnd(inf_neg_mask, args['input.image_axes']) as ninfs
			):

			# loop over input data and interpolate each frame
			slices = args['input.slice']
			for (idxs, data_slice), nan, pinf, ninf in zip(EnumerateSliceOf(data, slices), nans[slices], pinfs[slices], ninfs[slices]):
				print(f'Interpolating index {idxs[0] if idxs.size==1 else idxs}')

				# remember, Python creates temporary objects when using
				# iterators, therefore you have to index into the original
				# object if you want to mutate it. Hence "data[idxs] =" instead
				# of "data_slice =".
				data[idxs] = ut.sp.interpolate_at_mask(
					data_slice, 
					nan | pinf | ninf, 
					args['interpolate.edges']
				)

		# Save arguments to FITS header file so we can reference them
		# easily later
		hdr = hdu.header
		hdr.update(DictToFitsHeaderIter(args))

		# Create FITS Header Data Units
		hdu_interp = fits.PrimaryHDU(
			data = hdu.data,
			header = hdu.header
		)
		bool_hdr = hdu.header
		bool_hdr['BITPIX'] = 8
		hdu_nan_mask = fits.ImageHDU(
			data = nan_mask.astype('u1'),
			header = bool_hdr,
			name = 'NAN_PIXEL_MAP'
		)
		hdu_inf_pos_mask = fits.ImageHDU(
			data = inf_pos_mask.astype('u1'),
			header = bool_hdr,
			name = 'POS_INF_PIXEL_MASK'
		)
		hdu_inf_neg_mask = fits.ImageHDU(
			data = inf_neg_mask.astype('u1'),
			header = bool_hdr,
			name = 'NEG_INF_PIXEL_MASK'
		)

	# Combine HDUs into a Header Data Unit List (astropy's representation of a
	# FITS file)
	hdul_output = fits.HDUList([
		hdu_interp,
		hdu_nan_mask,
		hdu_inf_pos_mask,
		hdu_inf_neg_mask,
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
		
		'input.slice' 						: (np.array([229,230 ]),), 
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

		'input.image_axes'					: tuple([1,2]), # tuple(int,int)
			# The axes of the datacube that denote the image-plane, rather than
			# the wavelength axis. NOTE, this is in "astropy" ordering, i.e. if
			# NAXIS1 = 'wav', NAXIS2='ra', NAXIS3='dec', the image axes will be
			# (1,2). I.e. "astropy" ordering is zero-indexed, AND reversed from
			# the numbers in the FITS files (yes I know it's annoying).

		'output.file'				: './output/interpolated.fits', # str
			# File to output filtered data to, will put interpolated data in 
			# PRIMARY extension, and interpolated pixels map in NAN_PIXEL_MAP,
			# POS_INF_PIXEL_MAP, NEG_INF_PIXEL_MAP.

		'output.file.overwrite'		: True, # bool

		'interpolate.edges' 		: 'convolution' # None | str
			# How should we deal with NANs and INFs around the edges?
			# None -> ignore them
			# 'convolution' -> convolve with large rectangle to have some
			# 	sort of estimate.
		}
	main(args)







