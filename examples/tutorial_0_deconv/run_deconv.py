#!/usr/bin/env python3
"""
# DESCRIPTION #
	Front-end to deconvolution routines.

	## Example ##
		python3 ./deconv.py ../data/test_rebin.fits{1}[229:230] ../data/test_standard_star.fits{1}

		NOTE: Sometimes the shell will try and interpret the curly/square/round brackets as commands
		or globs etc. In that case put the arguments inside single or double quotes.

		E.g. python3 ./deconv.py "../data/test_rebin.fits{1}[229:230]" "../data/test_standard_star.fits{1}"


	## FITS File Path Format ##
		../path/to/datafile.fits{ext}[slice_tuple](img_axes_tuple)
			ext : int | str
				Extension of the FITS file to operate upon, if not present,
				will use the first extension that has some data.
			slice_tuple : tuple[Slice,...]
				Slice of the data to operate upon, useful for choosing
				a subset of wavelength channels. If not present, will
				assume all wavelengths are to be deconvolved.
			img_axes_tuple : tuple[int,...]
				Tuple of the spatial axes indices, uses FITS ordering
				if +ve, numpy ordering if -ve. Usually the RA,DEC axes.

		### Examples ###
			./some_datafile.fits{PRIMARY}[100:150]
				Select the extension called 'PRIMARY' and deconvolve the channels 100->150
			/an/absolute/path/to/this_data.fits[99:700:50]
				Try to guess the extension to use, deconvolve every 50th channel in the range 99->700
			./deconv/whole/file.fits{SCI}
				Use the extension 'SCI'
			./some/path/big_file.fits{1}[5:500:10](0,2)
				use the 1st extension (not the 0th), deconvolve every 10th channel in the range 5->500,
				the spatial axes are 0th and 2nd axis.
				

# END DESCRIPTION #
"""			


import sys, os
# Add folder that contains scripts to $PATH environment variable.
# need to do this so directory structure is easily transferrable
# between machines.
SCRIPTS_PATH = os.path.normpath(f"{os.path.dirname(__file__)}/../scripts")
sys.path.insert(0,SCRIPTS_PATH)
print('INFO: \tPython will search for modules in these locations:\n----: \t\t{}\n'.format("\n----: \t\t".join(sys.path)))

import utilities.logging_setup
#logging, _lgr = utilities.logging_setup.getLoggers(__name__, 'DEBUG') # Show debug logs and above (command is "_lgr.DEBUG")
logging, _lgr = utilities.logging_setup.getLoggers(__name__, 'INFO') # Show information logs and above (command is "_lgr.INFO")
#logging, _lgr = utilities.logging_setup.getLoggers(__name__, 'WARN') # Show warning logs and above (command is "_lgr.WARN")
#logging, _lgr = utilities.logging_setup.getLoggers(__name__, 'ERROR') # Show error logs and above (command is "_lgr.ERROR")
#logging, _lgr = utilities.logging_setup.getLoggers(__name__, 'CRIT') # Show critical logs and above (command is "_lgr.CRIT")

# Log this here as we can't log stuff before we set up logging :-p



import argparse
import inspect
import dataclasses as dc

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from fitscube.deconvolve.algorithms import CleanModified

import utilities as ut
import utilities.sp
import utilities.args
	
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
class IndicesOfSlice:
	"""
	Iterator that returns the sliced indices of a sliced array
	"""
	data : np.ndarray
	slice_tuple : tuple = None # can also be an np.ndarray[int]

	def __iter__(self):
		if self.slice_tuple is None:
			self.slice_tuple = tuple([slice(None,None,None)]*data.ndim)

		self.sliced_idxs = np.moveaxis(np.indices(self.data.shape[:len(self.slice_tuple)]), 0,-1)[self.slice_tuple]
		return((tuple(x) for x in self.sliced_idxs))

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


# This is useful to have as a global variable so 
# I can print the help message when there are errors.
parser = ut.args.RawArgParser(description=__doc__)

def parse_fits_path(apath):
	img_axes_str = None
	slice_str = None
	ext_str = None

	_idx = len(apath)
	img_axes_idx = _idx
	if apath.endswith(')'):
		img_axes_idx = apath.rfind('(')
		img_axes_str = apath[img_axes_idx:_idx]

	slice_idx = img_axes_idx
	if apath.endswith(']'):
		slice_idx = apath.rfind('[')
		slice_str = apath[slice_idx:]

	ext_idx = slice_idx
	if apath[slice_idx-1] == '}':
		ext_idx = apath.rfind('{')
		ext_str = apath[ext_idx:slice_idx]

	if img_axes_str is not None:
		try:
			img_axes_str = eval(img_axes_str)
		except:
			_lgr.ERROR(f'Malformed FITS path, img_axes_tuple should be of the form (int, int, ...). Example: (1,2)')
			parser.print_help()
			sys.exit()

	if ext_str is not None:
		try:
			ext_str = int(ext_str[1:-1])
		except ValueError:
			ext_str = ext_str[1:-1]

	if slice_str is not None:
		try:
			slice_str = eval(f'np.index_exp{slice_str}')
		except:
			_lgr.ERROR(f'Malformed FITS path, slice should be of form [start:stop:step]')
			parser.print_help()
			sys.exit()
	return(apath[:ext_idx], ext_str, slice_str, img_axes_str)
	

def parse_args(argv):
	parser.add_argument(
		'obs.file', 
		metavar='OBS_FITS_PATH',
		type=str, 
		help='Observation file to operate upon, uses FITS file path format',
	)
	parser.add_argument(
		'psf.file',
		metavar='PSF_FITS_PATH',
		type=str,
		help='Point Spread Function (PSF) to use during deconvolution, uses FITS file path format',
	)
	parser.add_argument(
		'--output.file',
		type=str,
		help='file to output deconvolved data to.',
		default="./output/deconv.fits",
	)
	parser.add_argument(
		'--output.file.overwrite',
		action=ut.args.ActionTf,
		prefix='output.file.',
		help='Should we overwrite the output file or not?',
	)

	ut.args.add_args_from_dataclass(parser, CleanModified, prefix='CleanModified', as_group=True)


	args = vars(parser.parse_args())

	args['obs.file'], args['obs.file.ext'], args['obs.file.slice'], args['obs.file.img_axes'] = parse_fits_path(args['obs.file'])
	args['psf.file'], args['psf.file.ext'], args['psf.file.slice'], args['psf.file.img_axes'] = parse_fits_path(args['psf.file'])

	# If neither slice parameter is set, assume they should be the whole data cube.
	# If we only set the slice parameter for one file, assume it applieds to the second as well.
	# If both are set, that's fine :-)
	if args['obs.file.slice'] is None and args['psf.file.slice'] is None:
		args['obs.file.slice'] = (slice(None),)
		args['psf.file.slice'] = (slice(None),)
	elif args['obs.file.slice'] is None:
		args['obs.file.slice'] = args['psf.file.slice']
	elif args['psf.file.slice'] is None:
		args['psf.file.slice'] = args['obs.file.slice']



	return(args)


def set_default_args_from_fits_hdul(args, hdul, prefix=''):
	if args[f'{prefix}ext'] is None:
		for i, hdu in enumerate(hdul):
			if hdu.data is not None and hdu.data.size > 0:
				args[f'{prefix}ext'] = i
				break
		if i >= len(hdul):
			_lgr.ERROR(f'Could not automatically assign {prefix}ext. No HDU has any data in it.')
			parser.print_help()
			sys.exit()

	hdu = hdul[args[f'{prefix}ext']]
	if args[f'{prefix}img_axes'] is None:
		args[f'{prefix}img_axes'] = ut.fits.hdr_get_celestial_axes(hdu.header)
	else:
		args[f'{prefix}img_axes'] = tuple(-i if i <0 else ut.fits.AxesOrdering(hdu.header['NAXIS'], i, 'fits').numpy for i in args[f'{prefix}img_axes'])
	return


def simple_bad_pixel_check(obs_data, psf_data):
	obs_bp_map = np.isnan(obs_data) | np.isinf(obs_data)
	psf_bp_map = np.isnan(psf_data) | np.isinf(psf_data)
	obs_data[obs_bp_map] = 0.0
	psf_data[psf_bp_map] = 0.0
	if np.any(obs_bp_map):
		_lgr.WARN(f'Observation has bad pixels in it, setting to zero as naive fix...')
	if np.any(psf_bp_map):
		_lgr.WARN(f'PSF has bad pixels in it, setting to zero as naive fix...')
	return(obs_data, psf_data)


def get_center_offset_brightest_pixel(a):
	if np.all(np.isnan(a)):
		return(np.zeros(a.ndim))
	offset = np.array([s//2 for s in a.shape]) - np.unravel_index(np.nanargmax(a), a.shape)
	return(offset)


def perform_psf_checks(psf_data, psf_slice):
	_lgr.DEBUG(f'{psf_data.shape=}')
	psf_data_img_oddness = np.array([_s%2 for _s in psf_data.shape[len(psf_slice):]])
	_lgr.DEBUG(f'{psf_data_img_oddness=}')
	if not np.all(psf_data_img_oddness):
		_lgr.WARN(f'PSF spatial image is not ODDxODD, using naive method (slice off last row/column) to fix...')
		aslice = tuple(slice(None) if _i < len(psf_slice) else slice(None,_s-(_s%2 + 1)) for _i, _s in enumerate(psf_data.shape))
		_lgr.DEBUG(f'{aslice=}')
		psf_data = psf_data[aslice]
		_lgr.DEBUG(f'{psf_data.shape=}')

	psf_center_offset = np.array([get_center_offset_brightest_pixel(x) for x in psf_data], dtype=int)
	_lgr.DEBUG(f'{psf_center_offset=}')
	if np.any(psf_center_offset != 0):
		_lgr.WARN(f'Brightest pixel of PSF {np.unravel_index(np.nanargmax(psf_data), psf_data.shape)} is not center pixel {np.array([s//2 for s in psf_data.shape])}. PSF has shape {psf_data.shape}, re-centering naively...')
		psf_data = np.array([np.roll(_x, _offset, tuple(range(_offset.size))) for _x, _offset in zip(psf_data, psf_center_offset)]) 
	return(psf_data)


def main(args):
	print(f'# ARGUMENTS #')
	for k, v in args.items():
		print(f'\t{k}\n\t\t{v}')
	print(f'# END ARGUMENTS #')

	# create class that can deconvolve our images

	deconvolver = CleanModified(
		**(ut.args.with_prefix(args, 'CleanModified'))
	)

	with fits.open(args['obs.file']) as obs_hdul, fits.open(args['psf.file']) as psf_hdul:
		set_default_args_from_fits_hdul(args, obs_hdul, prefix='obs.file.')
		set_default_args_from_fits_hdul(args, psf_hdul, prefix='psf.file.')
	
		deconv_components_raw = np.full_like(obs_hdul[args['obs.file.ext']].data, fill_value=np.nan)	
		deconv_residual_raw = np.full_like(obs_hdul[args['obs.file.ext']].data, fill_value=np.nan)	

		#for k,v in list(ut.args.with_prefix(args, 'obs.file').items()) + list(ut.args.with_prefix(args, 'psf.file').items()):
		#	print(f'{k}\n\t{v}')
		
		with (	ArrayAxesToEnd(obs_hdul[args['obs.file.ext']].data, args['obs.file.img_axes']) as obs_data,
				ArrayAxesToEnd(psf_hdul[args['psf.file.ext']].data, args['psf.file.img_axes']) as psf_data,
				ArrayAxesToEnd(deconv_components_raw, args['obs.file.img_axes']) as deconv_components,
				ArrayAxesToEnd(deconv_residual_raw, args['obs.file.img_axes']) as deconv_residual,
			):

			# peform checks on obs_data and psf_data here
			psf_data = perform_psf_checks(psf_data, args['psf.file.slice'])
			obs_data, psf_data = simple_bad_pixel_check(obs_data, psf_data)

			for i, j in zip(IndicesOfSlice(obs_data, args['obs.file.slice']), IndicesOfSlice(psf_data, args['psf.file.slice'])):
				_lgr.INFO(f'obs index {i} psf index {j}')
				deconvolver(obs_data[i], psf_data[j])
				deconv_components[i,...] = deconvolver.get_components()
				deconv_residual[i,...] = deconvolver.get_residual()


		hdr = obs_hdul[args['obs.file.ext']].header
		hdr.update(DictToFitsHeaderIter(args))
		
	hdu_components = fits.PrimaryHDU(
		header = hdr,
		data = deconv_components_raw
	)
	hdu_residual = fits.ImageHDU(
		header = hdr,
		data = deconv_residual_raw,
		name = 'RESIDUAL'
	)
	hdul_output = fits.HDUList([
		hdu_components,
		hdu_residual
	])
	hdul_output.writeto(args['output.file'], overwrite=args['output.file.overwrite'])

	_lgr.INFO(f'INFO: Written deconvolved cube to\n\t{args["output.file"]}\nOnly applied to channels in the union of the following ranges:\n\t{args["obs.file.slice"]} and {args["psf.file.slice"]}')
	_lgr.log_warned()



if __name__=='__main__':
	args = parse_args(sys.argv)
	main(args)
	
