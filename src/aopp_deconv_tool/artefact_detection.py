"""
Detects artefacts, returns a "badness map" that represents how much the algorithm thinks a particular pixel is an artefact.
"""
import sys, os
from pathlib import Path
import dataclasses as dc
from typing import Literal, Any, Type, Callable
from collections import namedtuple

import numpy as np
import scipy as sp
from astropy.io import fits

# TESTING
import skimage as ski

import matplotlib.pyplot as plt

from aopp_deconv_tool.algorithm.artefact_detection import difference_of_scale_filters, wavelet_decomposition

import aopp_deconv_tool.astropy_helper as aph
import aopp_deconv_tool.astropy_helper.fits.specifier
import aopp_deconv_tool.astropy_helper.fits.header
from aopp_deconv_tool.fpath import FPath
import aopp_deconv_tool.numpy_helper as nph
import aopp_deconv_tool.numpy_helper.axes
import aopp_deconv_tool.numpy_helper.slice
import aopp_deconv_tool.arguments

from aopp_deconv_tool.algorithm.bad_pixels.ssa_sum_prob import ssa2d_sum_prob_map, ssa2d_deviations

from aopp_deconv_tool.py_ssa import SSA

from aopp_deconv_tool.image_processing import otsu_thresholding
from aopp_deconv_tool import algorithm
import aopp_deconv_tool.algorithm.interpolate

import aopp_deconv_tool.cfg.logs
_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'WARN')


# First one of these is the default
artefact_detection_strategies= dict(
	ssa = {
		'description' : 'Uses singular spectrum analysis (SSA) to deterimine how likely a pixel is to belong to an artefact.',
		#'target' : ssa2d_sum_prob_map,
		'target' : ssa2d_deviations,
	},
	dummy = {
		'description' : 'Is a dummy option to be used when testing',
		'target' : lambda *a, **k: None,
	}
)

artefact_detection_strategy_choices = [x for x in artefact_detection_strategies]
artefact_detection_strategy_choices_help_str = '\n\t'+'\n\t'.join(f'{k}\n\t\t{v["description"]}' for k,v in artefact_detection_strategies.items())


def generate_masks_from_thresholds(data, thresholds):
	"""
	Generator that returns multiple masks when passed an iterable of numbers
	that can be used as thresholds. The first mask will be everything <= the first
	threshold; the last mask will be everything > the last threshold; intermediate
	masks will be everything in between neighbouring threshold values.
	"""
	thresholds = np.sort(thresholds)
	_lgr.debug(f'{thresholds=}')
	for i in range(0,len(thresholds)+1):
		if i==0:
			mask = data <= thresholds[i]
		elif i == len(thresholds):
			mask = data > thresholds[i-1]
		else:
			mask = (thresholds[i-1] < data) & (data <= thresholds[i])
		yield mask

def to_unit_range(data):
	"""
	Convert `data` to a unit range, will return the `converted_data`, `offset`, and `range` used in the conversion.
	"""
	offset = np.min(data)
	range = np.max(data) - offset
	return (data-offset)/range, offset, range

def undo_unit_range(data, offset, range):
	"""
	Given some `data` (between 0,1), an `offset` (from zero) and a `range` (max-min). Will undo a conversion to unit range.
	"""
	return (data * range) + offset

def to_dtype_range(data, dtype=np.uint16):
	"""
	Rescale data so it fits into the range of the specified `dtype`, default is `dtype=uint16`.
	"""
	final_range = np.iinfo(dtype).max
	offset = np.min(data)
	range = np.max(data) - offset
	return ((data-offset)*(final_range/range)).astype(np.uint16), offset, range, final_range

def undo_range(data, offset, range, final_range):
	"""
	Undo the coversion from one range of data to another.
	"""
	return (data.astype(float)/final_range)*range + offset

def get_ski_filter(ski_filter, undo_scaling=True, dtype=np.uint16):
	"""
	When given a scikit image filter, return a new function that will implement
	that filter on some data in a generic format, not just the one the scikit image
	filter wants.
	"""
	def new_filter(data, scale, *args, **kwargs):
		if scale == 0:
			return data
		d, o, r, rr = to_dtype_range(data, dtype)
		#plt.title('d')
		#plt.imshow(d)
		#plt.show()
		z = ski_filter(d, np.ones((scale,scale)), *args, **kwargs)
		#plt.title('z')
		#plt.imshow(z)
		#plt.show()
		if undo_scaling:
			z = undo_range(z, o, r, rr)
		return z.astype(data.dtype)
	return new_filter

def run(
		fits_spec,
		output_path,
		strategy : str,
		**kwargs : dict[str : Any]
	) -> None:
	"""
	Perform the operation associated with this module.
	"""
	
	_lgr.debug(f'{kwargs=}')
	
	strategy_callable = artefact_detection_strategies[strategy]['target']
	
	kwargs['start'] = int(kwargs['start']) if kwargs['start'] >= 0 else int(-1*kwargs['start']*kwargs['w_shape']**2)
	kwargs['stop'] = int(kwargs['stop']) if kwargs['stop'] >= 0 else int(-1*kwargs['stop']*kwargs['w_shape']**2)
	
	with fits.open(Path(fits_spec.path)) as data_hdul:
		
		
		_lgr.debug(f'{fits_spec.path=} {fits_spec.ext=} {fits_spec.slices=} {fits_spec.axes=}')
		#raise RuntimeError(f'DEBUGGING')
	
		data_hdu = data_hdul[fits_spec.ext]
		data = data_hdu.data
		
		# Allocate and set to all zeros
		badness_map = np.zeros_like(data, dtype=float)

		# Loop over the index range specified by `obs_fits_spec` and `psf_fits_spec`
		for i, idx in enumerate(nph.slice.iter_indices(data, fits_spec.slices, fits_spec.axes['CELESTIAL'])):
			_lgr.debug(f'{i=}')
			
			# Ignore data that is 99% or more NAN values
			data_nan_frac = np.count_nonzero(np.isnan(data[idx]))/data[idx].size
			_lgr.debug(f'{data_nan_frac=}')
			if data_nan_frac > 0.99:
				continue 
			
			data[idx] = algorithm.interpolate.quick_remove_nan_and_inf(data[idx])
			
			
			ssa = SSA(
				data[idx],
				w_shape = kwargs['w_shape'],
				grouping = {'mode':'elementary'}
				#grouping = {'mode':'similar_eigenvalues', 'tolerance' : 2E-1}
			)
			
			# TESTING
			#ssa.plot_ssa([0,1,2,3])
			#ssa.plot_component_slices([(0,5),(5,10),(10,20),(20,25),(25,45),(45,65),(65,75),(75,80),(80,85),(85,90),(90,95),(95,None)])
			#plt.savefig('uranus_ssa_test.png')
			#plt.show()
			#sys.exit()
			
			
			"""
			# Playing around with wavelet decomposition
			#w_data = data[idx]
			w_data = ssa2d_deviations(ssa, 25,100)
			wavelet_planes = wavelet_decomposition(w_data, 0, None, sp.ndimage.gaussian_filter, 0, False)
			#wavelet_planes = wavelet_decomposition(w_data, 1, 7, lambda x,s: sp.ndimage.rank_filter(x,s//2,s), 1, True)
			#wavelet_planes = wavelet_decomposition(w_data, 1, None, sp.ndimage.maximum_filter, 1, True)
			#wavelet_planes = wavelet_decomposition(w_data, 1, None, sp.ndimage.minimum_filter, 1, True)
			#wavelet_planes = wavelet_decomposition(w_data, 1, 4, sp.ndimage.median_filter, 1, True)
			#wavelet_planes = wavelet_decomposition(w_data, 1, None, sp.ndimage.uniform_filter, 1, True)
			
			plt.clf()
			f = plt.gcf()
			f.suptitle('wavelets components')
			n_wavelet_plots = wavelet_planes.shape[0]
			n_plots = n_wavelet_plots + 2
			nr = int(np.floor(np.sqrt(n_plots)))
			nc = int(np.ceil(n_plots / nr))
			ax = f.subplots(nr, nc).flatten()
			for i in range(n_plots):
				if i < n_wavelet_plots:
					wp = wavelet_planes[i]
					ax[i].set_title(f'wavelet component {i}\n[{np.nanmin(wp):06.2E}, {np.nanmax(wp):06.2E}]')
					ax[i].imshow(wp)
					ax[i].set_axis_off()
				elif i == n_wavelet_plots + 0:
					wp = np.sum(wavelet_planes, axis=0)
					ax[i].set_title(f'sum\n[{np.nanmin(wp):06.2E}, {np.nanmax(wp):06.2E}]')
					ax[i].imshow(wp)
					ax[i].set_axis_off()
				elif i == n_wavelet_plots + 1:
					wp = w_data - np.sum(wavelet_planes, axis=0)
					ax[i].set_title(f'data - sum\n[{np.nanmin(wp):06.2E}, {np.nanmax(wp):06.2E}]')
					ax[i].imshow(wp)
					ax[i].set_axis_off()
				# elif i == n_wavelet_plots + 2:
				# 	wp = np.std(wavelet_planes, axis=(1,2))
				# 	ax[i].set_title(f'sum over wavelet components')
				# 	ax[i].plot(wp)
				# 	ax[i].set_xlabel('wavelet component')
				# 	ax[i].set_yscale('log')
				else:
					raise RuntimeError('This should never happen')
			plt.show()
			"""
			
			
			
			
			"""
			j_count = 0
			s1,s2 = (5,75)
			plt.figure()
			plt.imshow(np.sum(ssa.X_ssa[s1:s2],axis=0))
			plt.figure()
			plt.imshow(data[idx] - np.sum(ssa.X_ssa[s1:s2],axis=0))
			plt.show()
			for j in range(s1,s2):
				_lgr.debug(f'{j=}')
				j_count += 1
				#badness_map[idx] += ssa.X_ssa[j]
				#temp = data[idx]
				temp = ssa.X_ssa[j]
				#temp, toff, tr= to_unit_range(ssa.X_ssa[j])
				
				s = 2000
				scale = 1
				ski_filter = get_ski_filter(ski.filters.rank.median, undo_scaling=True, dtype=np.uint8)
				#ski_filter = get_ski_filter(ski.filters.rank.entropy, undo_scaling=False, dtype=np.uint8)
				#r = temp
				#r = ski.filters.difference_of_gaussians(temp, 0, (scale+1)**2)
				#r = temp - sp.ndimage.uniform_filter(temp, size=scale)
				r = difference_of_scale_filters(temp, 0, scale, sp.ndimage.gaussian_filter)
				#r = difference_of_scale_filters(temp, 0, scale, ski_filter)
				#r = ski_filter(temp, scale)
				r_mean = sp.ndimage.uniform_filter(r, size=s)
				r_std = np.sqrt(sp.ndimage.uniform_filter((r - r_mean)**2, size=s))
				z = (r-r_mean)/r_std
				badness_map[idx] += z
				
				#plt.imshow(r)
				#plt.show()
				#break
			#badness_map[idx] = np.fabs(badness_map[idx]/50)
			badness_map[idx] = np.fabs(badness_map[idx]/j_count)
			continue # TESTING
			"""
			
			
			
						
			# Perform artefact detection on "background", "midground" and "foreground" separately
			thresholds = otsu_thresholding.n_exact(data[idx], 2, max_elements=10000)
			
			#for mask in generate_masks_from_thresholds(data[idx], ots):
			for mask in generate_masks_from_thresholds(data[idx], thresholds):
				
				bm = strategy_callable(
					ssa, 
					start=kwargs['start'], 
					stop=kwargs['stop'],
					mask=mask,
					size=max(data[idx].shape)//4,
				)
				
				x = np.full_like(bm, fill_value=np.nan)
				x[mask] = bm[mask]
				#plt.imshow(x)
				#plt.show()
				
				badness_map[idx] += bm
				
			
	
		hdr = data_hdu.header
		param_dict = {
			'original_file' : Path(fits_spec.path).name, # record the file we used
			'strategy' : strategy,
			**dict(('.'.join((strategy,k)),v) for k,v in kwargs.items())
		}
		#for i, x in enumerate(bad_pixel_map_binary_operations):
		#	param_dict[f'bad_pixel_map_binary_operations_{i}'] = x
		
		hdr.update(aph.fits.header.DictReader(
			param_dict,
			prefix='artefact_detection',
			pkey_count_start=aph.fits.header.DictReader.find_max_pkey_n(hdr)
		))
				

	
	# Save the products to a FITS file
	hdu_badness = fits.PrimaryHDU(
		header = hdr,
		data = badness_map
	)
	hdul_output = fits.HDUList([
		hdu_badness,
	])
	hdul_output.writeto(output_path, overwrite=True)



def parse_args(argv):
	"""
	Read command-line arguments when this module is called as a script.
	"""
	
	import aopp_deconv_tool.text
	import argparse
	
	DEFAULT_OUTPUT_TAG = '_artefactmap'
	DESIRED_FITS_AXES = ['CELESTIAL']
	OUTPUT_COLUMNS=80
	try:
		OUTPUT_COLUMNS = os.get_terminal_size().columns - 30
	except Exception:
		pass
	
	FITS_SPECIFIER_HELP = aopp_deconv_tool.text.wrap(
		aph.fits.specifier.get_help(DESIRED_FITS_AXES).replace('\t', '    '),
		OUTPUT_COLUMNS
	)
	
	parser = argparse.ArgumentParser(
		description=__doc__, 
		formatter_class=argparse.RawTextHelpFormatter,
		epilog=FITS_SPECIFIER_HELP
	)
	
	parser.add_argument(
		'fits_spec',
		help = '\n'.join((
			f'FITS Specifier of the data to operate upon . See the end of the help message for more information',
			f'required axes: {", ".join(DESIRED_FITS_AXES)}',
		)),
		type=str,
		metavar='FITS Specifier',
	)
	#parser.add_argument('-o', '--output_path', help=f'Output fits file path. By default is same as fie `fits_spec` path with "{DEFAULT_OUTPUT_TAG}" appended to the filename')
	parser.add_argument(
		'-o', 
		'--output_path', 
		type=FPath,
		metavar='str',
		default='{parent}/{stem}{tag}{suffix}',
		help = '\n    '.join((
			f'Output fits file path, supports keyword substitution using parts of `fits_spec` path where:',
			'{parent}: containing folder',
			'{stem}  : filename (not including final extension)',
			f'{{tag}}   : script specific tag, "{DEFAULT_OUTPUT_TAG}" in this case',
			'{suffix}: final extension (everything after the last ".")',
			'\b'
		))
	)
	
	
	
	parser.add_argument('--strategy', choices=artefact_detection_strategy_choices, default=artefact_detection_strategy_choices[0], help=f'Strategy to use when detecting artefacts {artefact_detection_strategy_choices_help_str}')

	ssa_group = parser.add_argument_group('ssa artefact detection', 'options for singular spectrum analysis (SSA) argument detection strategy')
	ssa_group.add_argument('--ssa.w_shape', type=int, default=10, help='Window size to calculate SSA for. Will generate `w_shape`^2 SSA components')
	ssa_group.add_argument('--ssa.start', type=float, default=-0.25, help='First SSA component to be included in artefact detection calc. Negative numbers are fractions of range')
	ssa_group.add_argument('--ssa.stop',  type=float, default=-0.75, help='Last SSA component to be included in artefact detection calc. Negative numbers are fractions of range')

	args = parser.parse_args(argv)
	
	args.fits_spec = aph.fits.specifier.parse(args.fits_spec, DESIRED_FITS_AXES)
	
	#if args.output_path is None:
	#	args.output_path =  (Path(args.fits_spec.path).parent / (str(Path(args.fits_spec.path).stem)+DEFAULT_OUTPUT_TAG+str(Path(args.fits_spec.path).suffix)))
	other_file_path = Path(args.fits_spec.path)
	args.output_path = args.output_path.with_fields(
		tag=DEFAULT_OUTPUT_TAG, 
		parent=other_file_path.parent, 
		stem=other_file_path.stem, 
		suffix=other_file_path.suffix
	)
	
	args.strategy_args = dict((k[len(args.strategy)+1:],v) for k,v in vars(args).items() if k.startswith(args.strategy))	
	
	
	return args


def go(
		fits_spec,
		output_path=None,
		strategy=None,
		**kwargs
	):
	"""
	Thin wrapper around `run()` to accept string inputs.
	As long as the names of the arguments to this function 
	are the same as the names expected from the command line
	we can do this programatically
	"""
	# Add stuff to kwargs here if needed
	kwargs['ssa.w_shape'] = None
	kwargs['ssa.start'] = None
	kwargs['ssa.stop'] = None
	
	# This must be first so we only grab the arguments to the function
	fargs = dict(locals().items())
	arglist = aopp_deconv_tool.arguments.construct_arglist_from_locals(fargs, n_positional_args=1)

	exec_with_args(arglist)

def exec_with_args(argv):
	"""
	Read arguments and run the module
	"""
	args = parse_args(argv)
	
	run(
		args.fits_spec, 
		output_path=args.output_path, 
		strategy=args.strategy, 
		**args.strategy_args
	)

if __name__ == '__main__':
	exec_with_args(sys.argv[1:])