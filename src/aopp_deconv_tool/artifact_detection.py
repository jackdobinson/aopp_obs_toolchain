"""
Detects artifacts, returns a "badness map" that represents how much the algorithm thinks a particular pixel is an artifact.
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

from aopp_deconv_tool.algorithm.artifact_detection import difference_of_scale_filters

import aopp_deconv_tool.astropy_helper as aph
import aopp_deconv_tool.astropy_helper.fits.specifier
import aopp_deconv_tool.astropy_helper.fits.header

import aopp_deconv_tool.numpy_helper as nph
import aopp_deconv_tool.numpy_helper.axes
import aopp_deconv_tool.numpy_helper.slice

from aopp_deconv_tool.algorithm.bad_pixels.ssa_sum_prob import ssa2d_sum_prob_map, ssa2d_deviations

from aopp_deconv_tool.py_ssa import SSA

from aopp_deconv_tool.image_processing import otsu_thresholding
from aopp_deconv_tool import algorithm
import aopp_deconv_tool.algorithm.interpolate

import aopp_deconv_tool.cfg.logs
_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'DEBUG')


# First one of these is the default
artifact_detection_strategies= dict(
	ssa = {
		'description' : 'Uses singular spectrum analysis (SSA) to deterimine how likely a pixel is to belong to an artifact.',
		#'target' : ssa2d_sum_prob_map,
		'target' : ssa2d_deviations,
	},
	dummy = {
		'description' : 'Is a dummy option to be used when testing',
		'target' : lambda *a, **k: None,
	}
)

artifact_detection_strategy_choices = [x for x in artifact_detection_strategies]
artifact_detection_strategy_choices_help_str = '\n\t'+'\n\t'.join(f'{k}\n\t\t{v["description"]}' for k,v in artifact_detection_strategies.items())


def generate_masks_from_thresholds(data, thresholds):
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
	offset = np.min(data)
	range = np.max(data) - offset
	return (data-offset)/range, offset, range

def undo_unit_range(data, offset, range):
	return (data * range) + offset

def to_dtype_range(data, dtype=np.uint16):
	final_range = np.iinfo(dtype).max
	offset = np.min(data)
	range = np.max(data) - offset
	return ((data-offset)*(final_range/range)).astype(np.uint16), offset, range, final_range

def undo_range(data, offset, range, final_range):
	return (data.astype(float)/final_range)*range + offset

def get_ski_filter(ski_filter, undo_scaling=True, dtype=np.uint16):
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
	
	_lgr.debug(f'{kwargs=}')
	
	strategy_callable = artifact_detection_strategies[strategy]['target']
	
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
			#plt.imshow(data[idx])
			#plt.show()
			
			ssa = SSA(
				data[idx],
				w_shape = kwargs['w_shape'],
				grouping = {'mode':'elementary'}
			)
			
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
			
			
			
						
			# Perform artifact detection on "background", "midground" and "foreground" separately
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
			prefix='artifact_detection',
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
	import aopp_deconv_tool.text
	import argparse
	
	DEFAULT_OUTPUT_TAG = '_artifactmap'
	DESIRED_FITS_AXES = ['CELESTIAL']
	FITS_SPECIFIER_HELP = aopp_deconv_tool.text.wrap(
		aph.fits.specifier.get_help(DESIRED_FITS_AXES).replace('\t', '    '),
		os.get_terminal_size().columns - 30
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
	parser.add_argument('-o', '--output_path', help=f'Output fits file path. By default is same as fie `fits_spec` path with "{DEFAULT_OUTPUT_TAG}" appended to the filename')
	
	parser.add_argument('--strategy', choices=artifact_detection_strategy_choices, default=artifact_detection_strategy_choices[0], help=f'Strategy to use when detecting artifacts {artifact_detection_strategy_choices_help_str}')

	ssa_group = parser.add_argument_group('ssa artifact detection', 'options for singular spectrum analysis (SSA) argument detection strategy')
	ssa_group.add_argument('--ssa.w_shape', type=int, default=10, help='Window size to calculate SSA for. Will generate `w_shape`^2 SSA components')
	ssa_group.add_argument('--ssa.start', type=float, default=-0.25, help='First SSA component to be included in artifact detection calc. Negative numbers are fractions of range')
	ssa_group.add_argument('--ssa.stop',  type=float, default=-0.75, help='Last SSA component to be included in artifact detection calc. Negative numbers are fractions of range')

	args = parser.parse_args(argv)
	
	args.fits_spec = aph.fits.specifier.parse(args.fits_spec, DESIRED_FITS_AXES)
	
	if args.output_path is None:
		args.output_path =  (Path(args.fits_spec.path).parent / (str(Path(args.fits_spec.path).stem)+DEFAULT_OUTPUT_TAG+str(Path(args.fits_spec.path).suffix)))
	
	args.strategy_args = dict((k[len(args.strategy)+1:],v) for k,v in vars(args).items() if k.startswith(args.strategy))	
	
	
	return args


if __name__ == '__main__':
	args = parse_args(sys.argv[1:])
	
	run(args.fits_spec, output_path=args.output_path, strategy=args.strategy, **args.strategy_args)
	