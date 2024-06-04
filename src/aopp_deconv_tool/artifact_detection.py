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

import aopp_deconv_tool.astropy_helper as aph
import aopp_deconv_tool.astropy_helper.fits.specifier
import aopp_deconv_tool.astropy_helper.fits.header

import aopp_deconv_tool.numpy_helper as nph
import aopp_deconv_tool.numpy_helper.axes
import aopp_deconv_tool.numpy_helper.slice

from aopp_deconv_tool.algorithm.bad_pixels.ssa_sum_prob import ssa2d_sum_prob_map

from aopp_deconv_tool.py_ssa import SSA

import aopp_deconv_tool.cfg.logs
_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'DEBUG')


# First one of these is the default
artifact_detection_strategies= dict(
	ssa = {
		'description' : 'Uses singular spectrum analysis (SSA) to deterimine how likely a pixel is to belong to an artifact',
		'target' : ssa2d_sum_prob_map,
	},
	dummy = {
		'description' : 'Is a dummy option to be used when testing',
		'target' : lambda *a, **k: None,
	}
)

artifact_detection_strategy_choices = [x for x in artifact_detection_strategies]
artifact_detection_strategy_choices_help_str = '\n\t'+'\n\t'.join(f'{k}\n\t\t{v["description"]}' for k,v in artifact_detection_strategies.items())


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
		
		badness_map = np.zeros_like(data, dtype=float)

		# Loop over the index range specified by `obs_fits_spec` and `psf_fits_spec`
		for i, idx in enumerate(nph.slice.iter_indices(data, fits_spec.slices, fits_spec.axes['CELESTIAL'])):
			_lgr.debug(f'{i=}')
			ssa = SSA(
				np.nan_to_num(data[idx]),
				w_shape = kwargs['w_shape'],
				grouping = {'mode':'elementary'}
			)
			
			badness_map[idx] = strategy_callable(
				ssa, 
				start=kwargs['start'], 
				stop=kwargs['stop']
			)
			
			
	
	
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
	