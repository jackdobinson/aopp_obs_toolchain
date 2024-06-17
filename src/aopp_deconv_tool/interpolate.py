"""
Quick tool for interpolating data in a FITS file
"""


import sys
from pathlib import Path
import dataclasses as dc
from typing import Literal, Callable
import functools

import numpy as np
import scipy as sp
from astropy.io import fits

from aopp_deconv_tool.task_strat import TaskStratInfo, TaskStratSet

import aopp_deconv_tool.astropy_helper as aph
import aopp_deconv_tool.astropy_helper.fits.specifier
from aopp_deconv_tool.astropy_helper.fits.specifier import FitsSpecifier
import aopp_deconv_tool.astropy_helper.fits.header
import aopp_deconv_tool.numpy_helper as nph
import aopp_deconv_tool.numpy_helper.axes
import aopp_deconv_tool.numpy_helper.slice

import aopp_deconv_tool.numpy_helper.array.grid

import aopp_deconv_tool.scipy_helper as sph
import aopp_deconv_tool.scipy_helper.interp
import aopp_deconv_tool.scipy_helper.label_ops

from aopp_deconv_tool.algorithm.interpolate.ssa_interp import ssa_interpolate_at_mask, ssa_deviations_interpolate_at_mask
from aopp_deconv_tool.algorithm.bad_pixels.ssa_sum_prob import ssa2d_sum_prob_map

from aopp_deconv_tool.py_ssa import SSA

import matplotlib.pyplot as plt

import aopp_deconv_tool.cfg.logs
_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'DEBUG')


def array_ssa_deviations_interpolate_at_mask(a : np.ndarray, mask : np.ndarray, **kwargs):
	return ssa_deviations_interpolate_at_mask(SSA(a, **kwargs), mask)

def array_ssa_interpolate_at_mask(a : np.ndarray, mask : np.ndarray, **kwargs):
	return ssa_interpolate_at_mask(a, SSA(a, **kwargs), mask)


@dc.dataclass
class InterpolationStratInfo (TaskStratInfo):
	callable : Callable[[np.ndarray['NM',float], np.ndarray['NM',bool]], np.ndarray['NM',float]]

interpolation_strategies = TaskStratSet('Performs interpolation upon an array at positions provided by a mask').add(
	InterpolationStratInfo(
		'scipy',
		'Uses scipy routies to interpolate over the masked regions, edge effects are minimised by convolving with a 3x3 square before interpolation',
		functools.partial(sph.interp.interpolate_at_mask, edges='convolution')
	),
	InterpolationStratInfo(
		'ssa',
		'[EXPERIMENTAL] Calculates singular spectrum analysis (SSA) components, and uses the first 25% of them to fill in masked regions.',
		functools.partial(array_ssa_interpolate_at_mask, w_shape=10, grouping = {'mode':'elementary'}),
	),
	InterpolationStratInfo(
		'ssa_deviation',
		'[EXPERIMENTAL] Within a masked region, replaces pixels of singular spectrum analysis (SSA) components with high deviations with median values, which are summed and replaced into the masked region on the original image.',
		functools.partial(array_ssa_deviations_interpolate_at_mask, w_shape=10, grouping = {'mode':'elementary'}),
	)
)




	




def run(
		data_fits_spec : FitsSpecifier,
		bpmask_fits_spec : FitsSpecifier,
		output_path : Path,
		interp_method : str = 'scipy',
	):
	
	
	
	with fits.open(Path(data_fits_spec.path)) as data_hdul, fits.open(Path(bpmask_fits_spec.path)) as bpmask_hdul:
		
		
		_lgr.debug(f'{data_fits_spec}')
		_lgr.debug(f'{bpmask_fits_spec}')
		
		
		
		#raise RuntimeError(f'DEBUGGING')
	
		data_hdu = data_hdul[data_fits_spec.ext]
		data = data_hdu.data
		
		bpmask_hdu = bpmask_hdul[bpmask_fits_spec.ext]
		if issubclass(bpmask_hdu.data.dtype.type, np.integer):
			bpmask = bpmask_hdu.data.astype(bool)
		elif issubclass(bpmask_hdu.data.dtype.type, np.bool_):
			bpmask = bpmask_hdu.data
		else:
			raise RuntimeError(f'Bad Pixel Mask "{bpmask_fits_spec.path}" is not a boolean mask. It is of type "{bpmask_hdu.data.dtype}", only integer types can be converted to a boolean mask.')
		
		# Ensure that the data and mask are compatible shapes
		data_shape = data[data_fits_spec.slices].shape
		bpmask_shape = bpmask[bpmask_fits_spec.slices].shape
		if (len(data_shape) != len(bpmask_shape)) or any(s1!=s2 for s1,s2 in zip(data_shape, bpmask_shape)):
			raise RuntimeError(f'data and bad pixel mask are of incompatible shape {data_shape} vs {bpmask_shape} with {data_fits_spec=} {bpmask_fits_spec=}')
			
		# Allocate array to hold interpolated data
		interp_data = np.full_like(data, fill_value=np.nan)
		residual = np.full_like(data, fill_value=np.nan)
		
		# Loop over the index range specified by `obs_fits_spec` and `psf_fits_spec`
		for i, idx in enumerate(nph.slice.iter_indices(data, data_fits_spec.slices, data_fits_spec.axes['CELESTIAL'])):
			_lgr.debug(f'{i=} current_idx={idx[0][tuple(0 for i in data_fits_spec.axes["CELESTIAL"])]}')
			
			interp_data[idx] = interpolation_strategies(interp_method, data[idx], bpmask[idx] | np.isnan(data[idx]) | np.isinf(data[idx]))
			residual[idx] = data[idx] - interp_data[idx]
			
	
	
		hdr = data_hdu.header
		param_dict = {
			'original_file' : Path(data_fits_spec.path).name, # record the file we used
			'bpmask_file' : Path(bpmask_fits_spec.path).name, 
			'interp_method' : interp_method,
		}

		hdr.update(aph.fits.header.DictReader(
			param_dict,
			prefix='interpolate',
			pkey_count_start=aph.fits.header.DictReader.find_max_pkey_n(hdr)
		))
				

	
	# Save the products to a FITS file
	hdu_interp = fits.PrimaryHDU(
		header = hdr,
		data = interp_data
	)
	hdu_residual = fits.ImageHDU(
		header = hdr,
		data = residual,
		name='RESIDUAL'
	)
	hdul_output = fits.HDUList([
		hdu_interp,
		hdu_residual
	])
	hdul_output.writeto(output_path, overwrite=True)
	

def parse_args(argv):
	import os
	import aopp_deconv_tool.text
	import argparse
	
	DEFAULT_OUTPUT_TAG = '_interp'
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
		'data_fits_spec',
		help = '\n'.join((
			f'FITS Specifier of the data to operate upon . See the end of the help message for more information',
			f'required axes: {", ".join(DESIRED_FITS_AXES)}',
		)),
		type=str,
		metavar='FITS Specifier',
	)
	parser.add_argument(
		'bpmask_fits_spec',
		help = '\n'.join((
			f'FITS Specifier of the data to operate upon. See the end of the help message for more information',
			f'required axes: {", ".join(DESIRED_FITS_AXES)}',
		)),
		type=str,
		metavar='FITS Specifier',
	)
	
	parser.add_argument('-o', '--output_path', help=f'Output fits file path. By default is same as the `data_fits_spec` path with "{DEFAULT_OUTPUT_TAG}" appended to the filename')
	parser.add_argument('--interp_method', choices=interpolation_strategies.names, default=interpolation_strategies.names[0], help=interpolation_strategies.format_description())


	args = parser.parse_args(argv)
	
	args.data_fits_spec = aph.fits.specifier.parse(args.data_fits_spec, DESIRED_FITS_AXES)
	args.bpmask_fits_spec = aph.fits.specifier.parse(args.bpmask_fits_spec, DESIRED_FITS_AXES)
	
	if args.output_path is None:
		fpath = Path(args.data_fits_spec.path)
		args.output_path =  (fpath.parent / (str(fpath.stem)+DEFAULT_OUTPUT_TAG+str(fpath.suffix)))
	
	return args


if __name__ == '__main__':
	args = parse_args(sys.argv[1:])
	
	run(
		args.data_fits_spec, 
		args.bpmask_fits_spec,
		output_path=args.output_path,
		interp_method=args.interp_method, 
	)
	
