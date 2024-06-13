"""
Given a badness map, will apply a value cut (or possibly a range of interpolated value cuts) to the badness map to give a boolean mask that defines "bad" pixels
"""

import sys, os
from pathlib import Path
import dataclasses as dc
from typing import Literal, Any, Type, Callable
from collections import namedtuple

import numpy as np
import scipy as sp
import scipy.ndimage
from astropy.io import fits

import matplotlib.pyplot as plt

import aopp_deconv_tool.astropy_helper as aph
import aopp_deconv_tool.astropy_helper.fits.specifier
import aopp_deconv_tool.astropy_helper.fits.header

import aopp_deconv_tool.numpy_helper as nph
import aopp_deconv_tool.numpy_helper.axes
import aopp_deconv_tool.numpy_helper.slice

from aopp_deconv_tool.algorithm.bad_pixels.ssa_sum_prob import get_bp_mask_from_badness_map

from aopp_deconv_tool.py_ssa import SSA

import aopp_deconv_tool.cfg.logs
_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'DEBUG')



def run(
		fits_spec,
		output_path,
		index_cut_values : list[list[float,float],...] | None = None,
	):
	
	if index_cut_values is None:
		index_cut_values = [[0,4.5]]
	
	
	with fits.open(Path(fits_spec.path)) as data_hdul:
		
		
		_lgr.debug(f'{fits_spec.path=} {fits_spec.ext=} {fits_spec.slices=} {fits_spec.axes=}')
		#raise RuntimeError(f'DEBUGGING')
	
		data_hdu = data_hdul[fits_spec.ext]
		data = data_hdu.data
		
		bad_pixel_mask = np.zeros_like(data, dtype=bool)
		
		cv_pos = 0
		next_cv_pos = 1
		

		# Loop over the index range specified by `obs_fits_spec` and `psf_fits_spec`
		for i, idx in enumerate(nph.slice.iter_indices(data, fits_spec.slices, fits_spec.axes['CELESTIAL'])):
			_lgr.debug(f'{i=}')
			current_data_idx = idx[0][tuple(0 for i in fits_spec.axes['CELESTIAL'])]
			
			
			_lgr.debug(f'{current_data_idx=}')
			
			
			
			while next_cv_pos < len(index_cut_values) and index_cut_values[next_cv_pos][0] < current_data_idx:
				next_cv_pos += 1
			cv_pos = next_cv_pos -1
			
			if next_cv_pos < len(index_cut_values):
				lo_cv_idx = index_cut_values[cv_pos][0]
				hi_cv_idx = index_cut_values[next_cv_pos][0]
				
				lo_cv_value = index_cut_values[cv_pos][1]
				hi_cv_value = index_cut_values[next_cv_pos][1]
				_lgr.debug(f'{lo_cv_idx=} {hi_cv_idx=} {lo_cv_value=} {hi_cv_value=}')
				cv_value = (current_data_idx-lo_cv_idx)*(hi_cv_value - lo_cv_value)/(hi_cv_idx - lo_cv_idx) + lo_cv_value
			else:
				cv_value = index_cut_values[cv_pos][1]
			
			_lgr.debug(f'{cv_value=}')
			
			
			
			# Large "badness values" should have a larger AOE than smaller "badness values"
			# Therefore, dilate according to pixel value, for every 1 larger than the
			# cut value, dilate the pixel once more.
			
			#bp_mask = np.zeros(data[idx].shape, dtype=bool)
			_lgr.debug(f'{(int(np.floor(np.nanmax(data[idx]))), int(np.ceil(cv_value+1)))=}')
			for t in range(int(np.floor(np.nanmax(data[idx]))), int(np.ceil(cv_value)), -1):
				_lgr.debug(f'{t=}')
				diff = t - np.ceil(cv_value)
				_lgr.debug(f'{diff=}')
				#plt.imshow(data[idx] >= t)
				#plt.show()
				bad_pixel_mask[idx] |= sp.ndimage.binary_dilation(data[idx] >= t, iterations = int(diff))
			
			bad_pixel_mask[idx] |= data[idx] >= cv_value
			#bp_mask = data[idx] >= cv_value
			
	
	
		hdr = data_hdu.header
		param_dict = {
			'original_file' : Path(fits_spec.path).name, # record the file we used
			#**dict((f'cut_value_of_index_{k}', v) for k,v in index_cut_values)
		}
		#for i, x in enumerate(bad_pixel_map_binary_operations):
		#	param_dict[f'bad_pixel_map_binary_operations_{i}'] = x
		
		hdr.update(aph.fits.header.DictReader(
			param_dict,
			prefix='artifact_detection',
			pkey_count_start=aph.fits.header.DictReader.find_max_pkey_n(hdr)
		))
				

	
	
	# Save the products to a FITS file
	hdu_bad_pixel_mask = fits.PrimaryHDU(
		header = hdr,
		data = bad_pixel_mask.astype(int)
	)
	hdu_cut_value_table = fits.BinTableHDU.from_columns(
		columns = [
			fits.Column(name='cut_index', format='I', array=[x[0] for x in index_cut_values]), 
			fits.Column(name=f'cut_value', format='D', array=[x[1] for x in index_cut_values])
		],
		name = 'CUT_VALUE_BY_INDEX',
		header = None,
	)
	
	hdul_output = fits.HDUList([
		hdu_bad_pixel_mask,
		hdu_cut_value_table
	])
	hdul_output.writeto(output_path, overwrite=True)
	
	


def parse_args(argv):
	import aopp_deconv_tool.text
	import argparse
	
	DEFAULT_OUTPUT_TAG = '_bpmask'
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
			f'FITS Specifier of the badness map to operate upon . See the end of the help message for more information',
			f'required axes: {", ".join(DESIRED_FITS_AXES)}',
		)),
		type=str,
		metavar='FITS Specifier',
	)
	parser.add_argument('-o', '--output_path', help=f'Output fits file path. By default is same as the `fits_spec` path with "{DEFAULT_OUTPUT_TAG}" appended to the filename')
	
	parser.add_argument('-x', '--value_cut_at_index', metavar='int float', type=float, nargs=2, action='append', default=[], help='[index, value] pair, for a 3D badness map `index` will be cut at `value`. Specify multiple times for multiple indices. The `value` for non-specified indices is interpolated with "extend" boundary conditions.')
	
	args = parser.parse_args(argv)
	
	args.fits_spec = aph.fits.specifier.parse(args.fits_spec, DESIRED_FITS_AXES)
	
	if args.output_path is None:
		args.output_path =  (Path(args.fits_spec.path).parent / (str(Path(args.fits_spec.path).stem)+DEFAULT_OUTPUT_TAG+str(Path(args.fits_spec.path).suffix)))
	
	if len(args.value_cut_at_index) == 0:
		args.value_cut_at_index = [[0,3]]
	for i in range(len(args.value_cut_at_index)):
		args.value_cut_at_index[i][0] = int(args.value_cut_at_index[i][0])
	
	return args


if __name__ == '__main__':
	args = parse_args(sys.argv[1:])
	
	_lgr.debug(f'{vars(args)=}')
	
	run(
		args.fits_spec, 
		output_path=args.output_path, 
		index_cut_values = args.value_cut_at_index
	)
	