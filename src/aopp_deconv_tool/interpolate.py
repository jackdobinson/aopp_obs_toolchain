"""
Quick tool for interpolating data in a FITS file
"""


import sys
from pathlib import Path

from typing import Literal

import numpy as np
from astropy.io import fits

import aopp_deconv_tool.astropy_helper as aph
import aopp_deconv_tool.astropy_helper.fits.specifier
import aopp_deconv_tool.astropy_helper.fits.header
import aopp_deconv_tool.numpy_helper as nph
import aopp_deconv_tool.numpy_helper.axes
import aopp_deconv_tool.numpy_helper.slice

import aopp_deconv_tool.numpy_helper.array.grid

import aopp_deconv_tool.scipy_helper as sph
import aopp_deconv_tool.scipy_helper.interp

from aopp_deconv_tool.algorithm.interpolate.ssa_interp import ssa_intepolate_at_mask
from aopp_deconv_tool.algorithm.bad_pixels.ssa_sum_prob import ssa2d_sum_prob_map

from aopp_deconv_tool.py_ssa import SSA

import matplotlib.pyplot as plt

import aopp_deconv_tool.cfg.logs
_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'DEBUG')



data_ssa = None

def get_data_ssa(a, **kwargs):
	global data_ssa
	if data_ssa is None:
		data_ssa = SSA(
			a, 
			**{
				'w_shape':20,
				#'grouping' : {'mode' : 'similar_eigenvalues', 'tolerance' : 1E-1},
				**kwargs
			},
		)
	return data_ssa

def set_data_ssa(value):
	global data_ssa
	data_ssa = value

def get_bad_pixel_map(a, method):
	nan_inf_mask = np.isinf(a) | np.isnan(a)
	match method:
		case 'ssa':
			ssa = get_data_ssa(np.nan_to_num(a))
			bad_pixel_map = ssa2d_sum_prob_map(ssa, value=3, start=0, stop=None, strategy='n_std_dev_from_median', show_plots=1, transform_value_as='identity')
		case 'simple':
			bad_pixel_map = nan_inf_mask
		case _:
			raise RuntimeError(f'Unknown value of {method=}')
	
	return bad_pixel_map | nan_inf_mask

def get_interp_at_mask(a, mask, method):
	match method:
		case 'ssa':
			interp = ssa_intepolate_at_mask(get_data_ssa(a), mask)
		case 'scipy':
			interp = sph.interp.interpolate_at_mask(a, mask, edges='convolution')
		case _:
			raise RuntimeError(f'Unknown value of {method=}')
	
	return interp
	


def run(
		fits_spec,
		output_path,
		bad_pixel_method = 'ssa',
		interp_method = 'ssa',
	):
	
	
	
	
	with fits.open(Path(fits_spec.path)) as data_hdul:
		
		
		_lgr.debug(f'{fits_spec.path=} {fits_spec.ext=} {fits_spec.slices=} {fits_spec.axes=}')
		#raise RuntimeError(f'DEBUGGING')
	
		data_hdu = data_hdul[fits_spec.ext]
		data = data_hdu.data
		
		bad_pixel_map = np.zeros_like(data, dtype=bool)
		interp_data = np.full_like(data, fill_value=np.nan)
		
		# Loop over the index range specified by `obs_fits_spec` and `psf_fits_spec`
		for idx in nph.slice.iter_indices(data, fits_spec.slices, fits_spec.axes['CELESTIAL']):
			_lgr.debug(f'{idx=}')
			set_data_ssa(None)
			
			plt.imshow(data[idx])
			plt.show()
			
			bad_pixel_map[idx] = get_bad_pixel_map(data[idx], bad_pixel_method)
			plt.imshow(bad_pixel_map[idx])
			plt.show()
			interp_data[idx] = get_interp_at_mask(data[idx], bad_pixel_map[idx], interp_method)
			plt.imshow(interp_data[idx])
			plt.show()
	
	
		hdr = data_hdu.header
		hdr.update(aph.fits.header.DictReader({
			'original_file' : Path(fits_spec.path).name, # record the file we used
			'bad_pixel_method' : bad_pixel_method,
			'interp_method' : interp_method,
		}))
		

	
	# Save the products to a FITS file
	hdu_interp = fits.PrimaryHDU(
		header = hdr,
		data = interp_data
	)
	hdu_bad_pixels = fits.ImageHDU(
		data = bad_pixel_map.astype(int),
		header = hdr,
		name='BAD_PIXELS'
	)
	hdul_output = fits.HDUList([
		hdu_interp,
		hdu_bad_pixels,
	])
	hdul_output.writeto(output_path, overwrite=True)
	

def parse_args(argv):
	import os
	import aopp_deconv_tool.text
	import argparse
	parser = argparse.ArgumentParser(
		description=__doc__, 
		formatter_class=argparse.RawTextHelpFormatter
	)
	
	parser.add_argument(
		'fits_spec', 
		help = aopp_deconv_tool.text.wrap(
			aph.fits.specifier.get_help(['CELESTIAL']).replace('\t', '    '),
			os.get_terminal_size().columns - 30
		)
	)
	parser.add_argument('-o', '--output_path', help='Output fits file path. By default is same as fie `fits_spec` path with "_interp" appended to the filename')
	
	parser.add_argument('--bad_pixel_method', choices=['ssa', 'simple'], default='ssa', help='Strategy to use when finding bad pixels to interpolate over')
	#parser.add_argument('--bad_pixel_args', nargs='*', help='Arguments to be passed to `--bad_pixel_method`')

	parser.add_argument('--interp_method', choices=['ssa', 'scipy'], default='ssa', help='Strategy to use when interpolating')


	args = parser.parse_args(argv)
	
	args.fits_spec = aph.fits.specifier.parse(args.fits_spec, ['CELESTIAL'])
	
	if args.output_path is None:
		args.output_path =  (Path(args.fits_spec.path).parent / (str(Path(args.fits_spec.path).stem)+'_interp'+str(Path(args.fits_spec.path).suffix)))
	
	return args


if __name__ == '__main__':
	args = parse_args(sys.argv[1:])
	
	run(args.fits_spec, bad_pixel_method=args.bad_pixel_method, interp_method=args.interp_method, output_path=args.output_path)
	
