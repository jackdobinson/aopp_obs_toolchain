"""
Quick tool for normalising a PSF in a FITS file
"""


import sys
from pathlib import Path

from typing import Literal

import numpy as np
import scipy as sp
from astropy.io import fits

import aopp_deconv_tool.astropy_helper as aph
import aopp_deconv_tool.astropy_helper.fits.specifier
import aopp_deconv_tool.astropy_helper.fits.header
import aopp_deconv_tool.numpy_helper as nph
import aopp_deconv_tool.numpy_helper.axes
import aopp_deconv_tool.numpy_helper.slice

import aopp_deconv_tool.psf_data_ops as psf_data_ops


import matplotlib.pyplot as plt

import aopp_deconv_tool.cfg.logs
_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'DEBUG')


def run(
		fits_spec,
		output_path,
		threshold : float = 1E-2,
		n_largest_regions : None | int = 1,
	):
	
	axes = fits_spec.axes['CELESTIAL']
	
	with fits.open(Path(fits_spec.path)) as data_hdul:
		
		
		_lgr.debug(f'{fits_spec.path=} {fits_spec.ext=} {fits_spec.slices=} {fits_spec.axes=}')
		#raise RuntimeError(f'DEBUGGING')
	
		data_hdu = data_hdul[fits_spec.ext]
		data = data_hdu.data
	
		roi_mask = psf_data_ops.get_roi_mask(data[fits_spec.slices], axes, threshold, n_largest_regions)
		com_offsets = psf_data_ops.get_center_of_mass_offsets(data[fits_spec.slices], axes, roi_mask)
		normalised_data = psf_data_ops.apply_offsets(data[fits_spec.slices], axes, com_offsets)
		roi_mask = psf_data_ops.apply_offsets(roi_mask, axes, com_offsets)
	
		# Loop over the index range specified by `obs_fits_spec` and `psf_fits_spec`
		#for i, idx in enumerate(nph.slice.iter_indices(data, fits_spec.slices, fits_spec.axes['CELESTIAL'])):
		#	_lgr.debug(f'{i=}')
			
	
	
		hdr = data_hdu.header
		param_dict = {
			'original_file' : Path(fits_spec.path).name, # record the file we used
			'roi_mask.threshold' : threshold,
			'roi_mask.n_largest_regions' : n_largest_regions,
		}
		
		hdr.update(aph.fits.header.DictReader(param_dict))
		

	
	# Save the products to a FITS file
	hdu_normalised_data = fits.PrimaryHDU(
		header = hdr,
		data = normalised_data
	)
	hdu_roi_mask = fits.ImageHDU(
		header = hdr,
		data = roi_mask.astype(int),
		name = 'ROI_MASK'
	)
	
	hdul_output = fits.HDUList([
		hdu_normalised_data,
		hdu_roi_mask,
	])
	hdul_output.writeto(output_path, overwrite=True)


def parse_args(argv):
	import os
	import aopp_deconv_tool.text
	import argparse
	
	DEFAULT_OUTPUT_TAG = '_normalised'
	DESIRED_FITS_AXES = 'CELESTIAL'
	
	parser = argparse.ArgumentParser(
		description=__doc__, 
		formatter_class=argparse.RawTextHelpFormatter
	)
	
	parser.add_argument(
		'fits_spec', 
		help = aopp_deconv_tool.text.wrap(
			aph.fits.specifier.get_help([DESIRED_FITS_AXES]).replace('\t', '    '),
			os.get_terminal_size().columns - 30
		)
	)
	parser.add_argument('-o', '--output_path', help=f'Output fits file path. By default is same as fie `fits_spec` path with "{DEFAULT_OUTPUT_TAG}" appended to the filename')
	
	#parser.add_argument('--bad_pixel_method', choices=['ssa', 'simple'], default='ssa', help='Strategy to use when finding bad pixels to interpolate over')
	#parser.add_argument('--bad_pixel_args', nargs='*', help='Arguments to be passed to `--bad_pixel_method`')

	#parser.add_argument('--interp_method', choices=['ssa', 'scipy'], default='ssa', help='Strategy to use when interpolating')


	args = parser.parse_args(argv)
	
	args.fits_spec = aph.fits.specifier.parse(args.fits_spec, [DESIRED_FITS_AXES])
	
	if args.output_path is None:
		args.output_path =  (Path(args.fits_spec.path).parent / (str(Path(args.fits_spec.path).stem)+DEFAULT_OUTPUT_TAG+str(Path(args.fits_spec.path).suffix)))
	
	return args


if __name__ == '__main__':
	args = parse_args(sys.argv[1:])
	
	run(
		args.fits_spec, 
		args.output_path
	)
	