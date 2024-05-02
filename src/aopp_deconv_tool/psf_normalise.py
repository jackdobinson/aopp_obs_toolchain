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
		n_sigma : float = 5
	):
	
	axes = fits_spec.axes['CELESTIAL']
	
	with fits.open(Path(fits_spec.path)) as data_hdul:
		
		
		_lgr.debug(f'{fits_spec.path=} {fits_spec.ext=} {fits_spec.slices=} {fits_spec.axes=}')
		#raise RuntimeError(f'DEBUGGING')
	
		data_hdu = data_hdul[fits_spec.ext]
		data = data_hdu.data[fits_spec.slices]
		hdr = data_hdu.header
		
		
		# Ensure data is of odd shape
		data = nph.array.ensure_odd_shape(data, axes)
		for ax in axes:
			aph.fits.header.set_axes_transform(hdr, ax, n_values = data.shape[ax])
	
		
		# Remove any outliers
		outlier_mask = psf_data_ops.get_outlier_mask(data[fits_spec.slices], axes, n_sigma)
		data[outlier_mask] = np.nan
		
		# Center around center of mass
		roi_mask = psf_data_ops.get_roi_mask(data, axes, threshold, n_largest_regions)
		com_offsets = psf_data_ops.get_center_of_mass_offsets(data, axes, roi_mask)
		normalised_data = psf_data_ops.apply_offsets(data, axes, com_offsets)
	
		# Recenter masks the same way for easy comparison
		roi_mask = psf_data_ops.apply_offsets(roi_mask, axes, com_offsets)
		outlier_mask = psf_data_ops.apply_offsets(outlier_mask, axes, com_offsets)
		
		# Normalise to unit sum
		original_sum = np.nansum(normalised_data, axis=axes)
		with nph.axes.to_start(normalised_data, axes) as (gdata, gaxes):
			gdata /= original_sum
		
		param_dict = {
			'original_file' : Path(fits_spec.path).name, # record the file we used
			'roi_mask.threshold' : threshold,
			'roi_mask.n_largest_regions' : n_largest_regions,
			'outlier_mask.n_sigma' : n_sigma,
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
	hdu_outlier_mask = fits.ImageHDU(
		header = hdr,
		data = outlier_mask.astype(int),
		name = 'OUTLIER_MASK'
	)
	hdu_original_sum = fits.BinTableHDU.from_columns(
		columns = [
			fits.Column(name='original_total_sum', format='D', array=original_sum),
		],
		name = 'ORIG_SUM',
		header = None,
	)
	
	hdul_output = fits.HDUList([
		hdu_normalised_data,
		hdu_roi_mask,
		hdu_outlier_mask,
		hdu_original_sum,
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
	