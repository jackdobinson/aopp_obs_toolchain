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
		background_threshold : float = 1E-3,
		background_noise_model : str = 'gennorm',
		n_sigma : float = 5,
		trim_to_shape : None | tuple[int,...] = None
	):
	
	ACCEPTABLE_TRIM_LOSS_FACTOR = 1E-2
	axes = fits_spec.axes['CELESTIAL']
	
	with fits.open(Path(fits_spec.path)) as data_hdul:
		
		
		_lgr.debug(f'{fits_spec.path=} {fits_spec.ext=} {fits_spec.slices=} {fits_spec.axes=}')
		#raise RuntimeError(f'DEBUGGING')
	
		data_hdu = data_hdul[fits_spec.ext]
		data = data_hdu.data[fits_spec.slices]
		hdr = data_hdu.header
		
		
		_lgr.info('Ensure data is of odd shape')
		data = nph.array.ensure_odd_shape(data, axes)
		for ax in axes:
			aph.fits.header.set_axes_transform(hdr, ax, n_values = data.shape[ax])
		
		_lgr.info('Remove any outliers')
		outlier_mask = psf_data_ops.get_outlier_mask(data, axes, n_sigma)
		data[outlier_mask] = np.nan
		
		if background_noise_model is not None:
			_lgr.info('Remove background offset')
			background_mask = ~psf_data_ops.get_roi_mask(data, axes, background_threshold, 1)
			noise_model_offsets, noise_model_parameters, noise_model_at_values, noise_model_cdf, noise_model_cdf_residual = psf_data_ops.remove_offset(data, axes, background_mask, background_noise_model)
		
		_lgr.info('Get offsets to center around center of mass')
		roi_mask = psf_data_ops.get_roi_mask(data, axes, threshold, n_largest_regions)
		com_offsets = psf_data_ops.get_center_of_mass_offsets(data, axes, roi_mask)
	
		_lgr.info('Recenter everything for easy comparison')
		normalised_data = psf_data_ops.apply_offsets(data, axes, com_offsets)
		roi_mask = psf_data_ops.apply_offsets(roi_mask, axes, com_offsets)
		background_mask = psf_data_ops.apply_offsets(background_mask, axes, com_offsets)
		outlier_mask = psf_data_ops.apply_offsets(outlier_mask, axes, com_offsets)
		
		
		if trim_to_shape is not None:
			_lgr.info('Trim everything around the center pixel')
			before_trim_sum = np.nansum(normalised_data)
			
			normalised_data = psf_data_ops.trim_around_center(normalised_data, axes, trim_to_shape)
			roi_mask = psf_data_ops.trim_around_center(roi_mask, axes, trim_to_shape)
			outlier_mask = psf_data_ops.trim_around_center(outlier_mask, axes, trim_to_shape)
			
			after_trim_sum = np.nansum(normalised_data)
			trim_loss_factor = 1 - after_trim_sum/before_trim_sum
			_lgr.debug(f'{trim_loss_factor=}')
			if trim_loss_factor > ACCEPTABLE_TRIM_LOSS_FACTOR:
				_lgr.warn(f'After trimming around center to shape {trim_to_shape}, have lost {trim_loss_factor} of original signal. This is above acceptable limit of {ACCEPTABLE_TRIM_LOSS_FACTOR}')
			
			
		
		_lgr.info('Normalise to unit sum')
		original_sum = np.nansum(normalised_data, axis=axes)
		with nph.axes.to_start(normalised_data, axes) as (gdata, gaxes):
			gdata /= original_sum
		
		param_dict = {
			'original_file' : Path(fits_spec.path).name, # record the file we used
			'roi_mask.threshold' : threshold,
			'roi_mask.n_largest_regions' : n_largest_regions,
			'background_mask.threshold' : background_threshold,
			'outlier_mask.n_sigma' : n_sigma,
			'trim_to_shape' : trim_to_shape,
		}
		
		hdr.update(aph.fits.header.DictReader(
			param_dict,
			prefix='psf_normalise',
			pkey_count_start=aph.fits.header.DictReader.find_max_pkey_n(hdr)
		))
		

	
	_lgr.info('save the products to a FITS file')
	hdus = []
	
	hdus.append(fits.PrimaryHDU(
		header = hdr,
		data = normalised_data
	))
	hdus.append(fits.ImageHDU(
		header = hdr,
		data = roi_mask.astype(int),
		name = 'ROI_MASK'
	))
	hdus.append(fits.ImageHDU(
		header = hdr,
		data = background_mask.astype(int),
		name = 'BACKGROUND_MASK'
	))
	hdus.append(fits.ImageHDU(
		header = hdr,
		data = outlier_mask.astype(int),
		name = 'OUTLIER_MASK'
	))
	hdus.append(fits.BinTableHDU.from_columns(
		columns = [
			fits.Column(name='original_total_sum', format='D', array=original_sum),
		],
		name = 'ORIG_SUM',
		header = None,
	))
	
	if background_noise_model is not None:
		hdus.append(fits.BinTableHDU.from_columns(
			columns = [
				fits.Column(name='noise_model_offsets', format='D', array=noise_model_offsets), 
			] + [
				fits.Column(name=f'noise_model_parameter_{k}', format='D', array=[x[k] for x in noise_model_parameters]) for k in noise_model_parameters[0].keys()
			],
			name = 'NOISE_MODEL_PARAMS',
			header = None,
		))
		
		hdus.append(fits.ImageHDU(
			header=None,
			name='NOISE_MODEL_CDF_VALUES',
			data = np.array(noise_model_at_values)
		))
		
		hdus.append(fits.ImageHDU(
			header=None,
			name='NOISE_MODEL_CDF',
			data = np.array(noise_model_cdf)
		))
		
		
		hdus.append(fits.ImageHDU(
			header=None,
			name='NOISE_MODEL_CDF_RESIDUAL',
			data = np.array(noise_model_cdf_residual)
		))
		
	
	hdul_output = fits.HDUList(hdus)
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
	
	parser.add_argument('--threshold', type=float, default=1E-2, help='When finding region of interest, only values larger than this fraction of the maximum value are included.')
	parser.add_argument('--n_largest_regions', type=int, default=1, help="""
		When finding region of interest, if using a threshold will only the n_largest_regions in the calculation.
		A region is defined as a contiguous area where value >= `threshold` along `axes`. I.e. in a 3D cube, if
		we recenter about the COM on the sky (CELESTIAL) axes the regions will be calculated on the sky, not in
		the spectral axis (for example)."""
	)
	parser.add_argument('--background_threshold', type=float, default=1E-3, help='Exclude the largest connected region with values larger than this fraction of the maximum value when finding the background')
	parser.add_argument('--background_noise_model', type=str, default='gennorm', 
		choices=['norm', 'gennorm', 'none'],
		help='Model of background noise to use when removing offset. "none" will not mean offset is not calculated or removed'
	)
	
	parser.add_argument('--n_sigma', type=float, default=5, help='When finding the outlier mask, the number of standard deviations away from the mean a pixel must be to be considered an outlier`')
	parser.add_argument('--trim_to_shape', type=int, nargs=2, default=None, help='After centering etc. will trim data to this shape around the center pixel. Used to reduce data volume for faster processing.')

	args = parser.parse_args(argv)
	
	args.fits_spec = aph.fits.specifier.parse(args.fits_spec, [DESIRED_FITS_AXES])
	
	if args.output_path is None:
		args.output_path =  (Path(args.fits_spec.path).parent / (str(Path(args.fits_spec.path).stem)+DEFAULT_OUTPUT_TAG+str(Path(args.fits_spec.path).suffix)))
	
	if args.background_noise_model == 'none':
		args.background_noise_model = None
	
	
	return args


if __name__ == '__main__':
	args = parse_args(sys.argv[1:])
	
	run(
		args.fits_spec, 
		args.output_path,
		threshold = args.threshold,
		n_largest_regions = args.n_largest_regions,
		background_threshold = args.background_threshold,
		background_noise_model = args.background_noise_model,
		n_sigma = args.n_sigma,
		trim_to_shape = args.trim_to_shape,
	)
	