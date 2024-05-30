"""
Quick tool for spectrally rebinning a FITS file

TODO:
Currently have what looks like an off-by one error between
python -m aopp_deconv_tool.spectral_rebin './example_data/ifu_observation_datasets/MUSE.2019-10-18T00:01:19.521.fits' -o ./test_pat_error_rebin.fits
and
python -m aopp_deconv_tool.spectral_rebin './example_data/ifu_observation_datasets/MUSE.2019-10-17T23:46:14.117.fits' -o ./test_pat_error_rebin.fits

Look at the matplotlib plots to see the problem
"""

import sys
from pathlib import Path
import dataclasses as dc
from typing import Literal, Callable



import numpy as np
import scipy as sp
import scipy.signal

from astropy.io import fits

import aopp_deconv_tool.astropy_helper as aph
import aopp_deconv_tool.astropy_helper.fits.specifier
import aopp_deconv_tool.astropy_helper.fits.header
import aopp_deconv_tool.numpy_helper as nph
import aopp_deconv_tool.numpy_helper.axes
import aopp_deconv_tool.numpy_helper.slice

import aopp_deconv_tool.numpy_helper.array.grid

import matplotlib.pyplot as plt

import aopp_deconv_tool.cfg.logs
_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'DEBUG')




named_spectral_binning_parameters = dict(
	spex = dict(
		bin_step = 1E-9,
		bin_width = 2E-9
	)
)


@dc.dataclass
class ResponseFunction:
	total_sum : float | None = None
	
	def as_array(self, pos_array : np.ndarray, trim : bool = True) -> np.ndarray:
		NotImplemented
	
	def get_pos_array(self, step:float) -> np.ndarray:
		NotImplemented
	

@dc.dataclass
class SquareResponseFunction (ResponseFunction):
	full_width_half_maximum : float = dc.field(default=2E-9, init=True)
	
	def as_array(self, pos_array : np.ndarray, trim : bool = True) -> np.ndarray:
		response_array = np.interp(pos_array, (0,self.full_width_half_maximum), (1,1), left=0, right=0)
		if self.total_sum is not None:
			response_array *= self.total_sum/np.sum(response_array)
		return response_array[(pos_array >= 0) & (pos_array <= self.full_width_half_maximum)] if trim else response_array
	
	def get_pos_array(self, step : float) -> np.ndarray:
		return np.linspace(0, self.full_width_half_maximum, int(np.ceil(self.full_width_half_maximum/step)))
	

@dc.dataclass
class TriangularResponseFunction (ResponseFunction):
	full_width_half_maximum : float = dc.field(default=2E-9, init=True)
	
	def as_array(self, pos_array : np.ndarray, trim=True) -> np.ndarray:
		response_array = np.interp(pos_array, (0,self.full_width_half_maximum,2*self.full_width_half_maximum), (0,1,0), left=0, right=0)
		_lgr.debug(f'{pos_array=}')
		_lgr.debug(f'{response_array=}')
		
		if self.total_sum is not None:
			response_array *= self.total_sum/np.sum(response_array)
		
		return response_array[(pos_array >= 0) & (pos_array <= 2*self.full_width_half_maximum)] if trim else response_array
	
	def get_pos_array(self, step : float) -> np.ndarray:
		return np.linspace(0, 2*self.full_width_half_maximum, int(np.ceil(2*self.full_width_half_maximum/step)))
	

def plot_rebin(old_bins, old_data, new_bins, new_data, title=None):
	plt.title(title)
	plt.plot(np.sum(old_bins, axis=0)/2, old_data, label='old_data')
	plt.plot(np.sum(new_bins, axis=0)/2, new_data, label='new_data')
	plt.legend()
	plt.show()


def overlap_add_convolve(data, response_1d, axis, mode='same') -> np.ndarray:
	n = response_1d.shape[0]
	full = np.zeros(tuple(s if i!= axis else s+2*n for i, s in enumerate(data.shape)))
	
	np.moveaxis(data, axis, 0)
	np.moveaxis(full, axis, 0)
	
	
	full[n:-n] = data
	np.moveaxis(data, 0, axis)
	
	n = response_1d.shape[0]
	for i in range(0, full.shape[0]):
		_lgr.info(f'rebinning {i}:{i+n} where total length is {full.shape[0]}')
		subset_of_data = full[i:i+n]
		full[i] = np.sum((subset_of_data.T * response_1d[:subset_of_data.shape[0]]).T, axis=0)
	
	dn = 0
	match mode:
		case 'same':
			return np.moveaxis(full[n//2+dn:-2*n+n//2+dn+1], 0, axis)
		case 'full':
			return np.moveaxis(full, 0, axis)
		case 'valid':
			return np.moveaxis(full[n+dn:-2*n+dn+1], 0, axis)
		case _:
			raise RuntimeError(f'Unknown mode "{mode}"')

def lin_interp(
		new_points : np.ndarray[['N']], 
		old_points : np.ndarray[['M']], 
		data : np.ndarray[[...,'M',...]], 
		axis : int, # should be axis of "M" in `data`,
		left : None | float = None,
		right : None | float = None,
	) -> np.ndarray:
	
	# find out which index new_points belong to
	idxs = np.arange(0,old_points.shape[0])
	_lgr.debug(f'{idxs=}')
	new_idx_pos = np.interp(new_points, old_points, idxs, left=-2, right=-1)
	_lgr.debug(f'{new_idx_pos=}')
	new_idxs = np.ceil(new_idx_pos-1).astype(int)
	_lgr.debug(f'{new_idxs=}')
	within_range_mask = new_idxs >= 0
	
	frac_to_next_idx = np.full_like(new_idxs, fill_value=np.nan)
	frac_to_next_idx[within_range_mask] = (new_points[within_range_mask] - old_points[new_idxs[within_range_mask]])/(old_points[new_idxs[within_range_mask]+1] - old_points[new_idxs[within_range_mask]])
	_lgr.debug(f'{frac_to_next_idx=}')
	
	
	np.moveaxis(data, axis, 0)
	
	interp_data = np.full((new_points.shape[0], *data.shape[1:]), fill_value=np.nan)
	
	
	for i, j in enumerate(new_idxs):
		_lgr.debug(f'{i=} {j=} n={new_idxs.shape[0]} {within_range_mask[i]=}')
		if within_range_mask[i]:
			interp_data[i] = data[j]*(1-frac_to_next_idx[i]) + frac_to_next_idx[i]*data[j+1]
		else:
			if new_idxs[i] == -2:
				interp_data[j] = data[0] if left is None else np.full(interp_data.shape[1:], fill_value=left)
			elif new_idxs[i] == -1:
				interp_data[j] = data[-1] if right is None else np.full(interp_data.shape[1:], fill_value=right)
			else:
				raise RuntimeError(f'Unknown index {new_idx_pos[i]} for {i}^th point in linear interpolation')
		
	"""
	interp_data[within_range_mask] = data[..., new_idxs[within_range_mask]]*(1-frac_to_next_idx) + frac_to_next_idx*data[..., new_idxs[within_range_mask]+1]
	interp_data[new_idxs == -2] = data[0] if left is None else left
	interp_data[new_idxs == -1] = data[-1] if right is None else right
	"""
	np.moveaxis(data, 0, axis)
	np.moveaxis(interp_data, 0, axis)
	
	return interp_data
	
	

def rebin_hdu_over_axis_with_response_function(
		data_hdu,
		axis,
		response_function : ResponseFunction,
		bin_start : float | None = None,
		bin_step : float = 1E-9,
		axis_unit_conversion_factors : float = 1
	) -> tuple[np.ndarray, np.ndarray]:
	
	ax_values = aph.fits.header.get_world_coords_of_axis(data_hdu.header, axis, wcs_unit_to_return_value_conversion_factor=axis_unit_conversion_factors)
	_lgr.debug(f'{ax_values=}')
	
	bin_start = ax_values[0] if bin_start is None else bin_start
	
	response_array = response_function.as_array(ax_values-ax_values[0], trim=True)
	_lgr.debug(f'{response_array=}')
	
	#plt.plot(response_array); plt.show()
	
	smoothed = overlap_add_convolve(
		data_hdu.data, 
		response_array,
		axis=axis,
		mode='same'
	)
	
	n_new_points = (ax_values[-1]-bin_start)/bin_step + 1
	_lgr.debug(f'{n_new_points=}')
	new_points = np.linspace(bin_start, ax_values[-1], int(np.floor(n_new_points)))
	
	#return None, smoothed # DEBUGGING
	return new_points, lin_interp(new_points, ax_values, smoothed, axis=axis, left=np.nan, right=np.nan)
	
	
	

def rebin_hdu_over_axis(
		data_hdu, 
		axis, 
		bin_step : float = 1E-9, 
		bin_width : float = 2E-9,
		operation : Literal['sum'] | Literal['mean'] | Literal['mean_err'] = 'mean',
		plot : bool = False,
		axis_unit_conversion_factors : float = 1
	) -> tuple[np.ndarray, np.ndarray]:
	ax_values = aph.fits.header.get_world_coords_of_axis(data_hdu.header, axis, wcs_unit_to_return_value_conversion_factor=axis_unit_conversion_factors)
	_lgr.debug(f'{ax_values=}')
	
	old_bins = nph.array.grid.edges_from_midpoints(ax_values)
	_lgr.debug(f'{old_bins=}')
	
	new_bins = nph.array.grid.edges_from_bounds(old_bins[0,0], old_bins[-1,-1], bin_step, bin_width)
	_lgr.debug(f'{new_bins=}')

	match operation:
		case 'sum':
			regrid_data, regrid_bin_weights = nph.array.grid.regrid(data_hdu.data, old_bins, new_bins, axis)
			new_data = regrid_data
		case 'mean':
			regrid_data, regrid_bin_weights = nph.array.grid.regrid(data_hdu.data, old_bins, new_bins, axis)
			new_data = (regrid_data.T/regrid_bin_weights).T
		case 'mean_err':
			regrid_data, regrid_bin_weights = nph.array.grid.regrid(data_hdu.data**2, old_bins, new_bins, axis)
			new_data = np.sqrt((regrid_data.T/(regrid_bin_weights)).T)
		case _:
			raise RuntimeError(f'Unknown binning operation "{operation}"')
	
	if plot:
		plot_rebin(
			old_bins, 
			data_hdu.data[:, data_hdu.data.shape[1]//2, data_hdu.data.shape[2]//2], 
			new_bins, 
			new_data[:, new_data.shape[1]//2, new_data.shape[2]//2]
		)
	
	return new_bins, new_data


def run(
		fits_spec : aph.fits.specifier.FitsSpecifier, 
		output_path : Path | str, 
		bin_step : float = 1E-9, 
		bin_width : float = 2E-9,
		operation : Literal['sum'] | Literal['mean'] | Literal['mean_err'] = 'mean',
		spectral_unit_in_meters : float = 1
	) -> tuple[np.ndarray, np.ndarray]:

	new_data = None
	with fits.open(Path(fits_spec.path)) as data_hdul:
	
		#_lgr.debug(f'{fits_spec.ext=}')
		#raise RuntimeError(f'DEBUGGING')
	
		data_hdu = data_hdul[fits_spec.ext]
	
		axes_ordering =  aph.fits.header.get_axes_ordering(data_hdu.header, fits_spec.axes['SPECTRAL'])
		axis = axes_ordering[0].numpy
	
		#new_spec_bins, new_data = rebin_hdu_over_axis(data_hdu, axis, bin_step, bin_width, operation, plot=False)
		new_spec_ax, new_data = rebin_hdu_over_axis_with_response_function(
			data_hdu, 
			axis, 
			TriangularResponseFunction(1, bin_width), 
			bin_start=None, 
			bin_step=bin_step, 
			axis_unit_conversion_factors=spectral_unit_in_meters
		)

		plt.plot(aph.fits.header.get_world_coords_of_axis(data_hdu.header, axis, wcs_unit_to_return_value_conversion_factor=spectral_unit_in_meters), data_hdu.data[:,161,168])
		plt.plot(new_spec_ax, new_data[:,161,168])
		#plt.plot(data_hdu.data[:,161,168]-new_data[:,161,168])
		plt.show()
	
		hdr = data_hdu.header
		axis_fits = axes_ordering[0].fits
		param_dict = {
			'original_file' : Path(fits_spec.path).name, # record the file we used
			'bin_axis' : axis_fits,
			'bin_step' : bin_step,
			'bin_width' : bin_width,
			'bin_operation' : operation
		}
		
		hdr.update(aph.fits.header.DictReader(
			param_dict,
			prefix='spectral_rebin',
			pkey_count_start=aph.fits.header.DictReader.find_max_pkey_n(hdr)
		))
		
		aph.fits.header.set_axes_transform(hdr, 
			axis_fits, 
			'Angstrom', 
			#np.mean(new_spec_bins[:,0])/1E-10,
			new_spec_ax[0]/1E-10,
			bin_step/1E-10,
			#new_spec_bins.shape[1],
			new_spec_ax.shape[0],
			1
		)

	
	# Save the products to a FITS file
	hdu_rebinned = fits.PrimaryHDU(
		header = hdr,
		data = new_data
	)
	hdul_output = fits.HDUList([
		hdu_rebinned,
	])
	hdul_output.writeto(output_path, overwrite=True)
	_lgr.info(f'Written processed file to "{output_path}"')




def parse_args(argv):
	import os
	import aopp_deconv_tool.text
	import argparse
	
	DEFAULT_OUTPUT_TAG = '_rebin'
	DESIRED_FITS_AXES = ['SPECTRAL']
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
	parser.add_argument('-o', '--output_path', help=f'Output fits file path. By default is same as the `fits_spec` path with "{DEFAULT_OUTPUT_TAG}" appended to the filename')
	
	parser.add_argument('--spectral_unit_in_meters', type=float, default=None, help='The conversion factor between the spectral unit and meters. Only required when the unit cannot be automatically determined, or there is a mismatch between unit and data. Any automatically found unit information will be overwritten.')
	parser.add_argument('--rebin_operation', choices=['sum', 'mean', 'mean_err'], default='mean', help='Operation to perform when binning.')
	
	rebin_group = parser.add_mutually_exclusive_group(required=False)
	rebin_group.add_argument('--rebin_preset', choices=list(named_spectral_binning_parameters.keys()), default='spex', help='Rebin according to the spectral resolution of the preset')
	rebin_group.add_argument('--rebin_params', nargs=2, type=float, metavar='float', help='bin_step and bin_width for rebinning operation (meters)')
	
	args = parser.parse_args(argv)
	
	args.fits_spec = aph.fits.specifier.parse(args.fits_spec, DESIRED_FITS_AXES)
	
	if args.rebin_preset is not None:
		for k,v in named_spectral_binning_parameters[args.rebin_preset].items():
			setattr(args, k, v)
	if args.rebin_params is not None:
		setattr(args, 'bin_step', args.rebin_params[0])
		setattr(args, 'bin_width', args.rebin_params[1])
	
	if args.output_path is None:
		args.output_path =  (Path(args.fits_spec.path).parent / (str(Path(args.fits_spec.path).stem)+DEFAULT_OUTPUT_TAG+str(Path(args.fits_spec.path).suffix)))
	
	print('INPUT PARAMETERS')
	for k,v in vars(args).items():
		print(f'    {k} : {v}')
	print('END')
	
	return args


if __name__ == '__main__':
	args = parse_args(sys.argv[1:])
	
	run(
		args.fits_spec, 
		bin_step=args.bin_step, 
		bin_width=args.bin_width, 
		operation=args.rebin_operation, 
		output_path=args.output_path, 
		spectral_unit_in_meters= 1 if args.spectral_unit_in_meters is None else args.spectral_unit_in_meters
	)
	
