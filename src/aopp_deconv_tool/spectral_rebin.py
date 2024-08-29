"""
Quick tool for spectrally rebinning a FITS file, convolves a triangular 
function of `bin_width` with the input data and samples it every `bin_step`.

NOTE: Supersampling is not done in any special way, so if that is desired, 
set `bin_width` to the distance between two datapoints and `bin_step` to the 
supersampling length. This will not preserve the sum, but it should work when
interpolation is required.
"""

import sys
from pathlib import Path
import dataclasses as dc
from typing import Literal, Callable, Sequence



import numpy as np
import scipy as sp
import scipy.signal

from astropy.io import fits
from astropy import units as u

import aopp_deconv_tool.astropy_helper as aph
import aopp_deconv_tool.astropy_helper.fits.specifier
import aopp_deconv_tool.astropy_helper.fits.header
import aopp_deconv_tool.numpy_helper as nph
import aopp_deconv_tool.numpy_helper.axes
import aopp_deconv_tool.numpy_helper.slice
import aopp_deconv_tool.numpy_helper.array.index
import aopp_deconv_tool.arguments

import aopp_deconv_tool.numpy_helper.array.grid
from aopp_deconv_tool.fpath import FPath

import matplotlib.pyplot as plt

import aopp_deconv_tool.cfg.logs
_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'WARN')



DEBUG_PIXEL = (161,168)


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
		if trim:
			min_idx = np.max(np.argwhere(pos_array<=0))
			max_idx = np.min(np.argwhere(pos_array>=self.full_width_half_maximum))+1
			response_array = response_array[min_idx:max_idx]
		return response_array
	
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
		
		if trim:
			min_idx = np.max(np.argwhere(pos_array<=0))
			max_idx = np.min(np.argwhere(pos_array>=2*self.full_width_half_maximum))+1
			response_array = response_array[min_idx:max_idx]
		
		return response_array
	
	def get_pos_array(self, step : float) -> np.ndarray:
		return np.linspace(0, 2*self.full_width_half_maximum, int(np.ceil(2*self.full_width_half_maximum/step)))
	

named_spectral_binning_parameters = dict(
	spex = dict(
		bin_step = 1E-9,
		bin_width = 2E-9,
		response_function_class = TriangularResponseFunction
	)
)

def indices_const_boundary(
		indices : tuple[Sequence[int]], 
		shape : tuple[int]
	) -> tuple[int]:
	"""
	Give a tuple with N entries `indices` like ((i_00,i_01,i_02,...,i_0M_0),(i_10,i_11,...,i_1M_1),...,(...,i_NM_N)), where N is the number of entries in `shape`
	which is like (s_0, s_1, ..., s_N), and M_0, M_1, ..., M_N are the number of indices present for each axis.
	
	Alter the indices so if i_nm < 0, i_nm=0. And if i_nm >= s_n, i_nm=s_n-1. I.e., indices "to the left" are set to the first entry in the axis, and
	indices "to the right" are set to the last entry in the axis.
	"""
	out_indices = []
	for j, (idxs, s) in enumerate(zip(indices, shape)):
		jdx = []
		for i in idxs:
			if i <0:
				jdx.append(0)
			elif i >= s:
				jdx.append(s-1)
			else:
				jdx.append(i)
		
		out_indices.append(tuple(jdx))
	return tuple(out_indices)

def array_const_boundary(a : np.ndarray, indices: tuple[Sequence[int]]) -> np.ndarray:
	"""
	Return the values on an arry such that any indices that are "to the left" return the first entry in the axis, and any that are "to the right" return
	the last entry in the axis.
	"""
	return a[indices_const_boundary(indices, a.shape)]


def overlap_add_convolve(
		data : np.ndarray[[...,'M',...]], 
		response_1d : np.ndarray[['N']], 
		axis : int, 
		mode : Literal['same'] | Literal['full'] | Literal['valid'] = 'same'
	) -> np.ndarray[[...,'L',...]]:
	"""
	Performs convolution over `axis` of `data` by the overlap-and-add method with `response_1d`. Is generally faster for large `data` and small `response_1d`.
	The shape of the output array depends on the `mode`:
	
	mode    | output shape
	--------|-----------------
	'same'  | [...,'M',...]
	'full'  | [...,'M+2N',...]
	'valid' | [...,'M-2N',...]
	"""
	data = np.moveaxis(data, 0, axis)
	
	n = response_1d.shape[0]
	full = np.zeros((data.shape[0]+2*n, *data.shape[1:]))
	
	
	
	for i in range(-n, data.shape[0]+n):
		subset_of_data = array_const_boundary(data, (np.arange(i,i+n,dtype=int),))
		full[i+n] = np.sum((subset_of_data.T * response_1d[:subset_of_data.shape[0]]).T, axis=0)
	
	
	dn = 1
	
	"""
	# DEBUGGING
	# Plot data and convolved data on the same axis to work out offset needed
	full_embed_data = np.zeros_like(full)
	full_embed_data[n//2+dn:-2*n+n//2+dn] = data
	plt.figure()
	plt.plot(full_embed_data[:,*DEBUG_PIXEL])
	plt.plot(full[:,*DEBUG_PIXEL])
	plt.show()
	"""
	
	match mode:
		case 'same':
			return np.moveaxis(full[n//2+dn:-2*n+n//2+dn], 0, axis)
		case 'full':
			return np.moveaxis(full, 0, axis)
		case 'valid':
			return np.moveaxis(full[n+dn:-2*n+dn], 0, axis)
		case _:
			raise RuntimeError(f'Unknown mode "{mode}"')

def lin_interp(
		new_points : np.ndarray[['N']], 
		old_points : np.ndarray[['M']], # assume this is sorted in ascending order
		data : np.ndarray[[...,'M',...]], 
		axis : int, # should be axis of "M" in `data`,
		boundary_conditions : Literal['constant'] | Literal['reflect'] | Literal['periodic'] | Literal['extrapolate'] = 'constant'
	) -> np.ndarray:
	"""
	Given `data` defined at `old_points` along `axis`, linearly interpolate at `new_points` subject to `boundary_conditions`.
	Works on N-dimensional arrays, but only over one axis at a time.
	"""
	
	data = np.moveaxis(data, axis, -1)
	_lgr.debug(f'Moved data axis')
	
	y = np.full((*data.shape[:-1], new_points.size), fill_value=np.nan)
	_lgr.debug(f'Output array allocated')
	
	new_points = np.array(new_points) # copy array soe we can process the points
	match boundary_conditions:
		case 'constant':
			gt = new_points >= old_points[-1]
			lt = new_points < old_points[0]
			new_points[gt] = old_points[-1]
			new_points[lt] = old_points[0]
		case 'reflect':
			gt = new_points >= old_points[-1]
			lt = new_points < old_points[0]
			while gt.size!=0 or lt.size !=0:
				new_points[lt] = 2*old_points[0] - new_points[lt]
				new_points[gt] = 2*old_points[1] - new_points[gt]
				gt = new_points >= old_points[-1]
				lt = new_points < old_points[0]
		case 'periodic':
			gt = new_points >= old_points[-1]
			lt = new_points < old_points[0]
			while gt.size!=0 or lt.size !=0:
				new_points[lt] = old_points[0] - new_points[lt] + old_points[1]
				new_points[gt] -=  old_points[-1] - old_points[0]
				gt = new_points >= old_points[-1]
				lt = new_points < old_points[0]
		case 'extrapolate':
			pass
		case _:
			raise RuntimeError(f'Unknown boundary conditions "{boundary_conditions}"')
	_lgr.debug(f'Calculated boundary conditions')
	
	
	_lgr.debug(f'{data.shape=} {old_points.shape=}')
	
	
	new_idxs = np.zeros_like(new_points, dtype=int)	
	for i in range(1, old_points.size-1):
		new_idxs += (old_points[i] <= new_points).astype(int)
		
	_lgr.debug(f'Calculated indices of new points')
	
	if (data.size < 1E6):
		# Do things the fast way if we can
		A = np.full((*data.shape[:-1], old_points.size-1, 2), fill_value=np.nan, dtype=float)
		
		A[...,:,0] = (data[...,1:] - data[...,:-1])/(old_points[1:] - old_points[:-1]) # gradient
		A[...,:,1] = data[...,:-1] - A[...,0]*old_points[:-1] # constant
		
		y[...] = np.sum(A[...,new_idxs,:]*np.stack((new_points, np.ones_like(new_points)), axis=-1), axis=-1)
		
		"""
		# DEBUGGING
		plt.figure()
		plt.title('linear interpolation parameters')
		ax = plt.gca()
		ax.plot(A[*DEBUG_PIXEL,:,0], color='tab:blue', alpha=0.5, label='gradient')
		ax.set_ylabel('gradient')
		ax2 = ax.twinx()
		ax2.plot(A[*DEBUG_PIXEL,:,1], color='tab:orange', alpha=0.5, label='constant')
		ax2.set_ylabel('constant')
		ax_h = ax.get_legend_handles_labels()
		ax2_h = ax2.get_legend_handles_labels()
		plt.legend(ax_h[0]+ax2_h[0], ax_h[1]+ax2_h[1])
		plt.show()
		plt.close('all')
		"""
		return np.moveaxis(y, -1, axis)
	
	# Otherwise, do things piecewise so we don't run out of memory
	
	for i, j in enumerate(new_idxs):
		if 1 <= j < old_points.size:
			gradient = (data[...,j] - data[...,j-1])/(old_points[j] - old_points[j-1])
		else:
			gradient = np.zeros(data.shape[:-1])

		if 0 <= j <= old_points.size:
			constant = data[...,j] - gradient*old_points[j]
		else:
			constant = 0
			
		y[...,i] = gradient*new_points[i] + constant
		
	return np.moveaxis(y,-1,axis)
	

def plot_rebin_response_function(data_hdu, axis, spectral_unit_in_meters, new_spec_ax, new_data):
	_lgr.debug(f'{new_spec_ax.shape=} {new_data[:,*DEBUG_PIXEL].shape=}')
	input_axis = aph.fits.header.get_world_coords_of_axis(
		data_hdu.header, 
		axis, 
		wcs_unit_to_return_value_conversion_factor=spectral_unit_in_meters
	)
	plt.plot(
		input_axis, 
		data_hdu.data[:,*DEBUG_PIXEL],
		color='tab:blue',
		label='original'
	)
	plt.plot(
		new_spec_ax, 
		new_data[:,*DEBUG_PIXEL],
		color='tab:orange',
		label='rebinned'
	)
	plt.plot(
		input_axis, 
		data_hdu.data[:,*DEBUG_PIXEL] - np.interp(input_axis, new_spec_ax, new_data[:,*DEBUG_PIXEL]),
		color='tab:green',
		label='original - rebinned'
	)
	plt.legend()
	plt.show()


def rebin_hdu_over_axis_with_response_function(
		data_hdu,
		axis,
		response_function : ResponseFunction,
		bin_start : float | None = None,
		bin_step : float = 1E-9,
		axis_unit_conversion_factors : tuple[float,...] = (1,),
		plot : bool = True
	) -> tuple[np.ndarray, np.ndarray]:
	
	_lgr.debug(f'{axis_unit_conversion_factors=}')
	ax_values = aph.fits.header.get_world_coords_of_axis(data_hdu.header, axis, wcs_unit_to_return_value_conversion_factor=axis_unit_conversion_factors)
	_lgr.debug(f'{data_hdu.data.shape=} {ax_values.shape=}')
	
	bin_start = ax_values[0] if bin_start is None else bin_start
	
	current_step = ax_values[1]-ax_values[0]
	if current_step > bin_step:
		_lgr.warning(f'Current rebinning does not work well for reducing step size, {current_step=} {bin_step=}. Using current data instead')
		return ax_values, data_hdu.data
	
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
	smoothed = lin_interp(new_points, ax_values, smoothed, axis=axis, boundary_conditions='constant')
	
	if plot:
		plot_rebin_response_function(data_hdu, axis, axis_unit_conversion_factors, new_points, smoothed)
	
	
	return new_points, smoothed


def run(
		fits_spec : aph.fits.specifier.FitsSpecifier, 
		output_path : Path | str, 
		bin_step : float = 1E-9, # Assume SI units
		bin_width : float = 2E-9, # Assume SI units
		operation : Literal['sum'] | Literal['mean'] | Literal['mean_err'] = 'mean',
		spectral_unit_in_meters : float = 1,
		response_function_class : Literal[SquareResponseFunction] | Literal[TriangularResponseFunction] = TriangularResponseFunction,
		output_unit : u.Unit | None = None,
		plot : bool = False,
	) -> tuple[np.ndarray, np.ndarray]:

	original_data_type=None

	new_data = None
	with fits.open(Path(fits_spec.path)) as data_hdul:
	
		#_lgr.debug(f'{fits_spec.ext=}')
		#raise RuntimeError(f'DEBUGGING')
	
		data_hdu = data_hdul[fits_spec.ext]
		
	
		axes_ordering =  aph.fits.header.get_axes_ordering(data_hdu.header, fits_spec.axes['SPECTRAL'])
		axis_unit = u.Unit(aph.fits.header.get_axes_unit_string(data_hdu.header, (axes_ordering[0].fits,))[0], format='fits')
		if output_unit is None:
			output_unit = axis_unit
		axis = axes_ordering[0].numpy
		original_data_type = data_hdu.data.dtype
				
		axis_si_unit = None
		match axis_unit.physical_type:
			case 'length':
				axis_si_unit = u.meter
			case 'frequency':
				axis_si_unit = u.Hz
			case _:
				raise RuntimeError(f'Unsupported physical type {axis_unit.physical_type} for unit {axis_unit}')
	
		#new_spec_bins, new_data = rebin_hdu_over_axis(data_hdu, axis, bin_step, bin_width, operation, plot=False)
		
		postprocess_data_mutator = lambda x: x
		
		response_function_sum = None
		match operation:
			case 'sum':
				pass # no need to alter anything
			case 'mean':
				response_function_sum = 1
			case 'mean_err':
				data_hdu.data = data_hdu.data**2 # square the data
				response_function_sum = 1
				postprocess_data_mutator = lambda x: np.sqrt(x) # square-root it after rebinning
			case _:
				raise RuntimeError(f'Unknown binning operation "{operation}"')
		
		new_spec_ax, new_data = rebin_hdu_over_axis_with_response_function(
			data_hdu, 
			axis, 
			response_function_class(response_function_sum, bin_width),
			bin_start=None, 
			bin_step=bin_step, 
			axis_unit_conversion_factors=1, # astropy always responds with SI units
			plot=plot
		)
		
		new_data = postprocess_data_mutator(new_data)
		
		
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
		
		#unit_factor = (1*output_unit / u.meter).value
		#unit_factor = (1*u.meter).to(axis_unit, equivalencies=u.spectral()).to(output_unit, equivalencies=u.spectral()).value
		unit_factor = (1*axis_si_unit).to(output_unit, equivalencies=u.spectral()).value
		aph.fits.header.set_axes_transform(hdr, 
			axis_fits, 
			u.format.Fits.to_string(output_unit), 
			new_spec_ax[0] * unit_factor,
			bin_step * unit_factor,
			new_spec_ax.shape[0],
			1
		)

	
	# Save the products to a FITS file
	hdu_rebinned = fits.PrimaryHDU(
		header = hdr,
		data = new_data.astype(original_data_type)
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
	#parser.add_argument('-o', '--output_path', help=f'Output fits file path. By default is same as the `fits_spec` path with "{DEFAULT_OUTPUT_TAG}" appended to the filename')
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
	
	parser.add_argument('--spectral_unit_in_meters', type=float, default=None, help='The conversion factor between the spectral unit and meters. Only required when the unit cannot be automatically determined, or there is a mismatch between unit and data. Any automatically found unit information will be overwritten.')
	parser.add_argument('--rebin_operation', choices=['sum', 'mean', 'mean_err'], default='mean', help='Operation to perform when binning.')
	parser.add_argument('--output_unit', type=str, default=None, help='Units to use for output datacube. If not specified will use the same units as the input datacube.')
	
	rebin_group = parser.add_mutually_exclusive_group(required=False)
	rebin_group.add_argument('--rebin_preset', choices=list(named_spectral_binning_parameters.keys()), default='spex', help='Rebin according to the spectral resolution of the preset')
	rebin_group.add_argument('--rebin_params', nargs=2, type=float, metavar='float', help='bin_step and bin_width for rebinning operation (meters)')
	
	args = parser.parse_args(argv)
	
	args.fits_spec = aph.fits.specifier.parse(args.fits_spec, DESIRED_FITS_AXES)
	
	if args.output_unit is not None:
		try:
			args.output_unit = u.Unit(args.output_unit, format='fits')
		except ValueError as e:
			try:
				desired_unit = u.Unit(args.output_unit)
			except Exception as e:
				raise e
			else:
				equiv_phys_types = {}
				fits_compat_units = u.format.Fits._generate_unit_names()[0]
				compat_units = desired_unit.compose(
					#equivalencies=u.spectral(), 
					units=fits_compat_units.values(), 
					include_prefix_units=False
				)
				e.add_note(f'Acceptible equivalent FITS units for "{args.output_unit}":\n\t'+'\n\t'.join([str(x) for x in compat_units]))
				raise e
	
	if args.rebin_preset is not None:
		for k,v in named_spectral_binning_parameters[args.rebin_preset].items():
			setattr(args, k, v)
	if args.rebin_params is not None:
		setattr(args, 'bin_step', args.rebin_params[0])
		setattr(args, 'bin_width', args.rebin_params[1])
		setattr(args, 'response_function_class', TriangularResponseFunction)
	
	#if args.output_path is None:
	#	args.output_path =  (Path(args.fits_spec.path).parent / (str(Path(args.fits_spec.path).stem)+DEFAULT_OUTPUT_TAG+str(Path(args.fits_spec.path).suffix)))
	other_file_path = Path(args.fits_spec.path)
	args.output_path = args.output_path.with_fields(
		tag=DEFAULT_OUTPUT_TAG, 
		parent=other_file_path.parent, 
		stem=other_file_path.stem, 
		suffix=other_file_path.suffix
	)
	
	
	print('INPUT PARAMETERS')
	for k,v in vars(args).items():
		print(f'    {k} : {v}')
	print('END')
	
	return args


def go(
		fits_spec,
		output_path=None,
		spectral_unit_in_meters=None,
		rebin_operation=None,
		output_unit=None,
		rebin_preset=None,
		rebin_params=None
	):
	"""
	Thin wrapper around `run()` to accept string inputs.
	As long as the names of the arguments to this function 
	are the same as the names expected from the command line
	we can do this programatically
	"""
	# This must be first so we only grab the arguments to the function
	fargs = dict(locals().items())
	arglist = aopp_deconv_tool.arguments.construct_arglist_from_locals(fargs, n_positional_args=1)
	
	exec_with_args(arglist)
	return

def exec_with_args(argv):
	args = parse_args(argv)
	
	run(
		args.fits_spec, 
		bin_step=args.bin_step, 
		bin_width=args.bin_width, 
		operation=args.rebin_operation, 
		output_path=args.output_path, 
		spectral_unit_in_meters= 1 if args.spectral_unit_in_meters is None else args.spectral_unit_in_meters,
		response_function_class = args.response_function_class,
		output_unit = args.output_unit,
	)

if __name__ == '__main__':
	exec_with_args(sys.argv[1:])
	
