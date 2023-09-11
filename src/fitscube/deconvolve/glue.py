#!/usr/bin/env python3
"""
Script for gluing "algorithms.py", "flag_bad_pixels.py", and "archive_crawler.py" together for end-to-end deconvolution
"""
import utilities.logging_setup
logging, _lgr = utilities.logging_setup.getLoggers(__name__, 'DEBUG')

import sys, os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import argparse
import dataclasses as dc
import datetime
import json
import typing # for type hints 

import fitscube.deconvolve.algorithms
import fitscube.deconvolve.flag_bad_pixels
import fitscube.deconvolve.helpers
import fitscube.deconvolve.archive_crawler
import py_ssa
import utilities as ut
import utilities.np
import utilities.sp
import utilities.plt
import utilities.fits
import utilities.args
from utilities.classes import Next
import utilities.text


# Setting debugging
#logging.setLevelExcept(__name__, 'WARNING')
#logging.setLevel(utilities.fits.__name__, 'INFO')
logging.setLevel(fitscube.deconvolve.archive_crawler, 'DEBUG')
logging.setLevel('fitscube.deconvolve.archive_crawler_config', 'DEBUG')
logging.setLevel(fitscube.deconvolve.algorithms, 'INFO')


# GLOBALS
# This should only ever get filled up when "parse_args()" is called at the start
# of the script. After that it's read-only so this should be OK if I'm careful.
ARGUMENT_OBJECT_CHOICES_DICTIONARY = {}

# -------

# TODO:
# * Set up method of re-running deconvolution with altered parameters from previous deconvolution file

class TimingSentry:
	def __init__(self, name, finish_message=True, output=print):
		self.name=name
		self.start_time = datetime.datetime.now()
		self.output=output
		self.finish_message = finish_message
		return
	def __enter__(self):
		self.output(f"### TimingSentry: {self.name} started at {self.start_time}")
		return
	def __exit__(self, type, value, traceback):
		self.end_time = datetime.datetime.now()
		if self.finish_message:
			self.output('\n'.join((	
				f"### TimingSentry: {self.name} finised at {self.end_time}.",
				f"### ------------: Elapsed time {self.end_time-self.start_time}"
			)))
		return


class TempUpdatedDict:
	def __init__(self, original_dict, update_dict):
		self.original_dict = original_dict
		self.update_dict = update_dict
		self.temp_dict = {}
		for k in self.update_dict.keys():
			self.temp_dict[k] = self.original_dict[k]
		return
	def __enter__(self):
		self.original_dict.update(self.update_dict)
		return(self.original_dict)
	def __exit__(self, etype, evalue, traceback):
		self.original_dict.update(self.temp_dict)
		return



def coerce_to_ssa2d_inputable(array, axis=None, show_plots=False):
	"""
	Want to make passed array inputtable to SSA2D, to do this I must remove all NANs and INFs
	from the data.

	# ARGUMENTS #
		array
			The numpy.ndarray to operate on
		axis
			<tuple<int>> A tuple (or None) of the axis that will be interpolated together.
			Usually I want to set this to axis=(1,2) as axis=(0,) is the
			spectral axis and the data can be treated separately in this
			direction.
		show_plots
			<bool> Should we display plots or not?

	# RETURNS #
		array
			The interpolated array
		nan_mask
			A boolean mask of where NAN values are
		posinf_mask
			A boolean mask of where +INF values are
		neginf_mask
			A boolean mask of ahwere -INF values are
	"""
	_lgr.INFO('Coercing array to correct format for SSA2D')

	# SSA2D does not work with NAN or INF elements
	_lgr.INFO('Finding masks for NAN, +INF, and -INF elements')
	nan_mask = np.isnan(array) | (array==0) # FITS sometimes uses zeros for nans, plus we should probably never have absolute zeros.
	posinf_mask = np.isposinf(array)
	neginf_mask = np.isneginf(array)

	mask = nan_mask | posinf_mask | neginf_mask

	# interpolate array at NAN or INF elements
	_lgr.INFO('Interpolating at combined NAN and INF masks')

	_lgr.INFO(f'Finding cube slices such that axes {axis} will be interpolated together')
	if axis is None:
		slices = tuple(slice(None) for n in array.shape)
	else:
		all_axis = tuple(range(array.ndim))
		permute_axis = tuple(i for i in all_axis if i not in axis)
		permute_axis_shape = tuple(array.shape[i] for i in permute_axis)
		permute_axis_idxs = np.array(np.nonzero(np.ones(permute_axis_shape))).T
		slices = tuple(tuple(y if i in permute_axis else slice(None) for i,y in enumerate(x)) for x in permute_axis_idxs)
		_lgr.DEBUG(f'{all_axis=}')
		_lgr.DEBUG(f'{permute_axis=}')
		_lgr.DEBUG(f'{permute_axis_shape=}')
		_lgr.DEBUG(f'{permute_axis_idxs=}')
		_lgr.DEBUG(f'{slices=}')

	_lgr.INFO('Interpolating found cube slices at combined NAN and INF masks')
	for i, aslice in enumerate(slices):
		print(f'Progress: {i}/{len(slices)}',end='\r')
		array[aslice] = ut.sp.interpolate_at_mask(array[aslice], mask[aslice], edges='convolution', method='linear')
	
	# Just interpolating the whole ND array can be very slow, interpolate over as few dimensions as is sensible.
	#array = ut.sp.interpolate_at_mask(array, mask, edges='convolution', method='linear')

	if show_plots: # DEBUGGING
		coerce_to_ssa2d_inputable_plots(array, nan_mask, posinf_mask, neginf_mask)

	return(array, nan_mask, posinf_mask, neginf_mask)


def coerce_to_ssa2d_inputable_plots(array, nan_mask, posinf_mask, neginf_mask):
	if array.ndim <= 2:
		array = array[None,...]

	img_shape = array.shape[:-2]
	img_idxs = tuple(tuple(x) for x in np.array(np.nonzero(np.ones(img_shape))).T)

	for img_idx in img_idxs:
		f1, a1 = ut.plt.figure_n_subplots(5)
		a1_iter=iter(a1.flatten())
		f1.suptitle(f'array[{img_idx}]')

		with Next(a1_iter) as ax:
			ax.set_title('Coerced array')
			ax.imshow(array[img_idx])
			ut.plt.remove_axes_ticks_and_labels(ax)
		
		with Next(a1_iter) as ax:
			ax.set_title('nan mask')
			ax.imshow(nan_mask[img_idx])
			ut.plt.remove_axes_ticks_and_labels(ax)

		with Next(a1_iter) as ax:
			ax.set_title('positive infinity mask')
			ax.imshow(posinf_mask[img_idx])
			ut.plt.remove_axes_ticks_and_labels(ax)

		with Next(a1_iter) as ax:
			ax.set_title('negative infinity mask')
			ax.imshow(neginf_mask[img_idx])
			ut.plt.remove_axes_ticks_and_labels(ax)

		with Next(a1_iter) as ax:
			ax.set_title('combined mask')
			ax.imshow(nan_mask[img_idx] | posinf_mask[img_idx] | neginf_mask[img_idx])
			ut.plt.remove_axes_ticks_and_labels(ax)

	plt.show()
	return

def plot_cube(array):
	if array.ndim <= 2:
		array = array[None,...]
	
	img_shape = array.shape[:-2]
	img_idxs = tuple(tuple(x) for x in np.array(np.nonzero(np.ones(img_shape))).T)

	f1, a1 = ut.plt.figure_n_subplots(len(img_idxs))
	a1_iter = iter(a1.flatten())
	f1.suptitle(f'plotting cube {id(array)}')
	for img_idx in img_idxs:
		with Next(a1_iter) as ax:
			ax.set_title(f'2d slice [{img_idx},:,:]')
			ax.imshow(array[img_idx])
			ut.plt.remove_axes_ticks_and_labels(ax)
	plt.show()
	return

# should really return a function and a 'argument_function' (i.e. a function that generates default arguments)
# but just get everything working for now and rely on good defaults.
def get_bp_flag_function_by_name(astring):
	bpf_dict = dict(
		cumulative_histrogram = fitscube.deconvolve.flag_bad_pixels.ssa2d_cumulative_histograms,
		probability_sum = fitscube.deconvolve.flag_bad_pixels.ssa2d_sum_prob_map,
		partial_sum_ratio = fitscube.deconvolve.flag_bad_pixels.ssa2d_ratio_bp_maps,
		chi_squared = fitscube.deconvolve.flag_bad_pixels.ssa2d_chisq_bp_map,
	)

	try:
		return(bpf_dict[astring])
	except KeyError:
		raise KeyError(f'Unknown bad pixel flagging function {repr(astring)}. Options are {list(bpf_dict.keys())}')
	return


def get_deconvolution_algorithm_by_name(astring):
	da_dict = dict(
		maximum_entropy = fitscube.deconvolve.algorithms.MaximumEntropy,
		lucy_richardson = fitscube.deconvolve.algorithms.LucyRichardson,
		clean_modified  = fitscube.deconvolve.algorithms.CleanModified,
	)
	try:
		return(da_dict[astring])
	except KeyError:
		raise KeyError(f'Unknown deconvolution algorithm name {repr(astring)}. Option are {list(da_dict.keys())}')
	return


def apply_deconvolution(
		cube_target : np.array, 
		cube_psf : np.array, 
		args : dict, 
		image_axis : tuple[int] =(1,2), 
		#signal_loss : float = 0.01,
		signal_loss : float = 0,
		cache_fpaths : dict = {}, # holds paths to cached files
	) -> tuple[np.array, tuple[slice], dict]:
	_lgr.INFO('In "apply_deconvolution()"')
	_lgr.INFO(f'{cube_target.shape=}')

	#reorder axes so that image axes are always last in cubes
	moved_image_axis = tuple(cube_target.ndim-i for i in range(len(image_axis),0,-1))
	non_image_axis = tuple(i for i in range(cube_target.ndim) if i not in moved_image_axis)
	cube_target = np.moveaxis(cube_target, image_axis, moved_image_axis)
	cube_psf = np.moveaxis(cube_psf, image_axis, moved_image_axis)


	# trim off excess 'frame', want to keep the non-image axes the same size.
	cube_applied_slices = []
	
	_lgr.INFO('Trimming off excess "frame" from cube_target and cube_psf')
	cube_target, target_frame_slices = ut.np.array_remove_frame(cube_target, axes=(1,2))
	cube_psf, psf_frame_slices = ut.np.array_remove_frame(cube_psf, axes=(1,2))
	_lgr.INFO(f'{cube_target.shape=}')
	
	cube_applied_slices.append(target_frame_slices)


	

	# now reduce the "excess" image around the target, do this by removing empty space
	cube_target, trim_slices = ut.np.trim_array(cube_target, signal_loss=signal_loss)
	cube_psf_tmp , trim_slices_psf_tmp = ut.np.trim_array(cube_psf, signal_loss=signal_loss)
	# alter the PSF slcies to have the same slice set in the spectral axis as the target slices
	trim_slices_psf = (*trim_slices[:moved_image_axis[0]],*trim_slices_psf_tmp[moved_image_axis[0]:])
	cube_psf = cube_psf[trim_slices_psf]

	cube_applied_slices.append(trim_slices)

	_lgr.INFO(f'{cube_target.shape=}')

	_lgr.INFO(f'Cube target trimmed slices {trim_slices}, resulting array shape {cube_target.shape}')
	_lgr.INFO(f'Cube psf trimmed slices {trim_slices_psf}, resulting array shape {cube_psf.shape}')



	_lgr.INFO('Adjusting PSF to ensure centrality and odd dimensions')
	#desired_center = ut.np.get_com(cube_psf).astype(int)
	slices = tuple(slice(None) if i not in moved_image_axis else slice(x-1-x%2) for i, x in enumerate(cube_psf.shape))
	cube_psf = cube_psf[slices]
	for i in range(cube_psf.shape[0]):
		desired_center = np.array(np.unravel_index(np.nanargmax(cube_psf[i]), cube_psf.shape[1:]))
		ut.np.center_on(cube_psf[i], point=desired_center)

	"""
	# DEBUGGING
	import matplotlib.pyplot as plt
	f, a = plt.subplots(1,2)
	a[0].imshow(cube_target[0])
	a[1].imshow(cube_psf[0])
	plt.show()
	"""

	_lgr.INFO('Finding bad pixel map')
	_lgr.INFO(f'Before ssa2d coersion {cube_target.shape=} {cube_psf.shape=}')
	with TimingSentry('ssa2d coersion') as TIMING:
		cube_target, nan_mask, posinf_mask, neginf_mask = get_cached_result(
			cache_fpaths.get('interp_target',None),
			lambda : coerce_to_ssa2d_inputable(
				cube_target,
				show_plots = args['plot_priority']>2,
				axis=moved_image_axis,
			)
		)

		cube_psf, nan_mask_psf, posinf_mask_psf, neginf_mask_psf = get_cached_result(
			cache_fpaths.get('interp_psf',None),
			lambda : coerce_to_ssa2d_inputable(
				cube_psf,
				show_plots=args['plot_priority']>2,
				axis=moved_image_axis,
			)
		)
		"""
		cube_target, nan_mask, posinf_mask, neginf_mask = coerce_to_ssa2d_inputable(
			cube_target, 
			show_plots=args['plot_priority']>2, 
			axis=moved_image_axis
		)
			
		cube_psf, nan_mask_psf, posinf_mask_psf, neginf_mask_psf = coerce_to_ssa2d_inputable(
			cube_psf, 
			show_plots=args['plot_priority']>2, 
			axis=moved_image_axis
		)
		"""

	_lgr.INFO(f'After ssa2d coersion {cube_target.shape=} {cube_psf.shape=}')


	_lgr.INFO('Setting deconvolution algorithm parameters')
	deconv = args['algorithm'](**ut.args.with_prefix(args, args['algorithm.parameter_prefix']))
		

	_lgr.INFO('Looping over each 2d image in cube')
	# cube_target and cube_psf should have the same size and shape for their non-image axes
	image_slices = tuple(zip(*np.nonzero(np.ones([cube_target.shape[_x] for _x in non_image_axis]))))

	bp_mask_target = np.zeros_like(cube_target, dtype=bool)
	bp_mask_psf = np.zeros_like(cube_psf, dtype=bool)

	with TimingSentry('Full Deconvolution') as TIMING:

		for i, image_slice in enumerate(image_slices):
			with TimingSentry(f"Deconvolution of slice {image_slice}/{[cube_target.shape[_x] for _x in non_image_axis]}") as TIMING:
				a2dslice_target = cube_target[image_slice]
				a2dslice_psf = cube_psf[image_slice]

				if np.all(np.isnan(a2dslice_target)):
					_lgr.INFO(f'Slice is all NANs, skipping deconvolution for this slice.')
					continue


				_lgr.INFO(''.join((
					f'Creating SSA of cube[{", ".join([str(_x) for _x in image_slice])}, ...]',
					f' of [{", ".join([str(cube_target.shape[_x]) for _x in non_image_axis])}, ..]',
				)))

				_lgr.INFO(f'Using SSA result to remove troublesome image areas')
				with TimingSentry('SSA calculation on science target') as TIMING:
					ssa_target = py_ssa.SSA2D(a2dslice_target, w_shape=args['ssa.wshape'], svd_strategy='eigval')
	
				with TimingSentry('SSA calculation on psf') as TIMING:
					ssa_psf = py_ssa.SSA2D(a2dslice_psf, w_shape=args['ssa.wshape'], svd_strategy='eigval')
			
				# Test to see if SSA can reduce convergence time by throwing away high frequency data
				#a2dslice_target = np.sum(ssa_target.X_ssa[:5], axis=0)
				#a2dslice_psf = np.sum(ssa_psf.X_ssa[:10], axis=0)
				
				_lgr.INFO('Finding bad pixel mask from SSA of cube slice')
				#bad_pixel_ff = get_bp_flag_function_by_name(args['bp_flag_func'])
				#bp_mask = bad_pixel_ff(ssa_target, show_plots=False)
				bp_mask_target[image_slice,...] = args['flag_bad_pixels_func'](ssa_target, **ut.args.with_prefix(args, args['flag_bad_pixels_func.parameter_prefix']))
				bp_mask_psf[image_slice,...] = args['flag_bad_pixels_func'](ssa_psf, **ut.args.with_prefix(args, args['flag_bad_pixels_func.parameter_prefix']))
		
				
				"""#DEBUGGING
				import matplotlib.pyplot
				f1, a1 = plt.subplots(2,2)
				ax_iter = iter(a1.flatten())

				ax = next(ax_iter)
				ax.imshow(a2dslice_psf.squeeze())
				ax.set_title('a2dslice_psf')

				ax = next(ax_iter)
				ax.imshow(bp_mask_psf.squeeze())
				ax.set_title('bp_mask_psf')

				ax = next(ax_iter)
				clims = (0, np.nanmax(a2dslice_psf))
				ax.imshow(a2dslice_psf.squeeze(), vmin=clims[0], vmax=clims[1])
				ax.set_title(f'a2dslice_psf\n{clims=}')
				"""# END DEBUGGING SECTION


				# delete ssa_target here to save memory
				_lgr.INFO('Deleting "ssa_target" and "ssa_psf" to save memory')
				del ssa_target
				del ssa_psf

				_lgr.INFO('Interpolating 2d image at bad pixel mask')

				a2dslice_target = ut.sp.interpolate_at_mask(a2dslice_target, bp_mask_target[image_slice], edges='convolution', method='cubic')
				a2dslice_psf = ut.sp.interpolate_at_mask(a2dslice_psf, bp_mask_psf[image_slice], edges='convolution', method='cubic')

				"""# CONTINUE DEBUGGING
				ax = next(ax_iter)
				ax.imshow(a2dslice_psf)
				ax.set_title('a2dslice_psf interpolated')
				plt.show()
				"""# END DEBUGGING


				_lgr.INFO(f'Normalising PSF')
				a2dslice_psf /= np.nansum(a2dslice_psf)



				_lgr.INFO('Running deconvolution algorithm')
				with TimingSentry('Deconvolution Algorithm') as TIMING:
					deconv(a2dslice_target, a2dslice_psf)

			
				_lgr.INFO('Applying constructed bad pixel mask')
				cube_target[image_slice,bp_mask_target[image_slice]] = np.nan
				cube_psf[image_slice,bp_mask_psf[image_slice]] = np.nan

				# store deconvolved components in original datacube memory to save space
				_lgr.INFO('storing deconvolved components in datacube memory')
				# DEBUGGING
				#cube_target_sliced_view = cube_target
				#for cube_applied_slice in cube_applied_slices:
				#	_lgr.INFO(f'{cube_applied_slice=}')
				#	cube_target_sliced_view = cube_target_sliced_view[cube_applied_slice]
				#	_lgr.INFO(f'{cube_target_sliced_view.shape=}')
				#_lgr.INFO(f'{image_slice=}')
				#_lgr.INFO(f'{np.shares_memory(cube_target_sliced_view, cube_target)=}')
				cube_target[image_slice,...] = deconv.get_components()

	
	_lgr.INFO('Cube has been fully deconvolved')

	_lgr.INFO('Re-applying nan_inf_mask')
	cube_target[nan_mask] = np.nan
	cube_target[posinf_mask] = np.inf
	cube_target[neginf_mask] =-np.inf
	_lgr.INFO(f'{cube_target.shape=}')
	
	"""
	# DEBUGGING
	import matplotlib.pyplot as plt
	plt.imshow(cube_target[0])
	plt.show()
	"""

	return(np.moveaxis(cube_target, moved_image_axis, image_axis), cube_applied_slices, deconv.get_params(), bp_mask_target, bp_mask_psf)


def apply_AOCD_to(args):
	"""
	Put in the functions and classes, not just function string names.
	"""
	for k, v in ARGUMENT_OBJECT_CHOICES_DICTIONARY.items():
		if k in args:
			args[f'{k}.parameter_prefix'] = ut.args.prefix_sep_str.join((k,args[k]))
			args[k] = v[args[k]]
	return

def write_algorithm_parameters(args, fname):
	algorithm_params = ut.args.with_prefix(args, 'algorithm', min_depth=2)

	with open(fname,'w') as f:
		json.dump(algorithm_params, f, indent='\t', separators=(', ',' : '))
	return

def read_algorithm_parameters(fname):
	with open(fname,'r') as f:
		params = json.load(f)
	algorithm_params = {}
	for k, v in params.items():
		algorithm_params['algorithm.'+k] = v

	apply_AOCD_to(algorithm_params) # shouldn't do much but I should be careful.
	return(algorithm_params)


def get_fits_writer(
		output_fits_mode : str, 
		output_file_exists : bool,
		output_fits_include_extensions : bool,
		output_hdul : typing.Optional[fits.HDUList] = fits.HDUList(fits.PrimaryHDU()),
		output_ext : typing.Optional[int] = 0,
	) -> typing.Callable[[str, list[np.ndarray], list[fits.Header]], typing.Any]:
	"""
	Get a callable that writes a fits file with the prototype "fits_writer(fpath, data, hdr)"
	"""
	if not output_fits_include_extensions:
		output_hdul = fits.HDUList([fits.PrimaryHDU()])
		output_ext = 0

	def hdul_to_write(data, hdr):
		output_hdul[output_ext].data = data
		output_hdul[output_ext].header = hdr
		return(output_hdul)

	def append_to_hdul(fpath, data, hdr):
		hdul = hdul_to_write(data, hdr)
		with fits.open(fpath, mode='update') as hdul_to_append_to:
			for hdu in hdul:
				hdul_to_append_to.append(hdu)
			hdul_to_append_to.flush(output_verify='warn')
		return

	if output_fits_mode == "overwrite":
		fits_writer = lambda fpath, data, hdr: hdul_to_write(data,hdr).writeto(fpath, checksum=False, output_verify='warn', overwrite=True)
	elif (not output_file_exists) and (output_fits_mode == 'no_overwrite'):
		fits_writer = lambda fpath, data, hdr: hdul_to_write(data,hdr).writeto(fpath, checksum=False, output_verify='warn', overwrite=False)
	elif output_fits_mode == 'append':
		fits_writer = append_to_hdul
	elif output_fits_mode == 'no_output':
		fits_writer = lambda fpath, data, hdr: _lgr.INFO(f'Argument "--output.fits.mode" = "no_output", no file has been written')
	else:
		fits_writer = None
	return(fits_writer)
		

def reduce_data_volume_for_testing(hdul, ext, reduction_spec=None):
	from utilities.classes import Slice

	oshape = hdul[ext].data.shape
	reduction_spec = f'[{oshape[0]//2}:{oshape[0]//2+2}, :, :]' if reduction_spec is None else reduction_spec
	"""
	slices = eval(f'np.s_{reduction_spec}')
	print(slices)
	#slices = (slice(oshape[0]//2,oshape[0]//2+1), slice(None), slice(None))
	j = hdul.index_of(hdul[ext])
	for i in range(len(hdul)):
		if hdul[i].data is None: continue
		if hdul[i].data.shape != oshape :continue
		hdul[i].data = hdul[i].data[slices]
		for j in range(len(slices)):
			smin, smax, sstep = slices[j].indices(hdul[i].data.shape[j])
			crpix_str = f'CRPIX{3-len(slices)-j}'
			cdelt_str = f'CDELT{3-len(slices)-j}'
			hdul[i].header[crpix_str] = hdul[i].header.get(crpix_str, 1) - smin
			hdul[i].header[cdelt_str] = hdul[i].header.get(cdelt_str, 1)*sstep
	"""
	hdul[ext].data = eval(f'Slice(hdul[ext].data)["{reduction_spec[1:-1]}"]')
	return(hdul)


def get_cached_result(cache_fpath, acallable):
	if cache_fpath is None:
		_lgr.INFO('No cache, running callable...')
		# no cache to use so just run the callable and return the result.
		return(acallable())
	
	recompute_result_flag = False
	import pickle
	if os.path.exists(cache_fpath):
		try:
			_lgr.INFO(f'Found cache at "{cache_fpath}" reading file...')
			# if the cache exists, we should grab data from it
			with open(cache_fpath, 'rb') as f:
				result = pickle.load(f)
		except Exception as E:
			_lgr.INFO(f'Cache seems corrupt (threw exception: {E}), running callable and re-creating cache file...')
			recompute_result_flag = True
	else:
		_lgr.INFO(f'No cache at "{cache_fpath}", running callable and creating cache file...')
		recompute_result_flag = True
	
	if recompute_result_flag:
		cache_dir = os.path.dirname(cache_fpath)
		if not os.path.exists(cache_dir):
			_lgr.INFO(f'Creating cache directory "{cache_dir}"...')
			os.makedirs(cache_dir)
			# we almost certainly do not want to crawl the cache file, so make the ".do_not_crawl" file.
			with open(os.path.join(cache_dir, ".do_not_crawl"), 'w') as f: pass
		# the cache does not exist, or is broken, so run the callable and store the result
		result = acallable()
		with open(cache_fpath, 'wb') as f:
			pickle.dump(result, f)
	
	# return the result
	return(result)


def get_cache_fpath_from(
		source_fpath, 
		fmt_str, 
		fmt_kwargs={},
		is_stale_callable = lambda source_fpath, cache_fpath: (os.path.getmtime(source_fpath) > os.path.getmtime(cache_fpath)),
	) -> typing.Optional[str]:
	_lgr.DEBUG(f'Getting cache file ...')
	cache_fpath = get_related_fpath_from_fmt(source_fpath, fmt_str, **fmt_kwargs)
	_lgr.DEBUG(f'Found cache file "{cache_fpath}" {os.path.exists(cache_fpath)=}')
	if (not os.path.exists(cache_fpath)) or (cache_fpath is None):
		# If the cache_fpath does not exist, then we want to make one
		# so returh the path.
		# if "fmt_str" is None then we don't have a way to get a cache_fpath anyway so
		# just return None to indicate there is no cache available
		return(cache_fpath)

	_lgr.DEBUG(f'{os.path.getmtime(source_fpath)=} {os.path.getmtime(cache_fpath)=} {is_stale_callable(source_fpath, cache_fpath)=}')
	# cache_fpath must exists if we get here
	if is_stale_callable(source_fpath, cache_fpath):
		_lgr.DEBUG('Cache is stale, deleting...')
		# the cache is stale so delete it
		os.remove(cache_fpath) 
		return(cache_fpath)
	else:
		# the cache is not stale so keep it
		return(cache_fpath)

	# if we've gotten something is weird so return None
	return(None)


def get_related_fpath_from_fmt(original_fpath, fmt_str, **kwargs):
	_lgr.DEBUG(f'Getting related fpath from {original_fpath} via format string "{fmt_str}" with extra params {kwargs}')
	if fmt_str is None: 
		# if we don't have a format string, we can't get a related file
		return(None)
	fdir, fnamefext = os.path.split(original_fpath)
	fname, fext = os.path.splitext(fnamefext)
	related_fpath = os.path.abspath(
		fmt_str.format(
			cwd = os.getcwd(),
			fdir = fdir,
			fname = fname,
			fext = fext,
			timestamp = datetime.datetime.now().isoformat(),
			**kwargs
		)
	)
	return(related_fpath)

def parse_args(argv):
	import argparse
	import inspect
	
	# can't directly pass functions or classes as arguments but can pass their names.

	parser = ut.args.DocStrArgParser(
		description=__doc__,
		fromfile_prefix_chars = '@',
		allow_abbrev=False,
	)


	## Add Positional Arguments ##
	parser.add_argument(
		'archive_roots', 
		type=str, 
		nargs='+',
		help='Root directory of an archive to be searched for data with "archive_crawler.py"',
	)

	## Add Optional Arguments ##
	parser.add_argument(
		'--ssa.wshape', 
		type=int, 
		nargs=2, 
		help='Dimensions of the window used in single spectrum analysis routines', 
		default=(10,10)
	)

	parser.add_argument(
		'--output.fits.mode',
		type=str,
		choices=('no_overwrite', 'overwrite', 'append', 'no_output'),
		help='How should we treat outputting a *.fits file. "overwrite" - overwrite any files with the same name, "append" - add a new image extension if the file exists.',
		default='no_overwrite',
	)

	parser.add_argument(
		'--output.fits.fmt_str', 
		type=ut.args.OptionalType(str), 
		help=' '.join((	
			'Format string for the output file name, used in "str.format()",',
			'any nonexistent folders will be created when the file is written. The following',
			'variables are avilable: "fdir" - the name of the file\'s directory, "fname" -',
			'the name of the file without it\'s extension, "fext" - the extension of the file',
			'including the ".", "cwd" - the current working directory, "algorithm" - the name',
			'of the deconvolution algorithm, "timestamp" - the iso format timestamp',
			'(YYYY-MM-DDTHH:mm:ss.ffffff) of the file creation time',
		)), 
		default="{fdir}/deconvolution/{fname}_deconv_{algorithm}{fext}"
	)

	parser.add_argument(
		'--output.cache.interp.fmt_str',
		type=ut.args.OptionalType(str),
		help = ' '.join((
			'Format string for the output file name of an optional "interpolated"',
			'fits file which contains the same data as the science observation, but',
			'interpolated at all NAN and +/-INF values. This can take while to compute',
			'and should not change unless the data does, so it is not a bad idea to cache',
			'the data. The same paramters as "--output.fits.fmt_str" apply to the format',
			'string. The data will be written as a FITS file.'
		)),
		default="{fdir}/cache/{fname}_interpolated.pkl"
	)

	parser.add_argument(
		'--output.cache.refresh',
		type=str,
		nargs='*',
		action='extend',
		#const = ['ALL',],
		help= ' '.join((
			'If present will refresh all caches, if passed a string will only refresh the passed string caches',
		)),
		default=None
	)

	parser.add_argument(
		'--output.fits.include_extensions',
		action=ut.args.ActionTf,
		prefix='output.fits.',
		help=''.join((
			'If present the output file will include the extensions present in the original file, any slicing ',
			'will be applied to extensions that have the same size and shape as the initial data',
		)),
	)

	parser.add_argument(
		'--output.argfile', 
		nargs='?', 
		type=ut.args.OptionalType(str), 
		help=f'Create a file containing the current command line arguments. Use on command line as {parser.fromfile_prefix_chars}<argfilename>', 
		default=None, const='args.txt'
	)
	parser.add_argument(
		'-v', 
		action='count', 
		help='Increases the verbosity level of logging', 
		default=0
	)
	parser.add_argument(
		'--!output.plots.show', 
		action=ut.args.ActionTF, 
		prefix='!output.plots.', 
		help='Global overwrite for showing (or not) plots'
	)
	parser.add_argument(
		'--!output.plots.save', 
		action=ut.args.ActionTF, 
		prefix='!output.plots.', 
		help='Global overwrite for saving (or not) plots'
	)
	parser.add_argument(
		'--!output.plots.dir', 
		type=ut.args.OptionalType(str), 
		help=''.join((
			'Global overwrite for plotting directory, if path starts with a ',
			'"/", is an absolute path. Otherwise path is relative to the current working directory.'
		)),
	)
	parser.add_argument(
		'--plot_priority', 
		type=int, 
		help='DEPRECIATED: controls how far "down the stack" plots are made', 
		default=0
	)

	parser.add_argument(
		'--algorithm.param_file',
		type=str,
		default='deconv.params',
		help=''.join((
			'If the specified file is present relative to the current observation *.fits file ',
			'then use parameters specified in that file instead of default parameters. Parameters ',
			'not in the file will take their default values. Pass an absolute path (e.g. ',
			'"/some/absolute/path") to use a non-relative file. Assumes JSON format.'
		))
	)

	parser.add_argument(
		'--algorithm.write_param_file',
		type=str,
		default=None,
		help=''.join((
			'File to write current algorithm parameters to, this should write the same format as ',
			'the "--algorithm.param_file" option reads. Use it to write a parameter file you can ',
			'alter easily and change locally for a dataset. JSON format.',
		))
	)

	parser.add_argument(
		'--testing.reduce_data_volume',
		type=str,
		nargs='?',
		const=None,
		help = "For testing purposes, reduce the data-volume of all inputs. Useful for testing code changes.",
		default=False
	)

	parser.add_argument(
		'--testing.exit_after',
		type=int,
		default=0,
		help='Number of items to deconvolve (if +ve) or visit (if -ve) before exiting, useful for testing.',
	)


	############### OPERATE ON ARGUMENTS BELOW THIS LINE ###################


	# Add all dataclasses from "fitscube.deconvolve.algorithms" that do not include "Base" in their name
	search_modules = (
		(fitscube.deconvolve.algorithms, "algorithm", "Choice of deconvolution algorithm"),
	)
	for i, (amodule, prefix_name, amodule_help) in enumerate(search_modules):
		ARGUMENT_OBJECT_CHOICES_DICTIONARY.update(
			ut.args.add_args_classes_from_module_set_as_choices(parser, amodule, prefix_name, amodule_help, 
				set_default='LucyRichardson', 
				class_filter=lambda cls_name_obj_tpl: (
					dc.is_dataclass(cls_name_obj_tpl[1]) 
					and (not 'Base' in cls_name_obj_tpl[0])
				)
			)
		)

	# Add all the "bad pixel flagging" routines that we have written
	arg_funcs = (
		(	'flag_bad_pixels_func', 
			(	fitscube.deconvolve.flag_bad_pixels.ssa2d_sum_prob_map,
				fitscube.deconvolve.flag_bad_pixels.ssa2d_ratio_bp_maps,
				fitscube.deconvolve.flag_bad_pixels.ssa2d_cumulative_histograms,
			), 
			'Choice of functions designed to identify bad pixels'
		),
	)
	# TODO: Move this into a function in ".../utilities/args.py"
	for module_prefix, function_list, function_list_help in arg_funcs:
		ARGUMENT_OBJECT_CHOICES_DICTIONARY.update(
			ut.args.add_args_callable_set_as_choices(
				parser, 
				function_list, 
				module_prefix, 
				function_list_help
			)
		)

	
	# add arguments from modules
	parser = utilities.logging_setup.add_arguments(parser, prefix='logging', suppress_help=False)



	# TESTING
	#print(parser.format_help())

	args = vars(parser.parse_args()) # I prefer a dictionary interface
	
	
	# Make cache refreshing work as intended
	args['output.cache.refresh'] = ['ALL'] if args['output.cache.refresh'] is not None and len(args['output.cache.refresh'])==0 else args['output.cache.refresh']


	# if we were instructed to output the current argument set to a file, do so.
	if args['output.argfile'] is not None:
		ut.args.output_current_parser_args_to_file(parser, args, args['output.argfile'])
		_lgr.INFO(f'Arguments written to {args["output.argfile"]}')
		sys.exit()

	# if we were instructed to write a human-readable algorithm parameter file, do so
	if args['algorithm.write_param_file'] is not None:
		algo_par_file = os.path.join(os.getcwd(),args['algorithm.write_param_file'])
		write_algorithm_parameters(args, algo_par_file)
		_lgr.INFO(f'Algorithm parameters written to {algo_par_file}')
		sys.exit()

	## Filter Arguments ##
	apply_AOCD_to(args)

	# print arguments
	_lgr.INFO('# ARGUMENTS #')
	for k, v in args.items():
		_lgr.INFO(f'{k} = {v!r}')
	_lgr.INFO('# --------- #')

	# call module-specific argument handlers
	utilities.logging_setup.handle_arguments(ut.args.with_prefix(args, prefix='logging'))

	return(args)


def main(**kwargs):
	sys.argv = kwargs.get('sys.argv', sys.argv)

	args = parse_args(sys.argv[1:])

	n_items_deconvolved = 0
	n_items_visited = args['testing.exit_after'] # -ve if using this option, therefore count up to zero

	with TimingSentry('Total archive crawl') as TIMING:

		# make sure that we have absolute paths to the archive files we want to deconvolve.
		for archive_root in [os.path.realpath(os.path.expanduser(_archive_root)) for _archive_root in args['archive_roots']]:
			_lgr.DEBUG('\n'+'#'*80) # separator for different archive roots
			_lgr.DEBUG(f'Operating on archive root "{archive_root}"')
			for found_sci_and_std_files in fitscube.deconvolve.archive_crawler.standard_star_crawl(archive_root):
				# loop over each science and standard file found in the current archive node
				for target_path, target_ext, psf_path, psf_ext in zip(*found_sci_and_std_files):
					_lgr.DEBUG('\n'+'='*80)
					_lgr.DEBUG(f'Operating on target {target_path} extension {target_ext}\nusing PSF {psf_path} extension {psf_ext}')
					#if 'smooth' not in target_path: continue # DEBUGGING, only run on smoothed cubes for now
					with fits.open(target_path) as target_hdul, fits.open(psf_path) as psf_hdul:

						# Create output fits file name
						deconv_fits_path = get_related_fpath_from_fmt(
							target_path, 
							args['output.fits.fmt_str'], 
							algorithm=args['algorithm'].__name__
						)

						# Get function to write, overwrite, or append to FITS file
						fits_writer = get_fits_writer(
							args['output.fits.mode'], 
							os.path.exists(deconv_fits_path), 
							args['output.fits.include_extensions'], 
							target_hdul, 
							target_ext
						)
						
						# If we don't have any function that would write a fits file, then skip this cube.
						if fits_writer is None:
							_lgr.INFO(''.join((
								f'Propopsed output file {deconv_fits_path} ',
								f'{"already exists" if os.path.exists(deconv_fits_path) else "not present"}, ',
								f'and argument "--output.fits.mode" = {args["output.fits.mode"]}, ',
								f'could not find a FITS writing function for this scenario. Skipping file...',
							)))
							n_items_visited += 1
							continue

						# if we are doing some testing, just take the middle slice of the spectral axis
						# put this in a function and apply it to the whole cube
						if type(args['testing.reduce_data_volume']) in (type(None), str):
							_lgr.INFO('reducing data volume for testing')
							#reduce_data_volume = lambda data: data[data.shape[0]//2:data.shape[0]//2+1]
							target_hdul = reduce_data_volume_for_testing(target_hdul, target_ext, args['testing.reduce_data_volume'])
							psf_hdul = reduce_data_volume_for_testing(psf_hdul, psf_ext, args['testing.reduce_data_volume'])

						if args['plot_priority'] > 1:
							plot_cube(target_hdul[target_ext].data)

						if ((args['algorithm.param_file'] is not None) 
							and os.path.exists(os.path.join(os.path.dirname(target_path), args['algorithm.param_file']))
							):
							local_algo_params = read_algorithm_parameters(os.path.join(os.path.dirname(target_path), args['algorithm.param_file']))
						else:
							local_algo_params = {}

						with TempUpdatedDict(args, update_dict = local_algo_params) as args:
							# perform deconvolution.
							_lgr.INFO(f'Deconvolving target {target_path} extension {target_ext} with PSF {psf_path} extension {psf_ext}')

							def is_stale_callable(source_fpath, cache_fpath, refresh_list = args['output.cache.refresh'], cache_name='interp'):
								#print(f'{source_fpath=}')
								#print(f'{cache_fpath=}')
								#print(f'{refresh_list=}')
								#print(f'{cache_name=}')
								#print(f'{any(tuple((x in refresh_list) for x in (cache_name, "ALL")))=}')
								if any(tuple((x in refresh_list) for x in (cache_name,'ALL'))):
									return(True)
								return(os.path.getmtime(source_fpath) > os.path.getmtime(cache_fpath))

							cache_fpaths = { # only pass cached files if they are newer than the files they are based on.
								'interp_target' : get_cache_fpath_from(
									target_path, 
									args['output.cache.interp.fmt_str'], 
									is_stale_callable = is_stale_callable,
								),
								'interp_psf' : get_cache_fpath_from(
									psf_path, 
									args['output.cache.interp.fmt_str'],
									is_stale_callable = is_stale_callable,
								),
							}


							(	target_cube, 
								target_applied_slices, 
								deconv_params,
								bp_mask_target,
								bp_mask_psf
							) = apply_deconvolution(
								target_hdul[target_ext].data, 
								psf_hdul[psf_ext].data, 
								args,
								cache_fpaths = cache_fpaths
							)
						

						# update target header data for slices and deconvolution parameters. May still need tweaking.
						_lgr.INFO('Updating FITS header of deconvolved data')
						deconv_hdr = target_hdul[target_ext].header
						for applied_slice in target_applied_slices:
							_lgr.INFO(f'{applied_slice}')
							deconv_hdr = ut.fits.header_apply_slice(deconv_hdr, applied_slice)

						if args['output.fits.include_extensions']:
							_lgr.INFO('Applying data transforms to relevant FITS extensions.')
							target_hdul.info() # DEBUGGING
							j = target_hdul.index_of(target_hdul[target_ext])
							for i in range(len(target_hdul)):
								if i == j: continue
								_lgr.INFO(f'Considering extension {i=} data extension {j=}')
								if target_hdul[i].data is None: continue

								if ((target_hdul[i].data.shape != target_hdul[target_ext].data.shape)
									and (target_hdul[i].data.shape != target_hdul[target_ext].data.shape[1:])
									): continue
								
								for applied_slice in target_applied_slices:
									_lgr.INFO(f'{applied_slice=}')
									target_hdul[i].data = target_hdul[i].data[applied_slice[3-target_hdul[i].data.ndim:]]
									target_hdul[i].header = ut.fits.header_apply_slice(
										target_hdul[i].header, 
										applied_slice[3-target_hdul[i].data.ndim:]
									)

						target_hdul.append(fits.ImageHDU(header=None, data=bp_mask_target.astype(float), name='Bad Pixel Mask'))


						
						# output FITS file with deconvolved data.
						_lgr.INFO(f'Writing deconvolved data to fits file at {repr(deconv_fits_path)}')
						if fits_writer is not None:
							deconv_output_dir = os.path.dirname(deconv_fits_path)
							if not os.path.exists(deconv_output_dir):
								_lgr.INFO(f'Directory {deconv_output_dir} does not exist, creating it.')
								os.makedirs(os.path.dirname(deconv_fits_path), exist_ok=True) # ensure containing directories are present
								_lgr.INFO(''.join((
									f'We almost certainly do not want to crawl into the newly created directory, so we will write a ".do_not_crawl"',
									f'file as a signal for archive crawlers that have configurations that inherit from ',
									f'"archive_crawler_config.ArchiveNodeConfigBase" to not search this folders or it\'s sub-folders',
								)))
								with open(os.path.join(deconv_output_dir, ".do_not_crawl"), 'w') as f:
									f.write("If this file is present in a directory, an archive crawler will not search it or it's chilren\n")
							fits_writer( # write fits file in desired mode
								deconv_fits_path,
								target_cube,
								deconv_hdr
							)
						else:
							_lgr.WARN(''.join((
								f'Could not write deconvolved observation to {deconv_fits_path} as file exists and ',
								f'"--output.fits.mode" is {args["output.fits.mode"]}, must be one of ',
								f'("overwrite", "append") to output when that file is already present.',
							)))

						# Print information to user about how "astropy" module will want to throw warnings at them
						_lgr.INFO(''.join((
							"Astropy does not understand the FITS conventions of long-string keywords ",
							"(i.e. the CONTINUE keyword) and the ESO-developed HIERARCH keyword. Astropy",
							" has therefore been set to warn users when a FITS file does not conform to ",
							"it's standards, but NOT to fix any errors it detects. If any warnings astropy ",
							"is giving you are of the form \"Card 'CONTINUE' is not FITS standard...\", then ",
							"those warnings can be safely ignored. See [the FITS standard](https://fits.",
							"gsfc.nasa.gov/fits_standard.html) for more information."
						)))

						# Finally, update counters
						n_items_deconvolved += 1
						n_items_visited += 1


					if (args['testing.exit_after'] is not None):
						if (args['testing.exit_after'] > 0) and (n_items_deconvolved >= args['testing.exit_after']):
							_lgr.INFO(f'Exiting after {n_items_deconvolved} deconvolutions')
							sys.exit(f'EXIT {__file__}-{sys._getframe().f_lineno} FOR TESTING') # DEBUGGING
						if (args['testing.exit_after'] < 0) and (n_items_visited >= 0):
							_lgr.INFO(f'Exit after {n_items_visited} items visited')
							sys.exit(f'EXIT {__file__}-{sys._getframe().f_lineno} FOR TESTING') # DEBUGGING

		

if __name__=='__main__':
	_lgr.INFO(f'"{__file__}" called as top-level entry point using "{sys.executable}"')

	test_params = {}
	# TESTING OVERWRITES
	if len(sys.argv) == 1:
		_lgr.INFO('No arguments passed, sending TESTING arguments.')
		test_params['sys.argv'] = [__file__,]
		#test_params['sys.argv'].insert(1,"/network/group/aopp/planetary/PGJI004_IRWIN_ICEGIANT/archive")
		#test_params['sys.argv'].insert(1,"/network/group/aopp/planetary/PGJI004_IRWIN_ICEGIANT/archive/Neptune/VLT_MUSE")
		#test_params['sys.argv'].insert(1,"/network/group/aopp/planetary/PGJI004_IRWIN_ICEGIANT/archive/Neptune/Gemini_NIFS/2009/20090902")
		#test_params['sys.argv'].insert(1,"/network/group/aopp/planetary/PGJI004_IRWIN_ICEGIANT/archive/Neptune/Gemini_NIFS/2009/20090901")
		#test_params['sys.argv'].insert(1,"/network/group/aopp/planetary/PGJI004_IRWIN_ICEGIANT/archive/Neptune/Gemini_NIFS/2009/20090907")
		#test_params['sys.argv'].insert(1,"/network/group/aopp/planetary/PGJI004_IRWIN_ICEGIANT/archive/Neptune/Gemini_NIFS/2009/")
		
		test_params['sys.argv'].insert(1,"/network/group/aopp/planetary/PGJI004_IRWIN_ICEGIANT/archive/Neptune/Gemini_NIFS/2009/20090907")
		test_params['sys.argv'].insert(2,"/network/group/aopp/planetary/PGJI004_IRWIN_ICEGIANT/archive/Neptune/Gemini_NIFS/2009/")
		test_params['sys.argv'].insert(3,"/network/group/aopp/planetary/PGJI004_IRWIN_ICEGIANT/archive/Uranus/Gemini_NIFS/2009/")
		
		#test_params['sys.argv'].insert(1,"/network/group/aopp/planetary/PGJI004_IRWIN_ICEGIANT/archive/Uranus/Gemini_NIFS/2009/")
		#test_params['sys.argv'] += ['--output.fits.fmt_str', '{cwd}/test_{fname}_deconv{fext}']
		test_params['sys.argv'] += ['--output.fits.fmt_str', '{fdir}/deconvolution/test_{fname}_deconv_{algorithm}{fext}']
		test_params['sys.argv'] += ['--output.fits.mode', 'no_overwrite']
		#test_params['sys.argv'] += ['--output.fits.mode', 'overwrite']
		#test_params['sys.argv'] += ['--output.fits.mode', 'no_output']
		test_params['sys.argv'] += ['--algorithm', 'LucyRichardson']
		test_params['sys.argv'] += ['--algorithm.LucyRichardson.n_iter', '200'] # DEBUGGING
		#test_params['sys.argv'] += ['--testing.reduce_data_volume',] # TESTING
		#test_params['sys.argv'] += ['--output.fits.fmt_str', '{fdir}/deconvolution/test_{fname}_deconv_{algorithm}_reduced{fext}']
		#test_params['sys.argv'] += ['--testing.exit_after', '1']
		#test_params['sys.argv'] += ['--algorithm.write_param_file', 'deconv.params']
		_lgr.INFO(f'TESTING arguments: {" ".join(test_params["sys.argv"])}')
	# TESTING

	main(**test_params)
	





