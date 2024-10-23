"""
Script that deconvolves the first argument with the second argument

Example invocation: 
	`python -m aopp_deconv_tool.deconvolve './example_data/test_rebin.fits{DATA}[10:12]{CELESTIAL:(1,2)}' './example_data/fit_example_psf_000.fits[10:12]{CELESTIAL:(1,2)}'`
"""

import sys, os
from pathlib import Path
from typing import Literal

from concurrent.futures import ProcessPoolExecutor as Executor, as_completed
from multiprocessing import Queue as MsgQueue
import multiprocessing as mp

import numpy as np
from astropy.io import fits

import aopp_deconv_tool.astropy_helper as aph
import aopp_deconv_tool.astropy_helper.fits.specifier
import aopp_deconv_tool.astropy_helper.fits.header
import aopp_deconv_tool.numpy_helper as nph
import aopp_deconv_tool.numpy_helper.axes
import aopp_deconv_tool.numpy_helper.slice
import aopp_deconv_tool.psf_data_ops as psf_data_ops
from aopp_deconv_tool.fpath import FPath
import aopp_deconv_tool.arguments as arguments


from aopp_deconv_tool.algorithm.deconv.clean_modified import CleanModified
from aopp_deconv_tool.algorithm.deconv.lucy_richardson import LucyRichardson

import matplotlib as mpl
import matplotlib.cm
import matplotlib.pyplot as plt
import copy
import aopp_deconv_tool.plot_helper as plot_helper
from aopp_deconv_tool.plot_helper.base import AxisDataMapping
from aopp_deconv_tool.plot_helper.plotters import PlotSet, Histogram, VerticalLine, Image, IterativeLineGraph, IterativeLogLineGraph, HorizontalLine

import aopp_deconv_tool.cfg.logs
_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'WARN')


deconv_methods = {
	'clean_modified' : CleanModified,
	'lucy_richardson' : LucyRichardson
}

class IterationTracker:
	def __init__(self, n_iter, print_interval):
		self.i_iter = 0
		self.n_iter = n_iter
		self.print_interval = print_interval
		
	def print(self):
		if self.i_iter % self.print_interval == 0:
			self.clear()
			print(f'Iteration {self.i_iter}/{self.n_iter} [{100*self.i_iter/self.n_iter}%]', end='')
			
	def in_notebook(self):
		try:
			from IPython import get_ipython
			if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
				return False
		except ImportError:
			return False
		except AttributeError:
			return False
		return True
	
	def reset(self):
		self.i_iter=0
	
	def clear(self):
		if self.in_notebook():
			from IPython.display import clear_output
			clear_output(True)
		else:
			print('\r', end='')
	
	def update(self, *args, **kwargs):
		self.print()
		self.i_iter += 1
	
	def complete(self, *args, **kwargs):
		self.clear()
		print(f'Iteration {self.i_iter}/{self.n_iter} [{100*self.i_iter/self.n_iter}%]', end='\n')


class Iterator:
	def __init__(self, obj):
		self._iterator = iter(obj)
		self._stack = []
		self._cached = None
		self._cached_valid = False
	
	def is_valid(self):
		if not self._cached_valid:
			raise RuntimeError("Current value of iterator is invalid, maybe iterator has not started or already ended?")
	
	@property
	def next(self):
		return next(self)
	
	@property
	def current(self):
		self.is_valid()
		return self._cached
		
	def push(self, obj):
		# Push a value into the iterator at the next location
		self._stack.append(obj)
		return self
	
	def __next__(self):
		if len(self._stack) > 0:
			self._cached = self._stack.pop()
			self._cached_valid = True
			return self._cached
		
		try:
			self._cached = next(self._iterator)
			self._cached_valid = True
		except StopIteration:
			self._cached_valid = False
			raise
		else:
			return self._cached

def create_plot_set(deconvolver, cadence = 1):
	"""
	Creates a set of plots that are updated every `cadence` steps. Useful to see exactly what a deconvolver is doing.
	"""
	fig, axes = plot_helper.figure_n_subplots(8)
	axes_iter = Iterator(axes)
	a7_2 = axes[7].twinx()
	
	try:
		cmap = mpl.colormaps['bwr_oob']
	except KeyError:
		cmap = copy.copy(mpl.colormaps['bwr'])
		cmap.set_over('magenta')
		cmap.set_under('green')
		cmap.set_bad('black')
		mpl.colormaps.register(name='bwr_oob', cmap=cmap)
	
	try:
		viridis_oob = mpl.colormaps['viridis_oob']
	except KeyError:
		viridis_oob = mpl.colormaps['viridis'].copy()
		viridis_oob.set_bad(color='magenta', alpha=1)
		#viridis_oob.set_under(color='black', alpha=1)
		#viridis_oob.set_over(color='black', alpha=1)
		mpl.colormaps.register(name='viridis_oob', cmap=viridis_oob)
	
	
	
	def selected_pixels_non_selected_are_nan(x):
		r = np.array(x._selected_px)
		r[r==0] = np.nan
		return r
		
	
	
	plot_set = PlotSet(
		fig,
		'clean modified step={self.n_frames}',
		cadence=cadence,
		plots = [	
			Histogram(
				'residual', 
				static_frame=False,
				axis_data_mappings = (AxisDataMapping('value','bins',limit_getter=plot_helper.lim), AxisDataMapping('count','_hist',limit_getter=plot_helper.LimRememberExtremes()))
			).attach(axes_iter.next, deconvolver, lambda x: x._residual),
		 	
			VerticalLine(
				None, 
				static_frame=False, 
				plt_kwargs={'color':'red'}
			).attach(axes_iter.current, deconvolver, lambda x: x._pixel_threshold),
			
			Histogram(
				'residual', 
				static_frame=False,
				plt_kwargs={'color' : 'green', 'alpha':0.3},
				axis_labels=(None, 'log(count)'),
				ax_funcs=[lambda ax: ax.set_yscale('log')],
				axis_data_mappings = (AxisDataMapping('value','bins',limit_getter=plot_helper.lim), AxisDataMapping('count','_hist',limit_getter=plot_helper.LimRememberExtremes()))
			).attach(axes_iter.current.twinx(), deconvolver, lambda x: x._residual),
			
			Image(
		 		'residual'
		 	).attach(axes_iter.next, deconvolver, lambda x: x._residual),
			
			Image(
		 		'current cleaned'
			).attach(axes_iter.next, deconvolver, lambda x: x._current_cleaned),
			
			Image(
		 		'components'
			).attach(axes_iter.next, deconvolver, lambda x: x._components),
			
			Image(
		 		'selected pixels',
				plt_kwargs = {'cmap': viridis_oob},
			).attach(axes_iter.next, deconvolver, lambda x: selected_pixels_non_selected_are_nan(x)),
			
			Image(
		 		'pixel choice metric',
		 		axis_data_mappings = (AxisDataMapping('x',None), AxisDataMapping('y',None), AxisDataMapping('brightness', '_z_data', plot_helper.LimSymAroundValue(0))),
		 		plt_kwargs={'cmap':'bwr_oob'}
			).attach(axes_iter.next, deconvolver, lambda x: x._px_choice_img_ptr.val),
			
			Histogram(
				'pixel choice metric', 
				static_frame=False,
			).attach(axes_iter.next, deconvolver, lambda x: x._px_choice_img_ptr.val),
			
			Histogram(
				'pixel choice metric', 
				static_frame=False,
				plt_kwargs={'color' : 'green', 'alpha':0.3},
				axis_labels=(None, 'log(count)'),
				ax_funcs=[lambda ax: ax.set_yscale('log')]
			).attach(axes_iter.current.twinx(), deconvolver, lambda x: x._px_choice_img_ptr.val),
			
			IterativeLogLineGraph(
				'metrics',
				datasource_name='fabs',
				axis_labels = (None, 'fabs value (blue)'),
				static_frame=False,
				plt_kwargs = {},
			).attach(axes_iter.next, deconvolver, lambda x: np.fabs(np.nanmax(x._residual))),
			
			HorizontalLine(
				None, 
				static_frame=False, 
				plt_kwargs={'linestyle':'--'}
			).attach(axes_iter.current, deconvolver, lambda x: x._fabs_threshold),
			
			IterativeLogLineGraph(
				'metrics',
				datasource_name='rms',
				axis_labels = (None,'rms value (red)'),
				static_frame=False,
				plt_kwargs={'color':'red'},
			).attach(axes_iter.push(axes_iter.current.twinx()).next, deconvolver, lambda x: np.sqrt(np.nansum(x._residual**2)/x._residual.size)),
			
			HorizontalLine(
				None, 
				static_frame=False, 
				plt_kwargs={'color':'red', 'linestyle':'--'}
			).attach(axes_iter.current, deconvolver, lambda x: x._rms_threshold),
		]
	)
	return plot_set



def init_process_context(msg_queue):
	global queue
	queue = msg_queue
	

def deconv_process(group_index, idx, deconvolver, obs, psf):
	#lambda idx, obs, psf: (idx, deconvolver(obs, psf), deconvolver.get_parameters()), obs_idx, processed_obs, normed_psf))
	import sys
	from contextlib import redirect_stdout
	from time import sleep
	
	class OutFileLikeQueue:
		def __init__(self, q, group_index):
			self.q = q
			self.group_index = group_index
		
		def write(self, msg):
			#print(f'queue in: {self.group_index=} {msg=}', file=sys.stderr)
			self.q.put_nowait((self.group_index, msg))
			#sleep(0.1)
		
		def close(self):
			self.q.put((self.group_index, None))
	
	with redirect_stdout(OutFileLikeQueue(queue, group_index)) as q:
	
		try:
			result = deconvolver(obs, psf)
			params = deconvolver.get_parameters()
		except Exception:
			print(e, file=sys.stdout)
		finally:
			q.write(None)
			q.close()
		
	return group_index, idx, result, params



def display_process_progress():
	from time import sleep
	from collections import OrderedDict
	
	stop = False
	empty = True

	max_line_len = 16
	display_line = []
	most_recent_msg = dict()
	display_msg = dict()
	#print('ONE')

	while not stop:
	
		#print('TWO')
		try:
			empty = queue.empty()
		except OSError as e:
			#print('queue stopping')
			#print(e)
			stop = True
			break
		
		#print('THREE')
		if empty:
			#print('queue is empty')
			sleep(0.1)
			continue
		
		#print('FOUR')
		
		group_index, msg = queue.get()
		
		#print('FIVE')
		
		if group_index is None and msg is None:
			#print('queue stopping')
			stop = True
			break
		
		#print('SIX')
		
		if msg is None:
			#print(f'msg is "None", deleting state tracker for {group_index=}')
			if group_index in most_recent_msg:
				del most_recent_msg[group_index]
			if group_index in display_msg:
				del display_msg[group_index]
	
		
		
		#print('SEVEN')
		#print(f'queue out: {group_index=} msg="{msg}"')

		if type(msg) is str:
			most_recent_msg[group_index] = msg.replace('\r', '').strip()

		#print('EIGHT')
		if len(most_recent_msg.get(group_index, '')) == 0:
			#print('continue')
			continue
		else:
			#print('keep message')
			display_msg[group_index] = most_recent_msg.get(group_index, 'X'*16)
		
		#print('NINE')
		
		#print(' '*max_line_len, end='\r')
		
		display_line = ['- Parallel Processes ' + '-'*20] + [f'{k} {v}' for k,v in display_msg.items()]
		show_line = '\n'.join(display_line)
		
		if len(show_line) > max_line_len:
			max_line_len = len(show_line)
		
		#print(show_line, end='')
		print(show_line, end='\n')
		
		
		#print('queue out: restart loop')
	
	print(f'queue ended')
	queue.close()
		


def run(
		obs_fits_spec : aph.fits.specifier.FitsSpecifier,
		psf_fits_spec : aph.fits.specifier.FitsSpecifier,
		deconvolver : Literal[CleanModified] | Literal[LucyRichardson],
		output_path : str | Path = './deconv.fits',
		plot : bool = True,
		progress : int = 0,
		n_processes : int = (mp.cpu_count() - 2),
	):
	"""
	Given a FitsSpecifier for an observation and a PSF, an output path, and a class that performs deconvolution,
	deconvolves the observation using the PSF.
	
	# ARGUMENTS #
		obs_fits_spec : aph.fits.specifier.FitsSpecifier
			FITS file specifier for observation data, format is PATH{EXT}[SLICE](AXES).
			Where:
				PATH : str
					The path to the FITS file
				EXT : str | int
					The name or number of the FITS extension (defaults to PRIMARY)
				SLICE : "python slice format" (i.e., [1:5, 5:10:2])
					Slice of the FITS extension data to use (defaults to all data)
				AXES : tuple[int,...]
					Axes of the FITS extension that are "spatial" or "celestial" (i.e., RA, DEC),
					by default will try to infer them from the FITS extension header.
		psf_fits_spec : aph.fits.specifier.FitsSpecifier
			FITS file specifier for PSF data, format is same as above
		output_path : str = './deconv.fits'
			Path to output deconvolution to.
		deconvolver : ClassInstance
			Instance of Class to use for deconvolving, defaults to an instance of CleanModified
		plot : bool = True
			If `True` will plot the deconvolution progress
		n_processes : int = N_cpu - 2
			Number of processes to parallelise the calculations over
	"""
	
	_lgr.debug(f'{obs_fits_spec=}')
	_lgr.debug(f'{psf_fits_spec=}')
	_lgr.debug(f'{output_path=}')
	_lgr.debug(f'{deconvolver=}')
	_lgr.debug(f'{plot=}')
	_lgr.debug(f'{progress=}')
	
	original_data_type=None
	
	# Set up plotting if we want it
	if plot:
		#plt.figure()
		#plt.imshow(obs_data[obs_idx])
		#plt.figure()
		#plt.imshow(psf_data[psf_idx])
		#plt.show()
		plt.close('all')
		plot_set = create_plot_set(deconvolver, cadence=1 if progress < 1 else progress)
		deconvolver.post_iter_hooks = []
		deconvolver.post_iter_hooks.append(lambda *a, **k: plot_set.update())
		plot_set.show()

	if progress > 0:
		iteration_tracker = IterationTracker(deconvolver.n_iter, progress)
		deconvolver.post_iter_hooks.append(iteration_tracker.update)
		deconvolver.final_hooks.append(iteration_tracker.complete)
	
	
	#idx_result=[[]]
	
	# Open the fits files
	with fits.open(Path(obs_fits_spec.path)) as obs_hdul, fits.open(Path(psf_fits_spec.path)) as psf_hdul:
		
		# pull out the data we want
		obs_data = obs_hdul[obs_fits_spec.ext].data
		psf_data = psf_hdul[psf_fits_spec.ext].data
		original_data_type=obs_data.dtype
		
		group_shape = tuple(s for x, s in enumerate(obs_data.shape) if x not in obs_fits_spec.axes['CELESTIAL'])
		idx_result = np.full(group_shape, dtype=object, fill_value=dict())
		
		_lgr.debug(f'{obs_data.shape=}')
		_lgr.debug(f'{psf_data.shape=}')
		_lgr.debug(f'{original_data_type=}')
		
		# Create holders for deconvolution products
		deconv_components = np.full_like(obs_data, np.nan)
		deconv_residual = np.full_like(obs_data, np.nan)
		
		msg_queue = MsgQueue()
		
		with Executor(max_workers=n_processes, initializer=init_process_context, initargs=(msg_queue,)) as executor:
			
			msg_responder_process = executor.submit(display_process_progress)
			
			futures = []
			# Loop over the index range specified by `obs_fits_spec` and `psf_fits_spec`
			for i, (obs_idx, psf_idx) in enumerate(zip(
					nph.slice.iter_indices(obs_data, obs_fits_spec.slices, obs_fits_spec.axes['CELESTIAL']),
					nph.slice.iter_indices(psf_data, psf_fits_spec.slices, psf_fits_spec.axes['CELESTIAL'])
				)):
				if progress > 0:
					iteration_tracker.reset()
					print(f'Deconvolving slice {i}/{obs_data[obs_fits_spec.slices].shape[tuple(x for x in range(obs_data.ndim) if x not in obs_fits_spec.axes['CELESTIAL'])[0]]}')
				_lgr.debug(f'Operating on slice {i}')
				
				
				# Get group index we can use to uniquely identify which
				# section of the array we are operating on
				gi_elems = tuple(0 for x in range(obs_data.ndim) if x not in obs_fits_spec.axes['CELESTIAL'])
				group_index = []
				group_obs_idx = obs_idx
				counter = 0
				for ax in sorted(obs_fits_spec.axes['CELESTIAL'], reverse=True):
					group_obs_idx = np.take(group_obs_idx, 0, ax-counter)
					counter += 1
				for _i,_v in enumerate(gi_elems):
					group_index.append(np.take(group_obs_idx, _i, axis=_v))
				group_index = tuple(group_index)
				
				
				# Ensure that we actually have data in this part of the cube
				if np.all(np.isnan(obs_data[obs_idx])) or np.all(np.isnan(psf_data[psf_idx])):
					_lgr.warn('All NAN obs or psf layer detected. Skipping...')
					continue
				
				# perform any normalisation and processing
				normed_psf = psf_data_ops.normalise(np.nan_to_num(psf_data[psf_idx]))
				processed_obs = np.nan_to_num(obs_data[obs_idx])
				
				"""
				# Store the deconvolution products in the arrays we created earlier
				deconv_components[obs_idx], deconv_residual[obs_idx], deconv_iters = deconvolver(processed_obs, normed_psf)
				
				# Save the parameters we used. NOTE: we are only saving the LAST set of parameters as they are all the same
				# in this case. However, if they vary with index they should be recorded with index as well.
				deconv_params = deconvolver.get_parameters()
			
				idx_result.append({**deconv_params})
				"""
				
				futures.append(executor.submit(deconv_process, group_index, obs_idx, deconvolver, processed_obs, normed_psf))
			
			print(f'Waiting for processes...')
			for future in as_completed(futures):
				if not future.done():
					_lgr.error(f'Deconv process {future} not completed, something went wrong')
					continue
				if future.cancelled():
					_lgr.warn(f'Deconv process {fugure} cancelled, skipping result...')
					continue
				
				_lgr.info(f'Process {group_index} ended')
				msg_queue.put((group_index, None))
				
				group_index, idx, (future_deconv_components, future_deconv_residual, future_deconv_iters), future_params = future.result()
				
				_lgr.debug(f'{group_index=}')
				_lgr.debug(f'{idx=}')
				_lgr.debug(f'{future_deconv_components=}')
				_lgr.debug(f'{future_deconv_residual=}')
				_lgr.debug(f'{future_deconv_iters=}')
				_lgr.debug(f'{future_params=}')
				deconv_components[idx] = future_deconv_components
				deconv_residual[idx] = future_deconv_residual
				
				idx_result[*group_index] = {**future_params}

			_lgr.info(f'Processes completed')
			msg_queue.put((None,None))
			msg_queue.close()
	

	_lgr.debug(f'{idx_result=}')
	
	param_names = None
	param_defaults = None
	for item in idx_result:
		if len(item) > 0:
			param_names = tuple(item.keys())
			param_defaults = item
			break
	
	params_that_change = []
	for k in param_names:
		if any((x[k] != param_defaults[k] for x in idx_result if len(x) != 0)):
			params_that_change.append(k)
	params_that_are_static=[k for k in param_names if k not in params_that_change]
	
	vp_type_missing = {}
	vp_type_codes = {}
	for k in params_that_change:
		v = param_defaults[k]
		if type(v) is str:
			vp_type_codes[k] = 'A'
			vp_type_missing[k] = 'MISSING'
		elif type(v) is int:
			vp_type_codes[k]='I'
			vp_type_missing[k] = -1
		elif type(v) is float:
			vp_type_codes[k]='D'
			vp_type_missing[k] = np.nan
		else:
			vp_type_codes[k]='1024A'
			vp_type_missing[k] = 'MISSING'
	
	static_params = dict(((k,param_defaults[k]) for k in param_names if k in params_that_are_static))
	volatile_params = dict((k, [x.get(k, vp_type_missing[k]) for x in idx_result]) for k in params_that_change)
	
	
	
	for k,v in vp_type_codes.items():
		if v[0]=='A':
			vp_type_codes[k]=f'{max(map(len,volatile_params[k]))}A'
	
	
	# Make sure we get all the observaiton header data as well as the deconvolution parameters
	hdr = obs_hdul[obs_fits_spec.ext].header
	hdr.update(aph.fits.header.DictReader(
		{
			'obs_file' : obs_fits_spec.path,
			'psf_file' : psf_fits_spec.path, # record the PSF file we used
			**static_params # record the deconvolution parameters we used
		},
		prefix='deconv',
		pkey_count_start=aph.fits.header.DictReader.find_max_pkey_n(hdr)
	))
	
	# Save the deconvolution products to a FITS file
	hdu_components = fits.PrimaryHDU(
		header = hdr,
		data = deconv_components.astype(original_data_type)
	)
	hdu_residual = fits.ImageHDU(
		header = hdr,
		data = deconv_residual.astype(original_data_type),
		name = 'RESIDUAL'
	)
	hdu_deconv_param_tbl = fits.BinTableHDU.from_columns(
		columns = [fits.Column(name=f'deconv.{k}', format=vp_type_codes[k], array=v) for k,v in volatile_params.items()],
		name = 'VOLATILE_PARAMS',
		header = None,
	)
	
	
	hdul_output = fits.HDUList([
		hdu_components,
		hdu_residual,
		hdu_deconv_param_tbl
	])
	hdul_output.writeto(output_path, overwrite=True)
	
	_lgr.info(f'Deconvolution completed, output written to "{output_path}"')
	

def parse_args(argv):
	import os
	import aopp_deconv_tool.text
	import argparse
	
	DEFAULT_OUTPUT_TAG = '_deconv'
	DESIRED_FITS_AXES = ['CELESTIAL']
	OUTPUT_COLUMNS=80
	try:
		OUTPUT_COLUMNS = os.get_terminal_size().columns - 30
	except Exception:
		pass
	
	FITS_SPECIFIER_HELP = aopp_deconv_tool.text.wrap(
		aph.fits.specifier.get_help(DESIRED_FITS_AXES).replace('\t', '    '),
		OUTPUT_COLUMNS
	)
	DECONV_METHOD_DEFAULT='clean_modified'
	
	class ArgFormatter (argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter, argparse.MetavarTypeHelpFormatter):
		def __init__(self, *args, **kwargs):
			super().__init__(*args, **kwargs)
	
	
	
	
	
	parser = argparse.ArgumentParser(
		description=__doc__, 
		formatter_class=ArgFormatter,
		epilog=FITS_SPECIFIER_HELP,
		exit_on_error=False
	)
	
	parser.add_argument(
		'obs_fits_spec',
		help = '\n'.join((
			f'The observation\'s (i.e., science target) FITS Specifier. See the end of the help message for more information',
			f'required axes: {", ".join(DESIRED_FITS_AXES)}',
		)),
		type=str,
		metavar='FITS Specifier',
	)
	
	parser.add_argument(
		'psf_fits_spec',
		help = '\n'.join((
			f'The psf\'s (i.e., calibration target) FITS Specifier. See the end of the help message for more information',
			f'required axes: {", ".join(DESIRED_FITS_AXES)}',
		)),
		type=str,
		metavar='FITS Specifier',
	)
	
	parser.add_argument(
		'-o', 
		'--output_path', 
		type=FPath,
		metavar='str',
		default='{parent}/{stem}{tag}{suffix}',
		help = '\n    '.join((
			f'Output fits file path, supports keyword substitution using parts of `obs_fits_spec` path where:',
			'{parent}: containing folder',
			'{stem}  : filename (not including final extension)',
			f'{{tag}}   : script specific tag, "{DEFAULT_OUTPUT_TAG}" in this case',
			'{suffix}: final extension (everything after the last ".")',
			'\b'
		))
	)
	parser.add_argument(
		'--plot', 
		action='store_true', 
		default=False, 
		help='If present will show progress plots of the deconvolution'
	)
	parser.add_argument(
		'--deconv_method', 
		type=str, 
		choices=deconv_methods.keys(), 
		default=DECONV_METHOD_DEFAULT, 
		help='Which method to use for deconvolution. For more information, pass the deconvolution method and the "--info" argument.'
	) 
	parser.add_argument(
		'--deconv_method_help', 
		action='store_true', 
		default=False, 
		help='Show help for the selected deconvolution method'
	)
	parser.add_argument(
		'--progress',
		type=int,
		default=0,
		help='Show progression of deconvolution on each `progress` step, 0 does not display progress'
	)
	
	parser.add_argument(
		'--n_processes',
		type=int,
		default=mp.cpu_count() - 2,
		help= 'Number of processors to parallelise the calculations over'
	)
	
	parser.successful = True
	parser.error_message = None
	
	def on_error(err_str):
		parser.successful = False
		parser.error_message = err_str
		
	parser.error = on_error
	
	args, deconv_args = parser.parse_known_args(argv)
	
	if not parser.successful:
		print(vars(args))
		if args.deconv_method_help:
			return args, deconv_args
		else:
			parser.print_usage()
			print(parser_error_message)
			sys.exit()
			
	
	
	args.obs_fits_spec = aph.fits.specifier.parse(args.obs_fits_spec, DESIRED_FITS_AXES)
	args.psf_fits_spec = aph.fits.specifier.parse(args.psf_fits_spec, DESIRED_FITS_AXES)
	
	other_file_path = Path(args.obs_fits_spec.path)
	args.output_path = args.output_path.with_fields(
		tag=DEFAULT_OUTPUT_TAG, 
		parent=other_file_path.parent, 
		stem=other_file_path.stem, 
		suffix=other_file_path.suffix
	)
	
	return args, deconv_args


def go(
		obs_fits_spec,
		psf_fits_spec,
		output_path=None, 
		plot=None, 
		deconv_method=None, 
		deconv_method_help_FLAG=None,
		n_processes=None,
		**kwargs
	):
	"""
	Thin wrapper around `run()` to accept string inputs.
	As long as the names of the arguments to this function 
	are the same as the names expected from the command line
	we can do this programatically
	"""
	# This must be first so we only grab the arguments to the function
	fargs = dict(locals().items())
	arglist = aopp_deconv_tool.arguments.construct_arglist_from_locals(fargs, n_positional_args=2)
	
	exec_with_args(arglist)
	return

def exec_with_args(argv):
	args, deconv_args = parse_args(argv)
	_lgr.debug('#### ARGUMENTS ####')
	for k,v in vars(args).items():
		_lgr.debug(f'\t{k} : {v}')
	_lgr.debug('#### END ARGUMENTS ####')
	
		
	deconv_class = deconv_methods[args.deconv_method]
	deconv_params = arguments.parse_args_of_dataclass(
		deconv_class, 
		deconv_args, 
		prog=f'deconvolve.py --deconv_method {args.deconv_method}',
		show_help=args.deconv_method_help
	)
	
	_lgr.debug(f'{deconv_params=}')
	deconvolver = deconv_class(**deconv_params)
	
	run(
		args.obs_fits_spec, 
		args.psf_fits_spec, 
		deconvolver = deconvolver,
		output_path = args.output_path, 
		plot = args.plot,
		progress = args.progress,
		n_processes = args.n_processes,
	)
	
	return
	
if __name__ == '__main__':
	argv = sys.argv[1:]
	exec_with_args(argv)
	
