"""
Script that deconvolves the first argument with the second argument

Example invocation: 
	`python -m aopp_deconv_tool.deconvolve './example_data/test_rebin.fits{DATA}[10:12]{CELESTIAL:(1,2)}' './example_data/fit_example_psf_000.fits[10:12]{CELESTIAL:(1,2)}'`
"""

import sys, os
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
import aopp_deconv_tool.psf_data_ops as psf_data_ops
from aopp_deconv_tool.fpath import FPath
import aopp_deconv_tool.arguments as arguments


from aopp_deconv_tool.algorithm.deconv.clean_modified import CleanModified
from aopp_deconv_tool.algorithm.deconv.lucy_richardson import LucyRichardson

import matplotlib as mpl
import matplotlib.pyplot as plt
import copy
import aopp_deconv_tool.plot_helper as plot_helper
from aopp_deconv_tool.plot_helper.base import AxisDataMapping
from aopp_deconv_tool.plot_helper.plotters import PlotSet, Histogram, VerticalLine, Image, IterativeLineGraph, HorizontalLine

import aopp_deconv_tool.cfg.logs
_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'WARN')


deconv_methods = {
	'clean_modified' : CleanModified,
	'lucy_richardson' : LucyRichardson
}

def create_plot_set(deconvolver, cadence = 1):
	"""
	Creates a set of plots that are updated every `cadence` steps. Useful to see exactly what a deconvolver is doing.
	"""
	fig, axes = plot_helper.figure_n_subplots(8)
	axes_iter = iter(axes)
	a7_2 = axes[7].twinx()
	
	try:
		cmap = mpl.colormaps['bwr_oob']
	except KeyError:
		cmap = copy.copy(mpl.colormaps['bwr'])
		cmap.set_over('magenta')
		cmap.set_under('green')
		cmap.set_bad('black')
		mpl.cm.register_cmap(name='bwr_oob', cmap=cmap)
	
	try:
		viridis_oob = mpl.colormaps['viridis_oob']
	except KeyError:
		viridis_oob = mpl.colormaps['viridis'].copy()
		viridis_oob.set_bad(color='magenta', alpha=1)
		#viridis_oob.set_under(color='black', alpha=1)
		#viridis_oob.set_over(color='black', alpha=1)
		mpl.cm.register_cmap(name='viridis_oob', cmap=viridis_oob)
	
	
	
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
			).attach(next(axes_iter), deconvolver, lambda x: x._residual),
		 	
			VerticalLine(
				None, 
				static_frame=False, 
				plt_kwargs={'color':'red'}
			).attach(axes[0], deconvolver, lambda x: x._pixel_threshold),
			
			Image(
		 		'residual'
		 	).attach(next(axes_iter), deconvolver, lambda x: x._residual),
			
			Image(
		 		'current cleaned'
			).attach(next(axes_iter), deconvolver, lambda x: x._current_cleaned),
			
			Image(
		 		'components'
			).attach(next(axes_iter), deconvolver, lambda x: x._components),
			
			Image(
		 		'selected pixels',
				plt_kwargs = {'cmap': viridis_oob},
			).attach(next(axes_iter), deconvolver, lambda x: selected_pixels_non_selected_are_nan(x)),
			
			Image(
		 		'pixel choice metric',
		 		axis_data_mappings = (AxisDataMapping('x',None), AxisDataMapping('y',None), AxisDataMapping('brightness', '_z_data', plot_helper.LimSymAroundValue(0))),
		 		plt_kwargs={'cmap':'bwr_oob'}
			).attach(next(axes_iter), deconvolver, lambda x: x._px_choice_img_ptr.val),
			
			Histogram(
				'pixel choice metric', 
				static_frame=False,
			).attach(next(axes_iter), deconvolver, lambda x: x._px_choice_img_ptr.val),
			
			IterativeLineGraph(
				'metrics',
				datasource_name='fabs',
				axis_labels = (None, 'fabs value (blue)'),
				static_frame=False,
				plt_kwargs = {},
				ax_funcs=[lambda ax: ax.set_yscale('log')]
			).attach(next(axes_iter), deconvolver, lambda x: np.fabs(np.nanmax(x._residual))),
			
			HorizontalLine(
				None, 
				static_frame=False, 
				plt_kwargs={'linestyle':'--'}
			).attach(axes[7], deconvolver, lambda x: x._fabs_threshold),
			
			IterativeLineGraph(
				'metrics',
				datasource_name='rms',
				axis_labels = (None,'rms value (red)'),
				static_frame=False,
				plt_kwargs={'color':'red'},
				ax_funcs=[lambda ax: ax.set_yscale('log')]
			).attach(a7_2, deconvolver, lambda x: np.sqrt(np.nansum(x._residual**2)/x._residual.size)),
			
			HorizontalLine(
				None, 
				static_frame=False, 
				plt_kwargs={'color':'red', 'linestyle':'--'}
			).attach(a7_2, deconvolver, lambda x: x._rms_threshold),
		]
	)
	return plot_set

def run(
		obs_fits_spec : aph.fits.specifier.FitsSpecifier,
		psf_fits_spec : aph.fits.specifier.FitsSpecifier,
		deconvolver : Literal[CleanModified] | Literal[LucyRichardson],
		output_path : str | Path = './deconv.fits',
		plot : bool = True,
		progress : int = 0,
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
	"""
	
	original_data_type=None
	# Open the fits files
	with fits.open(Path(obs_fits_spec.path)) as obs_hdul, fits.open(Path(psf_fits_spec.path)) as psf_hdul:
		
		# pull out the data we want
		obs_data = obs_hdul[obs_fits_spec.ext].data
		psf_data = psf_hdul[psf_fits_spec.ext].data
		original_data_type=obs_data.dtype
		
		# Create holders for deconvolution products
		deconv_components = np.full_like(obs_data, np.nan)
		deconv_residual = np.full_like(obs_data, np.nan)
		
		# Loop over the index range specified by `obs_fits_spec` and `psf_fits_spec`
		for i, (obs_idx, psf_idx) in enumerate(zip(
				nph.slice.iter_indices(obs_data, obs_fits_spec.slices, obs_fits_spec.axes['CELESTIAL']),
				nph.slice.iter_indices(psf_data, psf_fits_spec.slices, psf_fits_spec.axes['CELESTIAL'])
			)):
			_lgr.debug(f'Operating on slice {i}')
			
			"""
			# Playing with wiener filter deconvolution
			import scipy as sp
			from skimage import restoration
			import scipy.signal
			mask = np.zeros_like(psf_data[psf_idx], dtype=bool)
			mask[tuple(slice(s//4,3*s//4) for s in mask.shape)] = 1
			noise_var = 1E6
			_lgr.debug(f'{noise_var=}')
			wiener_filtered_data = restoration.wiener(obs_data[obs_idx], psf_data[psf_idx], 1E-2, clip=False)
			reconvolve =  sp.signal.fftconvolve(wiener_filtered_data, psf_data[psf_idx], mode='same')
			plt.clf()
			f = plt.gcf()
			ax = f.subplots(2,2).flatten()
			ax[0].imshow(obs_data[obs_idx])
			ax[1].imshow(wiener_filtered_data)
			ax[2].imshow(reconvolve)
			ax[3].imshow(obs_data[obs_idx] -reconvolve)
			plt.show()
			sys.exit()
			"""
		
			# Set up plotting if we want it
			if plot:
				#plt.figure()
				#plt.imshow(obs_data[obs_idx])
				#plt.figure()
				#plt.imshow(psf_data[psf_idx])
				#plt.show()
				plt.close('all')
				plot_set = create_plot_set(deconvolver)
				deconvolver.post_iter_hooks = []
				deconvolver.post_iter_hooks.append(lambda *a, **k: plot_set.update())
				plot_set.show()
			
			if progress > 0:
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
					
				iteration_tracker = IterationTracker(deconvolver.n_iter, progress)
				deconvolver.post_iter_hooks.append(iteration_tracker.update)
				deconvolver.final_hooks.append(iteration_tracker.complete)
			
			# Ensure that we actually have data in this part of the cube
			if np.all(np.isnan(obs_data[obs_idx])) or np.all(np.isnan(psf_data[psf_idx])):
				_lgr.warn('All NAN obs or psf layer detected. Skipping...')
			
			# perform any normalisation and processing
			normed_psf = psf_data_ops.normalise(np.nan_to_num(psf_data[psf_idx]))
			processed_obs = np.nan_to_num(obs_data[obs_idx])
			
			# Store the deconvolution products in the arrays we created earlier
			deconv_components[obs_idx], deconv_residual[obs_idx], deconv_iters = deconvolver(processed_obs, normed_psf)
			
		
		# Save the parameters we used. NOTE: we are only saving the LAST set of parameters as they are all the same
		# in this case. However, if they vary with index they should be recorded with index as well.
		deconv_params = deconvolver.get_parameters()
		
		# Make sure we get all the observaiton header data as well as the deconvolution parameters
		hdr = obs_hdul[obs_fits_spec.ext].header
		hdr.update(aph.fits.header.DictReader(
			{
				'obs_file' : obs_fits_spec.path,
				'psf_file' : psf_fits_spec.path, # record the PSF file we used
				**deconv_params # record the deconvolution parameters we used
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
	hdul_output = fits.HDUList([
		hdu_components,
		hdu_residual
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
	)
	
	return
	
if __name__ == '__main__':
	argv = sys.argv[1:]
	exec_with_args(argv)
	
