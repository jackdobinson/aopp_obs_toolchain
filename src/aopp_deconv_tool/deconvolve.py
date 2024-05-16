"""
Script that deconvolves the first argument with the second argument

Example invocation: 
	`python -m aopp_deconv_tool.deconvolve './example_data/test_rebin.fits{DATA}[10:12]{CELESTIAL:(1,2)}' './example_data/fit_example_psf_000.fits[10:12]{CELESTIAL:(1,2)}'`
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
import aopp_deconv_tool.psf_data_ops as psf_data_ops

from aopp_deconv_tool.algorithm.deconv.clean_modified import CleanModified
from aopp_deconv_tool.algorithm.deconv.lucy_richardson import LucyRichardson

import matplotlib as mpl
import matplotlib.pyplot as plt
import copy
import aopp_deconv_tool.plot_helper as plot_helper
from aopp_deconv_tool.plot_helper.base import AxisDataMapping
from aopp_deconv_tool.plot_helper.plotters import PlotSet, Histogram, VerticalLine, Image, IterativeLineGraph, HorizontalLine

import aopp_deconv_tool.cfg.logs
_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'DEBUG')


deconv_methods = {
	'clean_modified' : CleanModified,
	'lucy_richardson' : LucyRichardson
}

def create_plot_set(deconvolver, cadence = 10):
	fig, axes = plot_helper.figure_n_subplots(8)
	axes_iter = iter(axes)
	a7_2 = axes[7].twinx()
	
	try:
		cmap = mpl.cm.get_cmap('bwr_oob')
	except ValueError:
		cmap = copy.copy(mpl.cm.get_cmap('bwr'))
		cmap.set_over('magenta')
		cmap.set_under('green')
		cmap.set_bad('black')
		mpl.cm.register_cmap(name='bwr_oob', cmap=cmap)
	#mpl.rcParams['image.cmap'] = 'user_cmap'
	
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
		 		'selected pixels'
			).attach(next(axes_iter), deconvolver, lambda x: x._selected_px),
			
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
		output_path : str | Path = './deconv.fits',
		deconv_class : Literal[CleanModified] | Literal[LucyRichardson] = CleanModified,
		plot : bool = True,
		deconv_args : list[str,...] = []
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
				SLICE : "python slice format" (i.e. [1:5, 5:10:2])
					Slice of the FITS extension data to use (defaults to all data)
				AXES : tuple[int,...]
					Axes of the FITS extension that are "spatial" or "celestial" (i.e. RA, DEC),
					by default will try to infer them from the FITS extension header.
		psf_fits_spec : aph.fits.specifier.FitsSpecifier
			FITS file specifier for PSF data, format is same as above
		output_path : str = './deconv.fits'
			Path to output deconvolution to.
		deconv_class : Type
			Class to use for deconvolving, defaults to CleanModified
		plot : bool = True
			If `True` will plot the deconvolution progress
	"""
	deconv_params = parse_deconv_args(deconv_class, deconv_args)
	_lgr.debug(f'{deconv_params=}')
	deconvolver = deconv_class(**deconv_params)

	# Open the fits files
	with fits.open(Path(obs_fits_spec.path)) as obs_hdul, fits.open(Path(psf_fits_spec.path)) as psf_hdul:
		
		# pull out the data we want
		obs_data = obs_hdul[obs_fits_spec.ext].data
		psf_data = psf_hdul[psf_fits_spec.ext].data
		
		# Create holders for deconvolution products
		deconv_components = np.full_like(obs_data, np.nan)
		deconv_residual = np.full_like(obs_data, np.nan)
		
		# Loop over the index range specified by `obs_fits_spec` and `psf_fits_spec`
		for obs_idx, psf_idx in zip(
			nph.slice.iter_indices(obs_data, obs_fits_spec.slices, obs_fits_spec.axes['CELESTIAL']),
			nph.slice.iter_indices(psf_data, psf_fits_spec.slices, psf_fits_spec.axes['CELESTIAL'])
		):
		
		
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
		data = deconv_components
	)
	hdu_residual = fits.ImageHDU(
		header = hdr,
		data = deconv_residual,
		name = 'RESIDUAL'
	)
	hdul_output = fits.HDUList([
		hdu_components,
		hdu_residual
	])
	hdul_output.writeto(output_path, overwrite=True)


def parse_deconv_args(deconv_class, argv):
	import argparse
	import dataclasses as dc
	import re
	
	# Use this to grab only the first part of the docstring as that should be a short
	# description of the class
	re_empty_line = re.compile(r'^\s*$\s*', flags=re.MULTILINE)
	
	parser = argparse.ArgumentParser(
		description=re_empty_line.split(deconv_class.__doc__,2)[1], 
		formatter_class=argparse.RawTextHelpFormatter,
		add_help=False
	)
	def on_parser_error(err_str):
		print(err_str)
		parser.print_help()
		sys.exit(1)
	
	parser.error = on_parser_error
	
	parser.add_argument('--info', action='store_true', default=False, help='Show this information message')
	
	for field in dc.fields(deconv_class):
		if field.init != True:
			continue
			
		field_default = field.default if field.default != dc.MISSING else (field.default_factory() if field.default_factory != dc.MISSING else None)
		
		parser.add_argument(
			'--'+field.name, 
			type=field.type, 
			default= field_default,
			help=field.metadata.get('description', 'DESCRIPTION NOT FOUND') + f' (default = {field_default})',
			metavar=str(field.type)[8:-2]
		)
	
	deconv_args = parser.parse_args(argv)
	
	if deconv_args.info:
		parser.print_help()
		sys.exit()
	
	delattr(deconv_args, 'info')
	
	return vars(deconv_args)
	
	

def parse_args(argv):
	import os
	import aopp_deconv_tool.text
	import argparse
	
	DEFAULT_OUTPUT_TAG = '_deconv'
	DESIRED_FITS_AXES = ['CELESTIAL']
	FITS_SPECIFIER_HELP = aopp_deconv_tool.text.wrap(
		aph.fits.specifier.get_help(DESIRED_FITS_AXES).replace('\t', '    '),
		os.get_terminal_size().columns - 30
	)
	
	
	class ArgFormatter (argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
		def __init__(self, *args, **kwargs):
			super().__init__(*args, **kwargs)
	
	parser = argparse.ArgumentParser(
		description=__doc__, 
		formatter_class=ArgFormatter,
		epilog=FITS_SPECIFIER_HELP
	)
	
	parser.add_argument(
		'obs_fits_spec',
		help = 'The observation\'s (i.e. science target) FITS SPECIFIER, see the end of the help message for more information'
	)
	
	parser.add_argument(
		'psf_fits_spec', 
		help = 'The psf\'s (i.e. calibration target) FITS SPECIFIER, see the end of the help message for more information'
	)
	
	parser.add_argument('-o', '--output_path', type=str, help=f'Output fits file path. By default is same as the `fits_spec` path with "{DEFAULT_OUTPUT_TAG}" appended to the filename')
	parser.add_argument('--plot', action='store_true', default=False, help='If present will show progress plots of the deconvolution')
	parser.add_argument('--deconv_method', type=str, choices=deconv_methods.keys(), default='clean_modified', help='Which method to use for deconvolution. For more information, pass the deconvolution method and the "--info" argument.') 
	
	
	args, deconv_args = parser.parse_known_args(argv)
	
	args.obs_fits_spec = aph.fits.specifier.parse(args.obs_fits_spec, DESIRED_FITS_AXES)
	args.psf_fits_spec = aph.fits.specifier.parse(args.psf_fits_spec, DESIRED_FITS_AXES)
	
	if args.output_path is None:
		args.output_path =  (Path(args.obs_fits_spec.path).parent / (str(Path(args.obs_fits_spec.path).stem)+DEFAULT_OUTPUT_TAG+str(Path(args.obs_fits_spec.path).suffix)))
	
	
	return args, deconv_args


if __name__ == '__main__':

	args, deconv_args = parse_args(sys.argv[1:])
	

	
	run(
		args.obs_fits_spec, 
		args.psf_fits_spec, 
		output_path = args.output_path, 
		plot = args.plot,
		deconv_class = deconv_methods[args.deconv_method],
		deconv_args = deconv_args,
	)
	
