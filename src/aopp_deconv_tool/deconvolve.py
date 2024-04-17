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
	
	deconvolver = deconv_class()

	# Open the fits files
	with fits.open(Path(obs_fits_spec.path)) as obs_hdul, fits.open(Path(obs_fits_spec.path)) as psf_hdul:
		
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
		hdr.update(aph.fits.header.DictReader({
			'psf_file' : psf_fits_spec.path, # record the PSF file we used
			**deconv_params # record the deconvolution parameters we used
		}))
	
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



if __name__ == '__main__':

	class HelpString:
		def __init__(self, *args):
			self.help = list(args)

		def prepend(self, str):
			self.help = [str] + self.help
			return None
			
		def append(self, str):
			self.help.append(str)
			return None

		def print_and_exit(self, str=None):
			if str is not None:
				self.prepend(str)
			print('\n'.join(self.help))
			sys.exit()
			return
			
	help_string = HelpString(__doc__, aph.fits.specifier.get_help(['CELESTIAL']))

	print_help_and_exit_flag = False

	# Get the fits specifications from the command-line arguments
	if len(sys.argv) <= 1:
		help_string.print_and_exit()
	
	if any([any([x==y for y in sys.argv]) for x in ('-h', '-H', '--help', '--Help')]):
		help_string.print_and_exit()
		
	if len(sys.argv) > 5:
		help_string.print_and_exit(f'A maximum of 3 arguments are accepted: obs_fits_spec, psf_fits_spec, output_path, plot. But {len(sys.argv)-1} were provided')
	
	obs_fits_spec = aph.fits.specifier.parse(sys.argv[1], ['CELESTIAL']) if len(sys.argv) > 1 else help_string.print_and_exit('Need 2 arguments, 0 given')
	psf_fits_spec = aph.fits.specifier.parse(sys.argv[2], ['CELESTIAL']) if len(sys.argv) > 2 else help_string.print_and_exit('Need 2 arguments, 1 given')
	output_path = sys.argv[3] if len(sys.argv) > 3 else './deconv.fits'
	plot = bool(sys.argv[3]) if len(sys.argv) > 4 else False
	

	
	run(obs_fits_spec, psf_fits_spec, output_path=output_path, plot=plot)
	
