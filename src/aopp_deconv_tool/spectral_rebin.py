"""
Quick tool for spectrally rebinning a FITS file

TODO:
* Nicer argument passing.
* Operate on all extensions in passed file
* Have default operations for common extension names (i.e. "DATA" should have operation "mean", "ERROR" should have operation "mean_err")
* Add table of common rebin parameters. E.g. "spex" resolution, "gemini" resolution, etc.
* Add way to load custom rebin parameter definitions
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


def plot_rebin(old_bins, old_data, new_bins, new_data, title=None):
	plt.title(title)
	plt.plot(np.sum(old_bins, axis=0)/2, old_data, label='old_data')
	plt.plot(np.sum(new_bins, axis=0)/2, new_data, label='new_data')
	plt.legend()
	plt.show()




def rebin_hdu_over_axis(
		data_hdu, 
		axis, 
		bin_step : float = 1E-9, 
		bin_width : float = 2E-9,
		operation : Literal['sum'] | Literal['mean'] | Literal['mean_err'] = 'mean',
		plot : bool = False
	) -> tuple[np.ndarray, np.ndarray]:
	ax_values = aph.fits.header.get_world_coords_of_axis(data_hdu.header, axis)
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
			new_data = (regrid_data.T/(regrid_bin_weights**2)).T
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
		operation : Literal['sum'] | Literal['mean'] | Literal['mean_err'] = 'mean'
	) -> tuple[np.ndarray, np.ndarray]:

	new_data = None
	with fits.open(Path(fits_spec.path)) as data_hdul:
	
		#_lgr.debug(f'{fits_spec.ext=}')
		#raise RuntimeError(f'DEBUGGING')
	
		data_hdu = data_hdul[fits_spec.ext]
	
		axes_ordering =  aph.fits.header.get_axes_ordering(data_hdu.header, fits_spec.axes['SPECTRAL'])
		axis = axes_ordering[0].numpy
	
		new_spec_bins, new_data = rebin_hdu_over_axis(data_hdu, axis, bin_step, bin_width, operation, plot=False)

	
	
		hdr = data_hdu.header
		axis_fits = axes_ordering[0].fits
		hdr.update(aph.fits.header.DictReader({
			'original_file' : Path(fits_spec.path).name, # record the file we used
			'bin_axis' : axis_fits,
			'bin_step' : bin_step,
			'bin_width' : bin_width,
			'bin_operation' : operation
		}))
		
		aph.fits.header.set_axes_transform(hdr, 
			axis_fits, 
			'Angstrom', 
			np.mean(new_spec_bins[:,0])/1E10,
			bin_step/1E10,
			new_spec_bins.shape[1],
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




def parse_args(argv):
	import os
	import aopp_deconv_tool.text
	import argparse
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
	
	parser.add_argument(
		'fits_spec', 
		help = aopp_deconv_tool.text.wrap(
			aph.fits.specifier.get_help(['CELESTIAL']).replace('\t', '    '),
			os.get_terminal_size().columns - 30
		)
	)
	parser.add_argument('-o', '--output_path', help='Output fits file path. By default is same as fie `fits_spec` path with "_rebin" appended to the filename')
	
	args = parser.parse_args(argv)
	
	args.fits_spec = aph.fits.specifier.parse(args.fits_spec, ['SPECTRAL'])
	
	if args.output_path is None:
		args.output_path =  (Path(args.fits_spec.path).parent / (str(Path(args.fits_spec.path).stem)+'_rebin'+str(Path(args.fits_spec.path).suffix)))
	
	return args


if __name__ == '__main__':
	args = parse_args(sys.argv[1:])
	
	run(args.fits_spec, output_path=args.output_path)
	