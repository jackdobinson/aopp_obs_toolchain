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
		data_hdu = data_hdul[fits_spec.ext]
	
		axis =  fits_spec.axes['SPECTRAL'][0]
	
		new_spec_bins, new_data = rebin_hdu_over_axis(data_hdu, axis, bin_step, bin_width, operation, plot=False)

	
	
		hdr = data_hdu.header
		hdr.update(aph.fits.header.DictReader({
			'original_file' : Path(fits_spec.path).name, # record the file we used
			'bin_step' : bin_step,
			'bin_width' : bin_width,
			'bin_operation' : operation
		}))
	
	
		new_spec_hdr_keys= dict(
			CD3_3 = bin_step/1E-10,                       # Turn meters into Angstrom
			CUNIT3= 'Angstrom',                                # Tell FITS the units
			CRVAL3 = np.mean(new_spec_bins[:,0])/1E-10,   # Set the value of the reference pixel in the spectral direction in the FITS file (center of first bin)
			CRPIX3= 1,                                         # Tell FITS the index of the reference pixel in the spectral direction (first pixel, FITS is 1-index based)
			NAXIS3 = new_spec_bins.shape[1],              # Tell FITS the number of spectral planes
		)
		hdr.update(new_spec_hdr_keys) # Update the old header with the new values (in memory, not on disk) so we can use it to write the altered file.


	
	# Save the deconvolution products to a FITS file
	hdu_rebinned = fits.PrimaryHDU(
		header = hdr,
		data = new_data
	)
	hdul_output = fits.HDUList([
		hdu_rebinned,
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
		
	if len(sys.argv) > 3:
		help_string.print_and_exit(f'A maximum of 2 arguments are accepted: fits_spec, output_path. But {len(sys.argv)-1} were provided')
	
	fits_spec = aph.fits.specifier.parse(sys.argv[1], ['SPECTRAL']) if len(sys.argv) > 1 else help_string.print_and_exit('Need 2 arguments, 0 given')
	output_path = sys.argv[2] if len(sys.argv) > 2 else (Path(fits_spec.path).parent / (str(Path(fits_spec.path).stem)+'_rebin'+str(Path(fits_spec.path).suffix)))
	

	
	run(fits_spec, output_path=output_path)
	
