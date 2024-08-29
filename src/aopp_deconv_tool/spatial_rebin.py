"""
Quick tool for spatially rebinning a FITS file, useful for reducing data volume when testing.

Linearly interpolates at a new grid starting at the same location as the old one, but with a different step size. Uses the same units the fits file is in.

"""
import sys, os
import numpy as np
import numpy.linalg
import scipy as sp
import scipy.interpolate
import scipy.ndimage

from pathlib import Path

import matplotlib.pyplot as plt
from astropy.io import fits

from aopp_deconv_tool.fpath import FPath
import aopp_deconv_tool.astropy_helper as aph
import aopp_deconv_tool.astropy_helper.fits.specifier
import aopp_deconv_tool.astropy_helper.fits.header
import aopp_deconv_tool.arguments

import aopp_deconv_tool.numpy_helper as nph

import aopp_deconv_tool.cfg.logs
_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'WARN')




def run(
		fits_spec : aph.fits.specifier.FitsSpecifier,
		output_path : str | Path,
		steps : tuple[float,float],
	):
	with fits.open(Path(fits_spec.path)) as data_hdul:
	
		data_hdu = data_hdul[fits_spec.ext]
		data = data_hdu.data
		hdr = data_hdu.header
		axes_ordering =  aph.fits.header.get_axes_ordering(data_hdu.header, fits_spec.axes['CELESTIAL'])
		ax_values = aph.fits.header.get_world_coords_of_axis(data_hdu.header, fits_spec.axes['CELESTIAL'])
		iwc_matrix = aph.fits.header.get_iwc_matrix(hdr)
		
		_lgr.debug(f'{iwc_matrix=}')
		
		transform_matrix = np.array([
			[1,                                0,                                0],
			[0, np.abs(steps[0]/iwc_matrix[1,1]),                                0],
			[0,                                0, np.abs(steps[1]/iwc_matrix[2,2])],
		])
		_lgr.debug(f'{transform_matrix=}')
		
		result = np.full(tuple(int(np.floor(data.shape[i]/transform_matrix[i,i])) for i in range(data.ndim)), fill_value=np.nan)
		_lgr.debug(f'{result.shape=}')
		
		for i, (idx, group_slice) in enumerate(nph.slice.iter_indices_with_slices(data, fits_spec.slices, fits_spec.axes['CELESTIAL'])):
			_lgr.info(f'Operating on slice {i}')
			sp.ndimage.affine_transform(
				np.nan_to_num(data[idx]), 
				transform_matrix[fits_spec.axes['CELESTIAL'],fits_spec.axes['CELESTIAL']], 
				order=1, # linear interpolation
				output_shape=tuple(s for i,s in enumerate(result.shape) if i in fits_spec.axes['CELESTIAL']), 
				output=result[group_slice],
				mode='constant',
				cval=np.nan
			)

		axis_fits = tuple(ao.fits for ao in axes_ordering)
		param_dict = {
			'original_file' : Path(fits_spec.path).name, # record the file we used
			'axes' : axis_fits,
			'rebinned_steps' : steps,
		}
		
		hdr.update(aph.fits.header.DictReader(
			param_dict,
			prefix='spatial_rebin',
			pkey_count_start=aph.fits.header.DictReader.find_max_pkey_n(hdr)
		))
		
		for i, ax in enumerate(axis_fits):
			_lgr.debug(f'{ax=}')
			aph.fits.header.set_axes_transform(hdr, 
				ax, 
				None, 
				ax_values[0,i],
				steps[i],
				result.shape[i],
				1
			)
		
		aph.fits.header.set_iwc_matrix(hdr, transform_matrix @ iwc_matrix)
		

	
	# Save the products to a FITS file
	hdu_rebinned = fits.PrimaryHDU(
		header = hdr,
		data = result
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
	
	DEFAULT_OUTPUT_TAG = '_spatialrebin'
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
		
	parser.add_argument('--rebin_step', action='extend', nargs=2, type=float, metavar='float', help='step for rebinning operation in each direction, units are whatever the fits file is in, normally degrees')
	
	args = parser.parse_args(argv)
	
	args.fits_spec = aph.fits.specifier.parse(args.fits_spec, DESIRED_FITS_AXES)
		
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
		rebin_step=None
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
		args.output_path,
		args.rebin_step,	
	)

if __name__=='__main__':
	exec_with_args(sys.argv[1:])
	
	