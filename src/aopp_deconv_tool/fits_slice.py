"""
Apply slices to a FITS file, reducing the number of entries across a set of axes.
"""


import sys, os
from pathlib import Path


from astropy.io import fits

import aopp_deconv_tool.astropy_helper as aph
import aopp_deconv_tool.astropy_helper.fits.header
import aopp_deconv_tool.astropy_helper.fits.specifier
from aopp_deconv_tool.astropy_helper.fits.specifier import FitsSpecifier

from aopp_deconv_tool.fpath import FPath
from aopp_deconv_tool.numpy_helper.axes import AxesOrdering
import aopp_deconv_tool.arguments

import aopp_deconv_tool.cfg.logs
_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'WARN')



def run(
		fits_spec : FitsSpecifier,
		output_path : Path|str,
		#squeeze : bool = False
	):
	
	with fits.open(Path(fits_spec.path)) as hdul:
		hdu = hdul[fits_spec.ext]
		original_data_type = hdu.data.dtype
		original_data_shape = hdu.data.shape
		data = hdu.data[fits_spec.slices]
		assert data.size > 0, f"Slice selects no data elements: slice={fits_spec.slices}; data_shape={original_data_shape}"
		_lgr.debug(f'{data.shape=}')
		hdr = hdu.header
		hdr.update(aph.fits.header.DictReader(
			{
				'original_file' : fits_spec.path,
			},
			prefix='slice',
			pkey_count_start=aph.fits.header.DictReader.find_max_pkey_n(hdr)
		))
	
	for axis in fits_spec.axes['ALL']:
		ax = AxesOrdering(axis, hdr['NAXIS'], 'numpy')
		if ax.numpy >= len(fits_spec.slices):
			continue
		
		aph.fits.header.set_axes_transform(
			hdr,
			axis = ax.fits,
			reference_pixel = int(hdr[f'CRPIX{ax.fits}']) - fits_spec.slices[ax.numpy].start
		)
		
		
	
	
	hdu_sliced = fits.PrimaryHDU(
		header = hdr,
		data = data.astype(original_data_type)
	)
	hdul_output = fits.HDUList([
		hdu_sliced,
	])
	hdul_output.writeto(output_path, overwrite=True)


def parse_args(argv):
	import os
	import aopp_deconv_tool.text
	import argparse
	
	DEFAULT_OUTPUT_TAG = '_sliced'
	DESIRED_FITS_AXES = ['ALL']
	FITS_SPECIFIER_HELP = aopp_deconv_tool.text.wrap(
		aph.fits.specifier.get_help(DESIRED_FITS_AXES).replace('\t', '    '),
		os.get_terminal_size().columns - 30
	)
	
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
		'fits_spec',
		help = 'FITS file specifier, will slice the file as requested here',
		type=str,
		metavar='FITS Specifier',
	)
	
	#parser.add_argument(
	#	'--squeeze',
	#	help="If present, will remove any axes that end up with only 1 entry",
	#	action='store_true',
	#)
	
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

	
	args = parser.parse_args(argv)
	
	args.fits_spec = aph.fits.specifier.parse(args.fits_spec, DESIRED_FITS_AXES)
	
	other_file_path = Path(args.fits_spec.path)
	args.output_path = args.output_path.with_fields(
		tag=DEFAULT_OUTPUT_TAG, 
		parent=other_file_path.parent, 
		stem=other_file_path.stem, 
		suffix=other_file_path.suffix
	)
	
	return args

def go(
		fits_spec,
		output_path=None
	):
	"""
	Thin wrapper around `run()` to accept string inputs.
	As long as the names of the arguments to this function 
	are the same as the names expected from the command line
	we can do this programatically
	"""
	# Add stuff to kwargs here if needed
	
	# This must be first so we only grab the arguments to the function
	fargs = dict(locals().items())
	arglist = aopp_deconv_tool.arguments.construct_arglist_from_locals(fargs, n_positional_args=1)
	exec_with_args(arglist)

def exec_with_args(argv):
	args = parse_args(argv)
	_lgr.info('#### ARGUMENTS ####')
	for k,v in vars(args).items():
		_lgr.info(f'\t{k} : {v}')
	_lgr.info('#### END ARGUMENTS ####')
	
	run(
		args.fits_spec, 
		args.output_path, 
		#squeeze = args.squeeze
	)
	
if __name__ == '__main__':
	exec_with_args(sys.argv[1:])

	