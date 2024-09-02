"""
Copy the WCS information from one FITS file to another, useful when we know that two observations were taken with the same setup but one has no physical information.
"""



import sys, os
from pathlib import Path


from astropy.io import fits
from astropy.wcs import WCS

import aopp_deconv_tool.astropy_helper as aph
import aopp_deconv_tool.astropy_helper.fits.header
import aopp_deconv_tool.astropy_helper.fits.specifier
from aopp_deconv_tool.astropy_helper.fits.specifier import FitsSpecifier

from aopp_deconv_tool.fpath import FPath
from aopp_deconv_tool.numpy_helper.axes import AxesOrdering
import aopp_deconv_tool.arguments

import aopp_deconv_tool.cfg.logs
_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'WARN')



def get_fits_file_type(hdr):
	"""
	Return the type of a fits file, or `None` if type is not recognised.
	"""
	if hdr.get('HDUCLASS', None) == 'ESO':
		return 'ESO'
	return None


def run(
		donor_fits_spec : FitsSpecifier,
		acceptor_fits_spec : FitsSpecifier,
		output_path : Path | str
	):
	_lgr.debug('DEBUGGING')
	
	
	with fits.open(Path(donor_fits_spec.path)) as donor_hdul, fits.open(Path(acceptor_fits_spec.path)) as acceptor_hdul:
		donor_hdr = donor_hdul[donor_fits_spec.ext].header
		acceptor_hdr = acceptor_hdul[acceptor_fits_spec.ext].header
		
		# Want to make the resulting cube as close to "true" as possible, therefore use different strategies for different datasets
		donor_fits_file_type = get_fits_file_type(donor_hdr)
		acceptor_fits_file_type = get_fits_file_type(acceptor_hdr)
		
		donor_header_keys = aph.fits.header.get_wcs_header_keys(donor_hdr, donor_fits_spec.axes['ALL'])
		acceptor_header_keys = aph.fits.header.get_wcs_header_keys(acceptor_hdr, donor_fits_spec.axes['ALL'])
		
		_lgr.debug('donor_header_keys')
		for k,v in donor_header_keys.items():
			_lgr.debug(f'{k} = {v}')
		
		_lgr.debug('acceptor_header_keys')
		for k,v in acceptor_header_keys.items():
			_lgr.debug(f'{k} = {v}')
		
		# Copy donor header keys to acceptor header keys
		for k,v in donor_header_keys.items():
			if v is not None:
				acceptor_header_keys[k] = v
		
		
		match acceptor_fits_file_type:
			case 'ESO':
				# ESO files have the primary extension be information about the dataset, and the others hold the image data
				eso_dataset_hdr = acceptor_hdul['PRIMARY'].header
				eso_dataset_ra = eso_dataset_hdr.get('RA',None)
				eso_dataset_dec = eso_dataset_hdr.get('DEC',None)
				
				# Replace the starting points of the RA,DEC axes with these values
				for k in acceptor_header_keys:
					if 'CTYPE' in k:
						if 'RA' in acceptor_header_keys[k]:
							CRVAL_key = 'CRVAL'+k[5:]
							acceptor_header_keys[CRVAL_key] = eso_dataset_ra
						elif 'DEC' in acceptor_header_keys[k]:
							CRVAL_key = 'CRVAL'+k[5:]
							acceptor_header_keys[CRVAL_key] = eso_dataset_dec
						
		# update acceptor header with new keys, only update keys that are present
		for k,v in acceptor_header_keys.items():
			if (v is not None) and (k in acceptor_hdr):
				_lgr.debug(f'{k} = {v}')
				acceptor_hdr[k] = v
		
		
		acceptor_hdr.update(aph.fits.header.DictReader(
			{
				'original_file' : acceptor_fits_spec.path,
				'donor_file' : donor_fits_spec.path,
			},
			prefix='copy_wcs',
			pkey_count_start=aph.fits.header.DictReader.find_max_pkey_n(acceptor_hdr)
		))
		
		hdu_wcs_copy = fits.PrimaryHDU(
			header = acceptor_hdr,
			data = acceptor_hdul[acceptor_fits_spec.ext].data
		)
		hdul_output = fits.HDUList([
			hdu_wcs_copy,
		])
		hdul_output.writeto(output_path, overwrite=True)
				
			
	





def parse_args(argv):
	import os
	import aopp_deconv_tool.text
	import argparse
	
	DEFAULT_OUTPUT_TAG = '_wcs_paired'
	DESIRED_FITS_AXES = ['ALL']
	OUTPUT_COLUMNS=80
	try:
		OUTPUT_COLUMNS = os.get_terminal_size().columns - 30
	except Exception:
		pass
	
	FITS_SPECIFIER_HELP = aopp_deconv_tool.text.wrap(
		aph.fits.specifier.get_help(DESIRED_FITS_AXES).replace('\t', '    '),
		OUTPUT_COLUMNS
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
		'donor_fits_spec',
		help = 'FITS file specifier, will use the WCS data from this file',
		type=str,
		metavar='FITS Specifier',
	)
	
	parser.add_argument(
		'acceptor_fits_spec',
		help = 'FITS file specifier, will change the WCS data in this file',
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

	
	args = parser.parse_args(argv)
	
	args.donor_fits_spec = aph.fits.specifier.parse(args.donor_fits_spec, DESIRED_FITS_AXES)
	args.acceptor_fits_spec = aph.fits.specifier.parse(args.acceptor_fits_spec, DESIRED_FITS_AXES)
	
	other_file_path = Path(args.acceptor_fits_spec.path)
	args.output_path = args.output_path.with_fields(
		tag=DEFAULT_OUTPUT_TAG, 
		parent=other_file_path.parent, 
		stem=other_file_path.stem, 
		suffix=other_file_path.suffix
	)
	
	return args


def go(
		donor_fits_spec,
		acceptor_fits_spec,
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
	arglist = aopp_deconv_tool.arguments.construct_arglist_from_locals(fargs, n_positional_args=2)
	exec_with_args(arglist)

def exec_with_args(argv):
	args = parse_args(argv)
	_lgr.info('#### ARGUMENTS ####')
	for k,v in vars(args).items():
		_lgr.info(f'\t{k} : {v}')
	_lgr.info('#### END ARGUMENTS ####')
	
	run(
		args.donor_fits_spec, 
		args.acceptor_fits_spec,
		args.output_path, 
	)

if __name__ == '__main__':
	exec_with_args(sys.argv[1:])