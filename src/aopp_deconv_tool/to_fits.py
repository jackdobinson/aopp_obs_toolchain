"""
Convert an image to FITS format
"""

import sys, os
from pathlib import Path
import dataclasses as dc
import argparse
import io
import re

from astropy.io import fits
import PIL
import PIL.Image
import PIL.features


import numpy as np

import aopp_deconv_tool.cfg.logs
_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'DEBUG')




"""
loat = dc.field(default=1E-2, 	metadata={
		'description':'Fraction of maximum brightness to add to numerator and denominator to try and avoid numerical instability. This value should be in the range [0,1), and will usually be small. Larger values require more steps to give a solution, but suffer less numerical instability.',
		'domain' : (0,1),
	})
	dc.field(init=False, repr=False, hash=False, compare=False)
@dc.dataclass(slots=True, repr=False, eq=False,)
class ImageToFitsConverter:
	imag
"""

re_dashed_line = re.compile(r'\n-+\n') # lines just consisting of "-" characters
re_comma_space = re.compile(r',\s') # lines just consisting of "-" characters

def get_supported_formats():
	ss = io.StringIO()
	PIL.features.pilinfo(out = ss, supported_formats=True)
	
	supported_extensions = []
	for chunk in re_dashed_line.split(ss.getvalue()):
		_lgr.debug(f'{chunk=}')
		extensions_are_supported = False
		possibly_supported_extensions = []
		
		for line in chunk.split('\n'):
			if line.startswith('Extensions: '):
				possibly_supported_extensions = re_comma_space.split(line[12:])
				_lgr.debug(f'{possibly_supported_extensions=}')
			if line.startswith('Features: '):
				if 'open' in line[10:]:
					extensions_are_supported = True
		if extensions_are_supported:
			supported_extensions.extend(possibly_supported_extensions)
			
	return supported_extensions


def read_image_into_numpy_array(fpath : str | Path):
	with PIL.Image.open(fpath) as image:
		image = image.convert(mode='F')
		data = np.array(image).astype(np.float64)
	return data

def save_as_fits(data, output_path : str | Path, **kwargs):
	hdu_primary = fits.PrimaryHDU(
		header = None,
		data = data
	)
	
	hdul = fits.HDUList([
		hdu_primary,
		*(fits.ImageHDU(header=None, name=k, data=v) for k,v in kwargs.items())
	])
	
	hdul.writeto(output_path, overwrite=True)
	_lgr.info(f'Converted file written to "{output_path}"')


def parse_args(argv):
	parser = argparse.ArgumentParser(
		description=__doc__,
	)
	
	SUPPORTED_FORMATS = get_supported_formats()
	_lgr.debug(f'{SUPPORTED_FORMATS=}')
	
	parser.add_argument('image_path', type=Path, help=f'Path to the image to convert, can be one of {" ".join(SUPPORTED_FORMATS)}')
	parser.add_argument('-o', '--output_path', type=Path, default=None, help='Path to save fits conversion to, if not supplied will replace original file extension with ".fits"')
	
	args = parser.parse_args(argv)
	
	if args.image_path.suffix not in SUPPORTED_FORMATS:
		parser.print_help()
		_lgr.error(f'Unsupported image format {args.image_path.suffix}')
		sys.exit()
	
	if args.output_path is None:
		args.output_path = Path(str(args.image_path.stem) + '.fits')
	
	for k,v in vars(args).items():
		_lgr.debug(f'{k} = {v}')
	
	return args


if __name__=='__main__':
	args = parse_args(sys.argv[1:])
	
	save_as_fits(read_image_into_numpy_array(args.image_path), args.output_path)