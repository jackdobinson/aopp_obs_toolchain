"""
Convert an image to FITS format
"""

import sys, os
from pathlib import Path
import dataclasses as dc
import argparse
import io
import re

from collections import namedtuple

from astropy.io import fits
import PIL
import PIL.Image
import PIL.features
import PIL.ExifTags


import numpy as np

import aopp_deconv_tool.astropy_helper as aph
import aopp_deconv_tool.astropy_helper.fits.header

import aopp_deconv_tool.cfg.logs
_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'DEBUG')


DataBundle = namedtuple('DataBundle', ('data', 'header'))

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


def read_exif(image, header={}, exif_tag_reader=lambda k, v: (str(k), v)):
	exif = image.getexif()
	_lgr.debug(f'{exif=}')
	for k,v in exif.items():
		tag, value = exif_tag_reader(k,v)
		header[tag] = value
		_lgr.debug(f'{k} : {v}')
	_lgr.debug(f'DONE')
	return header

def read_image_into_numpy_array(fpath : Path) -> DataBundle:
	match fpath.suffix:
		case '.tif' | '.tiff':
			exif_tag_reader = lambda k, v: (PIL.TiffTags.lookup(k).name, v)
		case _:
			exif_tag_reader=lambda k, v: (str(k), v)
	
	
	data = None	
	header = {}
	with PIL.Image.open(fpath) as image:
		header = read_exif(image, header=header, exif_tag_reader=exif_tag_reader)
		image = image.convert(mode='F')
		data = np.array(image).astype(np.float64)
		
	return DataBundle(data, header)


def save_as_fits(output_path : str | Path, primary_data_bundle : DataBundle, **kwargs : dict[str, DataBundle]):

	hdr = fits.Header()
	hdr.update(aph.fits.header.DictReader(primary_data_bundle.header))
	hdu_primary = fits.PrimaryHDU(
		header = hdr,
		data = primary_data_bundle.data
	)
	
	hdu_list = [hdu_primary]
	
	for k,v in kwargs.items():
		new_hdr = fits.Header()
		new_hdr.update(aph.fits.header.DictReader(v.header))
		hdu_list.append(fits.ImageHDU(header=new_hdr, name=k, data=v.data))
	
	
	hdul = fits.HDUList(hdu_list)
	
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
	
	
	data_bundle = read_image_into_numpy_array(args.image_path)
	save_as_fits(args.output_path, data_bundle)