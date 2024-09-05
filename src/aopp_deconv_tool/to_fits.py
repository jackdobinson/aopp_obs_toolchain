"""
Convert an image to FITS format
"""

import sys, os
from pathlib import Path
import dataclasses as dc
import argparse
import io
import re
from typing import Any

from collections import namedtuple

from astropy.io import fits
import PIL
import PIL.Image
import PIL.features
import PIL.ExifTags

import matplotlib.pyplot as plt
import numpy as np

from aopp_deconv_tool import plot_helper
import aopp_deconv_tool.astropy_helper as aph
import aopp_deconv_tool.astropy_helper.fits.header
from aopp_deconv_tool.fpath import FPath
import aopp_deconv_tool.arguments

import aopp_deconv_tool.cfg.logs
_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'WARN')

# DataBundle is a grouping of related data and header information
DataBundle = namedtuple('DataBundle', ('data', 'header'))

re_dashed_line = re.compile(r'\n-+\n') # lines just consisting of "-" characters
re_comma_space = re.compile(r',\s') # comma then a space

def get_supported_formats():
	"""
	Return a list of supported file extensions that can be successfully converted to FITS format
	"""
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
	"""
	Read the EXIF tags of an image
	"""
	exif = image.getexif()
	_lgr.debug(f'{exif=}')
	for k,v in exif.items():
		tag, value = exif_tag_reader(k,v)
		header[tag] = value
		_lgr.debug(f'{k} : {v}')
	_lgr.debug(f'DONE')
	return header

def read_image_into_data_bundle(fpath : Path) -> DataBundle:
	"""
	Read an image at `fpath` into a numpy array. Store EXIF tags in the header field of a DataBundle, and the image data in "data" field of a DataBundle.
	"""
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

def plot_image_and_fits(image_path : str | Path, fits_path : str | Path, ext=None, slices=None, fig=None, ax=None, fig_kwargs : dict[str,Any] = {'figsize':(12,4)}):
	"""
	Uses matplotlib to display an image and a FITS file side-by-side
	"""
	
	image = read_image_into_data_bundle(image_path)
	fits_header = fits.getheader(fits_path, ext=ext)
	fits_data = fits.getdata(fits_path, ext=ext)
	
	# Note, can use slices=(0,slice(None),slice(None)) to reduce number of dimensions
	if slices is not None:
		fits_data = fits_data[*slices]
	
	assert fits_data.ndim == 2, f"{fits_path} does not have exactly 2 dimensions when read in. Need to slice or index the file."
	
	
	fig, ax = plot_helper.ensure_fig_and_ax(fig, ax, fig_kwargs, (1,3))
	fig.suptitle('Image and FITS data with data range.')
	
	ax[0].set_title(f'Image data [{np.nanmin(image.data):+9.2E},{np.nanmax(image.data):+9.2E}]')
	ax[0].imshow(image.data)
	ax[0].set_axis_off()
	
	ax[1].set_title(f'FITS data [{np.nanmin(fits_data):+9.2E},{np.nanmax(fits_data):+9.2E}]')
	ax[1].imshow(fits_data)
	ax[1].set_axis_off()
	
	residual = image.data - fits_data
	ax[2].set_title(f'Image - FITS [{np.nanmin(residual):+9.2E},{np.nanmax(residual):+9.2E}]')
	ax[2].imshow(image.data - fits_data)
	ax[2].set_axis_off()
	
	return fig, ax
	
	
	
	
	
	


def save_as_fits(output_path : str | Path, primary_data_bundle : DataBundle, **kwargs : dict[str, DataBundle]):
	"""
	Save a `primary_data_bundle` and a list of other data bundles to a specified path
	"""
	hdr = fits.Header()
	update_if_not_present = lambda hdr, key, value: hdr.__setitem__(key, value if key not in hdr else hdr[key])
	# If we do not already have CTYPEs for axes, we need to add them here
	# assume axis 0 is spectral, axis 1 is celestial, axis 2 is celestial
	ctypes = ('SPEC_PIX', 'SKY__PIX', 'SKY__PIX')
	cunits = ('PIXEL', 'PIXEL', 'PIXEL')
	for i in range(len(ctypes)):
		ctype_i_key = f'CTYPE{3 - i}' # convert from numpy order to FITS order
		cunit_i_key = f'CUNIT{3 - i}' # convert from numpy order to FITS order
		crval_i_key = f'CRVAL{3 - i}' # convert from numpy order to FITS order
		cdelt_i_key = f'CDELT{3 - i}' # convert from numpy order to FITS order
		crpix_i_key = f'CRPIX{3 - i}' # convert from numpy order to FITS order
		update_if_not_present(hdr, ctype_i_key, ctypes[i])
		update_if_not_present(hdr, cunit_i_key, cunits[i])
		update_if_not_present(hdr, crval_i_key, 1)
		update_if_not_present(hdr, cdelt_i_key, 1)
		update_if_not_present(hdr, crpix_i_key, 1)
			
	hdr.update(aph.fits.header.DictReader(primary_data_bundle.header))
	
	# Other routines rely on 3 axes, so make our output have 3 axes
	if primary_data_bundle.data.ndim == 2:
		output_data = primary_data_bundle.data[None,...]
	else:
		output_data = primary_data_bundle.data
	
	
	
	
	hdu_primary = fits.PrimaryHDU(
		header = hdr,
		data = output_data
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
	#parser.add_argument('-o', '--output_path', type=Path, default=None, help='Path to save fits conversion to, if not supplied will replace original file extension with ".fits"')
	parser.add_argument(
		'-o', 
		'--output_path', 
		type=FPath,
		metavar='str',
		default='{parent}/{stem}.fits',
		help = '\n    '.join((
			f'Output fits file path, supports keyword substitution using parts of `image_path` path where:',
			'{parent}: containing folder',
			'{stem}  : filename (not including final extension)',
			'{suffix}: final extension (everything after the last ".")',
			'\b'
		))
	)
	
	args = parser.parse_args(argv)
	
	if args.image_path.suffix not in SUPPORTED_FORMATS:
		parser.print_help()
		_lgr.error(f'Unsupported image format {args.image_path.suffix}')
		sys.exit()
	
	#if args.output_path is None:
	#	args.output_path = Path(str(args.image_path.stem) + '.fits')
	other_file_path = Path(args.image_path)
	args.output_path = args.output_path.with_fields(
		parent=other_file_path.parent, 
		stem=other_file_path.stem, 
		suffix=other_file_path.suffix
	)
	
	for k,v in vars(args).items():
		_lgr.debug(f'{k} = {v}')
	
	return args



def go(
		image_path,
		output_path=None,
		help_FLAG=None
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
	
	
	data_bundle = read_image_into_data_bundle(args.image_path)
	save_as_fits(args.output_path, data_bundle)

if __name__=='__main__':
	exec_with_args(sys.argv[1:])