#!/usr/bin/env python3
"""
# DESCRIPTION #
Fits a disk to a datacube to find LAT, LON, OBSEVER-ZEN angles.

	## FITS File Path Format ##
		../path/to/datafile.fits<info_ext>{ext}[slice_tuple](img_axes_tuple)
			info_ext : int | str
				Extension of the FITS file that contains information on 
				the object contained in the image. If not present, will
				use the first extension that has no data, or if all 
				extensions have data will be the same as "ext".
			ext : int | str
				Extension of the FITS file to operate upon, if not present,
				will use the first extension that has some data.
			slice_tuple : tuple[Slice,...]
				Slice of the data to operate upon, useful for choosing
				a subset of wavelength channels. If not present, will
				assume all wavelengths are to be deconvolved.
			img_axes_tuple : tuple[int,...]
				Tuple of the spatial axes indices, uses FITS ordering
				if +ve, numpy ordering if -ve. Usually the RA,DEC axes.

		### Examples ###
			./some_datafile.fits{PRIMARY}[100:150]
				Select the extension called 'PRIMARY' and the channels 
				100->150
			/an/absolute/path/to/this_data.fits[99:700:50]
				Try to guess the extension to use, use every 50th channel 
				in the range 99->700
			./deconv/whole/file.fits<0>{SCI}
				Use the extension 'SCI' as the data extension, use extension 0 
				as the infomation extension
			./some/path/big_file.fits{1}[5:500:10](0,2)
				Use the 1st extension (not the 0th), use every 10th channel 
				in the range 5->500, the spatial axes are 0th and 2nd axis.

# END DESCRIPTION #	
"""



import sys, os
# Add folder that contains scripts to $PATH environment variable.
# need to do this so directory structure is easily transferrable
# between machines.
SCRIPTS_PATH = os.path.normpath(f"{os.path.dirname(__file__)}/../scripts")
sys.path.insert(0,SCRIPTS_PATH)
print('INFO: \tPython will search for modules in these locations:\n----: \t\t{}\n'.format("\n----: \t\t".join(sys.path)))

import utilities.logging_setup
#logging, _lgr = utilities.logging_setup.getLoggers(__name__, 'DEBUG') # Show debug logs and above (command is "_lgr.DEBUG")
logging, _lgr = utilities.logging_setup.getLoggers(__name__, 'INFO') # Show information logs and above (command is "_lgr.INFO")
#logging, _lgr = utilities.logging_setup.getLoggers(__name__, 'WARN') # Show warning logs and above (command is "_lgr.WARN")
#logging, _lgr = utilities.logging_setup.getLoggers(__name__, 'ERROR') # Show error logs and above (command is "_lgr.ERROR")
#logging, _lgr = utilities.logging_setup.getLoggers(__name__, 'CRIT') # Show critical logs and above (command is "_lgr.CRIT")


from astropy.io import fits

import utilities as ut
import utilities.args
import utilities.str
import utilities.dict

import fitscube.geometry.ellipse
import fitscube.fit_region


# This is useful to have as a global variable so 
# I can print the help message when there are errors.
parser = ut.args.RawArgParser(description=__doc__)

def parse_fits_path(apath):
	"""
	Parses a file like:

	.../some/path/to/file.fits<info_ext>{data_ext}[slice](img_axes)
	"""
	img_axes_str = None
	slice_str = None
	ext_str = None
	info_ext_str = None

	_idx = len(apath)
	img_axes_idx = _idx
	if apath.endswith(')'):
		img_axes_idx = apath.rfind('(')
		img_axes_str = apath[img_axes_idx:_idx]

	slice_idx = img_axes_idx
	if apath.endswith(']'):
		slice_idx = apath.rfind('[')
		slice_str = apath[slice_idx:]

	ext_idx = slice_idx
	if apath[slice_idx-1] == '}':
		ext_idx = apath.rfind('{')
		ext_str = apath[ext_idx:slice_idx]

	info_ext_idx = ext_idx
	if apath[ext_idx-1] == '>':
		info_ext_idx = apath.rfind('<')
		info_ext_str = apath[info_ext_idx:ext_idx]

	if img_axes_str is not None:
		try:
			img_axes_str = eval(img_axes_str)
		except:
			_lgr.ERROR(f'Malformed FITS path, img_axes_tuple should be of the form (int, int, ...). Example: (1,2)')
			parser.print_help()
			sys.exit()

	if slice_str is not None:
		try:
			slice_str = eval(f'np.index_exp{slice_str}')
		except:
			_lgr.ERROR(f'Malformed FITS path, slice should be of form [start:stop:step]')
			parser.print_help()
			sys.exit()

	if ext_str is not None:
		try:
			ext_str = int(ext_str[1:-1])
		except ValueError:
			ext_str = ext_str[1:-1]

	if info_ext_str is not None:
		try: 
			info_ext_str = int(info_ext_str[1:-1])
		except ValueError:
			info_ext_str = info_ext_str[1:-1]


	return(apath[:ext_idx], info_ext_str, ext_str, slice_str, img_axes_str)
	
def set_default_args_from_fits_hdul(args, hdul, prefix=''):
	"""
	Sets default arguments for arguments that depend on values in an HDUL (fits file).
	"""

	# Find the first data extension with some actual data in it
	if args[f'{prefix}ext'] is None:
		for i, hdu in enumerate(hdul):
			if hdu.data is not None and hdu.data.size > 0:
				args[f'{prefix}ext'] = i
				break
		if i >= len(hdul):
			_lgr.ERROR(f'Could not automatically assign {prefix}ext. No HDU has any data in it.')
			parser.print_help()
			sys.exit()

		
	# find the first HDU with zero data and assume that is the correct
	# INFO extension. If none present, use the same as the data extension.
	if args[f'{prefix}info_ext'] is None:
		for i, hdu in enumerate(hdul):
			if hdu.data is None or hdu.data.size == 0:
				args[f'{prefix}info_ext'] = i
				break
		if i >= len(hdul):
			args[f'{prefix}info_ext'] = args[f'{prefix}ext']


	# get the image axes by reading the header data, or parse the integers we have been given.
	hdu = hdul[args[f'{prefix}ext']]
	if args[f'{prefix}img_axes'] is None:
		args[f'{prefix}img_axes'] = ut.fits.hdr_get_celestial_axes(hdu.header)
	else:
		args[f'{prefix}img_axes'] = tuple(-i if i <0 else ut.fits.AxesOrdering(hdu.header['NAXIS'], i, 'fits').numpy for i in args[f'{prefix}img_axes'])
	return


def parse_args(argv):
	parser.add_argument(
		'input.file', 
		metavar='INPUT_FITS_PATH',
		type=str, 
		help='FITS file to operate upon, uses FITS file path format',
	)
	parser.add_argument(
		'--output.file',
		type=str,
		help='FITS file to output deconvolved data to.',
		default="./output/disk.fits",
	)
	parser.add_argument(
		'--output.file.overwrite',
		action=ut.args.ActionTf,
		prefix='output.file.',
		help='Should we overwrite the output file or not?',
	)

	parser.add_argument(
		'--plots.show',
		action=ut.args.ActionTf,
		prefix='plots.',
		help='Should we show plots or not?'
	)

	parser.add_argument(
		'--output.data',
		action=ut.args.ActiontF,
		prefix='output.',
		true_tag = 'all_',
		false_tag = 'disk_',
		help='Should we output all of the data (including everything from "input.file"), or just the disk data?'
	)

	args = vars(parser.parse_args())

	args['input.file'], args['input.file.info_ext'], args['input.file.ext'], args['input.file.slice'], args['input.file.img_axes'] = parse_fits_path(args['input.file'])


	return(args)

	
def main(argv):
	args = parse_args(argv)
	print(ut.str.wrap_in_tag(ut.dict.to_str(args), 'ARGUMENTS'))

	with fits.open(args['input.file']) as hdul:
		set_default_args_from_fits_hdul(args, hdul, 'input.file.')

		hdul[args['input.file.ext']].data = fitscube.geometry.ellipse.remove_anomalous_data_1(hdul[args['input.file.ext']].data, axes=tuple(args['input.file.img_axes']), factor=1E2)

		ellipse_inputs = fitscube.geometry.ellipse.get_ellipse_inputs_from_fits(hdul, args['input.file.info_ext'], args['input.file.ext'], use_horizons_pos=False)
		ellipse_params, lats, lons, zens, xx, yy = fitscube.geometry.ellipse.project_lat_lon(hdul, *ellipse_inputs, show_plots=args['plots.show'], find_disk_manually=True, ext=args['input.file.ext'])



		info_hdr = hdul[args['input.file.info_ext']].header
		data_hdr = hdul[args['input.file.ext']].header
		if not args['output.data']:
			hdul = fits.HDUList([fits.PrimaryHDU(header=info_hdr, data=None)])

		for k, v in (('Latitude', lats), ('Longitude', lons), ('Zenith', zens), ('equatorial_offset',xx), ('polar_offset', yy)):
			_lgr.INFO(('{} extension '+f'"{k}"'+' {} datacube.').format(*(("Setting","in") if args["output.file.overwrite"] else ("Appending","to"))))
			if k in hdul and args['output.file.overwrite']:
				hdul[k] = fits.ImageHDU(header = data_hdr, data = v, name=k)
			elif k not in hdul:
				hdul.append(fits.ImageHDU(header = data_hdr, data = v, name=k))
			else:
				_lgr.WARN(f'An extension called {k} was already present in {args["output.file"]}, and we are not overwriting data. Skipping this extension...')

		_lgr.INFO(f'Writing fitted disk information to {args["output.file"]}')
		hdul.writeto(args['output.file'], overwrite=args['output.file.overwrite'])
	

			

if __name__=='__main__':
	main(sys.argv[1:])
