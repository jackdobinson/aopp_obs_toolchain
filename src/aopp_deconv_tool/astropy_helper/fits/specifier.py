"""
Routine for parsing a string containing the path, extension, slices, and axes
of a FITS file that we want to operate on.
"""
import os
from collections import namedtuple 

from astropy.io import fits


import aopp_deconv_tool.text as text
import aopp_deconv_tool.cast as cast

import aopp_deconv_tool.numpy_helper as nph
import aopp_deconv_tool.numpy_helper.slice

import aopp_deconv_tool.astropy_helper as aph
import aopp_deconv_tool.astropy_helper.fits.header

import aopp_deconv_tool.cfg.logs
_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'DEBUG')


FitsSpecifier = namedtuple('FitsSpecifier', ('path', 'ext', 'slices', 'axes'))



AxesInfo = namedtuple('AxesInfo', ('description', 'default_callable'))
axes_type_info={
	"SPECTRAL" : AxesInfo("wavelength/frequency varies along this axis", aph.fits.header.get_spectral_axes),
	"CELESTIAL" : AxesInfo("sky position varies along this axis", aph.fits.header.get_celestial_axes),
	"POLARISATION" : AxesInfo("polarisation varies along this axis, could be linear of circular", aph.fits.header.get_polarisation_axes),
	"TIME" : AxesInfo("time varies along this axis", aph.fits.header.get_time_axes),
}

help_fmt = """\
# FITS SPECIFIER #

	A string that describes which FITS file to load, the extension (i.e backplane) to use, the slices (i.e. sub-regions)
	that should be operated upon, and (if required) the semantic meaning of the data axes.

	Format:
		.../path/to/a.fits{{ext}}[slice0,slice1,...]{{axes_type_1:(ax11,ax12,...),axes_type_2:(ax21,ax22,...)}}
		
		Where:
			.../path/to/a.fits
				A path to a FITS file. Must be present.
			ext : str | int
				Fits extension, either a string or an index. If not present, will assume PRIMARY hdu, (hdu index 0)
			sliceM : str
				Slice of Mth axis in the normal python `start:stop:step` format. If not present will assume slice(None) for an axis.
			axes_type_N : axes_type
				Type of the Nth axes set. Axes types are detailed below. They are used to tell a program what to do with
				an axis when normal FITS methods of description fail (or as a backup). Note, the type (and enclosing curly backets
				`{{}}` can be ommited if a program only accepts one or two axes types. In that case, the specified axes will be
				assumed to be of the first type specified in the documentation, and the remaining axes of the second type (if there
				is a second type).
			axNL : int
				Axes of the extension that are of the specified type. If not present, will assume all axes are of the first type
				specified in the documentation.
		
		Accepted axes_type:
			{axes_types}
			
		NOTE: As the format for a FITS specifier uses characters that a terminal application may interpret as special characters,
			e.g. square/curly/round brackets, and colons. It can be better to wrap specifiers in quotes or single quotes. When
			doing this, it is important to un-escape any previously escaped characters.
			For example, specifies with timestamps in them would normally have the colons escaped, but when wrapped in quotes
			this is not required. E.g. the specifier ./example_data/MUSE.2019-10-17T23\:46\:14.117_normalised.fits(1,2) will not
			play nice with the bash shell due to the brackets. However, wrapping it in single quotes and removing the escaping
			slashes from the colons means it will work. E.g. './example_data/MUSE.2019-10-17T23:46:14.117_normalised.fits(1,2)'
		
		Examples:
			~/home/datasets/MUSE/neptune_obs_1.fits{{DATA}}[100:200,:,:](1,2)
				Selects the "DATA" extension, slices the 0th axis from 100->200 leaving the others untouched, and signals that axes 1 and 2 are important e.g. they are the RA-DEC axes.
			
			~/home/datasets/MUSE/neptune_obs_1.fits{{DATA}}[100:200](1,2)
				Does the same thing as above, but omits un-needed slice specifiers.
			
			~/home/datasets/MUSE/neptune_obs_1.fits{{DATA}}[100:200]{{CELESTIAL:(1,2)}}
				Again, same as above, but adds explicit axes type.
"""

def get_help(axes_types : list[str]):
	"""
	Generates the help string for a list of axes_types.
	"""
	for x in axes_types:
		if x not in axes_type_info:
			raise RuntimeError(f"axes_type '{x}' is not one of the known axes_types {list(axes_type_info.keys())}")
	
	return help_fmt.format(axes_types='\n\t\t'.join([k+'\n\t\t\t'+axes_type_info[k].description for k in axes_types]))



def parse_axes_type_list(axes_type_list : str, axes_types: list[str] | tuple[str]):
	"""
	Parses the axes type list at the end of the specifier string.
	
	Arguments:
		axes_type_list : str
			"axes_type_1:(ax11,ax22,...),axes_type_2:(ax21,ax22,...)" or "(ax11,ax22)"
		axes_types : list[str] | tuple[str]
			List of acceptable axes types
	"""
	
	# first split on commas
	axes = {}
	#_lgr.debug(f"{text.split_around_brackets(axes_type_list)=}")
	for i, axes_type_str in enumerate(text.split_around_brackets(axes_type_list)):
		# axes_type_str = "axes_type_1:(ax11,ax12,...)" or "(ax11,ax12,...)"
		#_lgr.debug(f'{axes_type_str=}')
		n_colon = axes_type_str.count(':')
		if n_colon == 0:
			axtype, axtuple_str = (axes_types[i], axes_type_str)
		elif n_colon == 1:
			axtype, axtuple_str = axes_type_str.split(':')
		else:
			raise RuntimeError("axes_type_str '{axes_type_str}' has more than one colon, it is malformed.")
		
		# now parse axtuple_str
		assert axtuple_str[0] == '(' and axtuple_str[-1] ==')', f"axes tuple '{axtuple_str}' must be enclosed in brackets"
		
		axtuple = text.to_tuple(axtuple_str[1:-1], int)
		axes[axtype] = axtuple
	
	return(axes)
		
	

def parse(specifier : str, axes_types : list[str]):
	f"""
	Parses a string that specifies a fits file, extension, and slices.
	
	# ARGUMENTS #
		specifier : str
			A FITS specifier string, has the following format:
				{text.indent(get_help(axes_types), 4)}
		axes_types : list[str]
			A list of axes types that we expect. Should be a list of entries from `axes_type_info.keys()`
	
	# RETURNS #
		FitsSpecifier(path, ext, slices, axes)
			
	"""
	help_string = get_help(axes_types)
	
	# Go through cases from end to start
	
	try:
		# get axes information
		j = len(specifier) - 1
		axes_type_list = None
		if specifier[j-1:] == ')}':
			# we have axes_types in the specifier.
			i = specifier.rfind('{',0,j)
			if i == -1:
				raise RuntimeError("FITS specifier is malformed, missing '{'")
			axes_type_list = specifier[i+1:j]
			specifier = specifier[:i]
			j = i-1
		elif specifier[j] == ')':
			i = specifier.rfind('(',0,j)
			if i == -1:
				raise RuntimeError("FITS specifier is malformed, missing '('")
			axes_type_list = specifier[i:j+1]
			specifier = specifier[:i]
			j = i-1
		#_lgr.debug(f'{axes_type_list=}')	
		if axes_type_list is not None:
			axes = parse_axes_type_list(axes_type_list, axes_types)
		else:
			axes = None
		_lgr.debug(f'{axes=}')
		if axes is not None:
			for k,v in axes.items():
				if len(v) == 0:
					raise RuntimeError(f'Could not work out axes "{k}" automatically, please specify axes explicitly')
		else:
			raise RuntimeError(f'Could not work out axes automatically, please specify axes explicitly')
			
		
		# get slice information
		if specifier[j] == ']':
			# we have slice information
			i = specifier.rfind('[',0,j)
			if i == -1:
				raise RuntimeError("FITS specifier is malformed, missing '['")
			slice_tuple_str = specifier[i+1:j] # exclude "[" and "]" to give "a:b:c,d:e:f,..."
			slices = nph.slice.from_string(slice_tuple_str)
			specifier = specifier[:i]
			j = i-1
		else:
			slices=None
		
		
		# get extension
		if specifier[j] == '}':
			i = specifier.rfind('{',0,j)
			if i == -1:
				raise RuntimeError("FITS specifier is malformed, missing '{'")
			ext_str = specifier[i+1:j]
			ext = cast.to_any(ext_str, (int, str))
			specifier = specifier[:i]
			j = i-1
		else:
			ext = "DATA"
		
	except Exception as e:
		e.add_note(help_string)
		raise
	
	# get path
	path = specifier
	
	if not os.path.isfile(path):
		raise FileNotFoundError(f'file "{path}" not found')

	if slices is None or axes is None:
		try:
			hdr = fits.getheader(path,ext=ext if type(ext) is int else (ext,1))
		except Exception as e:
			pass
		else:
			if slices is None:
				slices = tuple(slice(None) for _ in range(hdr['NAXIS']))
			if axes is None:
				axes = {}
				for ax_name in axes_types:
					axes[ax_name] = axes_type_info[ax_name].default_callable(hdr)


	return FitsSpecifier(path, ext, slices, axes)
