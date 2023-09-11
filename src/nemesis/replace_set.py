#!/usr/bin/env python3
"""
This docstring should describe the program

The `if __name__=='__main__':` statement allows execution of code if the script is called directly.
eveything else not in that block will be executed when a script is imported. 
Import statements that the rest of the code relies upon should not be in the if statement, python
is quite clever and will only import a given package once, but will give it multiple names if it
has been imported under different names.

Standard library documentation can be found at https://docs.python.org/3/library/

Packages used in this program are:
	sys
	os 
"""

import sys # https://docs.python.org/3/library/sys.html
import os # https://docs.python.org/3/library/os.html
import utils as ut # used for convenience functions
import shutil
import nemesis.write
import nemesis.read
def main(argv):
	"""This code will be executed if the script is called directly"""
	args = parse_args(argv)
	print(ut.str_wrap_in_tag(ut.str_dict(args), 'ARGUMENTS'))

	for sf in args['set_files']:
		ut.pINFO('Replacing file {}'.format(sf))
		if type(args['set_file_to_copy'])!=type(None):
			ut.pINFO('with file {}'.format(args['set_file_to_copy']))
			try:
				shutil.copyfile(args['set_file_to_copy'], sf)
			except shutil.SameFileError:
				pass #don't bother copying a file to itself
		else:
			ut.pINFO('with altered version of original')
			set_dict = nemesis.read.set(sf)
			#set_dict['n_zens'] = nzens
			#set_dict['quad_points'] = qcos
			#set_dict['quad_weights'] = qweights
			#set_dict['n_fourier_components'] = 6 # overwritten in code anyway
			#set_dict['n_azi_angles'] = 100 # set to 100 for years
			#set_dict['sunlight_flag'] = 1 # we have sunshine
			#set_dict['solar_dist'] = float(hdul[0].header['SUN_DIST'])
			#set_dict['lower_boundary_flag'] = 1
			#set_dict['ground_albedo'] = 0.0
			#set_dict['surf_temp'] = 0.0
			set_dict['base_altitude'] = -100.0 # unsure of units
			#set_dict['n_atm_layers'] = 39 # 39 is highest supported for scattering
			#set_dict['layer_type_flag'] = 1
			#set_dict['layer_int_flag'] = 1
			nemesis.write.set(set_dict, runname=sf)

	
	return

def parse_args(argv):
	"""Parses command line arguments, see https://docs.python.org/3/library/argparse.html"""
	import argparse as ap
	# =====================
	# FORMATTER INFORMATION
	# ---------------------
	# A formatter that inherits from multiple formatter classes has all the attributes of those formatters
	# see https://docs.python.org/3/library/argparse.html#formatter-class for more information on what each
	# of them do.
	# Quick reference:
	# ap.RawDescriptionHelpFormatter -> does not alter 'description' or 'epilog' text in any way
	# ap.RawTextHelpFormatter -> Maintains whitespace in all help text, except multiple new lines are treated as one
	# ap.ArgumentDefaultsHelpFormatter -> Adds a string at the end of argument help detailing the default parameter
	# ap.MetavarTypeHelpFormatter -> Uses the type of the argument as the display name in help messages
	# =====================	
	class RawDefaultTypeFormatter(ap.RawDescriptionHelpFormatter, ap.ArgumentDefaultsHelpFormatter, ap.MetavarTypeHelpFormatter):
		pass
	class RawDefaultFormatter(ap.RawDescriptionHelpFormatter, ap.ArgumentDefaultsHelpFormatter):
		pass
	class TextDefaultTypeFormatter(ap.RawTextHelpFormatter, ap.ArgumentDefaultsHelpFormatter, ap.MetavarTypeHelpFormatter):
		pass
	class TextDefaultFormatter(ap.RawTextHelpFormatter, ap.ArgumentDefaultsHelpFormatter):
		pass

	#parser = ap.ArgumentParser(description=__doc__, formatter_class = ap.TextDefaultTypeFormatter, epilog='END OF USAGE')
	# ====================================
	# UNCOMMENT to enable block formatting
	# ------------------------------------
	parser = ap.ArgumentParser	(	description=ut.str_block_indent_raw(ut.str_rationalise_newline_for_wrap(__doc__), wrapsize=79),
									formatter_class = RawDefaultTypeFormatter,
									epilog=ut.str_block_indent_raw(ut.str_rationalise_newline_for_wrap('END OF USAGE'), wrapsize=79)
								)
	# ====================================

	parser.add_argument('set_files', type=str, nargs='+', help='<runname>.set files to replace')
	parser.add_argument('--set_file_to_copy', type=str, help='<runname>.set file to replace them with')

	parsed_args = vars(parser.parse_args(argv)) # I prefer a dictionary interface
	return(parsed_args)

if __name__=='__main__':
	main(sys.argv[1:])
