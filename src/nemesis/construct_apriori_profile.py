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
import numpy as np
import logging
import logging_setup
import nemesis.read
import nemesis.write

def main(argv):
	"""This code will be executed if the script is called directly"""
	args = parse_args(argv)
	logging_setup.logging_setup()
	print(ut.str_wrap_in_tag(ut.str_dict(args), 'ARGUMENTS'))
	
	if 'aerosol.continuous' in args['mode']:
		if (args['aerosol.continuous.ref_file'] is None) or (not os.path.isfile(args['aerosol.continuous.ref_file'])):
			logging.error(f'Need to pass a reference file (<runname>.ref) to ensure correct values for npro and the pressure grid, this was passed "{args["aerosol.continuous.ref_file"]}" and is either not a file or could not be found.')
			raise FileNotFoundError
		ref_dd = nemesis.read.ref(args['aerosol.continuous.ref_file'])
		npro = ref_dd['npro']
		p_grid = ref_dd['press']
		clen = args['aerosol.continuous.clen']
		if 'aerosol.continuous.funnel' in args['mode']:
			x, err = make_aerosol_continuous_funnel(npro, p_grid, 	args['aerosol.continuous.funnel.deep_val'], 
																	args['aerosol.continuous.funnel.shallow_val'], 
																	args['aerosol.continuous.funnel.top_error'], 
																	args['aerosol.continuous.funnel.unpin_pressure'], 
																	args['aerosol.continuous.funnel.deep_pressure'], 
																	args['aerosol.continuous.funnel.shallow_pressure'],
																	args['aerosol.continuous.funnel.max_err_pressure']
																	)
		else:
			logging.error(f'Unknown sub-mode "{args["mode"]}" for mode "aerosol.continuous"')
		
		nemesis.write.continuous_profile({'npro':npro, 'clen':clen, 'p':p_grid, 'x':x, 'err':err}, filename=args['aerosol.continuous.filename'])
	else:
		logging.error(f'Unknown mode "{args["mode"]}", see if-elif-else block in main to determine which modes are valid (adding extra qualifiers by appending a ".qualifier" to the mode should let you choose a specific function etc.')

	return

def make_aerosol_continuous_funnel(npro, p_grid, deep_val, shallow_val, top_error, unpin_pressure, deep_pressure, shallow_pressure, max_err_pressure):
	"""
	Creates a continuous profile that smoothly transitions from a deep value to a shallow value,
	below a certain 'unpin_pressure' the error is small enough that NEMESIS will treat the value as
	a constant, above the 'unpin_pressure' the error will smoothly increase to the 'top_error' at the 
	level with the smallest pressure. How fast the value transitions between 'deep_val' and 'top_val'
	is controlled with 'deep_pressure' and 'shallow_pressure' parameters.

	ARGUMENTS:
		npro
			<int> Number of profile levels
		p_grid [npro]
			<float> Pressure grid of the profile
		deep_val
			<float> The value of the profile when pressure > 'deep_pressure'
		shallow_val
			<float> The value of the profile when pressure < 'shallow_pressure'
		top_error
			<float> The value of the error when the pressure grid is it's smallest value
		unpin_pressure
			<float> The pressure value below which the error becomes larger that 1E-6 times the value (when error < 1E-6*value nemesis treats the value as a constant)
		deep_pressure
			<float> The pressure, above which, the value is equal to 'deep_val'
		shallow_pressure
			<float> The pressure, below which, the value is equal to 'top_val'
		max_err_pressure
			<float>

	RETURNS:
		x [npro]
			<float> The value of the profile at each pressure level
		err [npro]
			<float> The error on the value of the profile at each pressure level
	"""
	if p_grid[0] < p_grid[-1]:
		p_increasing_flag = True
	else:
		p_increasing_flag = False
		p_grid = p_grid[::-1] # reverse order, remember to swap back later if we should

	transition_function = lambda z, z_t, z_s: 1.0/(1.0 + np.exp(-(z_t-z)/z_s))
	
	x = deep_val + (shallow_val - deep_val)*transition_function(p_grid, (deep_pressure+shallow_pressure)/2, (deep_pressure-shallow_pressure)/15)

	top_err_percent = np.log(top_error/x[0])
	unpin_pressure_err_percent = np.log(1E-6)
	err_grad = (top_err_percent - unpin_pressure_err_percent)/(max_err_pressure - unpin_pressure)
	err_const = top_err_percent - err_grad*max_err_pressure
	err_percent = np.exp(err_grad*p_grid + err_const)
	err_percent[err_percent<1E-6] = 1E-6 # if error is too small will get a problem with non-invertable matrix or something
	err_percent[p_grid <= max_err_pressure] = np.exp(top_err_percent)
	err = err_percent*x

	if not p_increasing_flag: # swap order if we swapped it earlier
		x = x[::-1]
		err = err[::-1]
	return(x, err)

	

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
									formatter_class = TextDefaultTypeFormatter,
									epilog=ut.str_block_indent_raw(ut.str_rationalise_newline_for_wrap('END OF USAGE'), wrapsize=79)
								)
	# ====================================

	#parser.add_argument('positional', help='A description of the argument')
	mode_choices = {'aerosol.continuous.funnel':'A continuous aerosol profile that has a small value and small error at high pressures, and smoothly translates to a larger value with larger errors at low pressure.'
					}
	mode_choices_help = '\n'.join([f'\t{k}\n\t\t{v}' for k,v in mode_choices.items()])
	parser.add_argument('mode', type=str, choices=mode_choices.keys(), help=f'What kind of profile should we create? options are:\n{mode_choices_help}')

	# arguments for "aerosol.continuous" mode
	parser.add_argument('--aerosol.continuous.ref_file', type=str, help='Use this <runname>.ref file to work out the values of "npro" and the pressure grid', default=None)
	parser.add_argument('--aerosol.continuous.clen', type=float, help='Correlation length to use for a continuous profile', default=1.5)
	parser.add_argument('--aerosol.continuous.filename', type=str, help='File to save continuous profile to', default='aerosol1apr.dat')

	# arguments for "make_aerosol_continuous_funnel()"
	parser.add_argument('--aerosol.continuous.funnel.deep_val', type=float, help='"deep_val" to be passed to function creating a "funnel" shaped profile', default= 1E-9)
	parser.add_argument('--aerosol.continuous.funnel.shallow_val', type=float, help='"shallow_val" to be passed to function creating a "funnel" shaped profile', default= 2.5E-4)
	parser.add_argument('--aerosol.continuous.funnel.top_error', type=float, help='"top_error" to be passed to function creating a "funnel" shaped profile', default= 5E-4)
	parser.add_argument('--aerosol.continuous.funnel.unpin_pressure', type=float, help='"unpin_pressure" to be passed to function creating a "funnel" shaped profile', default= 1)
	parser.add_argument('--aerosol.continuous.funnel.deep_pressure', type=float, help='"deep_pressure" to be passed to function creating a "funnel" shaped profile', default= 1)
	parser.add_argument('--aerosol.continuous.funnel.shallow_pressure', type=float, help='"shallow_pressure" to be passed to function creating a "funnel" shaped profile', default= 0.1)
	parser.add_argument('--aerosol.continuous.funnel.max_err_pressure', type=float, help='"max_err_pressure" to be passed to function creating a "funnel" shaped profile', default= 0.8)

	parsed_args = vars(parser.parse_args(argv)) # I prefer a dictionary interface
	return(parsed_args)

if __name__=='__main__':
	main(sys.argv[1:])
