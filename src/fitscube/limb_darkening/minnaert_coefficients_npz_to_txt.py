#!/usr/bin/env python3
import sys, os
import numpy as np



def parse_args(argv):
	"""Parses command line arguments, see https://docs.python.org/3/library/argparse.html"""
	import argparse as ap
	import plotutils
	"""
	# =====================
	# FORMATTER INFORMATION
	# ---------------------
	A formatter that inherits from multiple formatter classes has all the attributes of those formatters
	see https://docs.python.org/3/library/argparse.html#formatter-class for more information on what each
	of them do.
	Quick reference:
	ap.RawDescriptionHelpFormatter -> does not alter 'description' or 'epilog' text in any way
	ap.RawTextHelpFormatter -> Maintains whitespace in all help text, except multiple new lines are treated as one
	ap.ArgumentDefaultsHelpFormatter -> Adds a string at the end of argument help detailing the default parameter
	ap.MetavarTypeHelpFormatter -> Uses the type of the argument as the display name in help messages
	# =====================	
	"""
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
	parser = ap.ArgumentParser	(formatter_class = RawDefaultTypeFormatter)
	# ====================================

	parser.add_argument('npz_files', type=str, nargs='+', help='A list of minnaert input files to operate on')
	
	parser.add_argument('--overwrite',  action=plotutils.ActionTf, help='Should we overwrite the outputs')
	parsed_args = vars(parser.parse_args(argv))

	return(parsed_args)
		


if __name__=='__main__':
	args = parse_args(sys.argv[1:])
	
	for npz_file in args['npz_files']:
		txt_file = npz_file.rsplit('.',1)[0]+'.txt'
		if os.path.exists(txt_file) and (args['overwrite'] is False):
				print('INFO: File exists, skipping...')
		else:
			with np.load(npz_file) as dl:
				print(f'INFO: arrays found {[(k,dl[k].shape) for k in dl.keys()]}')
				wavs, IperF0s, ks, log_IperF0s_var, ks_var, IperF0s_var, n_points, n_used_points = (dl['wavs'], dl['IperF0s'], dl['ks'], 
																dl['log_IperF0s_var'], dl['ks_var'], dl['IperF0s_var'],
																dl['n_points'], dl['n_used_points'])
				with open(txt_file, 'w') as f:
					f.write('# wavs IperF0 k IperF0_err k_err\n')
					for i in range(wavs.size):
						f.write(f'{wavs[i]} {IperF0s[i]} {ks[i]} {IperF0s_var[i]} {ks_var[i]}\n')
						