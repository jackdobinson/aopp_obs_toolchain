#!/usr/bin/env python3
"""
Copies input files necessary for NEMESIS to run from one location to another. Will look in parent 
directories for missing files. That way you can set up variations of a model by creating a 
sub-folder that contains a modified file, and you don't have to copy all of the non-modified files
into the sub-folder.

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
import nemesis.read
import logging
import logging_setup
import shutil

log = logging_setup.log_init(__name__, 'INFO')

def main(argv):
	"""This code will be executed if the script is called directly"""
	print(argv)
	args = parse_args(argv)
	print(ut.str_wrap_in_tag(ut.str_dict(args), 'ARGUMENTS'))
	
	file_list = get_files_to_copy(args['src'], args['src.runname'], args['nemesis.tags'], args['src.n_search_parents'])

	for from_file in file_list:
		basefname = os.path.basename(from_file)
		if basefname.startswith(args['src.runname']):
			basefname = args['dest.runname']+basefname[len(args['src.runname']):]
		to_file = os.path.join(args['dest'], basefname)
		if os.path.exists(to_file) and (not args['dest.overwrite']):
			log.warning(f'Destination file {to_file} exists, we shall not copy file {from_file}')
		else:
			log.info(f'Copying file from {from_file} to {to_file}')
			if args['dry_run']:
				print('dry run, not copying files.')
			else:	
				shutil.copy(from_file, to_file)
	

	return

def get_files_to_copy(directory, runname, tags, n_search_parents):
	tags = list(tags)+['ALL']

	dirs_to_search = [os.path.abspath(directory)]
	for i in range(n_search_parents):
		dirs_to_search.append(os.path.dirname(dirs_to_search[-1]))

						# prototype, required for present tags, required for not present tags, conflicts with tags
	files_to_find = {	f'{runname}.inp':	(	('ALL',),		(), 			()		),
						f'{runname}.nam':	(	('ALL',),		(), 			()		),
						f'{runname}.set':	(	('ALL',),		(), 			()		),
						f'{runname}.ref':	(	('ALL',),		(), 			()		),
						f'{runname}.cia':	( 	('ALL',),		(), 			()		),
						f'{runname}.fla':	(	('ALL',),		(), 			()		),
						f'{runname}.zen':	( 	('ZEN',),		(), 			()		),
						f'{runname}.xsc':	(	('ALL',),		(), 			()		),
						f'{runname}.sur':	(	(),				('GP','LIMB'),	()		),
						f'{runname}.apr':	(	('ALL',),		(),				()		),
						f'{runname}.kls':	(	('ALL',),		(),				()		),
						f'{runname}.lls':	(	('LBL',),		(),				()		),
						f'{runname}.spx':	( 	('ALL',), 		(),				()		),
						f'{runname}.fil':	(	('CHI',),		(),				()		),
						f'{runname}.lbl':	(	('LBL',),		(),				()		),
						f'{runname}.sha':	(	('LBL',),		(),				()		),
						f'{runname}.key':	( 	('LBL',),		(),				()		),
						f'{runname}.pra':	(	('LBL',),		(),				()		),
						f'{runname}.sol':	(	('REF','TRA'),	(),				()		),
						f'{runname}.rfl':	(	('RFL',),		(),				()		),
						f'{runname}.vpf':	(	('VPF',),		(),				()		),
						f'{runname}.cel':	(	('SCR',),		(),				('RDW')	),
						f'{runname}.rdw':	(	('RDW',),		(),				('SCR')	),
						f'{runname}.pre':	(	('PRE',),		(),				()		),
						'aerosol.ref':		(	('ALL',),		(), 			()		),
						'fcloud.ref':		(	('SCA',),		(),				()		)
						}

	# build list of files to find depending on tags
	files_found = {}
	for k, v in files_to_find.items():
		req_present, req_not_present, conficts = v
		if any([_x in req_not_present for _x in req_present]):
			log.error('For file type {k} there is a tag in both the "requred for present" and "requred for not present" lists {req_present} vs. {req_not_present}')
		if any([_x in tags for _x in req_present]) and (not any([_x in tags for _x in req_not_present])):
			files_found[k] = None

	# loop through search directries and find files
	for adir in dirs_to_search:
		for afile in files_found.keys():
			test_file_path = os.path.join(adir,afile)
			if (files_found[afile] is None) and os.path.isfile(test_file_path):
				files_found[afile] = test_file_path

	# if we can't find all of our required files then say so and throw an error
	if any([_x is None for _x in files_found.values()]):
		dir_str = '\n'.join(dirs_to_search)
		#files_str = '\n'.join(filter(lambda x: files_found[x] is not None, files_found))
		files_str = '\n'.join([f'{k} {v}' for k,v in files_found.items()])
		log.error(f'Could not find all required files in search directories.\nSearched directories:\n{dir_str}\ncould not find files:\n{files_str}')
		raise FileNotFoundError

	# get the number of particle types
	ncont = nemesis.read.aerosol(files_found['aerosol.ref'])['ncont']
	
	# get wavenumber (0) or wavelength (1) from <runname>.inp
	ispace = nemesis.read.inp(files_found[f'{runname}.inp'])['ispace']

	# need to get number of particle types (defines in aerosol.ref or <runname>.xsc
	if ispace==0:
		extra_files = {'parah2.ref':		(	('GP',),			(), 			()		)} # only need this if we are using wavenumber not wavelength
	else:
		extra_files = {}
	for i in range(1,ncont+1):
		extra_files[f'hgphase{i}.dat'] =	(	('ALL'), 			(), 			()		)

	# update our "files_found" dictionary with the hgphase*.dat files to find and parah2.ref if needed	
	for k, v in extra_files.items():
		req_present, req_not_present, conficts = v
		if any([_x in req_not_present for _x in req_present]):
			log.error('For file type {k} there is a tag in both the "requred for present" and "requred for not present" lists {req_present} vs. {req_not_present}')
		if any([_x in tags for _x in req_present]) and (not any([_x in tags for _x in req_not_present])):
			files_found[k] = None

	# find any necessary files inside <runname>.apr that we need to find
	with open(files_found[f'{runname}.apr'], 'r') as f:
		for aline in f:
			awords = [_x.strip() for _x in aline.split('!')]
			if awords[0].endswith('.dat'):
				files_found[awords[0]] = None

	# find any necessary files inside <runname>.inp that we need to find
	with open(files_found[f'{runname}.inp'], 'r') as f:
		for aline in f:
			awords = [_x.strip() for _x in aline.split('!')]
			if awords[0].endswith('.dat'):
				files_found[awords[0]] = None

	# loop through search directries and find any unfound files
	for adir in dirs_to_search:
		for afile in files_found.keys():
			test_file_path = os.path.join(adir,afile)
			if (files_found[afile] is None) and os.path.isfile(test_file_path):
				files_found[afile] = test_file_path

	# if we can't find all our required files, then say so and throw an error
	if any([_x is None for _x in files_found.values()]):
		log.error('Could not find all required files in search directories.\nSearched directories:\n{"\n".join(dirs_to_search)}\ncould not find files:\n{"\n".join(filter(lambda x: files_found[x] is not None, files_found))}')
		raise FileNotFoundError

	# return paths to the required files
	return(files_found.values())



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

	parser.add_argument('src', type=str, help='Source directory of input files to copy')
	parser.add_argument('dest', type=str, help='Destination directory to copy file to')
	parser.add_argument('--src.runname', type=str, help='runname of the NEMESIS files to copy', default='neptune')
	parser.add_argument('--dest.runname', type=str, help='runname of the NEMESIS files to copy to', default='neptune')
	parser.add_argument('--dest.no_overwrite', action='store_false', dest='dest.overwrite', help='If present, will not overwrite files at destination')
	parser.add_argument('--src.n_search_parents', type=int, help='Sets the number of parent directories that will be searched for NEMESIS input files not found in the current directory', default=3)

	nemesis_tags = {'LBL':'Layer by layer calcuation',
					'LIMB':'Calculation is of a limb',
					'GP':'Calculation is of a giant planet',
					'REF':'Calcualtion is of a reflectance spectrum',
					'TRA':'Calculation is of a transit spectrum',
					'SCA':'Calculation includes scattering',
					'CHI':'Channel integratons are required',
					'RFL':'Include extra reflecting layer calculations the output',
					'VPF':'Include a file that limits specified gases volume mixing ratio by saturation',
					'RDW':'Includes a file that contains ranked wavelengths, NEMESIS will fit highest ranked wavelengths first, reduces computational time, incompatible with SCA',
					'ZEN':'Includes a file that specifies where the zenith angle is measured from, if not present zenith angle is measured from bottom of deepest layer'
					}
					
	parser.add_argument('--nemesis.tags', type=str, nargs='+', choices=nemesis_tags.keys(), 
						help=f'What type of run are you performing (changes files to be copied)\n{ut.str_dict(nemesis_tags)}', default=['SCA','GP','REF'])
	parser.add_argument('--dry_run', action='store_true', help='If present, will not actually copy files')

	parsed_args = vars(parser.parse_args(argv)) # I prefer a dictionary interface
	return(parsed_args)

if __name__=='__main__':
	main(sys.argv[1:])
