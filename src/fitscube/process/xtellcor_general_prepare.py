#!/usr/bin/env python3
"""
Creates input text files required by the IDL routine "xtellcor_general.pro" for a target and it's calibrators. Also
writes other ancilliary information about the calibrator/target that the routine requires such as V-band and B-band
flux, distance, etc. Uses the output of "pipeline_test.py" (a script that performs the esorex data reduction pipeline
for the SINFONI instrument) to collate observed targets with their calibrators. After this script has finished,
a message will be printed explaining the next steps for running "xtellcor_general.pro" to find the telluric spectrum
for each of the files created.

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
from astropy.io import fits
from astroquery.simbad import Simbad
import numpy as np
import fitscube.header
import textwrap
import pretty_table

def main(argv):
	"""This code will be executed if the script is called directly"""
	args = parse_args(argv)
	print(ut.str_wrap_in_tag(ut.str_dict(args), 'ARGUMENTS'))

	tc_dat_files = []
	cal_dat_files = []
	for tc in args['target_cubes']:
		ut.pINFO(f'Operating on target cube: "{tc}"')
		tc_dir = os.path.dirname(tc)
		tc_calibs_file = os.path.join(tc_dir, args['rel_calibrators_file'])
		tc_tellspec_folder = os.path.join(tc_dir, args['rel_tellspec_folder'])
		print(f'INFO: tc_dir {tc_dir}')
		print(f'INFO: tc_calibs_file {tc_calibs_file}')
		print(f'INFO: tc_tellspec_folder {tc_tellspec_folder}')
		
		# NEED: calibrator wavelength vs flux, object wavelength vs flux, B and V band magnitudes of calibrator, wavelength units, flux units should be the same but all calculations use relative flux so exact units unimportant, FWHM of instrument spectral response profile (assumed to be gaussian), rotation (Km s^-1) of calibrator, velocity along line of sight of object (km s^-1), velocity along line of sight of calibrator (km s^-1).

		cal_fits_files = get_calibrator_fits_files(tc_calibs_file, args['calibrators_relative_fix'], args['calibrators_relative_replace'])
		print('INFO: calibrator fits file paths')
		cal_not_found_flag = False
		for cff in cal_fits_files:
			if os.path.exists(cff):
				tag='EXISTS'
			else:
				tag = 'DOES NOT EXIST'
				cal_not_found_flag = True
			print(f'----:\t{cff}\t{tag}')
		if cal_not_found_flag:
			print('ERROR: Could not find one or more calibrator files (see above). Check "--calibrators_relative_fix" and paths to files.')
			sys.exit()


		cal_names, cal_wavs, cal_specs = zip(*[get_fits_obj_name_and_spectra(cal_fits_file) for cal_fits_file in cal_fits_files])
		print('INFO: Found the following calibrator names')
		for cal_name in cal_names:
			print(f'----:\t{cal_name}')
		tc_name, tc_wav, tc_spec = get_fits_obj_name_and_spectra(tc)
		print(f'INFO: Target cube name {tc_name}')

		cal_attrs = vector_query_simbad_to_ids(	cal_names, # object name list 
												[	'coordinates', # SIMBAD field list
													'propermotions', 
													'rv_value',
													'fluxdata(H)',
													'fluxdata(V)',
													'fluxdata(B)',
													'rot'
												],
												[	'FLUX_H', # SIMBAD data keyword list
													'FLUX_B', 
													'FLUX_V', 
													'RV_VALUE', 
													'RA',
													'DEC', 
													'PMRA', 
													'PMDEC',
													'ROT_mes'
												],
												[	'h_flux', #mag # keys to rename SIMBAD keywords to
													'b_flux', #mag
													'v_flux', #mag
													'radial_velocity', #km/s
													'ra', # h:m:s
													'dec', # d:m:s
													'proper_motion_ra', #mas/yr 
													'proper_motion_dec',#mas/yr
													'rotational_velocity' #km/s
												]
											)
		cal_attr_unit = {
			'h_flux':'mag',
			'b_flux':'mag',
			'v_flux':'mag',
			'radial_velocity':'km/s',
			'ra':'h:m:s',
			'dec':'d:m:s',
			'proper_motion_ra':'mas/yr',
			'proper_motion_dec':'mas/yr',
			'rotational_velocity':'km/s',
			'instrument_spec_fwhm':'um'
		}
		# prob. don't need target attributes as xtellcor_general.pro doesn't need it really
		# get a guess of FWHM of instrument wavelength response
		instrument_spec_fwhm = np.mean(np.abs(cal_wavs[0][1:] - cal_wavs[0][:-1]))

		tc_dat_file = os.path.normpath(os.path.join(tc_tellspec_folder, f'{tc_name.upper()}.dat'))
		make_xtellcor_general_spec_file(tc_dat_file, tc_wav, tc_spec)
		cal_dat_file_list = []
		for cal_name, cal_wav, cal_spec, cal_attr in zip(cal_names, cal_wavs, cal_specs, cal_attrs):
			cal_attr['instrument_spec_fwhm'] = instrument_spec_fwhm
			cal_dat_file = os.path.normpath(os.path.join(tc_tellspec_folder, f'{cal_name.upper()}.dat'))
			make_xtellcor_general_spec_file(cal_dat_file, cal_wav, cal_spec, cal_attr, cal_attr_unit)
			cal_dat_file_list.append(cal_dat_file)
		
		tc_dat_files.append(tc_dat_file)
		cal_dat_files.append(cal_dat_file_list)

	# Stuff below this line is just to help the user know what to do next
	print('')
	print(79*'#')
	next_steps = """\
	### NEXT STEPS ###
	
	Now you've grabbed the data necessary to run the IDL routine "xtellcor_general.pro", you need to:
	* Use the table below to help you get to the correct folder
	* `cd` to the folder of a given *.dat file set
	* Open IDL in that folder (command = `idl`)
	* run xtellcor_general (command = `xtellcor_general`)
	* Use the files provided in the table below find the inputs for each target-calibrator pair,
	  each calibrator *.dat file has a header with the calibrator star properties.
	  (command = `head -15 *.dat` will be helpful)
	* Save each in the format "<tc_tellspec_folder>/xtellcor_<CALIBRATOR_NAME_IN_CAPS>_adj*.dat"
	* For some reason when using the "Scale Lines" dialog, setting the velocity to 200 km/s will
	  avoid what look like red/blue shift artefacts in the lines. Not sure why as the velocities
	  involved are much smaller thatn 200 km/s.
	
	"""
	print(textwrap.dedent(next_steps))

	# each 'tc_dat_file' has multiple 'cal_dat_file' entries, split them up so they are 1:1
	tc_cal_file_table_data = [('Target *.dat file', 'Calibrator *.dat file')]
	for tc_dat_file, cal_dat_file_list in zip(tc_dat_files, cal_dat_files):
		for cal_dat_file in cal_dat_file_list:
			tc_cal_file_table_data.append((tc_dat_file, cal_dat_file))
	pretty_table.write({'Target-Calibrator Pairs':{'header':[tc_cal_file_table_data[0]], 'data':tc_cal_file_table_data[1:]}}, sys.stdout, frame=True)
	print(79*'#')

	
	return

def make_xtellcor_general_spec_file(afile, wavs, spec, attrs={}, attr_units={}):
	os.makedirs(os.path.dirname(afile), exist_ok=True)
	with open(afile, 'w') as f:
		if len(attrs) > 0:
			f.write('# OBJECT ATTRIBUTES\n')
			max_n_k = max([len(k) for k in attrs.keys()])
			kv_fmt = '# {:<'+f'{max_n_k+1}'+'} = {} ({})\n'
			for k, v in attrs.items():
				f.write(kv_fmt.format(k,v,attr_units.get(k,'')))
		f.write(f'# DATA COLUMNS\n')
		f.write(f'# wavelength flux\n')
		f.write(f'# um         relative_units\n')
		for w, s in zip(wavs, spec):
			f.write(f'{w:09.3E} {s:09.3E}\n')
	return
	

def vector_query_simbad_to_ids(objs, fields, sids, ids):
	"""
	Queries the SIMBAD archive about each object in list 'objs', and requests each field in list 'fields'.
	Then returns a dictionary populated with data from the archive, where each simbad id 'sid' is assigned
	a key in the returned dictonary. The keys are associated by position in the list 'ids'
	
	ARGUMENTS:
		objs	[n]
			<str> A list of object names understood by SIMBAD
		fields [m]
			<str> A list of field names understood by SIMBAD
		sids [l]
			<str> A list of data keywords understood by SIMBAD
		ids [l]
			<str> A list of keys to rename 'sids' to when repacked into a dictionary

	RETURNS:
		A dictionary containing the keys passed in 'ids'

	EXAMPLE:
		attribute_dict = vector_query_simbad_to_ids(['VEGA'], 
													['coordinates','fluxdata(V)'], 
													['RA','DEC','FLUX_V'], 
													['RightAscention','Declination','V_band_flux']
													)
	"""
	Simbad.add_votable_fields('typed_id')
	Simbad.remove_votable_fields('coordinates', 'main_id')
	Simbad.add_votable_fields(*fields)
	sbd_obj = Simbad.query_objects(objs)
	sbd_dict_list=[]
	for sbd in sbd_obj:
		sd = {}
		for s,i in zip(sids, ids):
			sd[i] = sbd[s]
		sbd_dict_list.append(sd)
	print(sbd_dict_list)
	return(sbd_dict_list)


def get_fits_obj_name_and_spectra(fits_file):
	"""
	Assume that the spectral dimension is the first dimension (fits usually uses zyx ordering)
	"""
	hdul = fits.open(fits_file)
	name = hdul[0].header['ESO OBS TARG NAME'].lower()
	w = fitscube.header.get_wavelength_grid(hdul[0])
	spec = np.nansum(hdul[0].data, axis=(1,2))
	return(name, w, spec)

def get_calibrator_fits_files(tc_calibs_file, calibrators_relative_fix=None, calibrators_relative_replace=None):
	calibrator_file_paths = []
	with open(tc_calibs_file, 'r') as f:
		for calib_file_path in f:
			if calibrators_relative_fix is not None:
				calibrator_file_paths.append(os.path.join(calibrators_relative_fix, os.path.relpath(calib_file_path.strip(), start=calibrators_relative_replace)))
			else:
				calibrator_file_paths.append(os.path.abspath(calib_file_path.strip()))
	return(calibrator_file_paths)

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

	parser.add_argument('target_cubes', type=str, nargs='+', help='Science observations to create "xtellcor_general.pro" inputs for')
	parser.add_argument('--rel_calibrators_file', type=str, help='A file (relative to target cubes) containing the calibrators that we should create "xtellcor_general.pro" inputs for.', default='./calibrators.txt')
	parser.add_argument('--rel_tellspec_folder', type=str, help='A folder (relative to target cubes) to store the "xtellcor_general.pro" input files in (and the routines output files)', default='./analysis/telluric_correction')
	parser.add_argument('--calibrators_relative_fix', type=str, nargs='?', help='If present will try to intelligently guess the parent folder of the calibrators, pass a path to overwrite the guessed value.', default=None, const=os.path.expanduser('~/scratch/reduced_images'))
	parser.add_argument('--calibrators_relative_replace', type=str, help='Section of calibrators path to overwrite with "--calibrators_relative_fix"', default='/network/group/aopp/planetary/PGJI002_IRWIN_TELESCOP/dobinson/sinfoni/esorex_test/neptune_4')
	parser.add_argument('--require_relative', nargs='+', type=str, help='If present will only return target cubes that have the required set of files present relative to them', default=[])

	parsed_args = vars(parser.parse_args(argv)) # I prefer a dictionary interface
	
	filtered_tcs = []
	for tc in parsed_args['target_cubes']:
		reqs_exist = [os.path.exists(os.path.join(os.path.dirname(tc), rel_file)) for rel_file in parsed_args['require_relative']]
		if all(reqs_exist):
			filtered_tcs.append(os.path.normpath(tc))
	parsed_args['target_cubes'] = filtered_tcs

	return(parsed_args)

if __name__=='__main__':
	main(sys.argv[1:])
