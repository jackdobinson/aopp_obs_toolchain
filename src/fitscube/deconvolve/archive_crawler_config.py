#!/usr/bin/env python3
import utilities.logging_setup
logging, _lgr = utilities.logging_setup.getLoggers(__name__, 'DEBUG')

import os, sys
import numpy as np
import re
import pexpect, shutil, glob
import datetime
import typing
from astropy.io import fits
import glob

import utilities as ut
import utilities.fits

"""
Configuration data for "archive_crawler.py"

The classes in this file describe how a certain type of node in an archive is configured. There should always be a class method
that can tell you if the config is valid for any node. It's called "<class>.is_config_valid()" in this case.

I'm including extension indices/names for data and information for science and standard star FITS files in the classes
because they are not indexed correctly in many AAREADME.txt files (lots of off-by-one errors).

Should only ever have class methods and class attributes. They should hold STATIC information and functions.

"""


########################### Global Variables ##################################

TEMPORARY_WORKING_DIRECTORY = os.path.join(os.getenv('HOME'), "scratch/archive_crawler_tmp")


####################### ArchiveNodeConfig Classes #############################


class ArchiveNodeConfigBase:
	# CLASS ATTRIBUTES
	DO_NOT_TRAVERSE_FILE : \
		str = '.do_not_crawl'					# If a file with this name is present in the archive node, the config will be invalid
												# I.e. have a file with this name in each folder you want to ignore.
	DO_NOT_INCLUDE : bool = True				# should this config not be included in the available configs for the archive?
	SCI_INFO_EXT : typing.Union[int,str] = 0	# FITS extension name or index that holds infomation about science observation
	STD_INFO_EXT : typing.Union[int,str] = 0	# FITS extension name or index that holds infomation about a standard star observation
	SCI_DATA_EXT : typing.Union[int,str] = 0	# FITS extension name or index that holds data from a science observation
	STD_DATA_EXT : typing.Union[int,str] = 0	# FITS extension name or index that holds data from a standard star observation
	SCI_SPECTRAL_AX_IDX : int = 0				# Axis index (numpy) for the spectral information of a science observation
	STD_SPECTRAL_AX_IDX : int = 0				# Axis index (numpy) for the spectral information of a standard star observation
	HDR_OBJECT_KEYS : typing.Union[str,tuple[str]] = 'OBJECT'	# FITS header keys that holds the name of the subject of the observation 
																# if a tuple, more than one key holds relevant information
	HDR_DATE_OBS_KEYS : typing.Union[str,tuple[str]] = 'DATE-OBS'	# FITS header keys that holds the date of the observation
																	# if a tuple, more than one key holds relelvant information
	
	
	# STATIC METHODS BELOW HERE
	@staticmethod
	def get_fits_hkeys_concat(hdr : fits.Header, fhkeys : typing.Union[str,tuple[str]], sep_str=' '):
		return(sep_str.join((hdr[x] for x in ((fhkeys,) if type(fhkeys) is str else fhkeys))))

	@staticmethod
	def get_fits_datetime(fpath : str, fext : typing.Union[int,str], fhkeys : typing.Union[str,tuple[str]]):
		hdr = fits.getheader(fpath, fext)
		return(iso2datetime(ArchiveNodeConfigBase.get_fits_hkeys_concat(hdr,fhkeys)))

	@staticmethod
	def _get_spec_grid(fpath : str, ext : typing.Union[int,str], spec_ax :typing.Union[int,tuple[int]]) -> np.ndarray:
		_lgr.INFO(f'Looking for spec_grid for file {fpath}\nabsolute path {os.path.abspath(fpath)}')
		hdr = fits.getheader(fpath, ext)
		# default units is Angstrom for some reason
		return(ut.fits.hdr_get_axis_world_coords(hdr, spec_ax)[0]*ut.fits.hdr_get_unit_si(hdr, spec_ax, default=1E-10))
	

	# CLASS METHODS BELOW HERE
	@classmethod
	def is_path_accessible(cls, archive_path:str) -> bool:
		# default to only be accessible unless otherwise stated
		if os.path.exists(os.path.join(archive_path, cls.DO_NOT_TRAVERSE_FILE)):
			_lgr.INFO(''.join((
				f"Found file '{cls.DO_NOT_TRAVERSE_FILE}' in directory '{archive_path}', ",
				f"therefore this directory and it's children are not accessible by a crawler ",
				f"with this config.",
			)))
			return(False)
		return(True)


	@classmethod
	def is_config_valid(cls, archive_path : str) -> bool:
		# assume config is valid for everything
		return(True)
	
	@classmethod
	def get_sci_std_filename(cls, fpath : str, tag : str ='_standard_star') -> str:
		"""Get name of STD file associated with SCI file"""
		sdir, sfile = os.path.split(os.path.realpath(os.path.expanduser(fpath)))
		sfname, sfext = os.path.splitext(sfile)
		sfilepath = os.path.join(sdir, f'{sfname}{tag}{sfext}')
		return(sfilepath)
	
	@classmethod
	def get_sci_datetime(cls, fpath : str) -> datetime.datetime:
		# default to assuming HDR_DATE_OBS_KEYS give a time in ISO format
		return(cls.get_fits_datetime(fpath, cls.SCI_INFO_EXT, cls.HDR_DATE_OBS_KEYS))
	
	@classmethod
	def get_std_datetime(cls, fpath : str) -> datetime.datetime:
		# default to assuming HDR_DATE_OBS_KEYS give a time in ISO format
		return(cls.get_fits_datetime(fpath, cls.STD_INFO_EXT, cls.HDR_DATE_OBS_KEYS))
	
	@classmethod
	def is_std(cls, fpath : str) -> bool:
		# default to assuming nothing is a standard star
		return(False)
	
	@classmethod
	def select_sci(cls, fpath : str) -> bool:
		"""
		should we select this science file as one to operate on
		"""
		# default to always operating on a science file
		return(True)
	
	@classmethod
	def get_sci_spec_set(cls, fpath : str) -> typing.Union[str, np.array]:
		"""Gets the filter name or set of wavelengths used by a science observation"""
		# return 'FILTER NAME' so everything matches
		return('FILTER NAME')
	
	@classmethod
	def get_std_spec_set(cls, fpath : str) -> typing.Union[str, np.array]:
		"""Gets the filter name or set of wavelengths used by a standard star observation"""
		# return 'FILTER NAME' so everything matches
		return('FILTER NAME')
	
	@classmethod
	def get_std_spec_grid(cls, fpath:str) -> np.ndarray:
		return(cls._get_spec_grid(fpath, cls.STD_DATA_EXT, cls.STD_SPECTRAL_AX_IDX))

	@classmethod
	def get_sci_spec_grid(cls, fpath:str)->np.ndarray:
		return(cls._get_spec_grid(fpath, cls.SCI_DATA_EXT, cls.SCI_SPECTRAL_AX_IDX))

	@classmethod
	def get_std_folders(cls, odirs : list[str]) -> list[str]:
		# default to not finding any standard star folders
		return([])
	
	@classmethod
	def create_std_cubes(cls, odirs : list[str]) -> list[str]:
		# default to not getting (i.e. making) any standard star cubes
		return([])

	@classmethod
	def sci_header_fix(cls, fpath : str) -> None:
		# default to no adjustment of science file header
		return


class ArchiveNodeConfig_AAREADME_Required(ArchiveNodeConfigBase):
	DO_NOT_INCLUDE = True # technically don't need to specify, but good for readability
	
	@classmethod
	def is_config_valid(cls, archive_path):
		return(AAREADME_is_present(archive_path) and super().is_config_valid(archive_path))


class Gemini_NIFS_ArchiveConfig(ArchiveNodeConfig_AAREADME_Required):
	DO_NOT_INCLUDE = False
	SCI_DATA_EXT = 1
	STD_DATA_EXT = 1
	HDR_DATE_OBS_KEYS = ('DATE-OBS', 'UT') # concatenate these keys to get full date+time
	
	@classmethod
	def is_config_valid(cls, archive_path):
		return(("Gemini_NIFS" in archive_path) and super().is_config_valid(archive_path))
	
	"""
	@classmethod
	def select_sci(cls, fpath):
		return('smooth' in fpath) # prefer to operate on smoothed cubes
	"""

	@classmethod
	def get_sci_datetime(cls, fpath):
		with fits.open(fpath) as hdul:
			obs_date = hdul[cls.SCI_INFO_EXT].header[cls.HDR_DATE_OBS_KEYS[0]]
			obs_time = hdul[cls.SCI_INFO_EXT].header[cls.HDR_DATE_OBS_KEYS[1]]
		return(iso2datetime(' '.join([obs_date, obs_time])))
	
	@classmethod
	def get_std_datetime(cls, fpath):
		with fits.open(fpath) as hdul:
			obs_date = hdul[cls.STD_INFO_EXT].header[cls.HDR_DATE_OBS_KEYS[0]]
			obs_time = hdul[cls.STD_INFO_EXT].header[cls.HDR_DATE_OBS_KEYS[1]]
		return(iso2datetime(' '.join([obs_date, obs_time])))
	
	@classmethod
	def get_sci_spec_set(cls, fpath : str) -> typing.Union[str, np.array]:
		"""Gets the filter name or set of wavelengths used by a science observation"""
		header_keys = ('FILTER','GRATING','OBSMODE')
		with fits.open(fpath) as hdul:
			spec_state = ' '.join([hdul[cls.SCI_INFO_EXT].header[hk] for hk in header_keys])
		return(spec_state)
	
	@classmethod
	def get_std_spec_set(cls, fpath : str):
		"""Gets the filter name or set of wavelengths used by a standard star observation"""
		header_keys = ('FILTER','GRATING','OBSMODE')
		with fits.open(fpath) as hdul:
			spec_state = ' '.join([hdul[cls.STD_INFO_EXT].header[hk] for hk in header_keys])
		return(spec_state)
	
	@classmethod
	def is_std(cls, fpath):
		return(os.path.basename(fpath).startswith('ctfbrsn'))
	
	@classmethod
	def get_std_folders(cls, odirs):
		return(odirs)
	
	@classmethod
	def sci_header_fix(cls, fpath):
		_lgr.DEBUG(f'Applying fixes to header of file {os.path.abspath(fpath)}')
		dirty_flag = False
		with fits.open(os.path.abspath(fpath), mode='update') as hdul:
			if hdul[cls.SCI_DATA_EXT].header['CRVAL3'] < 1000:
				if 'CUNIT3' not in hdul[cls.SCI_DATA_EXT].header:
					hdul[cls.SCI_DATA_EXT].header['CUNIT3'] = 'um'
					dirty_flag = True
			else:
				if 'CUNIT3' not in hdul[cls.SCI_DATA_EXT].header:
					hdul[cls.SCI_DATA_EXT].header['CUNIT3'] = 'Angstrom '
			hdul.flush()


	@classmethod
	def create_std_cubes(cls, odirs, logfile=sys.stdout):
		"""
		Runs routines to construct standard stars from telluric observations
		
		# ARGUMENTS #:
			original_dir
				The original directory that the observations in the archive are taken from, is specified in the "AAREADME" file
		
		# RETURNS #:
			std_files
				A lists of paths to standard start cubes. If the list is empty, could not create any standard star files.
		"""
		return_cubes = []
		for original_dir in odirs:
			# there should be a set of "irwin_science.cl", "irwin_telluric.cl" and "irwin_basecal.cl" in the parent directory
			ddir = os.path.normpath(os.path.join(original_dir, '../')) # data directory
			# TODO:
			# * Compare the file "irwin_science.cl" with "irwin_telluric.cl" to see the difference between
			#   creating a cube and creating a 1d spectrum. I think I just have to swap out the last command
			#   in the "irwin_telluric.cl" file with a "nifcube()" command, but make sure.
			# * Run a test with the newly installed IRAF version (remember stuff with pyenv) to see if I can
			#   make a cube manually. If I can, then write a programmatic version for this script to run.
			# * Remember to get this function to return the filename of the standard star, I need it for
			#   deconvolution.
			_lgr.DEBUG(f'{ddir=}')
			_lgr.DEBUG(f"{TEMPORARY_WORKING_DIRECTORY=}")
			
			
			if True: # DEBUGGING, don't want to copy the archive data every time during testing...
				_lgr.INFO(f'Copying archive data at {ddir} to temporary storage at {TEMPORARY_WORKING_DIRECTORY}')
				if os.path.exists(TEMPORARY_WORKING_DIRECTORY):
					shutil.rmtree(TEMPORARY_WORKING_DIRECTORY)
				
				# copy archived data to temporary directory
				shutil.copytree(
					ddir, 
					TEMPORARY_WORKING_DIRECTORY, 
					dirs_exist_ok=False, # we should have just deleted the temporary directory if it was there
					ignore = lambda adir, contents: [																			\
						c for c in contents 																					\
						if ( 																									\
							# Ignore "origina_dir" as it only has IRAF output in it 											\
							(os.path.normpath(os.path.join(adir,c))==os.path.normpath(original_dir)) 							\
							# Ignore files like "*.save" as they are probably read protected and are just for backup anyway		\
							or c.endswith('.save')																				\
						)																										\
					],
				) 
			
			# go to temporary directory
			os.chdir(TEMPORARY_WORKING_DIRECTORY)
			
			# copy file that calculates telluric spectrum to a new file we will modify
			telluric_spec_cl = os.path.join(TEMPORARY_WORKING_DIRECTORY, 'irwin_telluric.cl')
			telluric_cube_cl = os.path.join(TEMPORARY_WORKING_DIRECTORY, 'irwin_telluric_cube.cl')
			shutil.copy(telluric_spec_cl, telluric_cube_cl)
			
			# Pat copied these files from a different filesystem where the data is at the location "/Users/irwin/telescope_data/..."
			# the archive on the shared storage has a different prefix of "/network/group/aopp/planetary/PGJI002_IRWIN_TELESCOP/telescope_data/..."
			# therefore, we need to change the paths that are written in the file to point to the current path of the archive
			# then we need to change all the paths that point to the current path of the archive to point to the temporary directory
			
			mutated_paths = mutate_paths_in_file(telluric_cube_cl, mutating_path=ddir)
			
			# paths in file now point to the *archive* location, now we just need to swap them for the path to the temp directory
			replace_in_file(telluric_cube_cl, [(ddir, TEMPORARY_WORKING_DIRECTORY) for mp in mutated_paths])
			
			# We also have to swap out the variable name "fl_nscut" for "fl_cut". I'm not sure why, but
			# the version of IRAF and Gemini tools I have seem to use a different parameter name
			replace_in_file(telluric_cube_cl, [('fl_nscut','fl_cut')])
			
			# Now I need to comment out the "nfextract" and "gemcombine" commands and add in a "nscube" command
			cls._comment_and_add_cube_command(telluric_cube_cl)
			
			# Now I should be ready to run IRAF
			std_cubes = cls._run_iraf(logfile=logfile) # Don't run this until ready for testing
			
			_lgr.DEBUG(f'{std_cubes=}')
			
			_lgr.INFO(f'Copying created standard star files from {TEMPORARY_WORKING_DIRECTORY} to {original_dir}')
			
			for afile in std_cubes:
				dest_file = os.path.basename(afile)
				dest_file, dest_ext = os.path.splitext(dest_file)
				dest_file = os.path.normpath(os.path.join(original_dir, f'{dest_file}_standard_star{dest_ext}'))
				
				try:
					shutil.move(afile, dest_file, copy_function=shutil.copy) # don't care about copying metadata
					return_cubes.append(dest_file)
				except:
					_lgr.ERROR(f'Could not move standard star *.fits file "{afile}" to "{dest_file}"')
					pass
		
		#sys.exit(f'EXIT FOR DEBUGGING at line {sys._getframe().f_lineno}')
		return(return_cubes)
	
	@classmethod
	def _run_iraf(cls, logfile=None):
		_lgr.INFO(f'Running pyRAF to reduce dataset, logging output to {logfile}')	
		# set timeout really high later on when we run the command (maybe around 2 hours?)
		child = pexpect.spawn("/bin/bash", timeout=60, encoding='utf-8', echo=True)
		child.delaybeforesend = 1
		
		# log input and output, but don't log input if we're echoing to screen
		if logfile is not None:
			child.logfile_read = logfile
		
		# wait for prompt to appear first time.
		child.expect(r' $')
		
		# set up a minimal prompt and don't let pyenv change the prompt
		child.sendline(r'PS1="\[\033[01;32m\]\u@\h $\[\033[00m\] "')
		child.expect(r'\r\n.*$')
		child.sendline('export PYENV_VIRTUALENV_DISABLE_PROMPT=1')
		child.expect(r'$')
	
		# move into temporary directory
		child.sendline(f'cd {TEMPORARY_WORKING_DIRECTORY}')
		status = child.expect([r' $', r'cd: .* No such file or directory'])
		if status == 1:
			child.close(force=True)
			raise RuntimeError(f'Could not change directory to {TEMPORARY_WORKING_DIRECTORY} before starting IRAF (via PYRAF), is directory accessible?')

		_lgr.INFO('Checking for "RUN_PYRAF_SCRIPT" environment variable...')
		run_pyraf_script = os.environ.get('RUN_PYRAF_SCRIPT', None)
		_lgr.INFO(f'$RUN_PYRAF_SCRIPT = {run_pyraf_script}')
		if run_pyraf_script is not None:
			_lgr.INFO('Starting pyraf using $RUN_PYRAF_SCRIPT')
			child.sendline(f"{os.path.abspath(run_pyraf_script)}")
		else:
			# activate the correct pyenv environment, 
			_lgr.INFO('Starting pyraf using "pyenv activate miniconda-latest/envs/geminiconda2"')
			child.sendline('pyenv activate miniconda3-latest/envs/geminiconda2')
			status = child.expect([r' $', r'\r\ncommand \s*? not found, did you mean:'])
			if status == 1:
				child.close(force=True)
				raise RuntimeError('Could not activate correct python environment for IRAF')
			
			# start IRAF via PYRAF
			child.sendline('pyenv exec pyraf')
	
	
		# the control code "\x1b[?2004h" turns on "bracketed paste mode", 
		# text is not treated as commands. This is usually turned on whenever
		# there is user interaction expected (i.e. at a prompt)
		child.expect(r'\r\n\x1b\[\?2004h--> ') 		


		# send commands to IRAF, set timeout (in seconds) high to let command complete
		child.sendline('cl < irwin_telluric_cube.cl')
		status = child.expect([r'\r\n\x1b\[\?2004h--> ','ERROR'], timeout=60*60*3)
		if status == 1:
			child.close(force=True)
			raise RuntimeError(f'Telluric cube creation did not complete successfully')
		
		# exit iraf
		child.sendline('.exit')
		child.expect('$')
		
		# exit bash
		child.sendline('exit')
		child.expect(pexpect.EOF)
		child.close(force=True)
		
		
		# Alternatively, just try returning file names matching the glob "ctfbrsn*.fits"
		return(glob.glob(os.path.join(TEMPORARY_WORKING_DIRECTORY,'ctfbrsn*.fits')))
	
	@classmethod
	def _comment_and_add_cube_command(cls, afile):
		"""
		Comment out "nfextract" and "gemcombine" commands in ".../irwin_telluric_cube.cl" files and add
		a "nifcube" command.
		"""
		new_lineset = []
		command_found = None
		with open(afile) as f:
			for aline in f:
				# if we've encountered an empty line or a comment line, we are not in a command
				if aline.strip() == '' or aline.strip()[0]=='#':
					if command_found == 'gemcombine':
						# if we are after this command, we want to add a new command to the file
						new_lineset.append('\nnifcube("tfbrsn@telluriclist", logfile="nifs.log")\n')
					command_found=None

				if '(' in aline:
					if '=' in aline and aline.index('=') < aline.index('('):
						command_found = None
					else:
						command_found = aline.strip().split('(',1)[0]
				
				# look for the command "nfextract" or 'gemcombine' and comment them out
				if command_found in ('nfextract', 'gemcombine'):
					aline = '#'+aline
				new_lineset.append(aline)

		with open(afile, 'w') as f:
			f.write(''.join(new_lineset))
		return


class MUSE_ArchiveConfig(ArchiveNodeConfig_AAREADME_Required):
	DO_NOT_INCLUDE = False
	SCI_DATA_EXT = 1
	STD_DATA_EXT = 1
	HDR_OBJECT_KEYS = 'HIERARCH ESO OBS NAME'

	@classmethod
	def is_config_valid(cls, archive_path):
		return(("VLT_MUSE" in archive_path) and super().is_config_valid(archive_path))

	@classmethod
	def is_std(cls, fpath):
		return(os.path.basename(fpath) == 'DATACUBE_STD_0001.fits')

	@classmethod
	def get_std_folders(cls, odirs):
		std_folders = []
		for odir in odirs:
			std_folders += glob.glob(os.path.join(odir,"../*/data_with_raw_calibs/calib/std/std_products*/"))
		return(std_folders)

	@classmethod
	def _get_hdr_spec_set(cls, hdr :fits.Header):
		# see ftp://ftp.eso.org/pub/dfs/pipelines/instruments/muse/muse-pipeline-manual-2.8.3.pdf for useful header keys
		spec_hkeys = ('HIERARCH ESO INS MODE', 'HIERARCH ESO INS OPTI1 NAME', 'HIERARCH ESO INS OPTI2 NAME')
		return(cls.get_fits_hkeys_concat(hdr, spec_hkeys))
	
	@classmethod
	def get_sci_spec_set(cls, fpath):
		return(cls._get_hdr_spec_set(fits.getheader(fpath, cls.SCI_INFO_EXT)))
	
	@classmethod
	def get_std_spec_set(cls, fpath):
		return(cls._get_hdr_spec_set(fits.getheader(fpath, cls.STD_INFO_EXT)))

	@classmethod
	def sci_header_fix(cls, fpath):
		data_hkey_adjust_dict = {
			'CUNIT3' : 'um'
		}
		info_hkey_adjust_dict = {
		}
		_lgr.DEBUG(f'Scanning if fixes to header of file {fpath} are needed')
		dirty_flag = False
		with fits.open(fpath, mode='update') as hdul:
			for k, v in data_hkey_adjust_dict.items():
				v_old = hdul[cls.SCI_DATA_EXT].header.get(k, None)
				if v_old != v:
					hdul[cls.SCI_DATA_EXT].header[k] = v
					dirty_flag = True
			for k, v in info_hkey_adjust_dict.items():
				v_old = hdul[cls.SCI_INFO_EXT].header.get(k, None)
				if v_old != v:
					hdul[cls.SCI_INFO_EXT].header[k] = v
					dirty_flag = True
			if dirty_flag:
				_lgr.DEBUG('Some inconsistencies found, applying fixes...')
				hdul.flush()
				dirty_flag = False


	


class SINFONI_ArchiveConfig(ArchiveNodeConfig_AAREADME_Required):
	DO_NOT_INCLUDE = True
	HDR_OBJECT_KEYS = 'HIERARCH ESO OBS TARG NAME'
	HDR_DATE_OBS_KEYS = 'DATE-OBS'
	
	@classmethod
	def get_sci_datetime(cls, fpath):
		with fits.open(fpath) as hdul:
			obs_date = hdul[cls.SCI_INFO_EXT].header[cls.HDR_DATE_OBS_KEYS]
		return(iso2datetime())
	
	@classmethod
	def get_std_datetime(cls, fpath):
		with fits.open(fpath) as hdul:
			obs_date = hdul[cls.STD_INFO_EXT].header[cls.HDR_DATE_OBS_KEYS]
		return(iso2datetime())
	
	@classmethod
	def get_sci_spec_set(cls, fpath : str) -> typing.Union[str, np.array]:
		"""Gets the filter name or set of wavelengths used by a science observation"""
		header_keys = ('HIERARCH ESO INS FILT1 NAME', 'HIERARCH ESO INS GRAT1 NAME')
		with fits.open(fpath) as hdul:
			spec_state = ' '.join([hdul[cls.SCI_INFO_EXT].header[hk] for hk in header_keys])
		return(spec_state)
	
	@classmethod
	def get_std_spec_set(cls, fpath : str) -> typing.Union[str, np.array]:
		"""Gets the filter name or set of wavelengths used by a standard star observation"""
		header_keys = ('HIERARCH ESO INS FILT1 NAME', 'HIERARCH ESO INS GRAT1 NAME')
		with fits.open(fpath) as hdul:
			spec_state = ' '.join([hdul[cls.STD_INFO_EXT].header[hk] for hk in header_keys])
		return(spec_state)


class SINFONI_2014_ArchiveConfig(SINFONI_ArchiveConfig):
	DO_NOT_INCLUDE = False,
	
	@classmethod
	def is_config_valid(cls, archive_path):
		return(("VLT_SINFONI/2014" in archive_path) and super().is_config_valid(archive_path))
	
	@classmethod
	def is_std(cls, fpath):
		return(fpath.endswith('out_cube_obj00_0000.fits'))
	
	@classmethod
	def get_std_folders(cls, odirs):
		std_folders = []
		for odir in odirs:
			for x in os.listdir(os.path.join(odir, '../')):
				if x.startswith('standard'):
					std_folders.append(os.path.join(odir, '../',x))
		return(std_folders)


class SINFONI_2013_ArchiveConfig(SINFONI_ArchiveConfig):
	DO_NOT_INCLUDE = False
	
	@classmethod
	def is_config_valid(cls, archive_path):
		return(('VLT_SINFONI/2013' in archive_path) and super().is_config_valid())
	
	@classmethod
	def is_std(cls, fpath):
		return(re.search(r".*/\d+\.STD\.[HJK].*.fits",fpath))
	
	@classmethod
	def get_std_folders(cls, odirs):
		return(odirs)


class HST_STIS_ArchiveConfig(ArchiveNodeConfigBase):
	DO_NOT_INCLUDE = True
	SCI_DATA_EXT = 1
	STD_DATA_EXT = 1
	
	def is_config_valid(cls, archive_path):
		return(('HST_STIS' in archive_path) and super().is_config_valid())


########################### Helper Functions ##################################


def iso2datetime(iso_str):
	time_factors = {
		0 : 12, # years to months
		1 : {	# months to days
			0 : 31,
			1 : 28,
			2 : 31,
			3 : 30,
			4 : 31,
			5 : 30,
			6 : 31,
			7 : 31,
			8 : 30,
			9 : 30,
			10 : 31,
			11 : 31,
		},
		2 : 24, # days to hours
		3 : 60, # hours to minutes
		4 : 60, # minutes to seconds
		5 : 10000000, # seconds to microseconds
		6 : 0, # dummy, smaller than us not recorded
	}
	time_strs = re.split(r'[-:T .]', iso_str)
	time_numbers = [0]*7
	for i, ts in enumerate(time_strs):
		try:
			time_numbers[i] += int(ts)
		except ValueError:
			# must be a float so send fractions down the line
			tf = float(ts)
			j=i
			while j < 7:
				ti = int(tf//1)
				tf = (tf % 1) * (
					time_factors[j] if type(time_factors[j]) is not dict 
						else time_factors[j][ti] + (1 if (time_numbers[j-1]%4==0) else 0)
				)

				time_numbers[j] += ti
	return(datetime.datetime(*time_numbers,tzinfo=datetime.timezone.utc))


def AAREADME_is_present(archive_path):
	return(os.path.exists(os.path.join(archive_path, 'AAREADME.txt')))


def replace_in_file(afile, find_and_replacements):
	with open(afile) as f:
		astr = f.read()
	for find, replace in find_and_replacements:
		astr = astr.replace(find, replace)
	with open(afile,'w') as f:
		f.write(astr)
	return


def mutate_paths_in_file(afile, mutating_path):
	#path_pattern = re.compile(r'\.{0,2}(/[\w .-]+/??)+') # matches too much stuff
	path_pattern = re.compile(r'"(/[\w .-]+/??)+"') # thankfully I know that the paths I am interested in are absolute and quoted with the " character
	mutated_paths = []
	with open(afile) as f:
		astr = f.read()
		matches = tuple(path_pattern.finditer(astr))
		for match in matches:
			_lgr.DEBUG(match.group())
			_lgr.DEBUG(mutating_path)
			#i, j = match.span()
			#_lgr.DEBUG(f"{astr[i-20:i]}\u001b[41m{astr[i:j]}\u001b[0m{astr[j:j+20]}")
			mutated_path = '"'+path_mutate_by(match.group()[1:-1], mutating_path)+'"' # remember to add in quotes that surround original path
			mutated_paths.append(mutated_path)
			astr = astr.replace(match.group(), mutated_path)
	with open(afile,'w') as f:
		f.write(astr)
	return(mutated_paths)


def path_mutate_by(path, mutating_path):
	# strip of os.sep from either end as it interferes with matching
	sp = tuple(path.strip(os.sep).split(os.sep))
	smp = tuple(mutating_path.strip(os.sep).split(os.sep))
	spr = tuple(reversed(sp))
	smpr = tuple(reversed(smp))

	_lgr.DEBUG(sp)
	_lgr.DEBUG(smp)

	m_tail_idx = -1
	p_tail_idx = -1
	for i, m in enumerate(smpr):
		if m in sp:
			j = spr.index(m)
			_lgr.DEBUG(smpr[:i+1])
			_lgr.DEBUG(spr[j-i:j+1])
			if smpr[:i+1]==spr[j-i:j+1] and j+1 > p_tail_idx:
				p_tail_idx = j+1
				m_tail_idx = i+1
	_lgr.DEBUG(f"{m_tail_idx=} {p_tail_idx=}")
	i, j = len(smp)-m_tail_idx, len(sp)-p_tail_idx # undo reverse direction
	_lgr.DEBUG(f"{i=} {j=} {smp[i]=} {sp[j]=}")
	mutated_path = (os.sep if mutating_path[0]==os.sep else '')+os.path.join(*smp[:i],*sp[j:]) + (os.sep if path[-1]==os.sep else '')
	_lgr.DEBUG(mutated_path)
	return(mutated_path)


################################## NOTES ######################################
"""
# DETAILS ABOUT ARCHIVE #

Want NIFS, MUSE, and SINFONI datasets deconvolved, therefore need to get STD_STAR 
observations for them set up correctly. 

For NIFS, in the directory that contains the original file (odir) I want to
make a cube of the TELLURIC measurements. There are scripts in "odir/.." called
"irwin_science.cl" and "irwin_telluric.cl" that have the names of the raw files
for the telluric calibrator, and the commands on how to turn raw files into a
datacube. I want to read those files, and run IRAF to turn the telluric
callibrator into a cube. Then use that cube as my standard star. I have to
use IRAF to perform the data reduction (see below).

For MUSE, there is generally a folder called "odir/../cal/std/" that contains
cubes for standard stars. Try using those ones first.

For SINFONI, I will have to find which observations are standard stars and
choose the standard stars that best fit the observations.


# IRAF #

Iraf is depreciated software at this point, I cannot just install it normally. I
had to use "astroconda" inside "pyenv" to install it. See https://www.gemini.edu/observing/phase-iii/understanding-and-processing-data/data-processing-software/download-latest

Installation commands:
	# install miniconda as a pyenv python version
	$ pyenv install miniconda3-latest

	# activate the environment
	$ pyenv activate miniconda3-latest

	# set up conda and add channels for astroconda
	$ conda config --set auto_activate_base false
	$ conda config --add channels http://ssb.stsci.edu/astroconda
	$ conda config --add channels http://astroconda.gemini.edu/public

	# create a new virtual environment (with conda, but pyenv can talk to conda)
	# to hold gemini, stsci, iraf, pyraf packages. Call it "geminiconda2"
	# Note that the environment uses python2.7 as STSCI have depreciated IRAF
	# and do not support it any more.
	$ conda create -n geminiconda2 python=2.7 gemini stsci iraf-all pyraf-all

	# deactivate the "base" miniconda environment
	$ pyenv deactivate

	# activate the new miniconda environment where iraf etc. is installed
	$ pyenv activate miniconda3-latest/env/geminiconda2

	# Do any IRAF stuff as normal, however you need to prepend "pyenv exec" to
	# get ahold of any IRAF commands. The IRAF binaries are stored in the 
	# virtual environment, so you have to ask pyenv to get them for you.
	$ cd ~
	$ mkdir iraf
	$ pyenv exec mkiraf

If you need the location of the IRAF binaries, you can use the "pyenv prefix" command like so:
	# Be in the geminiconda2 (IRAF) environment
	$ pyenv activate miniconda3-latest/env/geminiconda2
	
	# _lgr.DEBUG the prefix, or store it somewhere
	$ PYENV_PREFIX=$(pyenv prefix)

	# The actual binary files are in the ./bin folder of the prefix
	$ $PYENV_PREFIX/bin/<executable>

	# Remember to deactivate the environment when you're done
	$ pyenv deactivate

You cannot use IRAF commands when not in the .../geminiconda2 environment, lots of
setup gets done in the background. The "pyraf" environment is probably the most 
useful. It can understand all the same things as normal IRAF "ecl", but has python
functionality.

IRAF and pyRAF have trouble reading files from the automouted directories at 
"/network/group/aopp/planetary/...". Therefore I will have to copy the relevant NIFS
observations to my a temporary space (probably "$HOME/scratch/nifs_temp/"), perform
the construction of the datacube for the telluric observations (probably using the
"pexpect" module to interface with pyRAF), copy the cubes back to the archive, and
then delete the temporary files (either after the cube creation, or before the next
cube creation). I should also log all the commands and output I get from pyRAF and
copy that log to the archive also. I will also have to update the file paths in
and *.cl files I make/modify to point to the new directory as opposed to their
original location in the archive.
"""

