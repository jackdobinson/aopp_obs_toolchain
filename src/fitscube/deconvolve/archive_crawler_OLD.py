#!/usr/bin/env python3
import utilities.logging_setup
logging, _lgr = utilities.logging_setup.getLoggers(__name__, 'DEBUG')

import os, sys
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import re
import itertools
import pexpect, shutil, glob
import datetime
from astropy.wcs import WCS

import utilities as ut
import utilities.fits
import utilities.np

"""
A program that will crawl through the telescope data archive at a given root directory.
Should emit fits file paths and relevent data extension indices for observations and their PSFs. 

# DETAILS ABOUT ARCHIVE #

Pat want's NIFS and MUSE datasets deconvolved, therefore need to get STD_STAR 
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
###############################################################################
############################ CONFIGURATION ####################################
###############################################################################
import typing
import types
import dataclasses as dc

########################### Global Variables ##################################

TEMPORARY_WORKING_DIRECTORY = os.path.join(os.getenv('HOME'), "scratch/archive_crawler_tmp")

####################### ArchiveNodeConfig Classes #############################
"""
These classes describe how a certain type of node in an archive is configured. There should always be a class method
that can tell you if the config is valid for any node. It's called "<class>.is_config_valid()" in this case.

I'm including extension indices/names for data and information for science and standard star FITS files in the classes
because they are not indexed correctly in many AAREADME.txt files (lots of off-by-one errors).

Should only ever have class methods and class attributes. They should hold STATIC information and functions.
"""
class ArchiveNodeConfigBase:
	# CLASS ATTRIBUTES
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
	
	
	# CLASS METHODS BELOW HERE
	@classmethod
	def is_config_valid(cls, archive_path : str) -> bool:
		# default to True -> valid for any archive node
		return(True)
	
	@classmethod
	def get_sci_std_filename(cls, fpath, tag='_standard_star'):
		"""Get name of STD file associated with SCI file"""
		sfile = os.path.basename(fpath)
		sfname, sfext = os.path.splitext(sfile)
		sfilepath = os.path.normpath(os.path.join(os.path.dirname(fpath), f'{sfname}{tag}{sfext}'))
		return(sfilepath)
	
	
	@classmethod
	def get_sci_datetime(cls, fpath : str) -> datetime.datetime:
		# default to assuming HDR_DATE_OBS_KEYS give a time in ISO format
		hdr = fits.getheader(fpath, cls.SCI_INFO_EXT)
		return(iso2datetime(' '.join([hdr[x] for x in ((cls.HDR_DATE_OBS_KEYS,) if type(cls.HDR_DATE_OBS_KEYS) is str else cls.HDR_DATE_OBS_KEYS)])))
	
	@classmethod
	def get_std_datetime(cls, fpath : str) -> datetime.datetime:
		# default to assuming HDR_DATE_OBS_KEYS give a time in ISO format
		hdr = fits.getheader(fpath, cls.STD_INFO_EXT)
		return(iso2datetime(' '.join([hdr[x] for x in ((cls.HDR_DATE_OBS_KEYS,) if type(cls.HDR_DATE_OBS_KEYS) is str else cls.HDR_DATE_OBS_KEYS)])))
	
	@classmethod
	def is_std(cls, fpath : str) -> bool:
		# default to assuming nothing is a standard star
		return(False)
	
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
	def get_std_folders(cls, odirs : list[str]) -> list[str]:
		# default to not finding any standard star folders
		return([])
	
	@classmethod
	def create_std_cubes(cls, odirs : list[str]) -> list[str]:
		# default to not getting (i.e. making) any standard star cubes
		return([])


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
		_lgr.INFO(f'Running pyRAF to reduce dataset, logging outpu to {logfile}')	
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
		
		# activate the correct pyenv environment, 
		child.sendline('pyenv activate miniconda3-latest/envs/geminiconda2')
		status = child.expect([r' $', r'\r\ncommand \s*? not found, did you mean:'])
		if status == 1:
			child.close(force=True)
			raise RuntimeError('Could not activate correct python environment for IRAF')
		
		# move into temporary directory
		child.sendline(f'cd {TEMPORARY_WORKING_DIRECTORY}')
		status = child.expect([r' $', r'cd: .* No such file or directory'])
		if status == 1:
			child.close(force=True)
			raise RuntimeError(f'Could not change directory to {TEMPORARY_WORKING_DIRECTORY} when running IRAF, is directory accessible?')
		
		# start IRAF
		child.sendline('pyenv exec pyraf')
		child.expect(r'\r\n\x1b\[\?2004h--> ') # the control code "\x1b[?2004h" turns on "bracketed paste mode", text is not treated as commands
		
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


###############################################################################
########################### END OF CONFIGURATION ##############################
###############################################################################





###############################################################################
############################### LIVE CODE #####################################
###############################################################################


########################### ArchiveNode Classes ###############################
"""
These classes describe how an archive node should retrieve information, you 
should pass an object of type "ArchiveNodeConfigBase" when creating the 
ArchiveNode object. The methods in here should use the config and information 
in the archive to get any data we want, and perform tasks.
"""
class ArchiveNodeBase:
	def __init__(self, config : ArchiveNodeConfigBase, archive_path : str, **kwargs):
		self.config = config
		self.path = archive_path
		return
	
	# prototype any functions here that we want to ensure any children have
	def get_sci_files(self) -> list[str]:
		raise NotImplementedError
	
	def get_orignal_files(self) -> list[str]:
		raise NotImplementedError
	
	def get_std_files(self) -> list[str]:
		raise NotImplementedError
	
	def associate_sci_std_files(self, sci_files : list[str], std_files : list[str]) -> list[tuple[str,str]]:
		raise NotImplementedError

class PlanetaryArchiveNode(ArchiveNodeBase):
	def __init__(self, config, archive_path, **kwargs):
		# do everything parent does on initialisation
		super().__init__(config, archive_path, **kwargs)

		# look for an AAREADME.txt file and add information from it to this object,
		# will create 
		readme_info = parse_AAREADME(os.path.join(self.path, 'AAREADME.txt'))
		for k, v in readme_info.items():
			set_attribute_from_dot_name(self, k, v)

	def get_sci_files(self):
		"""
		Get science files to operate on, prefer renormed if available
		"""
		if len(self.sci_files.renormed) > 0:
			return([os.path.normpath(os.path.join(self.path,afile)) for afile in self.sci_files.renormed])
		if len(self.sci_files.reduced) > 0:
			return([os.path.normpath(os.path.join(self.path,afile)) for afile in self.sci_files.reduced])
		return([])

	def get_original_files(self):
		# only have one original file so far, may need to change this logic eventually
		return([os.path.normpath(os.path.join(self.original_dirs[0],afile)) for afile in self.sci_files.original])

	def get_std_files(self) -> list[str]: # return full path to a set of std_files
		normalise_std_files = lambda folder, files: [os.path.normpath(os.path.join(folder,afile)) for afile in files]
		# first look for standard stars we have stored
		if hasattr(self, 'std_files'):
			if hasattr(self.std_files, 'renormed'):
				return(normalise_std_files(self.path, self.std_files.renormed))
			if hasattr(self.std_files, 'reduced'):
				return(normalise_std_files(self.path, self.std_files.reduced))

		# we're going to have to find our own standard star files
		found_std_files = []
		
		# second, look for associated files by name
		for afile in self.get_sci_files():
			std_fname = self.config.get_sci_std_filename(afile)
			if os.path.exists(std_fname):
				found_std_files.append(std_fname)
		if len(found_std_files) > 0:
			return(normalise_std_files(self.path, found_std_files))

		# third, look for files in the standard star directories
		std_folders = self.config.get_std_folders(self.original_folders)
		for afolder, afile_set in ((os.path.normpath(std_folder), os.listdir(std_folder)) for std_folder in std_folders):
			for afile in afile_set:
				afilepath = os.path.normpath(os.path.join(afolder,afile))
				if self.config.is_std(afilepath):
					found_std_files.append(afilepath)
		if len(found_std_files) > 0:
			return(found_std_files)

		# fourth, if we don't have any standard star files, we should see if we can make some
		# should get full path to files back from this function 
		found_std_files = self.config.create_std_cubes(self.original_folders)
		return(found_std_files) # if we haven't got any standard star files by now, we never will

	def associate_sci_std_files(self, sci_files, std_files) -> list[tuple[str,str]]:
		"""
		Use the information in the science and standard star files to associate each science file with a standard star file
		want to get the closest in observational parameters and at the same frequencies.
		"""
		sci_states = []
		for sci_file in sci_files:
			sci_states.append((sci_file, self.config.get_sci_datetime(sci_file), self.config.get_sci_spec_set(sci_file)))

		std_states = []
		for std_file in std_files:
			std_states.append((std_file, self.config.get_std_datetime(std_file), self.config.get_std_spec_set(std_file)))

		chosen_std_files = []
		for i, sci_state in enumerate(sci_states):
			candidate_std_states = std_states[:]

			# remove all std files that do not have matching spectral setups
			for j in range(len(candidate_std_states)-1, -1):
				std_state = candidate_std_states[j]
				if std_state[2] != sci_state[2]:
					candidate_std_states.remove(ij)

			# find closest in time std_file 
			chosen_std_state = candidate_std_states[0]
			for std_state in candidate_std_states:
				if abs(std_state[1]-sci_state[1]) < abs(chosen_std_state[1]-sci_state[1]):
					chosen_std_state = std_state
			
			# append best standard star file to chosen standard stars
			chosen_std_files.append(chosen_std_state[0])

		return(zip(sci_files, chosen_std_files))


############################ Helper Functions #################################


def set_attribute_from_dot_name(obj : typing.Any, name : str, value : typing.Any) -> None:
	"""
	Takes an object, takes an attribute name (e.g. "some.attribute.name") and value.
	Creates attributes of object that can be accessed via dot-notation in the same
	way the attribute name is specified. Will modify "obj"

	I.e.
	```set_attribute_from_dot_name(OBJ, "some.attribute.name", 100)```
	creates attributes such that the code
	```OBJ.some.attribute.name``` has a value of 100.
	i.e. ```OBJ.some.attribute.name == 100``` is true.
	"""
	idx = name.find('.')
	parent, child = (name[:idx], name[idx+1:]) if idx!=-1 else (name, '')
	if child == '':
		setattr(obj, parent, value)
	else:
		if not hasattr(obj, parent):
			setattr(obj, parent, types.SimpleNamespace())
		set_attribute_from_dot_name(getattr(obj,parent), child, value)
	return


def regex_tree_parser(regex_tree, text, filters={}):
	"""
	Use a dictionary of regular expressions to find information in 'text'

	filters will be applied to the found list of matches both before and after children
	are combined via dot notation.
	"""
	result = {}
	for parent_key, parent_v in regex_tree.items():
		_lgr.DEBUG(f"{parent_key = }")

		# ensure we are interating over a tuple/list of keys
		k_set = (parent_key,) if type(parent_key) is str else parent_key

		_lgr.DEBUG(f'{k_set=}')
		for k in k_set:
			_lgr.DEBUG(f"{k=}")
			# ensure that we are iterating over a tuple/list of values
			v_set = (parent_v,) if type(parent_v) not in (tuple,list) else parent_v
			for v in v_set:
				if type(v) is re.Pattern:
					# we just store the resulting matches in 'k'
					result[k] = result.get(k,[]) + [match.group(k) for match in v.finditer(text)]
				elif type(v) is dict:
					# we have to append the results of "use_regex_tree()" using "v" as the dictionary
					# to 'k' in a "parent.child" dot notation
					for match in v['parent'].finditer(text):
						for child_key, child_val in regex_tree_parser(v['children'], match.group(k)).items():
							parent_dot_child = f'{k}.{child_key}'
							result[parent_dot_child] = result.get(parent_dot_child, []) + child_val

	# filter results
	for k, v in filters.items():
		if k in result:
			result[k] = v(result[k])
	return(result)


def parse_AAREADME(fpath):
	# note that the keys and names of regex groups have to match except for 'parent', the regex groups in 'parent' should be the key of it's
	# encosing dictionary.
	regex_dict = {
		'original_folders' : re.compile(r"Copy of reduced files from .*? data in:\n(?P<original_folders>.*?)\n\n", flags=re.DOTALL),
		'sci_files' : (
			{	'parent': re.compile(r"\w main reduced files?:\n(?P<sci_files>(\S*\s+\(copy of \S*\))+)\n"), #text to pass to 'children' regexs
				'children': {
					# these get "dotted" to the parent. In this case the resulting key will be "file_names.reduced" and "file_names.original"
					('reduced','original') : re.compile(r'(?P<reduced>\S+)\s+\(copy of (?P<original>\S+)\)')
				},
			},
			{	'parent': re.compile(r"Photometric calibration .*? to:\n\n(?P<sci_files>.+?\n)*?\n", flags=re.DOTALL),
				'children': {
					'renormed' : re.compile(r'(?P<renormed>.+)\n')
				},
			},
		),
		'sci_fits_extensions' : {
			'parent':re.compile(r"There are a number of backplanes:\n(?P<sci_fits_extensions>(\d+: \S+\s+.*\S\s*\n)+)\n"),
			'children':{
				('idx','name','description'):re.compile(r'(?P<idx>\d+): (?P<name>\S+)\s+(?P<description>.*\S)\s*\n')
			},
		},
		'reference_spectra_path' : re.compile(r"The .* reference spectra are in \n(?P<reference_spectra_path>.*)\n"),
	}

	# these functions will be applied to the list of regex matches for the key stated
	filters = {
		'original_folders': lambda alist: [''.join(x.split('\n')) for x in alist] 
	}
	with open(fpath,'r') as f:
		fdata = f.read()
		data = regex_tree_parser(regex_dict, fdata, filters=filters)
	
	return(data)



######################### Load Configs ########################################
"""
Only load ArchiveNodeConfig classes if their ".DO_NOT_INCLUDE" attribute is
False. This attribute is used to denote config classes that are not meant to
be used in an actual node.
"""
import inspect
archive_configs = inspect.getmembers(
	sys.modules[__name__], 
	lambda member: (
		inspect.isclass(member) 
		and issubclass(member, ArchiveNodeConfigBase) 
		and (not member.DO_NOT_INCLUDE)
	)
)


################## Function That Crawls Archive ###############################
def crawl(root):
	for i, (path, folders, files) in enumerate(os.walk(root)):
		_lgr.DEBUG(f'######### Archive Site {i} ##########')
		_lgr.DEBUG(f'{path = }')

		# Find correct config for the archive
		archive_config = None
		for _names, _x in archive_configs:
			if _x.is_config_valid(path):
				archive_config = _x
				break
		if archive_config is None:
			_lgr.WARN(f'Could not find a valid config for archive node "{path}"')
			continue

		archive = PlanetaryArchiveNode(archive_config, path)
	
		_lgr.DEBUG(f'{archive=}')

		std_files = archive.get_std_files()
		sci_files = archive.get_sci_files()

		sci_std_files = list(archive.associate_sci_std_files(sci_files, std_files))
		_lgr.DEBUG('{sci_std_files=}')

		# ---------------------------------------------------------------------------
		# By now we should have either found some standard star files, and paired 
		# them up with the observation files, or we have exited.
		# 
		# Therefore, next we should copy the standard star files to the archive and 
		# rebin them to the same spectral resolution as input data
		# ---------------------------------------------------------------------------

		sci_files = [x[0] for x in sci_std_files]
		std_files = [x[1] for x in sci_std_files]
		for i, (sci_file, std_file) in enumerate(sci_std_files):
			dest_file = archive.config.get_sci_std_filename(sci_file)
			if os.path.abspath(std_file) == os.path.abspath(dest_file):
				_lgr.DEBUG('Standard star file is already named according to archive config, no need to copy.')
			else:
				try:
					_lgr.DEBUG("Copying standard star files to archive directory...")
					if os.path.exists(dest_file):
						std_files[i] = dest_file
						raise RuntimeError(f'Destination file "{dest_file}" already exists, not copying "{std_file}".')
					shutil.copy(std_file, dest_file)
					std_files[i] = dest_file
				
					_lgr.DEBUG(f"Re-binning the copied files so their spectral axes match those of their associated observations")
					# open the copied files and re-bin the spectral axes to be the same as the observations.
					with fits.open(dest_file, mode='update') as hdul:
						ut.fits.hdu_rebin_to(
							hdul[archive.config.STD_DATA_EXT], 
							fits.getheader(sci_file, archive.config.SCI_DATA_EXT), 
							ax_idx = (archive.config.SCI_SPECTRAL_AX_IDX, archive.config.STD_SPECTRAL_AX_IDX),
							combine_func = np.nansum
						)
						hdul.flush() # write all changes
				
				except Exception as e:
					_lgr.ERROR(f'Something went wrong when copying or rebinning the standard star file "{std_file}"')
					_lgr.ERROR(e)
		
		_lgr.DEBUG(f'{sci_files=}')
		_lgr.DEBUG(f'{std_files=}')

		yield(sci_files, [archive.config.SCI_DATA_EXT for f in sci_files], std_files, [archive.config.STD_DATA_EXT for f in std_files])


######################### Archive Crawler Testing #############################
if __name__=='__main__':
	for sci_files, sci_exts, std_files, std_exts in crawl("/network/group/aopp/planetary/PGJI004_IRWIN_ICEGIANT/archive"):
		print('#'*50)
		for i in range(len(sci_files)):
			print(f'{sci_files[i]}\n{sci_exts[i]}\n{std_files[i]}\n{std_exts[i]}\n')
			if i < len(sci_files)-1: print('-'*50)
			fits.info(sci_files[i])
			fits.info(std_files[i])
		print('='*50)

		sys.exit(f'EXIT FOR DEBUGGING at line {sys._getframe().f_lineno}')
