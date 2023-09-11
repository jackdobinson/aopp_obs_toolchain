#!/usr/bin/env python3
import utilities.logging_setup
logging, _lgr = utilities.logging_setup.getLoggers(__name__, 'DEBUG')

import os, sys
import numpy as np
import re
import shutil
import inspect
import typing
import types
from astropy.io import fits
import filecmp

import utilities as ut
import utilities.fits
import fitscube.deconvolve.archive_crawler_config as archive_crawler_config

# Set logging level for other modules
logging.setLevel(ut.fits, 'DEBUG')

"""
A program that will crawl through the telescope data archive at a given root directory.
Should emit fits file paths and relevent data extension indices for observations and their PSFs. 
"""


########################### ArchiveNode Classes ###############################
"""
These classes describe how an archive node should retrieve information, you 
should pass an object of type "ArchiveNodeConfigBase" when creating the 
ArchiveNode object. The methods in here should use the config and information 
available at a position in an archive to perform tasks and get data.
"""
class ArchiveNodeBase:
	"""
	Abstract base class that defines the ArchiveNode interface
	"""
	def __init__(self, config : archive_crawler_config.ArchiveNodeConfigBase, archive_path : str, **kwargs):
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
	"""
	Class that defines how to get various lumps of information in a Planetary Archive.
	"""
	def __init__(self, config, archive_path, **kwargs):
		# do everything parent does on initialisation
		super().__init__(config, archive_path, **kwargs)

		# look for an AAREADME.txt file and add information from it to this object,
		# will create 
		readme_info = parse_AAREADME(os.path.join(self.path, 'AAREADME.txt'))
		for k, v in readme_info.items():
			set_attribute_from_dot_name(self, k, v)

		# FIDDLE WITH THIS TO GET A SET OF EXISTING FILES AND THEN FILTER THEM IF NEEDED

		_lgr.INFO(f'Checking found science file for existence, and checking inclusion via config function "{self.config.__name__}.select_sci()"')
		# Only include files that actually exist and that we want to operate on as define in the config
		filter_func = lambda fname: (
			os.path.exists(os.path.join(self.path, fname)) 
		)
		self.sci_files.renormed = tuple(filter(filter_func, self.sci_files.renormed))
		self.sci_files.reduced = tuple(filter(filter_func, self.sci_files.reduced))
		# only record original files for files that exist
		self.sci_files.original = tuple(x[1] for x in filter(lambda x: filter_func(x[0]), zip(self.sci_files.reduced, self.sci_files.original))) 


		# want to find anything that looks like a science file here,
		# AAREADME does not neccessarily contain all of the science file names.
		# assume that all science files are named after self.sci_files.reduced except if the name contains the string "standard_star"
		
		sci_file_basename_roots = [os.path.splitext(os.path.basename(x))[0] for x in self.sci_files.reduced + self.sci_files.renormed]
		self.sci_files.all = list(filter(
			lambda x: (
				any([os.path.basename(x).startswith(y) for y in sci_file_basename_roots]) 
				and ("standard_star" not in x)
				#and (x.endswith('trim_smirtf_line.fits')) # DEBUGGING
				#and (x.endswith('H_renorm.fits')) # DEBUGGING
				#and (x.endswith('_trim.fits')) # DEBUGGING
			),
			os.listdir(self.path)
		))

		_lgr.DEBUG(f'{sci_file_basename_roots=}')
		_lgr.DEBUG(f'{self.sci_files.all=}')

		self.sci_files.chosen = list(filter(lambda x: self.config.select_sci(x), self.sci_files.all))


		# apply fixes to headers
		for afile in self.sci_files.all:
			self.config.sci_header_fix(os.path.join(self.path, afile))

		self.std_files = types.SimpleNamespace()
		self.std_files.chosen = []

		return
	
	
	def get_sci_files(self):
		"""
		Get science files to operate on, prefer renormed if available
		"""
		return([os.path.normpath(os.path.join(self.path, x)) for x in self.sci_files.chosen])
	
	
	def get_original_files(self):
		# only have one original folder so far, may need to change this logic eventually
		return([os.path.normpath(os.path.join(self.original_dirs[0],afile)) for afile in self.sci_files.original])
	

	def get_std_files_for(self, sci_file) -> list[str]:
		found_std_files = []
		acceptable_std_files = []
		sci_spec_grid = self.config.get_sci_spec_grid(sci_file)
		for method_for_finding_stds in (self.get_std_files_stored, self.get_std_files_name_associated, self.get_std_files_from_std_directory):
			found_std_files += method_for_finding_stds()
			#for i in range(len(found_std_files)-1, -1, -1):
			#	std_spec_grid = self.config.get_std_spec_grid(found_std_files[i])
			#	if not ut.np.is_regridable
				

		_lgr.DEBUG(f'{found_std_files=}')
		for std_file in found_std_files:
			#_lgr.DEBUG(f'{sci_file=}')
			_lgr.INFO(f'{sci_file=}')
			#_lgr.DEBUG(f'{std_file=}')
			_lgr.INFO(f'{std_file=}')
			std_spec_grid = self.config.get_std_spec_grid(std_file)
			#_lgr.DEBUG(f'{sci_spec_grid[0:3]=}\n{sci_spec_grid[-4:-1]=}')
			#_lgr.DEBUG(f'{std_spec_grid[0:3]=}\n{std_spec_grid[-4:-1]=}')
			#_lgr.DEBUG(f'{ut.np.is_regridable(std_spec_grid, sci_spec_grid)=}')
			if ut.np.is_regridable(std_spec_grid, sci_spec_grid):
				acceptable_std_files.append(std_file)


		if len(acceptable_std_files) ==0:
			found_std_files = self.config.create_std_cubes(self.original_folders)
			for std_file in found_std_files:
				if(ut.np.is_regridable(self.config.get_std_spec_grid(std_file), sci_spec_grid)):
					acceptable_std_files.append(std_file)

		_lgr.DEBUG(f'{acceptable_std_files=}')

		return(acceptable_std_files)
		

	def get_std_files_stored(self):
		_lgr.INFO('Looking for stored std_files in self.std_files.*')
		if hasattr(self, 'std_files'):
			if hasattr(self.std_files, 'chosen'):
				return(normalise_std_files(self.path, self.std_files.chosen))
		return([])

	def get_std_files_name_associated(self):
		found_std_files = []
		# second, look for associated files by name
		_lgr.INFO(f'Looking for std_files of form "{self.config.get_sci_std_filename("<scifilename>.fits")}"')
		for afile in self.get_sci_files():
			std_fname = self.config.get_sci_std_filename(afile)
			if os.path.exists(std_fname):
				found_std_files.append(std_fname)
		return(found_std_files)

	def get_std_files_from_std_directory(self):
		found_std_files = []
		std_folders = self.config.get_std_folders(self.original_folders)
		_lgr.INFO(f'Looking for std_files in the folders {std_folders}')
		for afolder, afile_set in ((os.path.normpath(std_folder), os.listdir(std_folder)) for std_folder in std_folders):
			for afile in afile_set:
				afilepath = os.path.abspath(os.path.join(afolder,afile))
				if self.config.is_std(afilepath):
					found_std_files.append(afilepath)
		return(found_std_files)


	def get_std_files(self) -> list[str]:
		"""
		I should re-write this to be more logical.
		Should find the best standard star file for 
		each science file here rather than later.
		But this works so I'm not going to fiddle right now.
		"""
		for a_sci_file in self.get_sci_files():
			for a_std_file in self.get_std_files_for(a_sci_file):
				if a_std_file not in self.std_files.chosen:
					self.std_files.chosen.append(a_std_file)
			#self.std_files.chosen += self.get_std_files_for(a_sci_file)
		return(self.std_files.chosen)


	def associate_sci_std_files(self, sci_files, std_files) -> list[tuple[str,str]]:
		"""
		Use the information in the science and standard star files to associate each science file with a standard star file
		want to get the closest in observational parameters and at the same frequencies.
		"""
		# create a holder for the chosen standard star files
		chosen_std_files = [None]*len(sci_files)


		# should probably use named tuples for this instead of having to remember the positions of stuff
		sci_states = []
		for sci_file in sci_files:
			sci_states.append((sci_file, self.config.get_sci_datetime(sci_file), self.config.get_sci_spec_set(sci_file)))

		std_states = []
		for std_file in std_files:
			std_states.append((std_file, self.config.get_std_datetime(std_file), self.config.get_std_spec_set(std_file)))

		for i, sci_state in enumerate(sci_states):
			_lgr.DEBUG(f'Finding matching std star for sci_state {i}\n{sci_state}')
			# find the filename of the standard star associated with the current science file
			std_associated_fname = self.config.get_sci_std_filename(sci_state[0])

			if chosen_std_files[i] is not None:
				continue # we already have one
			
			_lgr.DEBUG(f'Checking {len(std_states)} candidate standard states')
			candidate_std_states = std_states[:]

			# remove all std files that do not have matching spectral setups
			for j in range(len(candidate_std_states)-1, -1,-1):
				_lgr.DEBUG(f'Looking at spectral setup for candidate {j}\n{candidate_std_states[j]}')
				if candidate_std_states[j][2] != sci_state[2]:
					_lgr.DEBUG(f'Removing state {j}')
					candidate_std_states.pop(j)

			# remove all std files we cannot regrid to the sci spectral resolution
			sci_spec_grid = self.config.get_sci_spec_grid(sci_state[0])
			for j in range(len(candidate_std_states)-1, -1,-1):
				_lgr.DEBUG(f'Testing regriddability for candidate {j}\n{candidate_std_states[j]}')
				std_spec_grid = self.config.get_std_spec_grid(candidate_std_states[j][0])
				regriddable_flag = ut.np.is_regridable(std_spec_grid, sci_spec_grid)
				#_lgr.DEBUG(f'{ sci_spec_grid = }')
				#_lgr.DEBUG(f'{ std_spec_grid = }')
				#_lgr.DEBUG(f'{regriddable_flag = }')
				if not regriddable_flag:
					_lgr.DEBUG(f'Removing state {j}')
					candidate_std_states.pop(j)
				
			
			chosen_std_state = None
			
			# if there is already a std_file associated by name that is acceptable, then
			# set it as the chosen_std_state so that all subsequent tests must
			# be better than the existing file to become the chosen
			for std_state in candidate_std_states:
				if(std_associated_fname == std_state[0]):
					chosen_std_state = std_state

			# if there is not already a chosen_std_state, arbitrarily choose the first one
			if chosen_std_state is None:
				chosen_std_state = candidate_std_states[0]

			_lgr.DEBUG(f'')

			# find closest in time std_file 
			for std_state in candidate_std_states:
				_lgr.DEBUG(f'Testing time difference for candidate {j}\n{candidate_std_states}')
				if abs(std_state[1]-sci_state[1]) < abs(chosen_std_state[1]-sci_state[1]):
					chosen_std_state = std_state
		
			

			# append best standard star file to chosen standard stars
			chosen_std_files[i] = chosen_std_state[0]
			_lgr.DEBUG(f'Selected standard star file {chosen_std_files[i]} as best candidate')

		return(zip(sci_files, chosen_std_files))


############################ Helper Functions #################################


normalise_std_files = lambda folder, files: [os.path.normpath(os.path.join(folder,afile)) for afile in files]

def path_njoin_set(folder, files):
	return([os.path.normpath(os.path.join(folder,x)) for x in files])


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
	_lgr.DEBUG(f'##########\n{text=!r}\n##########')
	for parent_key, parent_v in regex_tree.items():
		_lgr.DEBUG(f"{parent_key = }")

		# ensure we are interating over a tuple/list of keys
		k_set = (parent_key,) if type(parent_key) is str else parent_key

		_lgr.DEBUG(f'{k_set=}')
		for k in k_set:
			_lgr.DEBUG(f"{k=}")
			# ensure that we are iterating over a tuple/list of values
			v_set = (parent_v,) if type(parent_v) not in (tuple,list) else parent_v
			_lgr.DEBUG(f'{v_set=}')
			for v in v_set:
				_lgr.DEBUG(f'{v=}')
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
				_lgr.DEBUG(f'{result=}')

	# filter results
	for k, v in filters.items():
		if k in result:
			result[k] = v(result[k])
	return(result)


def parse_AAREADME(fpath):
	"""
	Defines a dictionary of regular expressions that will be passed to 
	"regex_tree_parser()" to read the AAREADME.txt file. 
	
	Note that the keys and names of regex groups have to match except 
	for 'parent', the regex groups in 'parent' should be the key of 
	it's encosing dictionary.
	"""
	regex_dict = {
		'original_folders' : re.compile(r"Copy of reduced files from .*? data in:\n(?P<original_folders>.*?)\n\n", flags=re.DOTALL),
		'sci_files' : (
			{	'parent': re.compile(r"\w main reduced files?:\n(?P<sci_files>(\S*\s+\(copy of \S*\)\n)+)\n"), #text to pass to 'children' regexs
				'children': {
					# these get "dotted" to the parent. In this case the resulting key will be "file_names.reduced" and "file_names.original"
					('reduced','original') : re.compile(r'(?P<reduced>\S+)\s+\(copy of (?P<original>\S+)\)\n')
				},
			},
			{	'parent': re.compile(r"Photometric calibration [\s\S]*? to:\n\n(?P<sci_files>(\S+\n)+)\n"),
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
be used in an actual node. See "archive_crawler_config.py" for details.
"""
ARCHIVE_CONFIGS = inspect.getmembers(
	archive_crawler_config, 
	lambda member: (
		inspect.isclass(member) 
		and issubclass(member, archive_crawler_config.ArchiveNodeConfigBase) 
		and (not member.DO_NOT_INCLUDE)
	)
)


################# Functions That Crawl Archives ###############################

def standard_star_crawl(
		root : str, 
		overwrite_standard_stars : 																		\
			typing.Literal['never', 'different', 'incompatible_spectral_grid', 'always']	\
			= 'incompatible_spectral_grid',
	) -> tuple:
	"""
	Crawls through an archive and returns an iterator to the science 
	observations and associated standard star observations.

	# ARGUMENTS #
		root
			The base directory of the archive we want to crawl through

	# YIELDS #
		sci_files
			FITS files containing science observations
		sci_exts
			The FITS extension that holds the science observation for
			each file in "sci_files".
		std_files
			FITS files containing the "best" standard star observation
			for each file in "sci_files"
		std_exts
			The FITS extension that holds the standard star observation
			for each file in "std_files"
	
	# EXAMPLE #
		>>> for sci_files, sci_exts, std_files, std_exts in standard_star_crawl("/network/group/aopp/planetary/PGJI004_IRWIN_ICEGIANT/archive"):
		>>> 	for i in range(len(sci_files)):
		>>> 		print('SCIENCE FILE INFORMATION:')
		>>> 		fits.info(sci_files[i])
		>>> 		print('ASSOCIATED STANDARD STAR FILE INFORMATION:')
		>>> 		fits.info(std_files[i])
		>>> 		print('\n\n')

	"""
	for i, (path, folders, files) in enumerate(os.walk(root, topdown=True)):
		_lgr.DEBUG(f'######### Archive Site {i} ##########')
		_lgr.DEBUG(f'{path = }')

		# Find correct config for the archive_node
		archive_config = archive_crawler_config.ArchiveNodeConfigBase
		for _names, _x in ARCHIVE_CONFIGS:
			if _x.is_config_valid(path):
				archive_config = _x
				break
		if not archive_config.is_path_accessible(path):
			_lgr.INFO(f'Archive node {path} is inaccessible to crawler with config {archive_config}, skipping directory...')
			folders[:] = [] # remove child folders from traversable elements in tree. Only works if "topdown=True" in "os.walk()"
			continue

		if archive_config is archive_crawler_config.ArchiveNodeConfigBase:
			_lgr.WARN(f'Could not find a valid config for archive node "{path}", moving to next archive node')
			continue

		_lgr.INFO(f'Found valid config {archive_config} for archive node')

		# create the archive_node object
		archive_node = PlanetaryArchiveNode(archive_config, path)
	
		_lgr.DEBUG(f'{archive_node=}')

		std_files = archive_node.get_std_files()
		sci_files = archive_node.get_sci_files()

		_lgr.DEBUG(f'{sci_files=}')
		_lgr.DEBUG(f'{std_files=}')


		sci_std_files = list(archive_node.associate_sci_std_files(sci_files, std_files))
		_lgr.DEBUG(f'{sci_std_files=}')

		# ---------------------------------------------------------------------------
		# By now we should have either found some standard star files, and paired 
		# them up with the observation files, or we have exited.
		# 
		# Therefore, next we should copy the standard star files to the archive_node and 
		# rebin them to the same spectral resolution as input data
		# ---------------------------------------------------------------------------

		sci_files = [x[0] for x in sci_std_files]
		std_files = [x[1] for x in sci_std_files]


		for i, (sci_file, std_file) in enumerate(sci_std_files):
			dest_file = archive_node.config.get_sci_std_filename(sci_file)
			_lgr.INFO(f'Copying {std_file} to {dest_file}...')
			if os.path.abspath(std_file) == os.path.abspath(dest_file):
				_lgr.INFO(''.join((
					f'Standard star file "{std_file}" is already named according to ',
					f'{archive_node.config}.get_sci_std_filename(), no need to copy.',
				)))
				std_files[i] = dest_file
				continue
			dest_is_present_flag = os.path.exists(dest_file)
			if dest_is_present_flag:
				_lgr.INFO(f'Destination {dest_file} already exists')

			if (not dest_is_present_flag) or (overwrite_standard_stars == 'always'):
				shutil.copy2(std_file, dest_file)
			elif overwrite_standard_stars == 'never':
				_lgr.INFO(f'{overwrite_standard_stars = }, cannot copy standard star file')
				std_files[i] = dest_file
				continue
			elif overwrite_standard_stars == 'different':
				_lgr.INFO('Comparing file contents for potential copy')
				if filecmp.cmp(std_file, dest_file, shallow=False):
					_lgr.INFO('Both files have the same contents, no need to copy')
					std_files[i] = dest_file
					continue
				shutil.copy2(std_file, dest_file)
			elif overwrite_standard_stars == 'incompatible_spectral_grid':
				_lgr.INFO('Comparing spectral grids of sci_file and dest_file to see if an overwrite is needed.')
				try:
					sci_spec_grid = self.config.get_sci_spec_grid(sci_file)
					dest_spec_grid = self.config.get_std_spec_grid(dest_file)
					compatible_spec_grid_flag = ut.np.is_regridable(dest_spec_grid, sci_spec_grid)
				except:
					# if something went wrong, then assume it's a bad match
					compatible_spec_grid_flag = False
				if compatible_spec_grid_flag:
					_lgr.INFO('Spectral grids are compatible, no need to copy.')
					std_files[i] = dest_file
					continue # don't overwrite if grids are compatible
				shutil.copy2(std_file, dest_file)
			else:
				_lgr.WARN(f'Unknown value for argument {overwrite_standard_stars = }')

			# now we should use the desination file as the standard star file
			std_files[i] = dest_file

			# Now if we copied a file, try to re-bin the standard star to the same spectral grid as the observation.	

			try:	
				_lgr.INFO(f"Re-binning the copied file so it's spectral axes match those of it's associated observation")
				# open the copied files and re-bin the spectral axes to be the same as the observations.
				with fits.open(dest_file, mode='update') as hdul:
					ut.fits.hdu_rebin_to(
						hdul[archive_node.config.STD_DATA_EXT], 
						fits.getheader(sci_file, archive_node.config.SCI_DATA_EXT), 
						ax_idx = (archive_node.config.SCI_SPECTRAL_AX_IDX, archive_node.config.STD_SPECTRAL_AX_IDX),
						combine_func = np.nansum
					)
					hdul.flush() # write all changes
			except Exception as e:
				_lgr.ERROR(f'Something went wrong when re-binning the standard star file "{dest_file}" to the same grid as "{sci_file}"')
				_lgr.ERROR(e)
		
	
		_lgr.DEBUG(f'{sci_files=}')
		_lgr.DEBUG(f'{std_files=}')

		yield(sci_files, [archive_node.config.SCI_DATA_EXT for f in sci_files], std_files, [archive_node.config.STD_DATA_EXT for f in std_files])


######################### Archive Crawler Testing #############################
if __name__=='__main__':
	# Change this to test specific archive_configs
	ARCHIVE_CONFIGS = tuple(filter( 
		lambda archive_config: True,
		#lambda archive_config_member: issubclass(archive_config_member[1], archive_crawler_config.MUSE_ArchiveConfig),
	ARCHIVE_CONFIGS))
	_lgr.DEBUG(f"{ARCHIVE_CONFIGS=}")
	# -------------------------------------------

	#archive_root = "/network/group/aopp/planetary/PGJI004_IRWIN_ICEGIANT/archive"
	#archive_root = "/network/group/aopp/planetary/PGJI004_IRWIN_ICEGIANT/archive/Neptune/Gemini_NIFS/2009/20090907"
	archive_root = "/network/group/aopp/planetary/PGJI004_IRWIN_ICEGIANT/archive/Neptune/Gemini_NIFS/2009/20090901"


	for sci_files, sci_exts, std_files, std_exts in standard_star_crawl(archive_root):
		print('#'*50)
		for i in range(len(sci_files)):
			print(f'{sci_files[i]}\n{sci_exts[i]}\n{std_files[i]}\n{std_exts[i]}\n')
			if i < len(sci_files)-1: print('-'*50)
			fits.info(sci_files[i])
			fits.info(std_files[i])
		print('='*50)
		break # DEBUGGING
		sys.exit(f'EXIT FOR DEBUGGING at line {sys._getframe().f_lineno}')

