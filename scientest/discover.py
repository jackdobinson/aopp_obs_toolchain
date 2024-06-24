"""
Performs test discovery
"""

import os
from typing import Callable
import logging
from pathlib import Path
import inspect
import importlib

import scientest.cfg.logs
import scientest.cfg.settings

_lgr = scientest.cfg.logs.get_logger_at_level(__name__, 'DEBUG')

from scientest.term_output import terminal_right, terminal_left, terminal_center #,terminal_fill, terminal_wrap


class ModuleLocator:
	str_sep : str = '.'
		
	@classmethod
	def from_path(cls, path : Path, prefix : Path = Path()):
		rel_path = path.relative_to(prefix)
	
		if rel_path.suffix == '.py':
			rel_path = rel_path.with_suffix('')
		parts = rel_path.parts
		return cls(parts, prefix)
	
	@classmethod
	def from_str(cls, astr : str):
		parts = tuple(astr.split('.'))
		return cls(parts)
		
	def __init__(self,parts=tuple(),prefix=Path()):
		self.prefix = prefix
		self.parts = parts
	
	def __str__(self):
		return self.module
	
	@property
	def package(self) -> str:
		return ModuleLocator.str_sep.join(self.parts[:-1])
	
	@property
	def full_name(self) -> str:
		return ModuleLocator.str_sep.join(self.parts)
	
	@property
	def module(self):
		return importlib.import_module(self.full_name)
	
	@property
	def name(self) -> str:
		return self.parts[-1]
	
	@property
	def path(self) -> Path:
		return self.prefix.joinpath(*self.parts)
		
	def do_import(self):
		return self.module

def discover_tests(
		test_dir : str | Path, # directory to start searching for tests in

		 # If this is true, directory will be included in test search
		directory_search_predicate = \
			lambda adir: \
				not adir.startswith('__'),		
		
		# if this is true, modulefile will be included in test search
		modulefile_search_predicate = \
			lambda mf: \
				(not mf.startswith('__')) \
					and mf.endswith('.py') \
					and ('test' in mf), 
		
		# if this is true, the member of the module is a test and will be run
		member_search_predicate = \
			lambda m: \
				inspect.isfunction(m) \
					and ('test' in m.__name__),
	):

	test_dir = Path(test_dir)
	test_discovery_data = {}

	# Perform test discovery
	print(terminal_center(f' Starting test discovery from folder "{test_dir}" ', '='))
	
	for prefix, dirs, files in os.walk(test_dir, topdown=True, onerror=None, followlinks=False):
		_lgr.info(f'{prefix=}')
		_lgr.info(f'{dirs=}')
		_lgr.info(f'{files=}')
		
		dirs[:] = list(filter(directory_search_predicate, dirs))
		files[:] = list(filter(modulefile_search_predicate, files))
		
		for item in files:
			module_path = Path(prefix).relative_to(test_dir) / item
			module_locator = ModuleLocator.from_path(module_path)
			
			_lgr.info(f'Searching "{module_path}", discovering tests:')
			
			test_discovery_data[module_locator.full_name] = {}
			try:
				for test_name, test_member_callable in inspect.getmembers(module_locator.module, member_search_predicate):
					_lgr.info(f'    "{test_name}"')
					tid = f"{module_locator.full_name}::{test_name}"
					
					test_discovery_data[module_locator.full_name][tid] = {'callable':test_member_callable,'name':test_name}
			except ModuleNotFoundError as e:
				test_discovery_data[module_locator.full_name]['error'] = f'ERROR: Could not load module "{module_locator.full_name}", skipping all tests in this module'


	# process scientest_attributes on tests
	for module_name, module_tests in test_discovery_data.items():
		for test_id, test in module_tests.items():
			if test_id == 'error':
				continue
			for key, value in getattr(test['callable'], 'scientest_attributes', {}).items():
				match key:
					case 'debug':
						if value:
							_lgr.info(f'Debugging test {test_id}::{test["name"]}, removing all other tests...')
							return {module_name : {test_id : test}}
				

	return test_discovery_data