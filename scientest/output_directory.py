"""
Constructs and maintains the output directory for tests
"""
import os
from pathlib import Path
import time
import datetime as dt
from typing import Callable
from contextlib import contextmanager

import scientest
import scientest.cfg.logs
import scientest.cfg.settings

_lgr = scientest.cfg.logs.get_logger_at_level(__name__, 'INFO')


def get_last_modified_time_of_anything_in_dir(dir : Path) -> dt.datetime:

	get_last_modified_time = lambda path: dt.datetime.fromtimestamp(os.path.getmtime(path), dt.timezone.utc)

	last_modified_time = get_last_modified_time(dir)

	for root, dirnames, filenames in os.walk(dir, topdown=True, onerror=None, followlinks=False):
		for fname in dirnames + filenames:
			lmt = get_last_modified_time(dir / root / fname)
			if lmt > last_modified_time:
				last_modified_time = lmt
	
	return last_modified_time

def recursively_remove_directory(dir : Path, file_predicate = lambda path: True, dir_predicate = lambda path: True):
	for root, dirnames, filenames in os.walk(dir, topdown=False, onerror=None, followlinks=False):
		for name in filenames:
			if file_predicate((dir / root / name)):
				(dir / root / name).unlink()
		for name in dirnames:
			if dir_predicate((dir / root / name)):
				(dir / root / name).rmdir()
	dir.rmdir()
	return


class TestsOutputDirectory:
	"""
	Want to confine all test output to a specific directory, therefore don't allow absolute paths into this.
	"""
	@classmethod
	def remove_child_directory_predicate(cls, path):
		return scientest.cfg.settings.root_test_output_dir.resolve(strict=True) in path.resolve(strict=True).parents
	
	
	def __init__(self, 
			parent_dir : Path | str,
			dir_fmt : str = "test_{datetime_str}",
			format_parameter_providers : dict[str,Callable[[],str]] = {
				"datetime_str" :
					lambda : dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f%z")
			},
			remove_old_dirs_mode : None | int | dt.datetime = None,
		):
		self.format_parameters = dict((k,callable()) for k,callable in format_parameter_providers.items())
		self.parent_dir = Path(parent_dir)
		self.test_dir_str = dir_fmt.format(**self.format_parameters)
		self.test_dir = self.parent_dir / self.test_dir_str
		
		self.remove_old_dirs(remove_old_dirs_mode)
	
	
	def remove_old_dirs(self, method : None | int | dt.datetime):
		if (
			not self.parent_dir.exists()
			or (method is None)
			):
			return
	
		self.parent_dir == 'test_output', f'Ensure we only remove subfolders of a folder called "test_output"'
	
		potential_removal_list = []
	
		for child in self.parent_dir.iterdir():
			if child.is_dir():
				mod_time = get_last_modified_time_of_anything_in_dir(child)
				potential_removal_list.append((child, mod_time))
		
		# sort most recently modified directories first
		potential_removal_list.sort(key=lambda x:x[1], reverse=True)
	
		match method:
			case int():
				removal_list = potential_removal_list[method:]
			case dt.datetime():
				_lgr.debug(f'{[mod_time < method for i, (dir, mod_time) in enumerate(potential_removal_list)]=}')
				idx = min([i for i, (dir, mod_time) in enumerate(potential_removal_list) if mod_time < method])
				_lgr.debug(f'{idx=} {method=}')
				removal_list = potential_removal_list[idx:]
		
		for dir, mod_time in removal_list:
			_lgr.debug(f'Removing {dir=}')
			assert self.remove_child_directory_predicate(dir), f'{dir=} failed {self.__class__.__name__}.remove_child_directory_predicate(...) test.'
			recursively_remove_directory(dir)
		
		return
	
	@property
	def path(self):
		return self.test_dir
	
	def assert_path_is_relative(self, path : Path):
		p = Path(path)
		p_is_relative = (not p.is_absolute()) or (p.is_relative_to(self.test_dir))
		
		assert p_is_relative, f'Path "{path}" should be relative to {self.test_dir}'
			
	
	def __truediv__(self, path : Path | str):
		self.assert_path_is_relative(path)
		
		return self.test_dir / path
		
	
	def __str__(self):
		return str(self.test_dir.absolute)
	
	
	def ensure_dir(self, path : Path | str = Path(), parents=True, exist_ok=True):
		_lgr.debug(f'Ensuring directory "{path}"')
		p = self / path
		self.assert_path_is_relative(p)
		p.mkdir(parents=True, exist_ok=True)
	
	
	def _open(self, path : Path | str, mode='r', buffering=-1, encoding=None, errors=None, newline=None):
		fpath = (self / path)
		self.assert_path_is_relative(fpath)
		fpath.parent.mkdir(parents=True, exist_ok=True)
		return fpath.open(mode, buffering, encoding, errors, newline)
		
	
	@contextmanager
	def open(self, path : Path | str, mode='r', buffering=-1, encoding=None, errors=None, newline=None):
		f = self._open(path, mode, buffering, encoding, errors, newline)
		yield f
		f.close()


# Instantiate root directory
scientest.root_test_output_dir = TestsOutputDirectory(
	scientest.cfg.settings.root_test_output_dir, 
	remove_old_dirs_mode = scientest.cfg.settings.root_test_output_dir_remove_old_dirs_mode
)