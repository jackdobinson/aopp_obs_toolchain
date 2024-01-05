"""
Constructs and maintains the output directory for tests
"""

from pathlib import Path
import datetime as dt
from typing import Callable
from contextlib import contextmanager

import scientest.cfg.logs

_lgr = scientest.cfg.logs.get_logger_at_level(__name__, 'DEBUG')

class TestsOutputDirectory:
	"""
	Want to confine all test output to a specific directory, therefore don't allow absolute paths into this.
	"""
	def __init__(self, 
			parent_dir : Path | str = Path.cwd() / "test_output",
			dir_fmt : str = "test_{datetime_str}",
			format_parameter_providers : dict[str,Callable[[],str]] = {
				"datetime_str" :
					lambda : dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f%z")
			}
		):
		self.format_parameters = dict((k,callable()) for k,callable in format_parameter_providers.items())
		self.parent_dir = Path(parent_dir)
		self.test_dir_str = dir_fmt.format(**self.format_parameters)
		self.test_dir = self.parent_dir / self.test_dir_str
	
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