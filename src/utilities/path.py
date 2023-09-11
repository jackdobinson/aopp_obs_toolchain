#!/usr/bin/env python3
"""
Contains utility functions for operations on filesystem paths
"""
import utilities.logging_setup
logging, _lgr = utilities.logging_setup.getLoggers(__name__, 'INFO')

import os
import datetime
import typing

import utilities as ut
import utilities.type
import utilities.args

def append_to_filename(file_path, astr, ext=None):
	"""
	Appends "astr" to the end of a filename, but before the extension
	E.g. "my_file.txt" -> "my_file_modified.txt", where astr="_modified"
	"""
	if ext is None:
		newfname = astr.join(os.path.splitext(file_path))
	else:
		newfname = astr.join([file_path[:-len(ext)],file_path[-len(ext):]])
	return(newfname)


def fpath_from(source_path, fmt_str, **kwargs):
	"""
	Uses "fmt_str" to construct a path that is related to "source_path". "kwargs" may be
	used to pass additional strings to be used with the format spec in "fmt_str"

	Automatic format variables are:
		fdir
			the directory of "source_path" as found by `os.path.dirname(source_path)`
		fname
			the filename of "source_path" without any extensions as found 
			by `os.path.splitext(os.path.basename(source_path))[0]`
		fext
			the extension of "source_path" without as found by
			`os.path.splitext(os.path.basename(source_path))[1]`
		cwd
			the current working directory as found by `os.getcwd()`
		timestamp
			the current time in iso format as found by
			`datetime.datetime.now().isoformat()`
	"""
	if fmt_str is None: return None
	fdir = os.path.dirname(source_path)
	if fdir == '':
		# no path means the file is a local one, not that it is an absolute path,
		# an absolute path has an "fdir" of "os.sep" if nothing else.
		fdir = '.'
	fnamefext = os.path.basename(source_path)
	fname, fext = os.path.splitext(fnamefext)
	cwd = os.getcwd()
	timestamp = datetime.datetime.now().isoformat()
	_lgr.DEBUG(f'{fdir=}')
	_lgr.DEBUG(f'{fname=}')
	_lgr.DEBUG(f'{fext=}')
	_lgr.DEBUG(f'{cwd=}')
	_lgr.DEBUG(f'{timestamp=}')
	for k, v in kwargs.items():
		_lgr.DEBUG(f'{k}={v}')
	return(fmt_str.format(fdir=fdir, fname=fname, fext=fext, cwd=cwd, timestamp=timestamp, **kwargs))


def file_writer(fpath, data, append_flag):
	try:
		with open(fpath, 'a' if append_flag else 'w') as f:
			f.write(data)
		return(True)
	except Exception as e:
		raise RuntimeError('Could not write data to file "{fpath}"') from e
	return(False)


def add_args(parser, prefix='', defaults={}):
	"""
	Adds arguments that work with the routines in this file to "parser", will prepend them with "prefix"
	to enable scoping.
	"""
	parser.add_argument(f'--{prefix}fmt', type=str, help=fpath_from.__doc__, default=defaults.get('fmt',"{fdir}/{fname}modified{fext}"))
	parser.add_argument(f'--{prefix}mode', type=str, choices = typing.get_type_hints(write_with_mode)['mode'].__args__, help='How should we write the data?', default=defaults.get('mode','no_overwrite'))
	parser.add_argument(f'--{prefix}make_dirs', action=(ut.args.ActionTf if defaults.get('make_dirs',True) else ut.args.ActiontF), prefix=prefix, help='Should intermediate directories be created?')
	return(parser)

def write_with_mode(
		fpath : str, 
		data : typing.Any, 
		writer : typing.Callable[[str, typing.Any, bool], bool], 
		mode : typing.Literal['no_overwrite','overwrite','append','no_output'] = 'no_overwrite',
		make_dirs : bool = True,
	) -> bool:
	"""
	Writes "data" to "fpath" using the callable "writer" in a way as decided by "mode".

	# ARGUMENTS # 
		fpath
			<str> Path to write data to
		data
			<Any> data to write (should be an input to "writer")
		writer
			<Callable[[fpath, data, append_flag], bool]> A callable that accepts a path, the
			data to write to it, and a flag determining if data should be appended or overwritten.
			Should return True if data was written successfully, False otherwise.
		mode
			<str> Mode that determines under what circumstances the data is written.
		make_dirs
			<bool> Should we make intermediate directories if "fpath" does not exist?

	# RETURNS #
		data_written_flag
			<bool> True if data has been written, false otherwise.
	"""

	fpath_exists = os.path.exists(fpath)
	fpath_dir = os.path.dirname(fpath)
	fpath_dir_exists = os.path.exists(fpath_dir)

	if ((fpath_exists and (mode == 'no_overwrite'))
			or (mode == 'no_output')):
		_lgr.INFO(f'No output written to "{fpath}", as {mode=}, and destination file {"does" if fpath_exists else "does not"} exist.')
		return(False)

	if not fpath_dir_exists:
		_lgr.INFO(f'Destination directory "{fpath_dir}" does not exist. {"Creating it..." if make_dirs else "Returning..."}')
		if make_dirs:
			os.makedirs(fpath_dir)
		else:
			return(False)

	_lgr.INFO(f'{"Writing" if (mode!="append") else "Appending"} data to "{fpath}"...')
	if writer(fpath, data, mode=='append'):
		return(True)
	return(False)


	




def write_to(fpath, data, mode='w', create_dirs=True):
	"""
	Writes "data" to the location "fpath" in mode "mode", if "create_dirs" is True 
	will create any missing intermediate directories.
	"""
	fdir = os.dirname(fpath)
	if (not os.path.exists(fdir)) and (create_dirs):
		os.makedirs(fdir)

	with open(fpath, mode=mode) as f:
		if ut.type.is_iter(data):
			for item in data:
				f.write(item)
		elif callable(data):
			if ut.type.is_gen_func(data):
				# write each item from the generator
				for item in data():
					f.write(item)
			else:
				# write the return value of the callable
				f.write(data())
		else:
			# data is just some object, lets hope we can convert it to a string
			f.write(data)
	return

def ensure_exists(fpath, isdir=True):
	"""
	Make sure the path "fpath" exist, create it if needed,
	if "isdir" is True, then will make the path as a dir
	"""
	if isdir:
		if not os.path.exists(fpath):
			os.makedirs(fpath)
	else:
		fdir = os.path.dirname(fpath)
		if not os.path.exists(fdir): 
			os.makedirs(fdir)
		if not os.path.exists(fpath):
			with open(fpath, 'w') as f: pass

	return


if __name__=='__main__':
	_lgr.setLevel("DEBUG")
