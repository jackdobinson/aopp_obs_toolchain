#!/usr/bin/env python3
"""
Contains utility functions for creating caches
"""
import typing

import utilities as ut
import utilities.cfg
import utilities.path

if ut.cfg.cache_type == 'pickle':
	import pickle
	def default_writer(fcache, data):
		with open(fcache,'wb') as f:
			pickle.dump(data, f)
		return

	def default_reader(fcache):
		with open(fcache, 'rb') as f:
			return(pickle.load(f))
else:
	raise ValueError('Unknown value {ut.cfg.cache_type} for module variable "utilities.cfg.cache_type"')


def remove_if_stale(
		fcache : str, 
		is_stale : typing.Callable[[str],bool] = lambda fcache: False
	) -> None:
	"""
	Checks the return value of "is_stale" if True, deletes the old cache-file
	if there is one.

	# ARGUMENTS #
		fcache
			path to the cache to be tested for staleness
		is_stale
			A callable that takes "fcache" and returns a bool.
			the default always assumes the cache is never stale.
			
			An example of a callable that returns true
			when fcache is older than another file is
			`lamdba fcache : os.path.getmtime(other_file) > os.path.getmtime(fcache)`.
	"""
	# don't do anything if we don't have an existing cache file
	if fcache is None: return
	if not os.exists(fcache): return

	# if we're here the cache file exists, so check it's state
	if is_stale(fcache):
		# remove it if it's stale
		os.remove(fcache)
	
	return


R = typing.TypeVar('R') # return type
def get_result(
		fcache : str, 
		acallable : typing.Callable[[], R], 
		cache_writer : typing.Callable[[str, R], None],= default_writer, 
		cache_reader : typing.Callable[[str], R] = default_reader
	) -> R:
	# if we don't have a cache, just return the result
	if fcache is None: return(acallable())

	recompute_flag = False
	if os.path.exists(fcache):
		# the file exists, so get the result from it
		# N.B. does not check for stale cache, use "remove_if_stale()" to do that
		try:
			result = cache_reader(fcache)
		except Exception:
			# something went wrong, so we need to recompute the result anyway
			recompute_flag = True
	else:
		# we need to compute the result as cache does not exist
		recompute_flag = True

	if recompute_flag:
		result = acallable()
		ut.path.ensure_exists(os.dirname(fcache), isdir=True)
		cache_writer(fcache, result)
	
	# return the result
	return(result)

	



