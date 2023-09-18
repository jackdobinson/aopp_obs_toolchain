"""
Simplest we can think of, nans and infs are set to zero
"""
from typing import Any

import numpy as np


def get_map(a : np.ndarray):
	"""
	Gets the map of bad pixels

	Arguments:
		a : np.ndarray
			A numpy array of pixels
	
	Returns:
		A boolean numpy array that is True where `a` is NAN and INF.
	"""
	return np.isnan(a) | np.isinf(a)

def fix(a : np.ndarray, bp_map : np.ndarray[Any, bool]):
	"""
	Fixes the bad pixels of an array of pixel data in place.

	Arguments:
		a : np.ndarray
			A numpy array of pixel data
		bp_map : np.ndarray
	"""
	a[bp_map] = 0.0
	return(a)




