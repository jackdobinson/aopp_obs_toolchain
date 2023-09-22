"""
Simplest we can think of, nans and infs are set to zero
"""
from typing import Any

import numpy as np


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




