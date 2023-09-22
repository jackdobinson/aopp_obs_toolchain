"""
Very easy, nans and infs are the mean of surrounding values
"""
from typing import Any

import numpy as np
import scipy as sp
import scipy.interpolate

def fix(a : np.ndarray, bp_map : np.ndarray[Any, bool]):
	"""
	Fixes the bad pixels of an array of pixel data in place.

	Arguments:
		a : np.ndarray
			A numpy array of pixel data
		bp_map : np.ndarray
	"""
	raise NotImplementedError
	ii = np.indices(a.shape)
	a = sp.ndimage.uniform_filter(
		tuple(ii[:,~bp_map]),
		a[~bp_map],
		tuple(ii[:,bp_map])
	)
		
	return(a) 
