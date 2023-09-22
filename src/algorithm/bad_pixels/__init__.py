"""
Routines for identifying and fixing bad pixels in observation data.
"""
import numpy as np

def get_map(a : np.ndarray['N',float]) -> np.ndarray['N',bool]:
	"""
	Gets the map of bad pixels

	Arguments:
		a : np.ndarray
			A numpy array of pixels
	
	Returns:
		A boolean numpy array that is True where `a` is NAN and INF.
	"""
	return np.isnan(a) | np.isinf(a)
