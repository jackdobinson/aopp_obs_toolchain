"""
Routines that use masks to assist in array operations
"""
import numpy as np

def from_nan_and_inf(a : np.ndarray[[...],float]) -> np.ndarray[[...],bool]:
	"""
	Gets a mask of all nan and inf values

	Arguments:
		a : np.ndarray
			A numpy array of pixels
	
	Returns:
		A boolean numpy array that is True where `a` is NAN and INF.
	"""
	return np.isnan(a) | np.isinf(a)
