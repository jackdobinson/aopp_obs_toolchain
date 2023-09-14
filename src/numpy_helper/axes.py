"""
Helper functions for axes operations on numpy arrays
"""

import numpy as np
from contextlib import contextmanager

@contextmanager
def to_end(a : np.ndarray, axes : tuple[int,...]):
	"""
	Reorders the axes of `a` so that the axes specified in the `axes` tuple are 
	at the end (rhs) of the ordering for purposes of slicing and indexing.
	"""
	axes_dest = tuple(a.ndim - 1 -i for i in range(len(axes)))[::-1]
	
	yield np.moveaxis(a, axes, axes_dest)
	
	# as np.moveaxis returns a view of `a` we don't have to clean up anything


def reverse(a : np.ndarray):
	print(a.ndim)
	axes = tuple(range(a.ndim))
	return np.moveaxis(a, axes, axes[::-1])

def group(a : np.ndarray, axes : tuple[int,...], axis : int = 0):
	"""
	Group a set of axes into a single axis.
	
	a : np.ndarray
		Numpy array to group axes of
	axes : tuple[int:...]
		axes to group together
	axis : int
		location of the final grouped axis
	"""
	not_group_shape = (a.shape[i] for i in range(a.ndim) if i not in axes)
	
	return np.moveaxis(np.moveaxis(a, axes, tuple(range(len(axes)))).reshape(-1, *not_group_shape), 0, axis)
