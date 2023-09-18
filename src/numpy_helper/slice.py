"""
Helper functions for slice operations on numpy arrays
"""

import numpy as np
import numpy_helper as nph
import numpy_helper.axes

import cast

def from_string(slice_tuple_str : str):
	try:
		return tuple(slice(*tuple(cast.to(z,int) if z != '' else None for z in y.split(':'))) for y in slice_tuple_str.split(','))
	except Exception as e:
		e.add_note(f"slice tuple string '{slice_tuple_str}' is malformed")
		raise

def get_indices(
		a : np.ndarray, 
		slice_tuple : tuple[slice|int,...] | np.ndarray[int] | None,
		group : tuple[int,...] | None = None
	):
	"""
	Iterator that returns the sliced indices of a sliced array.
	
	a
		The array to get the indicies of a slice of
	slice_tuple
		A tuple of length `a.ndim` that specifies the slices
	group
		Axes (after slice) that we should group together
	"""
	
	if slice_tuple is None:
		slice_tuple = tuple([slice(None)]*a.ndim)
	
	sliced_idxs = np.indices(a.shape)[(slice(None),*slice_tuple)]
	
	if group is not None:
		#print(f'BEFORE {sliced_idxs.shape=}')
		sliced_idxs = nph.axes.group(sliced_idxs, tuple(1+x for x in group), 0)
		#print(f'AFTER {sliced_idxs.shape=}')
		return (tuple(x) for x in sliced_idxs)
	else:
		return tuple(sliced_idxs)

