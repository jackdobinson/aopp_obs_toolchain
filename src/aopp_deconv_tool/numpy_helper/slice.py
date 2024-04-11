"""
Helper functions for slice operations on numpy arrays
"""
from typing import Iterator
import numpy as np

import aopp_deconv_tool.numpy_helper as nph
import aopp_deconv_tool.numpy_helper.axes

import aopp_deconv_tool.cast as cast

def around_center(big_shape : tuple[int,...], small_shape : tuple[int,...]) -> tuple[slice,...]:
	"""
	returns a slice of a in the shape of b about the center of a
	Accepts np.ndarray or tuple for each argument
	"""
	s_diff = tuple(s1-s2 for s1,s2 in zip(big_shape, small_shape))
	return(tuple([slice(d//2, s-(d//2+d%2)) for s, d in zip(big_shape,s_diff)]))

def from_string(slice_tuple_str : str):
	try:
		return tuple(slice(*tuple(cast.to(z,int) if z != '' else None for z in y.split(':'))) for y in slice_tuple_str.split(','))
	except Exception as e:
		e.add_note(f"slice tuple string '{slice_tuple_str}' is malformed")
		raise

def squeeze(slice_tuple: tuple[slice | int,...]) -> tuple[slice | int]:
	return tuple(s.start if type(s) is slice and ((s.stop - s.start) == (1 if s.step is None else s.step)) else s for s in slice_tuple )

def unsqueeze(slice_tuple: tuple[slice | int,...]) -> tuple[slice]:
	return tuple(slice(s,s+1) if type(s) is int else s for s in slice_tuple)

def get_indices(
		a : np.ndarray, 
		slice_tuple : tuple[slice|int,...] | np.ndarray[int] | None,
		as_tuple : bool = True
	) -> np.ndarray | tuple[np.ndarray,...]:
	"""
	Returns the indices of an array slice. Indicies will select the sliced part of an array.
	
	a
		The array to get the indicies of a slice of
	slice_tuple
		A tuple of length `a.ndim` that specifies the slices
	as_tuple
		If True will return an `a.ndim` length tuple, otherwise will return a numpy array.
	"""
	
	if slice_tuple is None:
		slice_tuple = tuple([slice(None)]*a.ndim)
	
	print(f'BEFORE {slice_tuple=}')
	slice_tuple = unsqueeze(slice_tuple) # want a.ndim == a[sliced_idxs].ndim
	print(f'AFTER {slice_tuple=}')
	
	slice_idxs = np.indices(a.shape)[(slice(None),*slice_tuple)] 
	
	return tuple(slice_idxs) if as_tuple else slice_idxs 
	

def iter_indices(
		a : np.ndarray, 
		slice_tuple : tuple[slice|int,...] | np.ndarray[int] | None = None,
		group : tuple[int,...] = tuple(),
		squeeze=True
	) -> Iterator[tuple[np.ndarray]]:
	"""
	Iterator that returns the sliced indices of a sliced array.
	
	a
		The array to get the indicies of a slice of
	slice_tuple
		A tuple of length `a.ndim` that specifies the slices
	group
		Axes that are not iterated over (i.e. they are grouped together). E.g.
		if a.shape=(3,5,7,9) and group=(1,3), then on iteration the indices
		`idx` select a slice from `a` such that a[idx].shape=(5,9).
	squeeze
		Should non-grouped axes be merged (need one for loop for all of them),
		or should they remain separate (need a.ndim - len(group) loops).
	"""
	
	group = tuple(group)
	print(f'{a.shape=} {slice_tuple=} {group=} {squeeze=}')
	sliced_idxs = get_indices(a, slice_tuple, as_tuple=False)
	
	print(f'BEFORE {sliced_idxs.shape=}')
	if squeeze:
		sliced_idxs = nph.axes.merge(sliced_idxs, tuple(1+x for x in nph.axes.not_in(a,group)), 0)
	else:
		sliced_idxs = np.moveaxis(sliced_idxs, (1+x for x in group), (x for x in range(len(group),sliced_idxs.ndim)))
		sliced_idxs = np.moveaxis(sliced_idxs, 0, -(len(group)+1))
		
	print(f'AFTER {sliced_idxs.shape=}')
	return (tuple(x) for x in sliced_idxs)
	
