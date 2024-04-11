"""
Helpers for array operations involving indexing
"""
from typing import TypeVar, NewType

import numpy as np


import aopp_deconv_tool.cfg.logs
_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'WARNING')


from aopp_deconv_tool.typedef import NumVar, ShapeVar

T = TypeVar('T')

type N = NewType('N', NumVar)
type M = NewType('M', NumVar)

type S[X] = NewType('ShapeS', ShapeVar[X])
type Q[X] = NewType('ShapeQ', ShapeVar[X])
type R[X] = NewType('ShapeR', ShapeVar[X])


def pacman(a : np.ndarray[S[N],T], indices : np.ndarray[[N,M],int]) -> np.ndarray[[N,M],T]:
	"""
	Get values of array `a` at `indices`, out of bounds indices obey pacman physics,
	i.e. topology of a torus. Same as periodic boundaries
	"""
	return a[tuple(i % s for i,s in zip(indices, a.shape))]

def periodic(a : np.ndarray[S[N],T], indices : np.ndarray[[N,M],int]) -> np.ndarray[[N,M],T]:
	"""
	Get values of array `a` at `indices`, out of bounds indices are wrapped periodically
	"""
	return pacman(indices, a)

def const(a : np.ndarray[S[N],T], indices : np.ndarray[[N,M],int], const : T = 0) -> np.ndarray[[N,M],T]:
	"""
	Get values of array `a` at `indices`, out of bounds indices return a constant
	"""
	# get a mask that has the out of bounds elements, set them to a in-bound
	# index to get the values, then set the values from the out of bounds
	# indices to a constant.
	idx_out_of_bounds_mask = np.array([(i < 0) | (i >= s) for i,s in zip(indices, a.shape)],bool)
	indices[idx_out_of_bounds_mask] = 0
	oob_mask = np.array(idx_out_of_bounds_mask.sum(axis=0),bool)
	v = a[(*indices,)]
	v[oob_mask] = const
	return v

def reflect(a : np.ndarray[S[N],T], indices : np.ndarray[[N,M],int]) -> np.ndarray[[N,M],T]:
	"""
	Get values of array `a` at `indices`, out of bounds indices are reflected at the edges.
	"""
	# if i < s, then i_reflect = -i
	# if i >=s, then i_reflect = 2*(s-1) - i
	# Use multiplication to do if statements in arrays
	return a[tuple(i + 2*(s-1)*(i>=s) + i*(-2*(~((0<=i) & (i<s)))) for i,s in zip(indices, a.shape))] 
