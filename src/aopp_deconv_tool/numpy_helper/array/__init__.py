"""
Helper routines that calculate values or give information about arrays
"""
from typing import TypeVar, Literal, NewType

import numpy as np

from aopp_deconv_tool.typedef import NumVar, ShapeVar

T = TypeVar('T')

type N = NewType('N', NumVar)
type M = NewType('M', NumVar)

type S[X] = NewType('ShapeS', ShapeVar[X])
type Q[X] = NewType('ShapeQ', ShapeVar[X])
type R[X] = NewType('ShapeR', ShapeVar[X])




def get_center_offset_brightest_pixel(a : np.ndarray[S[N],T]) -> np.ndarray[[N],T]:
	"""
	Get the offset from the center of array `a` of the brightest element of `a`
	"""
	if np.all(np.isnan(a)):
		return(np.zeros(a.ndim))
	offset = np.array([s//2 for s in a.shape]) - np.unravel_index(np.nanargmax(a), a.shape)
	return(offset)


def apply_offset(a : np.ndarray[S[N],T], offset : np.ndarray[[N],int]) -> np.ndarray[S[N],T]:
	"""
	Apply `offset` to array `a`, shifting the elements with periodic boundary conditions
	"""
	return	np.roll(a, offset, tuple(range(a.ndim)))


def ensure_odd_shape(a : np.ndarray[S[N],T], axes : tuple[int,...] | None = None) -> np.ndarray[Q[N],T]:
	"""
	Get a slice of array `a` such that each axis has an odd number of entries.
	The resulting slice is always the same size or smaller than `a`
	"""
	if axes is None:
		axes = tuple(range(a.ndim))
	slices = tuple(slice(s-1+s%2) if i in axes else slice(None) for i, s in enumerate(a.shape))
	return a[slices]


def offsets_from_point(shape : S[N], point : np.ndarray[[N],T] = None, scale : S[N] = None) -> np.ndarray[S[N],T]:
	"""
	For an array of shape `shape` get the offsets from a specific point in array coordinates.
	"""
	if point is None:
		point = np.array([s//2 for s in shape]) # center
	if scale is None:
		scale = np.ones_like(point, dtype=float)
	else:
		# -ve values are cell sizes, +ve are array axis sizes.
		scale = np.array([float(x)/s if x > 0 else -x for x,s in zip(scale, shape)], dtype=float) 
	#print(f'{scale=}')
	return np.moveaxis((np.moveaxis(np.indices(shape),0,-1) - point)*scale,-1,0)

def indices_of_mask(m : np.ndarray[S[N],bool]) -> tuple[np.ndarray[[N],int],...]:
	"""
	Get the indices that are masked by mask `m`
	"""
	return np.where(m)

def manhattan_distance_mask(ii : np.ndarray[[*S[N],N,*Q[M]],T], dist : int | tuple[int,...] = 1, index_axis : int = 0) -> np.ndarray[[*S[N],*Q[M]],T]:
	"""
	Calculate a mask that is the manhattan distance `dist` from the array `ii` holding coord information, where the axis `index_axis`
	is has the `N` coordinates.
	"""
	if index_axis != 0:
		np.moveaxis(ii,index_axis,0)
		
	m = np.ones(ii.shape[1:],bool)
	
	if type(dist) is int:
		dist = tuple([dist]*m.ndim)
	elif len(dist) < m.ndim:
		dist = tuple(*dist,*([0]*(m.ndim-len(dist))))
	
	for b, d in zip(ii,dist):
		m &= np.abs(b) <= d
	m &= (np.abs(ii).sum(axis=0) <= np.max(dist))
	return m

def offsets_manhattan_distance(dist : int | tuple[int,...] = 1, ndim : Literal[N] = 3) -> np.ndarray[[N,M],T]:
	"""
	Get the offsets required to visit each element of an array of dimension `ndim` that is `dist` manhattan distance away or less from a point.
	"""
	if type(dist) is int:
		dist = tuple([dist]*ndim)
	elif len(dist) < ndim:
		dist = (*dist,*([0]*(ndim-len(dist))))
	shape = tuple(2*d + 1 for d in dist)
	return offsets_from_center_of_mask(manhattan_distance_mask(offsets_from_point(shape), dist))

def manhattan_distance(shape : S[N], point : np.ndarray[[N],T] = None) -> np.ndarray[S[N],T]:
	"""
	Generate an array of `shape` where each element is the manhattan distances from `point`
	"""
	if point is None:
		point = tuple(s//2 for s in shape)
	return np.abs(np.moveaxis(np.indices(shape),0,-1) - np.array(point)).sum(axis=-1)

def offsets_from_center_of_mask(mask : np.ndarray[S[N],bool]) -> np.ndarray[[N,*S[N]],int]:
	"""
	Get the offsets to all masked elements of `mask` from the center of `mask`.
	"""
	assert all([s%2 == 1 for s in mask.shape]), "mask must have an odd shape to have a unique center"
	return np.argwhere(mask) - np.array([s//2 for s in mask.shape])




