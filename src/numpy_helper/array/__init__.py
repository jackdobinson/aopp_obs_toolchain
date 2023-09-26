"""
Helper routines that calculate values or give information about arrays
"""
from typing import TypeVar, Generic
import numpy as np


def get_center_offset_brightest_pixel(a : np.ndarray['N',[...]]) -> np.ndarray[1,['N']]:
	if np.all(np.isnan(a)):
		return(np.zeros(a.ndim))
	offset = np.array([s//2 for s in a.shape]) - np.unravel_index(np.nanargmax(a), a.shape)
	return(offset)


def apply_offset(a : np.ndarray['N',[...]], offset : np.ndarray[1,['N']]) -> np.ndarray['N',[...]]:
	return	np.roll(a, offset, tuple(range(a.ndim)))


def ensure_odd_shape(a : np.ndarray['N',['A','B',...]]) -> np.ndarray['N', ['A-1+A%2','B-1+B%2',...]]:
	slices = tuple(slice(s-1+s%2) for s in a.shape)
	return a[slices]




