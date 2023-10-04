"""
Helpers for array operations
"""
from typing import TypeVar

import numpy as np

from typedef import NumVar, ShapeVar
import cfg.logs
_lgr = cfg.logs.get_logger_at_level(__name__, 'WARNING')


T = TypeVar('T')
M = NumVar('M')
N = NumVar('N')
S = ShapeVar('S')
Q = ShapeVar('Q')


def pacman(a : np.ndarray[S[N],T], delta : np.ndarray[[N],int]) -> np.ndarray[S[N],T]:
	"""
	Shift an array by some delta using pacman (periodic) boundary conditions.
	E.g. tiles of the same array exist to either side.
	"""
	return np.roll(a, delta, tuple(i for i in range(a.ndim)))

def periodic(a : np.ndarray[S[N],T], delta : np.ndarray[[N],int]) -> np.ndarray[S[N],T]:
	"""
	Shift an array by some delta using periodic (pacman) boundary conditions
	E.g. tiles of the same array exist to either side.
	"""
	return pacman(a,delta)

def const(a : np.ndarray[S[N],T], delta : np.ndarray[[N],int], const : T = 0) -> np.ndarray[S[N],T]:
	"""
	Shift and array by some delta using constant boundary conditions
	E.g. the array is surrounded by a constant value
	"""
	# Shift as in periodic boundary conditions, then fill the "empty" region
	# with a constant value
	s = pacman(a, delta)
	for i, d in enumerate(delta):
		wrap_region_t = tuple((slice(d,None) if d < 0 else slice(None,d)) if i==j else slice(None) for j, d in enumerate(delta))
		s[wrap_region_t] = const
	return s

def reflect(a : np.ndarray[S[N],T], delta : np.ndarray[[N],int]) -> np.ndarray[S[N],T]:
	"""
	Shift an array by some delta using reflecting boundary conditions.
	E.g. the array is surrounded by reflected versions of itself.
	"""
	# Shift using pacman as normal, then set the 'wrapped' region to the data 
	# for reflected boundary conditions
	s = pacman(a, delta)
	for i, d in enumerate(delta):
		n = a.shape[i]
		wrap_region_t = tuple((slice(d,n,1) if d < 0 else slice(0,d,1)) if i==j else slice(None) for j, d in enumerate(delta))
		reflect_region_t = tuple((slice(n-2,n+d-2,-1) if d < 0 else slice(d,0,-1)) if i==j else slice(None) for j, d in enumerate(delta))
		s[wrap_region_t] = a[reflect_region_t]
	return s
 
 
