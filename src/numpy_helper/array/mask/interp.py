"""
ND interpolation for numpy arrays
"""
from typing import TypeVar
import numpy as np

import cfg.logs
_lgr = cfg.logs.get_logger_at_level(__name__, 'DEBUG')

T = TypeVar('T')


def constant(a : np.ndarray[[-1,-1],T], m : np.ndarray[[...],bool], value : T = 0 ) -> np.ndarray[[...],T]:
	a[m] = value
	return a


def mean(a : np.ndarray[[-1,-1],T], m : np.ndarray[[...],bool], window_shape : tuple[int,...] | int = 3 ) -> np.ndarray[[...],T]:
	"""
	Replaces masked elements with mean of surrounding values in place.

	Arguments:
		a : np.ndarray
			A numpy array of pixel data
		, : np.ndarray
	"""
	if type(window_shape) is int:
		window_shape = tuple(window_shape for _ in a.shape)
		
	for s in window_shape:
		assert s % 2 == 1, "window shape must be odd"
	
	idxs = np.indices(a.shape)
	m_idxs = idxs[:,m].T
	
	accumulator = np.zeros((np.count_nonzero(m),))
	
	
	for j in range(a.ndim):
		for i, m_idx in enumerate(m_idxs):
			st = tuple(slice(x-s//2 if x > s//2 else 0, x+s//2+1 if x+s//2+1 <= a.shape[k] else a.shape[k]) if k==j else x for k, (x, s) in enumerate(zip(m_idx, window_shape)))
			nm = ~m[st]
			_lgr.debug(f'{st=} {nm=}')
			_lgr.debug(f'{m_idx[j]=} {idxs[j][st][nm]=} {a[st][nm]=}')
			accumulator[i] += np.mean(a[st][nm])
	
	_lgr.debug(f'{accumulator=}')
	
	if np.any(np.isnan(accumulator)):
		raise RuntimeError(f'Could not replace masked elements with means at {[tuple(m_idx) for bp_idx in m_idxs[np.isnan(accumulator)]]} with {window_shape=}')
		
	accumulator /= a.ndim
	
	for i, m_idx in enumerate(m_idxs):
		_lgr.debug(f'{i=} {m_idx=} {accumulator[i]=}')
		a[tuple(m_idx)] = accumulator[i]
	
	return(a)



def interp(a : np.ndarray[[-1,-1],T], m : np.ndarray[[...],bool], fix_edges=True, **kwargs) -> np.ndarray[[...],T]:
	"""
	Replaces masked elements with linear interpolation of surrounding values in place.

	Arguments:
		a : np.ndarray
			A numpy array of pixel data
		m : np.ndarray
	"""
	from scipy.interpolate import griddata
	
	idxs = np.indices(a.shape)
	
	accumulator = griddata(np.moveaxis(idxs[:,~m],0,-1), a[~m], np.moveaxis(idxs[:,m],0,-1), **kwargs)
	a[m] = accumulator
	
	
	if np.any(np.isnan(accumulator)) and fix_edges:
		_lgr.debug('Fixing edges...')
		a = interp(a, np.isnan(a), fix_edges=False, **{**kwargs, 'method':'nearest'})
	
	if np.any(np.isnan(a)):
		raise RuntimeError('Could not interpolate masked elements, this may be due to NANs at the edges')
	
	return a
