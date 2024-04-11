"""
ND interpolation for numpy arrays
"""
from typing import TypeVar, NewType
import numpy as np


import aopp_deconv_tool.cfg.logs
_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'DEBUG')

import aopp_deconv_tool.numpy_helper as nph
import aopp_deconv_tool.numpy_helper.array
import aopp_deconv_tool.numpy_helper.array.index

from aopp_deconv_tool.typedef import NumVar, ShapeVar

T = TypeVar('T')

type N = NewType('N', NumVar)
type M = NewType('M', NumVar)

type S[X] = NewType('ShapeS', ShapeVar[X])
type Q[X] = NewType('ShapeQ', ShapeVar[X])
type R[X] = NewType('ShapeR', ShapeVar[X])


def get_index_boundary_func(name : str):
	return {	
		'pacman' : nph.array.index.pacman,
		'periodic' : nph.array.index.periodic,
		'const' : nph.array.index.const,
		'reflect' : nph.array.index.reflect
	}[name]

def constant(a : np.ndarray[S[N],T], m : np.ndarray[S[N],bool], value : T = 0 ) -> np.ndarray[S[N],T]:
	"""
	Set masked elements to a constant value
	"""
	a[m] = value
	return a

def mean(
		a : np.ndarray[S[N],T], 
		m : np.ndarray[S[N],bool], 
		window : np.ndarray[Q[M],bool] | Q[M] | int = 3, # Where M <= N
		boundary : str ='reflect', 
		const : T = 0 
	) -> np.ndarray[S[N],T]:
	"""
	Replaces masked elements with mean of surrounding values in place.
	
	Arguments:
		a : np.ndarray[T]
			A numpy array of pixel data
		m : np.ndarray[bool]
			A mask of the elements of a to replace
		window : np.ndarray[bool] | tuple[int,...] | int
			The surrounding values we should use in the mean calculation. NANs
			and INFs will be ignored. If an `np.ndarray[bool]` will use the
			True elements as memmbers of the calculation, if a tuple will create
			a mask by using	each element of the tuple as a manhattan distance 
			from the point whose mean is being calculated, if an integer will
			use the same manhattan distance for each axis.
		boundary : str
			How boundaries are handled, one of ['pacman','const','reflect']
		const : T
			Value
	"""
	a = np.array(a)
	b_func = get_index_boundary_func(boundary)
	if boundary == 'const' : 
		index_boundary_func = lambda *args, **kwargs: b_func(*args, const = const, **kwargs)
	else:
		index_boundary_func = b_func
	
	idx_array = nph.array.indices_of_mask(m)
	if type(window) is np.ndarray:
		_lgr.debug(f'{a.ndim=} {window.ndim=}')
		while window.ndim < a.ndim:
			window = window[None,...]
		_lgr.debug(f'{window=}')
		deltas = nph.array.offsets_from_center_of_mask(np.array(window, bool))
	else:
		deltas = nph.array.offsets_manhattan_distance(window, a.ndim)
	_lgr.debug(f'{deltas=}')
	
	n = idx_array[0].size
	contributions = np.zeros((n,),int)
	accumulator = np.zeros((n,),dtype=a.dtype)
	
	
	#dbg_array = []
	for i, delta in enumerate(deltas):
		_lgr.debug(f'calculating contributions from offset {delta} number {i}/{deltas.shape[0]}')
		offset_idx_array = (np.array(idx_array).T-delta).T
		values = index_boundary_func(a, offset_idx_array)
		_lgr.debug(f'{values=}')
		# Things that are not masked should contribute
		contrib = ~index_boundary_func(m, offset_idx_array)
		_lgr.debug(f'{contrib=}')
		values[~contrib] = 0
		accumulator += values
		contributions += contrib
		#dbg_array.append((delta, offset_idx_array, values, contrib))
	
	#a_old = np.array(a)
	_lgr.debug(f'{accumulator=}')
	_lgr.debug(f'{contributions=}')
	invalid = contributions == 0
	contributions[invalid] = 1
	accumulator[invalid] = const
	a[idx_array] = (accumulator/contributions)
	
	
	"""# Uncomment for debugging
	a_new = a
	m_plt = m
	if a.ndim > 2:
		st = tuple(slice(None) if i >= (a.ndim-2) else s//2 for i,s in enumerate(a.shape))
		_lgr.debug(f'{st=}')
		a_old = a_old[st]
		a_new = a_new[st]
		m_plt = m[st]
	import matplotlib.pyplot as plt
	import plot_helper as ph
	fig, ax = ph.figure_n_subplots(3)#+2*len(deltas))
	ax[0].imshow(a_old, origin='lower')
	ax[0].get_xaxis().set_visible(False)
	ax[0].get_yaxis().set_visible(False)
	
	ax[1].imshow(m_plt, origin='lower')
	ax[1].get_xaxis().set_visible(False)
	ax[1].get_yaxis().set_visible(False)
	
	ax[2].imshow(a_new, origin='lower')
	ax[2].get_xaxis().set_visible(False)
	ax[2].get_yaxis().set_visible(False)
	'''
	for i, (delta, offset_idx_array, values, contrib) in enumerate(dbg_array):
		print(f'{offset_idx_array} {values} {contrib}')
	
	for i, (delta, offset_idx_array, values, contrib) in enumerate(dbg_array):
		ax[3+2*i].imshow(contrib[None,:], interpolation='nearest', origin='lower')
		ax[3+2*i].set_title(f'{delta[0]}, {delta[1]}')
		ax[3+2*i].get_xaxis().set_visible(False)
		ax[3+2*i].get_yaxis().set_visible(False)
		ax[3+2*i+1].imshow(values[None,:], interpolation='nearest', vmin=np.min(a_new), vmax=np.max(a_new), origin='lower')
		ax[3+2*i+1].set_title(f'{delta[0]}, {delta[1]}')
		ax[3+2*i+1].get_xaxis().set_visible(False)
		ax[3+2*i+1].get_yaxis().set_visible(False)
		_lgr.debug(f'{contrib=}')
		_lgr.debug(f'{values=}')
	'''
	plt.show()
	"""
	
	

	return a



def interp(a : np.ndarray[S[N],T], m : np.ndarray[S[N],bool], fix_edges : bool = True, **kwargs) -> np.ndarray[S[N],T]:
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
