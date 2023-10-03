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





def indices_from_point(shape, point=None):
	if point is None:
		point = np.array([s//2 for s in shape]) # center
	return np.moveaxis(np.moveaxis(np.indices(shape),0,-1) - point,-1,0)

def manhattan_distance_mask(a, dist=1, index_axis=0):
	if index_axis != 0:
		np.moveaxis(a,index_axis,0)
		
	m = np.ones(a.shape[1:],bool)
	
	if type(dist) is int:
		dist = tuple([dist]*m.ndim)
	elif len(dist) < m.ndim:
		dist = tuple(*dist,*([0]*(m.ndim-len(dist))))
	
	for b, d in zip(a,dist):
		m &= np.abs(b) <= d
	m &= (np.abs(a).sum(axis=0) <= np.max(dist))
	return m

def offsets_manhattan_distance(dist=1,ndim=3):
	if type(dist) is int:
		dist = tuple([dist]*ndim)
	elif len(dist) < ndim:
		dist = tuple(*dist,*([0]*(ndim-len(dist))))
	shape = tuple(2*d + 1 for d in dist)
	return offsets_from_center_of_mask(manhattan_distance_mask(indices_from_point(shape), dist))

def manhattan_distance(shape, point=None):
	if point is None:
		point = tuple(s//2 for s in shape)
	return np.abs(np.moveaxis(np.indices(shape),0,-1) - np.array(point)).sum(axis=-1)

def offsets_from_center_of_mask(mask):
	assert all([s%2 == 1 for s in mask.shape]), "mask must have an odd shape to have a unique center"
	return np.argwhere(mask) - np.array([s//2 for s in mask.shape])

def index_array_from_mask(m : np.ndarray[[...],bool]) -> tuple[np.ndarray[int],...]:
	return np.where(m)

def index_pacman(indices, a):
	# Works like pacman, topology of a torus. Same as periodic boundarys.
	return a[tuple(i % s for i,s in zip(indices, a.shape))]

def index_periodic(indices, a):
	# Topology of a torus, same as pacman.
	return index_pacman(indices, a)

def index_const(indices, a, const=0):
	# get a mask that has the out of bounds elements, set them to a in-bound
	# index to get the values, then set the values from the out of bounds
	# indices to a constant.
	idx_out_of_bounds_mask = np.array([(i < 0) | (i >= s) for i,s in zip(indices, a.shape)],bool)
	indices[idx_out_of_bounds_mask] = 0
	oob_mask = np.array(idx_out_of_bounds_mask.sum(axis=0),bool)
	v = a[(*indices,)]
	v[oob_mask] = const
	return v

def index_reflect(indices, a):
	# if i < s, then i_reflect = -i
	# if i >=s, then i_reflect = 2*(s-1) - i
	# Use multiplication to do if statements in arrays
	return a[tuple(i + 2*(s-1)*(i>=s) + i*(-2*(~((0<=i) & (i<s)))) for i,s in zip(indices, a.shape))]

def shift_pacman(a, delta):
	"""
	Shift an array by some delta using pacman (periodic) boundary conditions.
	E.g. tiles of the same array exist to either side.
	"""
	return np.roll(a, delta, tuple(i for i in range(a.ndim)))

def shift_periodic(a,delta):
	"""
	Shift an array by some delta using periodic (pacman) boundary conditions
	E.g. tiles of the same array exist to either side.
	"""
	return shift_pacman(a,delta)

def shift_const(a, delta, const=0):
	"""
	Shift and array by some delta using constant boundary conditions
	E.g. the array is surrounded by a constant value
	"""
	# Shift as in periodic boundary conditions, then fill the "empty" region
	# with a constant value
	s = shift_pacman(a, delta)
	for i, d in enumerate(delta):
		wrap_region_t = tuple((slice(d,None) if d < 0 else slice(None,d)) if i==j else slice(None) for j, d in enumerate(delta))
		s[wrap_region_t] = const
	return s

def shift_reflect(a, delta):
	"""
	Shift an array by some delta using reflecting boundary conditions.
	E.g. the array is surrounded by reflected versions of itself.
	"""
	# Shift using pacman as normal, then set the 'wrapped' region to the data 
	# for reflected boundary conditions
	s = shift_pacman(a, delta)
	for i, d in enumerate(delta):
		_lgr.debug(f'{delta=}')
		n = a.shape[i]
		wrap_region_t = tuple((slice(d,n,1) if d < 0 else slice(0,d,1)) if i==j else slice(None) for j, d in enumerate(delta))
		reflect_region_t = tuple((slice(n-2,n+d-2,-1) if d < 0 else slice(d,0,-1)) if i==j else slice(None) for j, d in enumerate(delta))
		s[wrap_region_t] = a[reflect_region_t]
	return s


def get_index_boundary_func(name : str):
	return {	
		'pacman' : index_pacman,
		'periodic' : index_periodic,
		'const' : index_const,
		'reflect' : index_reflect
	}[name]

def mean(a : np.ndarray[[-1,-1],T], m : np.ndarray[[...],bool], window : np.ndarray[bool] | tuple[int,...] | int = 3, boundary='reflect', const : T = 0 ) -> np.ndarray[[...],T]:
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
	
	idx_array = index_array_from_mask(m)
	if type(window) is np.ndarray:
		_lgr.debug(f'{a.ndim=} {window.ndim=}')
		while window.ndim < a.ndim:
			window = window[None,...]
		_lgr.debug(f'{window=}')
		deltas = offsets_from_center_of_mask(np.array(window, bool))
	else:
		deltas = offsets_manhattan_distance(window, a.ndim)
	_lgr.debug(f'{deltas=}')
	
	n = idx_array[0].size
	contributions = np.zeros((n,),int)
	accumulator = np.zeros((n,),dtype=a.dtype)
	
	
	#dbg_array = []
	for i, delta in enumerate(deltas):
		_lgr.debug(f'calculating contributions from offset {delta} number {i}/{deltas.shape[0]}')
		offset_idx_array = (np.array(idx_array).T-delta).T
		values = index_boundary_func(offset_idx_array, a)
		_lgr.debug(f'{values=}')
		contrib = ~(np.isnan(values) | np.isinf(values))
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
