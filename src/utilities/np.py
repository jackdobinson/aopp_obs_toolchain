#!/usr/bin/env python3
"""
Contains utility functions to help with numpy routines
"""
import sys
sys.path = sys.path[1:] + sys.path[:1] # move current directory to end of search path.

import utilities.logging_setup
logging, _lgr = utilities.logging_setup.getLoggers(__name__, 'DEBUG')
#logging, _lgr = utilities.logging_setup.getLoggers(__name__, 'INFO')

import numpy as np
import typing
import matplotlib.pyplot as plt


class ShiftedAxes:
	def __init__(self, array, axes_pos, axes_dest=None):
		if type(axes_pos) is int:
			axes_pos = (axes_pos,)
		if axes_dest is None:
			axes_dest = tuple(range(len(axes_pos)))
		self.array = array
		self.axes_pos = axes_pos 
		self.axes_dest = axes_dest
	def __enter__(self):
		return(np.moveaxis(self.array, self.axes_pos, self.axes_dest))
	def __exit__(self, etype, evalue, traceback):
		return(False) # don't swallow any exceptions


def array_axes_iterator(*arrays, axes=None, **kwargs):
	import itertools
	ungrouped_axes = [i for i in range(arrays[0].ndim) if i not in axes]
	n = np.prod([arrays[0].shape[_x] for _x in ungrouped_axes],dtype=int)
	for a in arrays[1:]:
		if n !=  np.prod([arrays[0].shape[_x] for _x in ungrouped_axes],dtype=int):
			raise ValueError('Arrays must have same ungrouped axes structure')

	flags = kwargs.pop('flags',[]) + ['external_loop','buffered']
	bufsize = kwargs.pop('buffersize',None)
	shapes = [tuple(a.shape[_x] for _x in axes) for a in arrays]
	buffersizes = [np.prod([a.shape[_x] for _x in axes],dtype=int) if bufsize is None else bufsize for a in arrays]
	iterators = [np.nditer(a, flags=flags, buffersize=_bufsize, **kwargs) for a, _bufsize in zip(arrays, buffersizes)]
	iter_flag = True

	for i in range(n):
		def _inner():
			for it, shape in zip(iterators, shapes):
				try:
					x = next(it)
				except StopIteration:
					iter_flag = False
					return
				yield(x.reshape(shape))
		yield(_inner())

def trim_array(data, signal_loss=1E-3, r_tol=1, iter_max=100):
	if signal_loss == 0:
		return(data, tuple(slice(None) for i in data.shape))
	data_idxs = idx_grid(data)
	data_sum = np.nansum(data)
	centroid = get_centroid(data, data_idxs)
	
	find_zero_func = lambda r: np.array([np.nansum(data[is_within_range_of_point(centroid, _r, data_idxs)])/data_sum - (1-signal_loss) for _r in r])
	
	r = newton_raphson(find_zero_func, x0=(0,np.nanmax(data_idxs)), x_tol=r_tol, iter_max=iter_max)[0]
	slice_lims = [(_c-r, _c+r) for _c in centroid]
	slices = [slice(int(smin if smin >0 else 0), int(smax if smax < s else s)) for (smin, smax), s in zip(slice_lims, data.shape)]
	
	return(data[tuple(slices)], tuple(slices))


def array_remove_frame(array, axes=None, frame_pixel_func=lambda px: (px==np.nan or px==0)):
	# TODO: make this configurable over axes
	axes = tuple(range(array.ndim)) if axes is None else axes
	vframe_pixel_func = np.vectorize(frame_pixel_func)
	#vframe_pixel_func = frame_pixel_func
	# find first LH vertical column that is all rejectable pixels
	x = np.array([np.zeros((len(axes),),dtype=int), np.ones((len(axes),),dtype=int)], dtype=int)
	x[1] *= np.array([array.shape[ax] for ax in axes], dtype=int)
	_lgr.DEBUG(f'{x=}')
	_lgr.DEBUG(f'{x[0]=}')

	"""
	import matplotlib.pyplot as plt
	plt.imshow(array[0])
	plt.show()
	"""

	for j, ax in enumerate(axes):
		ax_shape = array.shape[ax]
		ax_slice = [slice(None) if k != ax else 0 for k in range(array.ndim)]
		for i in range(ax_shape):
			ax_slice[ax] = i
			test_array = vframe_pixel_func(array[tuple(ax_slice)])
			if not np.all(test_array):
				break
			x[0,j] = i
			
		_lgr.DEBUG(f'{j=} {ax=} {x=} {x[j]=}')
		for i in range(ax_shape):
			ax_slice[ax] = ax_shape - (i+1)
			if not np.all(vframe_pixel_func(array[tuple(ax_slice)])):
				break
			x[1,j] = ax_shape-i

	_lgr.DEBUG(f'{x=}')
	frame_slice = []
	for k in range(array.ndim):
		frame_slice.append(slice(None) if k not in axes else slice(x[0,axes.index(k)],x[1,axes.index(k)]))
	
	"""
	plt.imshow(array[frame_slice][0])
	plt.show()
	"""
	_lgr.DEBUG(f'{frame_slice=}')
	return(array[tuple(frame_slice)], tuple(frame_slice))

def newton_raphson(obj_func, x0=None, x_tol=1E-5, y_tol=1E-3, iter_max=100, make_plots=True):
	if x0 is None:
		trial_x = np.concatenate([-np.logspace(300, 0, 30), np.logspace(0,300, 30)])
		trial_y = obj_func(trial_x)
		trial_y_nan = np.isnan(trial_y)
		diff_sign_as_neighbour = (np.sign(trial_y[:-1]) != np.sign(trial_y[1:])) & ~(trial_y_nan[:-1] | trial_y_nan[1:])
		n_zero_crossings = np.sum(diff_sign_as_neighbour)
		
		if n_zero_crossings == 0:
			raise RuntimeError('Could not automatically find zero crossing intervals, please supply "x0" argument to begin with known bounds')
			
		x0 = np.zeros((2,int(n_zero_crossings)))
		x0[0] = trial_x[:-1][diff_sign_as_neighbour]
		x0[1] = trial_x[1:][diff_sign_as_neighbour]
	if type(x0) in (tuple, list):
		x0 = np.array(x0)
	if type(x0) is np.ndarray:
		if x0.ndim == 1:
			x0 = x0[:,None] # add a single dimension
		if x0.ndim > 2:
			raise ValueError('"x0" argument must be a 1d or 2d numpy array, tuple, list, or None')
	else:
		raise TypeError('"x0" argument must be a 1d or 2d numpy array, tuple, list, or None')
	
	x = x0
	y = obj_func(x)
	
	i=0
	while (np.any(np.fabs(x[1]-x[0]) > x_tol) and np.any(np.fabs(y) > y_tol)) and (i < iter_max):
		xi = np.mean(x, axis=0)
		yi = obj_func(xi)
		
		y_matching_sign = np.sign(yi) == np.sign(y)
		if not np.all(np.logical_xor(*y_matching_sign)):
			raise RuntimeError(f'"utilities.np.newton_raphson()" could not find different sign interval at {i=} {xi=} {yi=}')
		
		y[y_matching_sign] = yi
		x[y_matching_sign] = xi
		
		i += 1
	return(np.mean(x, axis=0))
		
	
	
def is_within_range_of_point(p, r, positions):
	positions = np.mgrid[tuple([slice(x) for x in positions])] if (type(positions) in (tuple,list)) else positions
	return(np.logical_and(*((positions - p[tuple([slice(None)]+[None for x in range(p.size)])])**2 < r**2)))
	
def get_centroid(data, positions=None):
	positions = idx_grid(data) if positions is None else positions
	return(np.nansum(data*positions, axis=tuple([_i+1 for _i in range(data.ndim)]))/np.nansum(data, axis=None))

def remove_value_frame(data, value=np.nan, axes=None):
	axes = list(range(data.ndim)) if axes is None else axes
	is_value = (lambda d: np.isnan(d)) if np.isnan(value) else (lambda d: d==value)
	data_slices = []
	
	inv_axes = [tuple([_x for _x in range(data.ndim) if _x!=_y]) for _y in axes]
	#print(f'{inv_axes=}')
	for ax_n, ax_i in enumerate(inv_axes):
		#print(f'{ax_n=} {ax_i=}')
		mask = np.all(is_value(data), axis=ax_i)
		#print(f'{mask.shape=} {mask=}')
		# only remove edges of mask
		for i, b in enumerate(mask): 
			if not b: break
		for j, b in enumerate(mask[::-1]): 
			if not b: break
		#print(f'{i=} {data.shape[ax_n]-j=}')
		data_slices.append(slice(i,data.shape[ax_n]-j))
	#print(data_slices)
	return(data[tuple(data_slices)])
	
def with_continuous_domain(x, func):
	_x = get_integer_range_from(x)
	return(np.interp(x, _x, func(_x)))

def get_integer_range_from(x):
	return(np.linspace(np.floor(np.min(x)), np.ceil(np.max(x)), int(np.ceil(np.max(x)))+1))

def get_bin_edges(x):
	"""
	get edges around x such that edges contain x values. E.g. if o are x values
	and | are edges:
		| o | o | o | o | .... o |
	"""
	edges = np.zeros([s+1 for s in x.shape])
	edges[1:-1] = 0.5*(x[1:]+x[:-1])
	edges[0] = 2*x[0] - edges[1]
	edges[-1] = 2*x[-1] - edges[-2]
	return(edges)

def get_bin_widths(e):
	return(e[1:] - e[:-1])

def get_bin_mids(e):
	return(0.5*(e[:-1]+e[1:]))

def simple_area(x, y):
	w = get_bin_widths(get_bin_edges(x))
	return(w*y)

def slice_center(a, b):
	"""
	returns a slice of a in the shape of b about the center of a
	Accepts np.ndarray or tuple for each argument
	"""
	a_shape = np.array(a.shape if type(a) is np.ndarray else a)
	b_shape = np.array(b.shape if type(b) is np.ndarray else b)
	s_diff = a_shape - b_shape
	return(tuple([slice(d//2, s-(d//2+d%2)) for s, d in zip(a_shape,s_diff)]))

def slice_outer_mask(a, b):
	"""
	Returns a mask of array a that are not included an a
	region of shape b in the center
	"""
	a_shape = np.array(a.shape if type(a) is np.ndarray else a)
	b_shape = np.array(b.shape if type(b) is np.ndarray else b)
	x = np.ones(a_shape, dtype=bool)
	x[slice_center(a,b)] = False
	return(x)

def idx_center(a):
	"""
	Get center index of array "a", will return index before midpoint
	for even size axes
	"""
	return(tuple([(x)//2 - (x-1)%2 for x in a.shape]))

def idx_grid(a):
	return(np.mgrid[tuple([slice(0,s) for s in a.shape])])

def idxs_of_grid(a, as_tuples=True):
	idxs = np.reshape(idx_grid(a), (len(a.shape),a.size)).T
	if as_tuples:
		for idx in idxs:
			yield(tuple(idx))
	else:
		return(idxs)

def array_from_sequence(s, fill_value=0, dtype=None, order='C', like=None):
	"""
	Create an array from a sequence 's'.

	Will create a rectangular array of (hopefully) basic types, the 
	shape will be large enough to hold all the elements of "s", indices
	not covered by "s" will be set to 'fill_value'
	"""
	shape = get_iterable_shape(s)
	a = np.full(shape, fill_value=fill_value, dtype=dtype)
	fill_from(a, s)
	return(a)

def fill_from(a, s):
	"""
	Will fill the array 'a' with values from 's'. Does not change the values
	of 'a' where 'a' and 's' do not overlap.
	"""
	n = min(a.shape[0], len(s))
	if a.ndim == 1:
		a[:n] = s[:n]
		return(a)
	elif a.ndim > 1:
		for i in range(n):
			a[i,...] = fill_from(a[i],s[i])
	return


def get_iterable_shape(iterable):
	"""
	Returns the *rectangular* shape of 'iterable', this is the shape
	that is large enough to hold the maximum number of elements that
	iterable has in each dimension.
	"""
	s = []
	if hasattr(iterable, '__iter__'):
		s.append(len(iterable))
		el_s = []
		for el in iterable:
			if hasattr(el, '__iter__'):
				el_s += get_iterable_shape(el)
		if len(el_s) > 0:
			s.append(max(el_s))
	return(s)

def fractional_slice(array, start, stop):
	starti, stopi = int(start//1), int(stop//1)
	startf, stopf = start%1, stop%1
	result = array[starti:stopi].copy()
	#_lgr.DEBUG(f'{start=} {stop=} {starti=} {startf=} {stopi=} {stopf=} {result.shape=}')
	result[0] *= (1-startf)
	result[-1] *= stopf
	return(result)

def new_shape(old_shape, axes, sizes):
	if len(axes) != len(sizes): raise ValueError(f'Arguments "axes" and "sizes" must be of same size, currently {len(axes)=} {len(sizes)=}')
	n_new = axes.count(None) # if an axes_index is "None" then create a new axis.
	ns = np.ones((len(old_shape)+n_new,), dtype=int) # always 1d
	for i in range(ns.size):
		if i in axes: ns[i] = sizes[axes.index(i)]
		else: ns[i] = old_shape[i]
	return(tuple(ns[ns>=0])) # remove axes who's new sizes are negative

def rebin_to(values, old_bin_edges, new_bin_edges, combine_func=np.sum, axis=0):
	_lgr.DEBUG(f'{values.shape=} {old_bin_edges.shape=} {new_bin_edges.shape=} {combine_func=} {axis=}')
	idxs = np.interp(new_bin_edges, old_bin_edges, np.linspace(0, old_bin_edges.size-1, old_bin_edges.size))
	result = np.empty(new_shape(values.shape, (axis,), (idxs.size-1,)))
	_lgr.DEBUG(f'{result.shape=}')
	#_lgr.DEBUG(f'{idxs=}')
	with ShiftedAxes(values, axis) as _values, ShiftedAxes(result, axis) as _result:
		#_lgr.DEBUG(f'{_result.shape=}')
		for i in range(_result.shape[0]):
			#_lgr.DEBUG(f"{i=} {idxs[i]=} {idxs[i+1]=}")
			_result[i, ...] = combine_func(fractional_slice(_values, idxs[i], idxs[i+1]), axis=0)
			#_lgr.DEBUG(f"{result[i]=}")
	return(result)

def is_regridable(from_grid, to_grid):
	"""
	Can "from_grid" be regridded to "to_grid" without splitting cells into multiple points?
	"""
	grid_diff = np.diff(np.interp(to_grid, from_grid, np.linspace(0, from_grid.shape[0]-1, from_grid.shape[0]))//1) 
	_lgr.DEBUG(f'{grid_diff=}')
	_lgr.DEBUG(f'{np.all(grid_diff==0)=}')
	return(np.all(grid_diff==0))

def get_com(array):
	"""
	Zero indexed
	"""
	ax_idxs = np.mgrid[:array.ndim]
	com = np.mgrid[:array.ndim:1.0]
	idxs = idx_grid(array)
	for i in ax_idxs:
		not_i_idxs = tuple(np.concatenate((ax_idxs[:i],ax_idxs[i+1:]),axis=0))
		x = np.mgrid[0:array.shape[i]:1.0]
		y = np.nansum(array, axis=not_i_idxs)
		com[i] = np.nansum(x*y)/np.nansum(y)
		
	return(com)

def get_exclude_axes(a, axes):
	ax_idxs = np.mgrid[:a.ndim]
	#print(axes)
	for ax in ax_idxs:
		#print(ax)
		if ax not in axes:
			yield(ax)


def center_on(a, point, pad=0, axes=None):
	axes = np.mgrid[:a.ndim] if axes is None else axes
	ex_axes = tuple(get_exclude_axes(a, axes))
	#print(axes)
	#print(ex_axes)

	cidx = np.array(idx_center(a))
	diff = point.astype(int) - cidx
	a_slices = tuple(slice(None) if i in ex_axes else slice(d,None) if d>0 else slice(0,a.shape[i]+d) for i, d in enumerate(diff))
	a_adj_slices = tuple(slice(None) if i in ex_axes else slice(0,a.shape[i]-d) if d > 0 else slice(-d,None) for i, d in enumerate(diff))
	#print(a_slices)
	#print(a_adj_slices)
	temp_a = np.array(a[a_slices])
	#print(temp_a)
	a.fill(pad)
	a[a_adj_slices] = temp_a
	
	return(a)
	

