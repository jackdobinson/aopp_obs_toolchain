"""
Helper functions for axes operations on numpy arrays.

Axes referrs to the indexed dimensions of a numpy array. I.e. `a.ndim` is the number
of dimensions in a, and also the number of axes. `a.shape` is the length of each
axis.
"""
import typing
import numpy as np
from contextlib import contextmanager

import aopp_deconv_tool.cfg.logs
_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'WARN')



class AxesOrdering:
	"""
	A class that can shift between numpy, fits, and fotran axes ordering
	"""
	__slots__ = (
		'_idx',
		'_n'
	)
	def __init__(self, 
			idx : int, 
 			n : int, 
			ordering : typing.Literal['numpy','fits','fortran']
		) -> None:
		self.n = n
		setattr(self, ordering, idx)
		return

	@classmethod
	def range(cls, start, stop=None, step=1):
		# make interface work like "range()" command
		if stop is None:
			stop = start
			start = 0
		return(cls(x, stop-start, 'numpy') for x in range(start,stop,step))

	@property
	def n(self) -> int:
		return(self._n)
	@n.setter
	def n(self, value: int) -> None:
		self._n = value
		return

	# numpy representation is the internal one, so just return attribute
	@property
	def numpy(self) -> int:
		return(self._idx)
	@numpy.setter
	def numpy(self,value:int)->None:
		self._idx = value
		return

	@property
	def fits(self)->int:
		if isinstance(self._idx, typing.Sequence):
			return(type(self._idx)([self._n - _i for _i in self._idx]))
		return(self._n - self._idx)
		
	@fits.setter
	def fits(self,value:int)->None:
		# convert to numpy representation and store
		if isinstance(value, typing.Sequence):
			self._idx = type(value)([self._n - _v for _v in value])
		else:
			self._idx = self._n - value
		return

	@property
	def fortran(self)->int:
		if isinstance(self._idx, typing.Sequence):
			return(type(self._idx)([1+_i for _i in self._idx]))
		return(1+self._idx)
		
	@fortran.setter
	def fortran(self, value:int)->None:
		if isinstance(value, typing.Sequence):
			self._idx = type(value)([_v-1 for _v in value])
		else:
			self._idx = value-1
		return
# END class AxesOrdering





@contextmanager
def to_end(a : np.ndarray, axes : tuple[int,...]):
	"""
	Reorders the axes of `a` so that the axes specified in the `axes` tuple are 
	at the end (rhs) of the ordering for purposes of slicing and indexing.
	"""
	axes_dest = tuple(a.ndim - 1 -i for i in range(len(axes)))[::-1]
	
	yield np.moveaxis(a, axes, axes_dest)
	
	# as np.moveaxis returns a view of `a` we don't have to clean up anything

@contextmanager
def to_start(a : np.ndarray, axes : tuple[int,...]):
	"""
	Reorders the axes of 'a' so the axes specified in the 'axes' tuple are
	at the start (lhs) of the ordering for slicing and indexing
	"""
	axes_dest = tuple(range(len(axes)))
	
	yield np.moveaxis(a, axes, axes_dest), axes_dest


def iter_axes_group(
		a : np.ndarray,
		axes : tuple[int,...],
	):
	"""
	Iterate over a group of axes
	"""
	with to_end(a, axes) as b:
		n = b.ndim - len(axes)
		shape = (*b.shape[:-len(axes)],1)
		size = np.product(shape)
		strides = tuple(np.cumprod(shape[::-1])[:-1][::-1])
		steps = [0]*n
		#print(f'{n=} {strides=} {steps=}')
		for i in range(size):
			for j, stride in enumerate(strides):
				#print(f'{i=}')
				step = i//stride
				steps[j] = step
				i -= step*stride
				#print(f'{j=} {step=} {steps=} {i=}')
			yield (tuple(steps), b)
				


def reverse(a : np.ndarray):
	"""
	Reverse the axes of `a`
	"""
	axes = tuple(range(a.ndim))
	return np.moveaxis(a, axes, axes[::-1])

def not_in(a : np.ndarray, group : tuple[int,...]) -> tuple[int,...]:
	"""
	Returns the axes of `a` that are not in `group`
	"""
	return tuple(i for i in range(a.ndim) if i not in group)

def merge(a : np.ndarray, axes : tuple[int,...], axis : int = 0):
	"""
	Merge a set of axes into a single axis.
	
	a : np.ndarray
		Numpy array to group axes of
	axes : tuple[int:...]
		axes to group together
	axis : int
		location of the final grouped axis
	"""
	not_merge_shape = (a.shape[i] for i in range(a.ndim) if i not in axes)
	
	return np.moveaxis(np.moveaxis(a, axes, tuple(range(len(axes)))).reshape(-1, *not_merge_shape), 0, axis)



