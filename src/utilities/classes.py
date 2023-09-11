#!/usr/bin/python3
"""
File containing helper classes that I can use for useful behaviour and
as building blocks of other classes
"""
import utilities.logging_setup
logging, _lgr = utilities.logging_setup.getLoggers(__name__, 'INFO')


import collections
import collections.abc
import typing
import types

class Next:
	"""
	Context manager wrapper around the next(iter) function, to be used in
	conjunction 
	"""
	slots=('el',)
	def __init__(self, iterator : iter):
		self.el = next(iterator)
	def __enter__(self):
		return(self.el)
	def __exit__(self, etype : type, evalue : Exception, traceback : types.TracebackType):
		return(False)

class Alias:
	"""
	Context manager wrapper around a variable to temporarily change it's name
	"""
	slots=('ptr',)
	def __init__(self, var : typing.Any):
		self.ptr = Pointer(var)
	def __enter__(self):
		return(self.ptr.val)
	def __exit__(self, etype : type, evalue : Exception, traceback : types.TracebackType):
		return(False)

# TODO: ensure correct behaviour when using these slices to index objects
class Slice:
	"""
	Works the same way as numpy slicing but keeps the dimensionality of the
	return array constant. Use [None] to increase dimensions and np.squeeze()
	to remove singleton dimensions, or np.reshape() to have more control.
	"""
	_slots_ = tuple('_target',)

	@classmethod
	def add_args(cls, parser, prefix='', defaults={}):
		parser.add_argument(
			f'--{prefix}slices', 
			nargs='+', 
			type=str, 
			action='extend', 
			help="""\
				A set of slices that detail how to split up the data, should be
				of the format:
					[slice|int|tuple, slice|int|tuple, ...]
				where:
					slice
						A "x:y" style slice.
					int
						An integer that can index the data array
					tuple
						A tuple of either slices or ints. The sliced array
						will have each of the specified slices and indices
						stacked together. E.g. an array sliced with the slice-set
						"[([10:15],[34:40],7)]" will have a size of 5+6+1 = 12 
						in the first dimension.
				""",
			default=defaults.get('slices',['[:]']))
		return(parser)

	@classmethod
	def _coerce_to_slice_tuple(cls, key : typing.Union[slice, int, None, typing.Sequence], depth : int = 0):
		if type(key) is slice:
			_lgr.DEBUG(f'key is slice = {key}')
			new_key = key
		elif type(key) is int:
			_lgr.DEBUG(f'key is int = {key}')
			new_key = slice(key,key+1)
		elif key is None:
			_lgr.DEBUG(f'key is None = {key}')
			new_key = None #slice(key)
		elif issubclass(type(key),typing.Sequence):
			_lgr.DEBUG(f'key is sequence = {key}')
			new_key = []
			for el in key:
				new_key.append(cls._coerce_to_slice_tuple(el))
			return(*new_key,)
		else:
			raise TypeError(f'Unknown type {type(key)} to be coerced to a tuple of slices')
		if depth > 0:
			return((new_key,))
		else:
			return(new_key)
		
	@classmethod
	def __class_getitem__(cls, key):
		_lgr.DEBUG(f'{key=} {type(key)=}')
		transformed_key = cls._coerce_to_slice_tuple(key)
		_lgr.DEBUG(f'{transformed_key=}')
		return(transformed_key)
		return(cls._coerce_to_slice_tuple(key))

	def __init__(self, a):
		self._target = a
		return

	def __getitem__(self, key):
		if type(key) is str:
			key = key.replace('[', 'Slice[')
			#print(key)
			#print(f'self[{key}]')
			return(eval(f'self[{key}]'))
		new_key = self._coerce_to_slice_tuple(key)
		_lgr.DEBUG(f'{new_key=}')
		new_key = (new_key,) if type(new_key) is not tuple else new_key
		indices_key = self._slices_to_indices(new_key, self._target.shape)
		#print(self._target[indices_key[0],indices_key[1], indices_key[2]].shape)
		#print(self._target[(indices_key[0],)][:,indices_key[1]][:,:,indices_key[2]].shape)
		#print(indices_key)
		r = self._target[indices_key[0],]
		s = [slice(None)]
		for i in range(1,len(indices_key)):
			r = r[(*(s*i), indices_key[i])]
		return(r)
		#return(self._target[indices_key])

	@classmethod
	def _slices_to_indices(cls, ktuple, ns):
		#print(ktuple)
		#print(ns)

		indices_key = [[] for i in range(len(ns))]
		for i, x in enumerate(ktuple):
			_lgr.DEBUG(f'{i=} {x=}')
			if type(x) is tuple:
				for y in x:
					if type(y) is slice:
						_lgr.DEBUG(f'interpreting slice {y} as ints')
						y = tuple(range(*y.indices(ns[i])))
						#print("slice", i, y)
					if type(y) is tuple:
						_lgr.DEBUG(f'combining {indices_key[i]=} with {y}')
						indices_key[i] += list(y)
			else:
				_lgr.DEBUG(f"not tuple so can be slice or integer {i=} {x=}")
				indices_key[i] = x

		indices_key = tuple((tuple(x) if len(x)>0 else slice(None)) if type(x) is list else x for x in indices_key)
		_lgr.DEBUG(f'{indices_key=}')
		return(indices_key)
	
	

class Pointer:
	"""
	Lets the user set and retrieve referenced variables like a pointer does.
	Not sure exactly how to emulate pointers to normal variables yet.

	USAGE
	a_ptr = Pointer(7)
	b_ptr = Pointer(8)

	a_ptr.ref = b_ptr.ref
	print(a_ptr) # '8'

	b_ptr.val = 10
	print(a_ptr) # '10'
	"""
	__slots__=("__ref",)
	def __init__(self, val):
		self.ref = [val]
	@property
	def ref(self):
		""" Get the "reference" object, i.e. the internal list"""
		return(self.__ref)
	@ref.setter
	def ref(self, ref_list):
		""" Set the reference object, i.e. change the internal list"""
		self.__ref = ref_list
	@property
	def val(self):
		""" get the value stored in the reference, i.e. the single list element"""
		return(self.__ref[0])
	@val.setter
	def val(self, val):
		""" set the value stored in the referece, i.e. the single list element"""
		self.__ref[0] = val
	
	# END CLASS Pointer	
		
class IndexableDict(collections.UserList):
	"""
	A dictionary that you can also access the elements of via indicies
	"""
	__slots__ = ('_dict')
	def __init__(self, data=None):
		self.data = []
		self._dict = dict()
		if data is None:
			pass
		elif isinstance(data, collections.abc.Sequence):
			self.data = list(data)
			for i in range(len(self.data)):
				self._dict[str(i)] = i
		elif isinstance(data, collections.abc.Mapping):
			keys, values = tuple(zip(*data.items()))
			data = list(values)
			for k, i in enumerate(keys):
				self._dict[k] = i
		else:
			raise TypeError('IndexableDict can only be created with a list or a dict')
		return
	
	def __getitem__(self, idx):
		if isinstance(idx, slice) or isinstance(idx, int):
			return(self.data[idx])
		elif isinstance(idx, collections.abc.Hashable):
			return(self.data[self._dict[idx]])
	
	def __setitem__(self, idx, value):
		if isinstance(idx, slice) or isinstance(idx, int):
			self.data[idx] = value
		elif isinstance(idx, collections.abc.Hashable):
			self._dict[idx] = self._dict.get(idx,len(self.data))
			if self._dict[idx] < len(self.data):
				self.data[self._dict[idx]] = value
			else:
				self.data.append(value)
		else:
			raise TypeError(f'Cannot set IndexableDict with an index of type {type(idx)}')
			
	def set_idx_key(self, idx, key):
		self._dict[key] = idx
	def get_idx_key(self, idx):
		for k, v in self._dict.items():
			if idx == v:
				return(k)
		raise IndexError(f'Index {idx} not present in key-index dictionary of IndexableDict')
	def keys(self):
		return(self._dict.keys())
	def items(self):
		return(zip(self.keys(),self.data))
	def values(self):
		return(self.data)
	
	# END CLASS IndexableDict



if __name__=='__main__':
	import numpy as np
	_lgr.setLevel("DEBUG") # always debug when testing...

	a = np.random.random((100,30,40))
	print(a[Slice[30:50],Slice[1],Slice[2]].shape)

	print(Slice(a)[(Slice[30:50], Slice[70:80]), 1, 2].shape)

	print(Slice(a)['([30:50],[70:85]), 1:6, (5,6,7)'].shape)

