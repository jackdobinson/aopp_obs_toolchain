

from __future__ import annotations
from typing import TypeVar, TypeVarTuple, Type, Generic
import numpy as np

from aopp_deconv_tool.typedef import NumVar, ShapeVar


IntVar = TypeVar('IntVar', bound=int)
T = TypeVar('T')
Ts = TypeVarTuple('Ts')
S = Generic[IntVar]
N = TypeVar('N',bound=int)



class BoundingBox(Generic[N,T]):
	"""
	Class to handle the concept of a bounding box in some coordinate system
	"""
	def __init__(self, 
			corner_min : np.ndarray[S[N], T],
			corner_max : np.ndarray[S[N], T],
			dtype : Type | None = None
		):
		self.coords = np.array([corner_min, corner_max],dtype=dtype if dtype is not None else corner_min.dtype)
	
	@classmethod
	def from_min_max_tuple(cls, 
			min_max_tuple : tuple[*Ts]
		) -> BoundingBox[N,T]:
		n = int(len(min_max_tuple)//2)
		return	cls(np.array(min_max_tuple[:n]), np.array(min_max_tuple[n:]))

	@classmethod
	def from_mpl_rect(cls, 
			mpl_rect : tuple[tuple[T,T],T,T]
		) -> BoundingBox[N,T]:
		n = 2
		min_corner = np.array(mpl_rect[0])
		max_corner = min_corner + np.array(mpl_rect[1:])
		return cls(min_corner, max_corner)
	
	@classmethod
	def from_slices(cls, 
			slices : tuple[slice,...]
		) -> BoundingBox[N,T]:
		min_corner = np.array([s.start if s.start is not None else 0 for s in slices])
		max_corner = np.array([s.stop if s.stop is not None else -1 for s in slices])
		return cls(min_corner, max_corner)
		
	def to_mpl_rect(self) -> tuple[tuple[T,T],T,T]:
		return tuple(self.coords[0][::-1]), *tuple((self.coords[1]-self.coords[0])[::-1])
	
	def to_slices(self) -> tuple[slice,...]:
		return tuple(slice(*x) for x in self.coords.T)
	
	def inflate(self, factor, n) -> BoundingBox:
		center = (self.coords[1] + self.coords[0])/2
		half_extent = self.extent/2
		self.coords = np.array(((center - (factor*half_extent+n)),(center + (factor*half_extent+n))), dtype=self.coords.dtype)
		return self

	def __repr__(self) -> str:
		return repr(self.coords)
	
	@property
	def min_corner(self) -> np.ndarray[S[N],T]:
		return self.coords[0]

	@property
	def max_corner(self) -> np.ndarray[S[N],T]:
		return self.coords[1]
	
	@property
	def mpl_min_corner(self) -> np.ndarray[S[N],T]:
		return self.coords[0][::-1]

	@property
	def mpl_max_corner(self) -> np.ndarray[S[N],T]:
		return self.coords[1][::-1]
	
	@property
	def extent(self) -> np.ndarray[S[N],T]:
		return self.coords[1] - self.coords[0]
