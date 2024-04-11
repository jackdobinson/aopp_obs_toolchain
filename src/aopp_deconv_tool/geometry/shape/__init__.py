"""
Shapes for use with geometry package
"""
from typing import Callable
import dataclasses as dc

import numpy as np

import aopp_deconv_tool.geometry as geo

@dc.dataclass(slots=True)
class GeoShape:
	"""
	Represents a shape that has an inside and outside
	"""
	metric : Callable[[np.ndarray,np.ndarray],float] = dc.field(default=geo.euclidean_metric, init=False)
	
	def __contains__(self, p : np.ndarray) -> bool:
		raise NotImplementedError


@dc.dataclass(slots=True)
class PolyShape(GeoShape):
	"""
	Represents a shape that can have a position in space
	"""
	center : np.ndarray = dc.field(default_factory = lambda : np.zeros((2,)))


@dc.dataclass(slots=True)
class Circle(PolyShape):
	radius : float = 1
	
	@classmethod
	def of_radius(cls, radius=1):
		return cls(np.zeros((2,)), radius)
	
	def __contains__(self, p : np.ndarray) -> bool:
		return self.metric(p,self.center) <= self.radius
	
	@property
	def diameter(self) -> float:
		return 2.0*self.radius
	
	def __str__(self):
		return f'{self.__class__.__name__.rsplit(".",1)[-1]}({self.radius})'

@dc.dataclass(slots=True)
class CompoundShape(GeoShape):
	_members : list[GeoShape] = dc.field(default_factory=list)
	_center : np.ndarray = dc.field(default_factory = lambda : np.zeros((2,)))
	
	@property
	def center(self) -> np.ndarray:
		return self._center
	
	@center.setter
	def center(self, value:np.ndarray) -> None:
		diff = value - self._center
		for shape in self._members:
			shape.center += diff
	
	def __add__(self, shape : GeoShape):
		if len(self._members) == 0:
			self.center = np.array(shape.center)
		self._members.append(shape)
		return self
	
	def __contains__(self, p: np.ndarray) -> bool:
		for shape in self._members:
			if p in shape: return True
		return False
