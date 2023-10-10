"""
Holds a representation of optical components that can be combined together
into instruments.
"""

import dataclasses as dc
from typing import Literal

import numpy as np

import numpy_helper as nph
from numpy_helper import S,N
from geometry import GeoShape, Circle



@dc.dataclass(slots=True)
class OpticalComponent:
	position : float = 0
	name : str = 'optical_component'
	sense : Literal[-1] | Literal[0] | Literal[1] = 1 # reflecting (-1), blocking (0), refracting (1)
	
	_otf : np.ndarray[S[2],float] = dc.field(default=None, init=False, repr=False, compare=False) # optical transfer function
	_psd : np.ndarray[S[2],float] = dc.field(default=None, init=False, repr=False, compare=False) # power spectral distribution
	_psf : np.ndarray[S[2],float] = dc.field(default=None, init=False, repr=False, compare=False) # point spread function
	
	def on_insert(self, optical_component_set):
		# Perform this function when inserting the component into a component set.
		pass
	
	def get_otf(self):
		# Calculates the optical transfer function of the component
		raise NotImplementedError
	
	def get_psd(self):
		# Calculates the power-spectral-distribution of the component
		raise NotImplementedError
	
	def get_psf(self):
		# Calculates the point-spread-function of the component
		raise NotImplementedError


@dc.dataclass(slots=True)
class OpticalComponentSet:
	_optical_path : list[OpticalComponent] = dc.field(default=list, init=False, repr=False, compare=False) # the ordered list of components in the optical path
	
	def __post_init__(self, optical_path : list[OpticalComponent] = []):
		self._optical_path = sorted(optical_path, key=lambda oc: oc.position)
	
	def _component_name_present(self, name):
		for oc in self._optical_path:
			if oc.name == name:
				return True
		return False
	
	
	def _do_insert(self, idx : int, oc : OpticalComponent):
		self._optical_path.insert(idx, oc)
		oc.on_insert(self)
	
	def insert(self, oc : OpticalComponent):
		inserted = False
		if self._component_name_present(oc.name):
			raise RuntimeError(f'A component with the name "{oc.name}" already exists in the component set.')
			
		for i in reversed(range(len(self._optical_path))):
			if self._optical_path[i].position <= oc.position:
				self._do_insert(i+1, oc)
				inserted=True
		if not inserted:
			self._do_insert(0, oc)
	
	def get_components_by_class(self, aclass):
		for oc in enumerate(self._optical_path):
			if isinstance(oc, aclass):
				yield oc
	
	def __getitem__(self, idx : str | int):
		t_idx = type(idx)
		if t_idx == str:
			for oc in self._optical_path:
				if oc.name == idx:
					return oc
			return None
		if t_idx == int:
			return self._optical_path[idx]
		
		raise IndexError(f'Cannot index an OpticalComponentSet with an index of type {type(idx)}')




@dc.dataclass(slots=True)
class ImagePlane(OpticalComponent):
	pass

@dc.dataclass(slots=True)
class Refractor(OpticalComponent):
	shape : GeoShape = dc.field(default_factory=lambda: Circle.of_radius(8)) # meters. Cross-sectional shape of the light in the path
	focal_length : float = 120 # meters. Distance in the direction of the optical path that this refractor will focus light to.
	
	def on_insert(self, ocs: OpticalComponentSet):
		super(Refractor,self).on_insert(ocs)
		ocs.insert(ImagePlane(self.position + self.focal_length, self.name+'_image_plane'))

@dc.dataclass(slots=True)
class Aperture(OpticalComponent):
	sense : Literal[-1] | Literal[0] | Literal[1] = 0 # reflecting (-1), blocking (0), refracting (1)
	shape : GeoShape = dc.field(default_factory=lambda: Circle.of_radius(2.52/18)) # NOTE: This number is in fraction of pupil exit diameter.
