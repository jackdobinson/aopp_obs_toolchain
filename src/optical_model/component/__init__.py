"""
Holds a representation of optical components that can be combined together
into instruments.
"""

import dataclasses as dc
from typing import Literal

import numpy as np

import numpy_helper as nph
from numpy_helper.array import S,N
from geometry.shape import GeoShape, Circle
		

def sign(a):
	return -1 if a < 0 else 1

# TODO: Move this to better location
"""
@dc.dataclass(slots=True)
class LightBeam:
	h_a : float = 0
	nu_a : float = 0
	h_b : float = 1
	nu_b : float = 0
	o : float = 0
	
	def w_a(self, optical_position):
		return self.h_a*(1-self.nu_a*(optical_position-self.o))
	
	def w_b(self, optical_position):
		return self.h_b*(1-self.nu_b*(optical_position-self.o))
	
	def __call__(self, optical_position):
		return self.w_a(optical_position), self.w_b(optical_position)
"""

@dc.dataclass(slots=True)
class LightBeam:
	c_a : float = 0
	m_a : float = 0
	c_b : float = 1
	m_b : float = 0
	o : float = 0
	
	def w_a(self, optical_position):
		return self.c_a + self.m_a*(optical_position-self.o)
	
	def w_b(self, optical_position):
		return self.c_b + self.m_b*(optical_position-self.o)
	
	def __call__(self, optical_position):
		return self.w_a(optical_position), self.w_b(optical_position)


@dc.dataclass(slots=True)
class LightBeamSet:
	light_beams : list[LightBeam] = dc.field(default_factory=list)
	
	def get_optical_position_range(self):
		vmin = np.inf
		vmax = -np.inf
		for lb in self.light_beams:
			if lb.o < vmin: vmin = lb.o
			if lb.o > vmax: vmax = lb.o
		return vmin, vmax
	
	
	def __call__(self, optical_position):
		# Assume that list is ordered by optical distance at this point
		if optical_position <= self.light_beams[0].o:
			return np.nan, np.nan
			
		for i, lb in enumerate(self.light_beams):
			if lb.o >= optical_position:
				print(optical_position, i, lb.o)
				return self.light_beams[i-1](optical_position)
		return self.light_beams[-1](optical_position)



@dc.dataclass(slots=True)
class OpticalComponent:
	position : float = 0
	name : str = 'optical_component'
	
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
	_optical_path : list[OpticalComponent] = dc.field(default_factory=list, init=False, repr=False, compare=False) # the ordered list of components in the optical path

	@classmethod
	def from_components(cls, optical_components : list[OpticalComponent]):
		"""
		Example:
			OpticalComponentSet.from_components([
				Aperture(0, 'objective aperture', Circle.of_radius(7)), 
				Obstruction(3, 'secondary mirror back', Circle.of_radius(2)), 
				Refractor(100, 'primary mirror', Circle.of_radius(7), 120),
				Aperture(197, 'secondary mirror front', circle.of_radius(2))
			])
		"""
		ocs = cls()
		for oc in optical_components:
			print(oc)
			ocs.insert(oc)
		return ocs
	
	def get_evaluation_positions(self, delta=1E-5):
		x = np.ndarray((2*len(self)+1,))
		xmax = -np.inf
		for i, oc in enumerate(self._optical_path):
			_xmax = oc.position + oc.focal_length if hasattr(oc,'focal_length') else 0
			if xmax < _xmax : xmax = _xmax
			x[2*i] = oc.position-delta
			x[2*i+1] = oc.position+delta
		x[-1] = xmax
		return x
	
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
		
		print(f'{[x.position for x in self._optical_path]}')
		for i in range(len(self._optical_path)):
			print(i, self._optical_path[i].position, oc.position)
			if self._optical_path[i].position > oc.position:
				self._do_insert(i, oc)
				inserted=True
		if not inserted:
			self._do_insert(len(self._optical_path), oc)
	
	def get_components_by_class(self, aclass):
		for oc in enumerate(self._optical_path):
			if isinstance(oc, aclass):
				yield oc
	
	def __len__(self):
		return len(self._optical_path)
	
	def __iter__(self):
		return iter(self._optical_path)
	
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
	
	def get_light_beam(self, in_light_beam : LightBeam) -> LightBeamSet:
		lb_list = [in_light_beam]
		for i, oc in enumerate(self._optical_path):
			lb_list.append(oc.out_light_beam(lb_list[i]))
		return LightBeamSet(lb_list)


@dc.dataclass(slots=True)
class Refractor(OpticalComponent):
	shape : GeoShape = dc.field(default_factory=lambda: Circle.of_radius(8)) # meters. Cross-sectional shape of the light in the path
	focal_length : float = 120 # meters. Distance in the direction of the optical path that this refractor will focus light to.
	
	def out_light_beam(self, in_light_beam : LightBeam) -> LightBeam:
		# Gradient of the high and low edges of the beam are adjusted
		# as they now focus to a new position.
		c_a = in_light_beam.w_a(self.position)
		c_b = in_light_beam.w_b(self.position)
		return LightBeam(
			c_a, 
			-c_a/self.focal_length, 
			c_b, 
			-c_b/self.focal_length, 
			self.position
		)
	
	def __str__(self):
		return f'{self.__class__.__name__.rsplit(".",1)[-1]}({self.name},{self.position},{str(self.shape)},focal_length={self.focal_length})'

@dc.dataclass(slots=True)
class Aperture(OpticalComponent):
	shape : GeoShape = dc.field(default_factory=lambda: Circle.of_radius(2.52/18)) # NOTE: This number is in fraction of pupil exit diameter.
	
	def out_light_beam(self, in_light_beam : LightBeam) -> LightBeam:
		# high edge of the light beam should be adjusted to exclude the occluded region
		# gradient of the high edge of the beam adjusted as it should focus in the
		# same place as before.
		w_b_pos = in_light_beam.w_b(self.position)
		c_b = min(abs(w_b_pos), self.shape.radius)*sign(w_b_pos)
		if in_light_beam.m_b == 0:
			m_b=0
		else:
			m_b = c_b*in_light_beam.m_b/(in_light_beam.c_b + (self.position - in_light_beam.o)*in_light_beam.m_b)
		lb= LightBeam(
			in_light_beam.w_a(self.position), 
			in_light_beam.m_a, 
			c_b, 
			m_b,
			self.position
		)
		print(f'{lb=}')
		return lb
	
	def __str__(self):
		return f'{self.__class__.__name__.rsplit(".",1)[-1]}({self.name},{self.position},{str(self.shape)})'


@dc.dataclass(slots=True)
class Obstruction(OpticalComponent):
	shape : GeoShape = dc.field(default_factory=lambda: Circle.of_radius(2.52/18)) # NOTE: This number is in fraction of pupil exit diameter.
	
	def out_light_beam(self, in_light_beam : LightBeam) -> LightBeam:
		# low edge of beam must be adjusted to exclude the occluded part,
		# the gradient of the low edge of the beam should be adjusted to
		# ensure it still focuses at the same point as before
		w_a_pos = in_light_beam.w_a(self.position)
		c_a = max(abs(w_a_pos), self.shape.radius)*sign(w_a_pos)
		if in_light_beam.m_a == 0:
			m_a = 0
		else:
			m_a = c_a*in_light_beam.m_a/(in_light_beam.c_a + (self.position - in_light_beam.o)*in_light_beam.m_a)
		return LightBeam(
			c_a, 
			m_a,
			in_light_beam.w_b(self.position), 
			in_light_beam.m_b,
			self.position
		)
	
	def __str__(self):
		return f'{self.__class__.__name__.rsplit(".",1)[-1]}({self.name},{self.position},{str(self.shape)})'
