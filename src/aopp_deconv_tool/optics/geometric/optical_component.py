"""
Holds a representation of optical components that can be combined together
into a set of optical components that describe an optical system.
"""

import dataclasses as dc
from typing import Literal, Any

import numpy as np

import aopp_deconv_tool.numpy_helper as nph
import aopp_deconv_tool.numpy_helper.array
from aopp_deconv_tool.numpy_helper.array import S,N
from aopp_deconv_tool.geometry.shape import GeoShape, Circle
from aopp_deconv_tool.optics.geometric.light_beam import LightBeam, LightBeamSet

def sign(a):
	return -1 if a < 0 else 1



@dc.dataclass(slots=True)
class OpticalComponent:
	"""
	Represents a component of an optical system. As I am working in "optical
	distance of the undeviated ray", there are only 3 basic things to model. 
	Apertures, obstructions, and reflector/refractors (which have the same 
	behaviour on optical distance).
	"""
	position : float = 0 # in units of "optical distance of undeviated ray"
	name : str = 'optical_component' # Name of this component
	_is_aperture_stop : bool = dc.field(default=False, init=False, repr=False, compare=False)
	
	
	def out_light_beam(self, in_light_beam : LightBeam) -> LightBeam:
		"""
		For an incident light beam `in_light_beam`, calculate the parameters of 
		a beam after passing through this component.
		"""
		raise NotImplementedError
	
	def __str__(self) -> str:
		"""
		A textual representation of this component
		"""
		raise NotImplementedError

@dc.dataclass(slots=True)
class Refractor(OpticalComponent):
	"""
	A refractor or reflector (as position is in "optical distance of undeviated
	ray")
	"""
	shape : GeoShape = dc.field(default_factory=lambda: Circle.of_radius(8)) # meters. Cross-sectional shape of the light in the path
	focal_length : float = 120 # meters. Distance in the direction of the optical path that this refractor will focus light to.
	
	def out_light_beam(self, in_light_beam : LightBeam) -> LightBeam:
		# Gradient of the high and low edges of the beam are adjusted
		# as they now focus to a new position.
		w_a_pos = in_light_beam.w_a(self.position)
		w_b_pos = in_light_beam.w_b(self.position)
		
		# Need to know if this aperture is the aperture stop or not
		print(f'{self.shape=}')
		if (sign(w_a_pos)*(w_b_pos)) <= (self.shape.radius):
			self._is_aperture_stop=False
			c_b = w_b_pos
		else:
			self._is_aperture_stop=True
			c_b = sign(w_a_pos)*self.shape.radius
			
		c_a = min(sign(w_a_pos)*(w_a_pos), self.shape.radius)*sign(w_a_pos)
		return LightBeam(
			c_a, 
			in_light_beam.m_a-c_a/self.focal_length, 
			c_b, 
			in_light_beam.m_b-c_b/self.focal_length, 
			self.position
		)
	
	def __str__(self):
		return f'{self.__class__.__name__.rsplit(".",1)[-1]}({self.name},{self.position},{str(self.shape)},focal_length={self.focal_length})'

@dc.dataclass(slots=True)
class Aperture(OpticalComponent):
	"""
	A transparent opening surrounded by opaque material. Can also model flat mirrors
	"""
	shape : GeoShape = dc.field(default_factory=lambda: Circle.of_radius(2.52/18)) # NOTE: This number is in fraction of pupil exit diameter.
	
	def out_light_beam(self, in_light_beam : LightBeam) -> LightBeam:
		# high edge of the light beam should be adjusted to exclude the occluded region
		# gradient of the high edge of the beam adjusted as it should focus in the
		# same place as before.
		w_a_pos = in_light_beam.w_a(self.position)
		w_b_pos = in_light_beam.w_b(self.position)
		if (sign(w_a_pos)*w_a_pos) >= self.shape.radius and (sign(w_b_pos)*w_b_pos) >= self.shape.radius:
			# We have no light that can get through the aperture, return null beam
			return LightBeam(np.nan,np.nan,np.nan,np.nan,self.position)
			
		w_b_pos = in_light_beam.w_b(self.position)
		w_pos = w_b_pos - w_a_pos
		
		# Need to know if this aperture is the aperture stop or not
		if sign(w_a_pos)*(w_b_pos) <= self.shape.radius:
			self._is_aperture_stop=False
			c_b = w_b_pos
		else:
			self._is_aperture_stop=True
			c_b = sign(w_a_pos)*self.shape.radius
		
		c_a = min(sign(w_a_pos)*(w_a_pos), self.shape.radius)*sign(w_a_pos)
		w_b_pos_frac = (c_b-w_a_pos) / w_pos
		w_a_pos_frac = (c_a-w_a_pos) / w_pos
		
		m_b_pos = (in_light_beam.m_b - in_light_beam.m_a)*w_b_pos_frac + in_light_beam.m_a
		m_a_pos = (in_light_beam.m_b - in_light_beam.m_a)*w_a_pos_frac + in_light_beam.m_a
		
		lb= LightBeam(
			c_a,
			m_a_pos,
			c_b, 
			m_b_pos,
			self.position
		)
		print(f'{lb=}')
		return lb
	
	def __str__(self):
		return f'{self.__class__.__name__.rsplit(".",1)[-1]}({self.name},{self.position},{str(self.shape)})'


@dc.dataclass(slots=True)
class Obstruction(OpticalComponent):
	"""
	Opaque material surrounded by transparent material. Used to model e.g. the back of secondary mirrors.
	"""
	shape : GeoShape = dc.field(default_factory=lambda: Circle.of_radius(2.52/18)) # NOTE: This number is in fraction of pupil exit diameter.
	
	def out_light_beam(self, in_light_beam : LightBeam) -> LightBeam:
		# low edge of beam must be adjusted to exclude the occluded part,
		# the gradient of the low edge of the beam should be adjusted to
		# ensure it still focuses at the same point as before
		w_a_pos = in_light_beam.w_a(self.position)
		w_b_pos = in_light_beam.w_b(self.position)
		if (sign(w_b_pos)*w_a_pos) <= self.shape.radius and (sign(w_b_pos)*w_b_pos) <= self.shape.radius:
			# We have no light that can get through the aperture, return null beam
			return LightBeam(np.nan,np.nan,np.nan,np.nan,self.position)
			
		w_b_pos = in_light_beam.w_b(self.position)
		w_pos = w_b_pos - w_a_pos
		
		c_b = max(sign(w_b_pos)*(w_b_pos), self.shape.radius)*sign(w_b_pos)
		c_a = max(sign(w_b_pos)*(w_a_pos), self.shape.radius)*sign(w_b_pos)
		w_b_pos_frac = (c_b-w_a_pos) / w_pos
		w_a_pos_frac = (c_a-w_a_pos) / w_pos
		
		m_b_pos = (in_light_beam.m_b - in_light_beam.m_a)*w_b_pos_frac + in_light_beam.m_a
		m_a_pos = (in_light_beam.m_b - in_light_beam.m_a)*w_a_pos_frac + in_light_beam.m_a
		
		return LightBeam(
			c_a, 
			m_a_pos,
			c_b,
			m_b_pos,
			self.position
		)
	
	def __str__(self):
		return f'{self.__class__.__name__.rsplit(".",1)[-1]}({self.name},{self.position},{str(self.shape)})'



@dc.dataclass(slots=True)
class OpticalComponentSet:
	"""
	A set of optical components that make up an optical system. The components 
	are placed in order of the "optical distance of the undeviated ray". I.e.
	an [aperture]->[mirror] system, will be modelled as 
	[aperture]->[refractor]->[aperture] as the undeviated ray encounters the 
	[aperture] twice. This also enables refractors and reflectors to be modelled
	as the same thing.
	"""
	_optical_path : list[OpticalComponent] = dc.field(default_factory=list, init=False, repr=False, compare=False) # the ordered list of components in the optical path, position is in terms of "optical distance of undeviated ray"
	_aperture_stop_idx : int | None = None # index of the component that is the aperture stop of the system. If None, then aperture stop has not been determined
	
	_lbs : Any = dc.field(default=None, init=False, repr=False, compare=False)

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
		"""
		When plotting all light beams in a system, want to have positions of 
		beam envelope before and after each optical component.
		"""
		x = np.ndarray((2*len(self)+1,))
		xmax = -np.inf
		for i, oc in enumerate(self._optical_path):
			_xmax = oc.position + oc.focal_length if hasattr(oc,'focal_length') else 0
			if xmax < _xmax : xmax = _xmax
			x[2*i] = oc.position-delta
			x[2*i+1] = oc.position+delta
		x[-1] = xmax
		return x
			
	
	def calc_full_light_beam(self):
		if self._lbs is None:
			self._lbs = self.get_light_beam(LightBeam(0,0,np.inf,0,-1))
		
	
	
	def get_pupil_function_scale(self, expansion_factor):
		self.calc_full_light_beam()
		lb = self._lbs[-1]
		w_max = max(abs(lb.c_a),abs(lb.c_b)) # this is a radius
		scale = ((2*w_max)*expansion_factor, (2*w_max)*expansion_factor) # need a diameter
		return scale
	
	def pupil_function(self, 
			shape=(101,101), 
			scale=None,
			expansion_factor = 1.1,
			supersample_factor = 1
		):
		"""
		Calculates the pupil function at the last optical component.
		
		Returned scale is in units of length (normally meters)
		"""
		
		
		self.calc_full_light_beam()
		# get the final light beam
		lb = self._lbs[-1]
		print(f'{lb=}')
		pf_pos = lb.o
		if scale is None:
			scale = self.get_pupil_function_scale(expansion_factor)
		else:
			scale = tuple(s*expansion_factor for s in scale)
		print(f'{scale=}')
		
		# assume cylindrically symmetric
		center_offsets = nph.array.offsets_from_point(
			tuple(int(s*expansion_factor*supersample_factor) for s in shape),
			None, 
			np.array(scale,dtype=float)
		)
		print(f'{center_offsets}')
		pf_rad = np.sqrt(np.sum(center_offsets**2, axis=0))
		
		
		lb = self._lbs.get_light_beam_at_position(pf_pos)	
		w_a, w_b = lb(pf_pos) if lb is not None else (np.nan, np.nan)
		
		print(f'{w_a=} {w_b=}')
		pf_val = np.zeros_like(pf_rad, dtype=float)
		pf_val[pf_rad < (w_b)] = 1
		pf_val[pf_rad < (w_a)] = 0
		print(pf_rad)
		print(pf_val)
		return pf_val
	
	def psf(self, 
			shape=(101,101), 
			scale = None,
			expansion_factor = 1.1,
			supersample_factor = 1
		):
		"""
		Calculates the PSF.
		
		Returned scale is in radians/wavelength, multiply by light wavelength
		to get the PSF in radians.
		"""
		# NOTE: Not accounting for wavelength of light in working out the PSF
		scale, pf = self.pupil_function(self, shape, scale, expansion_factor, supersample_factor)
		pf_fft = np.fft.fftshift(np.fft.fftn(pf))
		psf = (np.conj(pf_fft)*pf_fft)
		return scale, psf
	
	def otf(self, 
			shape=(101,101), 
			scale=None,
			expansion_factor = 1.1,
			supersample_factor = 1
		):
		"""
		Gets the optical transfer function.
		
		Returned scale is in UNKNOWN AT THIS TIME
		"""
		scale, psf = self.psf(shape, scale, expansion_factor, supersample_factor)
		otf = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(psf)))
		return scale, otf
	
	def _component_name_present(self, name):
		"""
		returns True if a component of `name` is present in the set
		"""
		for oc in self._optical_path:
			if oc.name == name:
				return True
		return False
	
	
	def _do_insert(self, idx : int, oc : OpticalComponent):
		"""
		Perform insertion of optical component `oc` at positoin `idx`
		"""
		self._optical_path.insert(idx, oc)
	
	def insert(self, oc : OpticalComponent):
		"""
		Insert an optical component `oc` based upon the value of `oc.position`.
		"""
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
	
	def get_component_before_pos(self, pos) -> OpticalComponent:
		"""
		For some `pos` (units are "optical distance of undeviated ray"), find the
		component that is before this position.
		"""
		prev = None
		for oc in self:
			if oc.position < pos:
				prev = oc
			else:
				break
		if prev is not None:
			return prev
		raise RuntimeError(f'No optical components before position {pos}, first component is at position {self._optical_path[0]}')
	
	def get_component_index_before_pos(self, pos) -> int:
		"""
		For some `pos` (units are "optical distance of undeviated ray"), find the
		index of the component that is before this position.
		"""
		prev = None
		for i, oc in enumerate(self):
			if oc.position < pos:
				prev = i
			else:
				break
		if prev is not None:
			return prev
		raise RuntimeError(f'No optical components before position {pos}, first component is at position {self._optical_path[0]}')
	
	def get_components_by_class(self, aclass):
		"""
		Return all components in this set that are of `aclass`
		"""
		for oc in enumerate(self._optical_path):
			if isinstance(oc, aclass):
				yield oc
	
	def __len__(self):
		"""
		Return the number of components in this set
		"""
		return len(self._optical_path)
	
	def __iter__(self):
		"""
		Get an iterator over the components in this set
		"""
		return iter(self._optical_path)
	
	def __getitem__(self, idx : str | int):
		"""
		Get a component in this set by either index, or name.
		"""
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
		"""
		Get the set of light beams that result from an incident light beam 
		`in_light_beam`. This set of light beams respresents the geometric 
		optics approximation of light's behaviour in the system.
		"""
		lb_list = [in_light_beam]
		for i, oc in enumerate(self._optical_path):
			lb_list.append(oc.out_light_beam(lb_list[i]))
			
			# Later aperture stops in the system replace earlier ones.
			if oc._is_aperture_stop:
				self._aperture_stop_idx = i
		return LightBeamSet(lb_list)

	def get_aperture_stop(self):
		if self._aperture_stop_idx is None:
			raise RuntimeError("Aperture stop has not been determined yet, run method 'self.get_light_beam(...)' first")
		return self._optical_path[self._aperture_stop_idx]

 
