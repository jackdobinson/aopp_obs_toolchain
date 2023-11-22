"""
Classes for dealing with a beam of light. I.e. a bundle of rays enveloped by 2
straight lines
"""


import dataclasses as dc
import numpy as np




# TODO: Move this to better location
@dc.dataclass(slots=True)
class LightBeam:
	"""
	A beam of light defined by the two lines that limit the extent of the beam,
	see (FIG 1). The beam behaves according to geometric optics. Beam position
	is defined in terms of optica path distance "o". Therefore a beam never goes
	"backwards".
	
	w_a = c_a + o*m_a
	w_b = c_b + o*m_b
	
	w_a - The line that defines the beams "inner" edge, closest to the optical axis
	w_b - The line that defines the beams "outer" edge, farthest from the optical axis
	
	
	
	
	
	FIG 1 ######################################################################
	
	      |......................... w_b - outer edge of light beam
	     /|
	    / |
	   /  |       _________________ w_a - iner edge of light beam
	  /   |      X
	 /    |      X
	*-----|------X----------------- optical axis (o)
	
	             ^
	             Obstruction
	      ^
	      Lens
	^
	Light Source
	
	############################################################################
	
	
	"""
	c_a : float = 0 # y intercept of inner edge
	m_a : float = 0 # gradient of inner edge
	c_b : float = 1 # y intercept of outer edge
	m_b : float = 0 # gradient of outer edge
	o : float = 0 # starting point (on optical axis) of this light beam
	
	def w_a(self, optical_position):
		"""
		The width (distance from the undeviated ray/optical axis) of the inner edge of the beam envelope
		"""
		return self.c_a + self.m_a*(optical_position-self.o)
	
	def w_b(self, optical_position):
		"""
		The width (distance from the undeviated ray/optical axis) of the outer edge of the beam envelope
		"""
		return self.c_b + self.m_b*(optical_position-self.o)
	
	def is_null(self) -> bool:
		"""
		Returns TRUE if the beam is null. I.e. represents no light in a region.
		"""
		return any(np.isnan(x) for x in (self.c_a, self.m_a, self.c_b, self.c_b, self.o))
	
	def __call__(self, optical_position):
		"""
		Calculate the beam envelope at `optical_position`
		"""
		return self.w_a(optical_position), self.w_b(optical_position)
	
	def get_focal_length(self) -> float:
		"""
		For a beam, calculate the focal length that it's angle w.r.t the 
		undeviated ray/optical axis would correspond to.
		"""
		f_a =  self.c_a / self.m_a
		f_b = self.c_b / self.m_b
		assert f_a - f_b < 1E-6, f"Focal lengths of inner and outer edge of beam need to match to within tolerance {f_a=} {f_b=}"
		return f_a


@dc.dataclass(slots=True)
class LightBeamSet:
	"""
	A set of light beams that are present in an optical system. Each beam exists
	between two components (or before the first and after the final components).
	
	Note, a light beam set is defined in optical distance "o", like a beam so the
	members of a beam set are monotonically increasing in "o".
	"""
	light_beams : list[LightBeam] = dc.field(default_factory=list)
	
	def get_optical_position_range(self):
		"""
		Get the range of optical positions covered by the beams in the set
		"""
		vmin = np.inf
		vmax = -np.inf
		for lb in self.light_beams:
			if lb.o < vmin: vmin = lb.o
			if lb.o > vmax: vmax = lb.o
		return vmin, vmax
	
	def get_light_beam_at_position(self, optical_position) -> LightBeam:
		"""
		For some `optical_position`, return the beam in the set that is at that
		position.
		"""
		if optical_position <= self.light_beams[0].o:
			return None
			
		for i, lb in enumerate(self.light_beams):
			if lb.o >= optical_position:
				return self.light_beams[i-1]
		return self.light_beams[-1]
	
	def __getitem__(self, idx):
		"""
		Return the `idx`-th beam from the start of the beam set.
		"""
		return self.light_beams[idx]
	
	def __call__(self, optical_position):
		"""
		Get w_a, w_b for light beam that is part of the set at `optical_position`
		"""
		# Assume that list is ordered by optical distance at this point
		lb = self.get_light_beam_at_position(optical_position)
		if lb is None:
			return np.nan, np.nan
		return lb(optical_position)
 
