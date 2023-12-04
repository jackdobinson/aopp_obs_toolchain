
import numpy as np

from instrument_model.instrument_base import InstrumentBase
from geometry.shape import Circle
from optics.geometric.optical_component import OpticalComponentSet, Aperture, Obstruction, Refractor
from optics.telescope_model import optical_transfer_function_of_optical_component_set

class VLT(InstrumentBase):
	"""
	Instrument description for very large telescope.
	"""
	n_actuators = 42
	obj_diameter = 8 # meters
	f_ao = n_actuators / (2*obj_diameter)
	primary_mirror_focal_length = 120 # meters
	primary_mirror_pos = 100 # meters
	primary_mirror_diameter = 8
	secondary_mirror_diameter_frac_of_primary = 2.52/18
	secondary_mirror_dist_from_primary =50 # meters
	secondary_mirror_diameter_meters = primary_mirror_diameter*secondary_mirror_diameter_frac_of_primary
	ocs = OpticalComponentSet.from_components([
		Aperture(
			0, 
			'objective aperture', 
			shape=Circle.of_radius(obj_diameter/2)
		), 
		Obstruction(
			primary_mirror_pos - secondary_mirror_dist_from_primary, 
			'secondary mirror back', 
			shape=Circle.of_radius(secondary_mirror_diameter_meters/2)
		), 
		Refractor(
			primary_mirror_pos, 
			'primary mirror', 
			shape=Circle.of_radius(primary_mirror_diameter/2), 
			focal_length=primary_mirror_focal_length
		),
	])
	
	
	@classmethod
	def muse(cls,
			expansion_factor,
			supersample_factor,
			obs_shape = (301,301),
			obs_pixel_size = 0.025 / (3600) *np.pi/180,
			ref_wavelength = 5E-7
		):
		"""
		Description of MUSE instrument on the VLT telescope
		"""
		
		obs_scale = np.array(obs_shape)*np.array(obs_pixel_size)/ref_wavelength
		return cls(obs_shape, obs_scale, ref_wavelength, expansion_factor, supersample_factor)


