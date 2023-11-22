
import numpy as np

from geometry.shape import Circle
from optics.geometric.optical_component import OpticalComponentSet, Aperture, Obstruction, Refractor
from optics.telescope_model import optical_transfer_function_of_optical_component_set

class VLT:
	n_actuators = 24
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
	
	def __init__(self,
			obs_shape,
			obs_scale,
			ref_wavelength
		):
		self.obs_shape = obs_shape
		self.obs_scale = obs_scale
		self.ref_wavelength = ref_wavelength
	
	
	@classmethod
	def muse(cls,
		  obs_shape = (201,201),
		  obs_pixel_size = 0.0125 / (60*60) *np.pi/180,
		  ref_wavelength = 5E-7
		):
		obs_scale = np.array(obs_shape)*np.array(obs_pixel_size)/ref_wavelength
		return cls(obs_shape, obs_scale, ref_wavelength)
	
	
	def optical_transfer_function(self, 
			expansion_factor : float, 
			supersample_factor : float
		):
		self.expansion_factor = expansion_factor
		self.supersample_factor= supersample_factor
		return optical_transfer_function_of_optical_component_set(self.obs_shape, expansion_factor, supersample_factor, self.ocs, self.obs_scale)

