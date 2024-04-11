

from aopp_deconv_tool.optics.geometric.optical_component import OpticalComponentSet
from aopp_deconv_tool.optics.telescope_model import optical_transfer_function_of_optical_component_set


class InstrumentBase:
	"""
	Interface for instrument descriptions to be used with the PSFModel class.
	"""
	f_ao : float | None = None # adaptive optics spatial freq (if required)
	ocs : OpticalComponentSet | None = None # optical component set that describes the instrument
	
	def __init__(self,
			obs_shape,
			obs_scale,
			obs_pixel_size,
			ref_wavelength,
			expansion_factor : float, 
			supersample_factor : float
		):
		self.obs_shape = obs_shape
		self.obs_scale = obs_scale
		self.obs_pixel_size = obs_pixel_size
		self.ref_wavelength = ref_wavelength
		self.expansion_factor = expansion_factor
		self.supersample_factor= supersample_factor
	
	def optical_transfer_function(self):
		
		return optical_transfer_function_of_optical_component_set(
			self.obs_shape, 
			self.expansion_factor, 
			self.supersample_factor, 
			self.ocs, 
			self.obs_scale
		)
