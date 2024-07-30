
from aopp_deconv_tool.psf_model.dependency_injector.params_and_psf_model_di import *
from aopp_deconv_tool.psf_model.turbulence_psf_model import TurbulencePSFModel


class TurbulencePSFModelDependencyInjector(ParamsAndPsfModelDependencyInjector):
	"""
	Models the PSF as the result of von-karman turbulence. Assumes the PSF is at the centre of the model.
	"""
	
	def __init__(self, psf_data):
		super().__init__(psf_data)
		
		self._params = PriorParamSet(
			PriorParam(
				'wavelength',
				(0, np.inf),
				True,
				750E-9,
				"Wavelength (meters) that properties are calculated at, will be automatically set if spectral information is present in the FITS cube"
			),
			PriorParam(
				'r0',
				(0, 1),
				True,
				0.1,
				"Fried Parameter"
			),
			PriorParam(
				'turbulence_ndim',
				(0, 3),
				False,
				1.5,
				"Number of dimensions the turbulence has"
			),
			PriorParam(
				'L0',
				(0, 50),
				False,
				8,
				"Von-Karman turbulence parameter"
			)
		)
		
		self._psf_model = TurbulencePSFModel(
			SimpleTelescope(
				8, 
				200, 
				CCDSensor.from_shape_and_pixel_size(psf_data.shape, 2.5E-6)
			),
			turbulence_model
		)
		
	def get_parameters(self):
		return self._params
	
	def get_psf_model_flattened_callable(self): 
		return self._psf_model
	
	def get_psf_result_postprocessor(self): 
		return None
