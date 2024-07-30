

from aopp_deconv_tool.psf_model.dependency_injector.params_and_psf_model_di import *#ParamsAndPsfModelDependencyInjector
from aopp_deconv_tool.psf_model.radial_psf_model import RadialPSFModel


class RadialPSFModelDependencyInjector(ParamsAndPsfModelDependencyInjector):
	"""
	Models the PSF as a radial histogram with `nbins` from point (`x`,`y`)
	"""
	
	def __init__(self, psf_data):
		
		super().__init__(psf_data)
		
		self._params = PriorParamSet(
			PriorParam(
				'x',
				(0, psf_data.shape[0]),
				False,
				psf_data.shape[0]//2,
				"centre point of radial histogram on x-axis"
			),
			PriorParam(
				'y',
				(0, psf_data.shape[1]),
				False,
				psf_data.shape[1]//2,
				"centre point of radial histogram on y-axis"
			),
			PriorParam(
				'nbins',
				(0, np.inf),
				True,
				50,
				"Number of bins in radial histogram"
			)
		)
		
		self._psf_model = RadialPSFModel(
			psf_data
		)
		
	
	def get_parameters(self):
		return self._params
	
	def get_psf_model_flattened_callable(self): 
		return self._psf_model
	
	def get_psf_result_postprocessor(self): 
		def psf_result_postprocessor(params, psf_model_flattened_callable, fitted_vars, consts):
			params.apply_to_callable(
				psf_model_flattened_callable, 
				fitted_vars,
				consts
			)
			return psf_model_flattened_callable.centreed_result
			
		return psf_result_postprocessor