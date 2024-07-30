from aopp_deconv_tool.psf_model.dependency_injector.params_and_psf_model_di import *
from aopp_deconv_tool.psf_model.gaussian_psf_model import GaussianPSFModel


class GaussianPSFModelDependencyInjector(ParamsAndPsfModelDependencyInjector):
	"""
	Models the PSF as a 2d gaussian with mean (`x`,`y`), standard deviation `sigma` an offset `const` from zero, and a multiplication factor `factor`
	"""
	
	def __init__(self, psf_data):
		
		super().__init__(psf_data)
		
		self._params = PriorParamSet(
			PriorParam(
				'x',
				(0, psf_data.shape[0]),
				True,
				psf_data.shape[0]//2,
				"Position of gaussian mean on x-axis"
			),
			PriorParam(
				'y',
				(0, psf_data.shape[1]),
				True,
				psf_data.shape[1]//2,
				"Position of gaussian mean on y-axis"
			),
			PriorParam(
				'sigma',
				(0, np.sum([x**2 for x in psf_data.shape])),
				False,
				5,
				"Standard deviation of gaussian in x and y axis"
			),
			PriorParam(
				'const',
				(0, 1),
				False,
				0,
				"Constant added to gaussian"
			),
			PriorParam(
				'factor',
				(0, 2),
				False,
				1,
				"Scaling factor"
			)
		)
		
		self._psf_model = GaussianPSFModel(psf_data.shape, float)
	
	
	
	def get_parameters(self):
		return self._params
	
	def get_psf_model_flattened_callable(self): 
		def psf_model_flattened_callable(x, y, sigma, const, factor):
			return self._psf_model(np.array([x,y]), np.array([sigma,sigma]), const)*factor
		return psf_model_flattened_callable
	
	def get_psf_result_postprocessor(self): 
		return None


