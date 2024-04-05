








from typing import TypeVar, TypeVarTuple, ParamSpec, Generic, Callable, Any, NewType, Protocol

import numpy as np

from optimise_compat import PriorParam, PriorParamSet


T = TypeVar('T')
Ts = TypeVarTuple('Ts')

# Integers that may or may not be equal
type N = TypeVar('N',bound=int)
type M = TypeVar('M',bound=int)

# A length N tuple of integers
type S[N] = GenericAlias(tuple, (int,)*N)






# Need to have some way to describe how to communicate with the dependency injection mechanism,
# this isn't the best, but it's what I could come up with. 
# `ParamsAndPsfModelDependencyInjector` is a base class that defines the interface
# I could probably do this with protocols, but I think it would be harder to communicate my intent

# We don't know the shape of the PSF data, but it must be a have two integer values
type Ts_PSF_Data_Array_Shape = S[Literal[2]]

# we require that this argument set is compatible with parameters specified in a 'PriorParamSet' instance
type P_ArgumentsLikePriorParamSet = ParamSpec('ArgumentsLikePriorParamSet')

# PSF data is a numpy array of some specified shape and type 'T'
type T_PSF_Data_NumpyArray = np.ndarray[Ts_PSF_Data_Array_Shape, T] 

# We want the callable we are given to accept parameters in a way that is compatible with 'PriorParamSet', and return a numpy array that is like our PSF Data
type T_PSF_Model_Flattened_Callable = Callable[P_ArgumentsLikePriorParamSet, T_PSF_Data_NumpyArray] 

# Fitted varaibles from `psf_data_ops.fit_to_data(...)` are returned as a dictionary
type T_Fitted_Variable_Parameters = dict[str,Any] 

# Constant paramters to `psf_data_ops.fit_to_data(...)` are returned as a dictionary
type T_Constant_Parameters = dict[str,Any] 

# If we want to postprocess the fitted PSF result, we will need to know the PriorParamSet used, the callable used, the fitted variables, and the constant paramters resulting from the fit.
type P_PSF_Result_Postprocessor_Arguments = [PriorParamSet, T_PSF_Model_Flattened_Callable, T_Fitted_Variable_Parameters, T_Constant_Parameters] 

# If we preprocess the fitted PSF, we must return something that is compatible with the PSF data.
type T_PSF_Result_Postprocessor_Callable = Callable[
	P_PSF_Result_Postprocessor_Arguments, 
	T_PSF_Data_NumpyArray
]


class ParamsAndPsfModelDependencyInjector:
	def __init__(self, psf_data : T_PSF_Data_NumpyArray):
		self.psf_data = psf_data
		self._psf_model = NotImplemented
		self._params = NotImplemented # PriorParamSet()
	
	def get_psf_model_name(self):
		return self._psf_model.__class__.__name__

	def get_parameters(self) -> PriorParamSet:
		return self._params
	
	def get_psf_model_flattened_callable(self) -> T_PSF_Model_Flattened_Callable : 
		NotImplemented
	
	def get_psf_result_postprocessor(self) -> None | T_PSF_Result_Postprocessor_Callable : 
		NotImplemented


class RadialPSFModelDependencyInjector(ParamsAndPsfModelDependencyInjector):
	from radial_psf_model import RadialPSFModel
	
	def __init__(self, psf_data):
		
		super().__init__(psf_data)
		
		self._params = PriorParamSet(
			PriorParam(
				'x',
				(0, psf_data.shape[0]),
				False,
				psf_data.shape[0]//2
			),
			PriorParam(
				'y',
				(0, psf_data.shape[1]),
				False,
				psf_data.shape[1]//2
			),
			PriorParam(
				'nbins',
				(0, np.inf),
				True,
				50
			)
		)
		
		self._psf_model = RadialPSFModelDependencyInjector.RadialPSFModel(
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
			return psf_model_flattened_callable.centered_result
			
		return psf_result_postprocessor


class GaussianPSFModelDependencyInjector(ParamsAndPsfModelDependencyInjector):
	from gaussian_psf_model import GaussianPSFModel
	
	def __init__(self, psf_data):
		
		super().__init__(psf_data)
		
		self._params = PriorParamSet(
			PriorParam(
				'x',
				(0, psf_data.shape[0]),
				True,
				psf_data.shape[0]//2
			),
			PriorParam(
				'y',
				(0, psf_data.shape[1]),
				True,
				psf_data.shape[1]//2
			),
			PriorParam(
				'sigma',
				(0, np.sum([x**2 for x in psf_data.shape])),
				False,
				5
			),
			PriorParam(
				'const',
				(0, 1),
				False,
				0
			),
			PriorParam(
				'factor',
				(0, 2),
				False,
				1
			)
		)
		
		self._psf_model = GaussianPSFModelDependencyInjector.GaussianPSFModel(psf_data.shape, float)
	
	
	
	def get_parameters(self):
		return self._params
	
	def get_psf_model_flattened_callable(self): 
		def psf_model_flattened_callable(x, y, sigma, const, factor):
			return self._psf_model(np.array([x,y]), np.array([sigma,sigma]), const)*factor
		return psf_model_flattened_callable
	
	def get_psf_result_postprocessor(self): 
		return None


class TurbulencePSFModelDependencyInjector(ParamsAndPsfModelDependencyInjector):
	from turbulence_psf_model import TurbulencePSFModel, SimpleTelescope, CCDSensor
	from optics.turbulence_model import phase_psd_von_karman_turbulence as turbulence_model
	
	def __init__(self, psf_data):
		super().__init__(psf_data)
		
		self._params = PriorParamSet(
			PriorParam(
				'wavelength',
				(0, np.inf),
				True,
				750E-9
			),
			PriorParam(
				'r0',
				(0, 1),
				True,
				0.1
			),
			PriorParam(
				'turbulence_ndim',
				(0, 3),
				False,
				1.5
			),
			PriorParam(
				'L0',
				(0, 50),
				False,
				8
			)
		)
		
		self._psf_model = TurbulencePSFModelDependencyInjector.TurbulencePSFModel(
			TurbulencePSFModelDependencyInjector.SimpleTelescope(
				8, 
				200, 
				TurbulencePSFModelDependencyInjector.CCDSensor.from_shape_and_pixel_size(psf_data.shape, 2.5E-6)
			),
			TurbulencePSFModelDependencyInjector.turbulence_model
		)
		
	def get_parameters(self):
		return self._params
	
	def get_psf_model_flattened_callable(self): 
		return self._psf_model
	
	def get_psf_result_postprocessor(self): 
		return None

