"""
Dependency injectors make communicating with the routines in `psf_data_ops.py` more managable
by providing a consistent interface. As long as the dependency injector provides parameters
and functions as specified in this file, the fitting functions should work.

"""








from typing import TypeVar, TypeVarTuple, ParamSpec, Generic, Callable, Any, NewType, Protocol

import numpy as np

from aopp_deconv_tool.optimise_compat import PriorParam, PriorParamSet

from aopp_deconv_tool.gaussian_psf_model import GaussianPSFModel
from aopp_deconv_tool.radial_psf_model import RadialPSFModel
from aopp_deconv_tool.turbulence_psf_model import TurbulencePSFModel, SimpleTelescope, CCDSensor
from aopp_deconv_tool.optics.turbulence_model import phase_psd_von_karman_turbulence as turbulence_model
from aopp_deconv_tool.adaptive_optics_psf_model import PSFModel as AOInstrumentPSFModel
from aopp_deconv_tool.optics.turbulence_model import phase_psd_von_karman_turbulence
from aopp_deconv_tool.optics.adaptive_optics_model import phase_psd_fetick_2019_moffat_function
from aopp_deconv_tool.instrument_model.vlt import VLT


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

# A callable that accepts a dictionary of fitted parameter values
type T_Fitted_Parameter_Callable = Callable[[dict[str,float], ...], T_PSF_Data_NumpyArray] 




def prior_param_args_from_param_spec(
		param_name : str, 
		is_const_default : bool | None, 
		initial_value_default : float | None, 
		range_default : tuple[float,float] | None, 
		var_params : list[str] | tuple[str], 
		const_params : list[str] | tuple[str], 
		initial_values : dict[str, float], 
		range_values : dict[str, tuple[float,float]]
	) -> tuple[str, tuple[float,float], bool, float]:
	
	if (is_const_default == None) and ((param_name not in var_params) | (param_name not in const_params)):
		raise RuntimeError(f'Parameter "{param_name}" cannot be assumed to be constant or variable, it must be defined as such.')
	if (initial_value_default == None) and (param_name not in initial_values):
		raise RuntimeError(f'Parameter "{param_name}" does not have an initial value assigned and cannot use a default value')
	if (range_default == None) and (param_name not in range_values):
		raise RuntimeError(f'Parameter "{param_name}" does not have a range assigned and cannot use a default range')

	return (
		param_name,
		range_values.get(param_name, range_default),
		False if param_name in var_params else True if param_name in const_params else is_const_default,
		initial_values.get(param_name, initial_value_default)
	)



class ParamsAndPsfModelDependencyInjector:
	"""
	Subclass this and overwrite it's methods to get something that works like something in the ".../examples/psf_model_example.py" script.
	"""
	def __init__(self, psf_data : T_PSF_Data_NumpyArray):
		self.psf_data = psf_data
		self._psf_model = NotImplemented # this will be defined here in a subclass
		self._params = NotImplemented # PriorParamSet(), this will be defined here in a subclass
	
	def get_psf_model_name(self):
		"""
		Returns the name of the PSF model. Defaults to the class name.
		"""
		return self._psf_model.__class__.__name__

	def get_parameters(self) -> PriorParamSet:
		"""
		Returns the PriorParamSet that represents the parameters for the `self._psf_model`
		"""
		return self._params
	
	def get_psf_model_flattened_callable(self) -> T_PSF_Model_Flattened_Callable :
		"""
		Returns a wrapper around `self._psf_model` that accepts all of it's parameters as simple-data arguments (no lists or objects)
		"""
		NotImplemented
	
	def get_psf_result_postprocessor(self) -> None | T_PSF_Result_Postprocessor_Callable :
		"""
		Returns a callable that postprocesses the result from `self._psf_model`
		"""
		NotImplemented
	
	def get_fitted_parameters_callable(self) -> T_Fitted_Parameter_Callable:
		"""
		Return a callable that accepts a dictionary of fitted parameteter values
		"""
		def fitted_parameters_callable(fitted_params : dict[str,float]):
			return self._params.apply_to_callable(self.get_psf_model_flattened_callable(), fitted_params, self._params.consts)
		
		return fitted_parameters_callable


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
			return psf_model_flattened_callable.centered_result
			
		return psf_result_postprocessor


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
		
		self._psf_model = GaussianPSFModel(psf_data.shape, float)
	
	
	
	def get_parameters(self):
		return self._params
	
	def get_psf_model_flattened_callable(self): 
		def psf_model_flattened_callable(x, y, sigma, const, factor):
			return self._psf_model(np.array([x,y]), np.array([sigma,sigma]), const)*factor
		return psf_model_flattened_callable
	
	def get_psf_result_postprocessor(self): 
		return None


class TurbulencePSFModelDependencyInjector(ParamsAndPsfModelDependencyInjector):
	"""
	Models the PSF as the result of von-karman turbulence. Assumes the PSF is at the center of the model.
	"""
	
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



class MUSEAdaptiveOpticsPSFModelDependencyInjector(ParamsAndPsfModelDependencyInjector):
	"""
	Models the PSF using a model of the adaptive optics MUSE instrument on the VLT telescope.
	"""
	
	def __init__(self, 
			psf_data, 
			var_params : list[str] | tuple[str] = [], 
			const_params : list [str] | tuple [str] = [],
			initial_values : dict[str,float] = {},
			range_values : dict[str,tuple[float,float]] = {}
		):
		super().__init__(psf_data)
		
		instrument = VLT.muse(
			expansion_factor = 3,
			supersample_factor = 2,
			obs_shape=psf_data.shape[-2:]
		)
		
		self._psf_model = AOInstrumentPSFModel(
			instrument.optical_transfer_function(),
			phase_psd_von_karman_turbulence,
			phase_psd_fetick_2019_moffat_function,
			instrument
		)
		
		self._params = params = PriorParamSet(
			PriorParam(
				*prior_param_args_from_param_spec('wavelength', True, 750E-9, (0,np.inf), var_params, const_params, initial_values, range_values)
			),
			PriorParam(
				*prior_param_args_from_param_spec('r0', True, 0.15, (0.01,10), var_params, const_params, initial_values, range_values)
			),
			PriorParam(
				*prior_param_args_from_param_spec('turb_ndim', True, 1.3, (1,2), var_params, const_params, initial_values, range_values)
			),
			PriorParam(
				*prior_param_args_from_param_spec('L0', True, 1.5, (0,10), var_params, const_params, initial_values, range_values)
			),
			PriorParam(
				*prior_param_args_from_param_spec('alpha', False, 0.4, (0.1,3), var_params, const_params, initial_values, range_values) 
			),
			PriorParam(
				*prior_param_args_from_param_spec('beta', True, 1.6, (1.1, 10), var_params, const_params, initial_values, range_values)
			),
			PriorParam(
				*prior_param_args_from_param_spec('ao_correction_frac_offset', False, 0, (-1,1), var_params, const_params, initial_values, range_values)
			),
			PriorParam(
				*prior_param_args_from_param_spec('ao_correction_amplitude', False, 2.2, (0,5), var_params, const_params, initial_values, range_values)
			),
			PriorParam(
				*prior_param_args_from_param_spec('factor', False, 1, (0.7,1.3), var_params, const_params, initial_values, range_values)
			),
			PriorParam(
				*prior_param_args_from_param_spec('s_factor', True, 0, (0,np.inf), var_params, const_params, initial_values, range_values)
			),
			PriorParam(
				*prior_param_args_from_param_spec('f_ao', True, instrument.f_ao, (24.0/(2*instrument.obj_diameter),52.0/(2*instrument.obj_diameter)), var_params, const_params, initial_values, range_values)
			)
		)
		
		
		
	def get_parameters(self):
		return self._params
	
	def get_psf_model_flattened_callable(self):
		parent = self
		class PSFModelFlattenedCallable:
			def __init__(self):
				self.specific_model = None
				self.result = None
				self.cached_args = np.zeros((10,))
				
			
			def __call__(
					self,
					wavelength,
					r0, 
					turb_ndim, 
					L0, 
					alpha, 
					beta,
					f_ao,
					ao_correction_amplitude, 
					ao_correction_frac_offset, 
					s_factor,
					factor
				):

				# if we should recalculate everything, do so. Otherwise use the saved model
				if np.any(np.abs(np.array((r0, turb_ndim, L0, alpha, beta, f_ao,ao_correction_amplitude, ao_correction_frac_offset, s_factor,factor)) - self.cached_args) > 1E-5):
					self.specific_model = parent._psf_model(
						None,
						(r0, turb_ndim, L0),
						(alpha, beta),
						f_ao,
						ao_correction_amplitude,
						ao_correction_frac_offset,
						s_factor
					)
				
				
				#_lgr.debug(f'{(wavelength,r0, turb_ndim, L0, alpha, beta, f_ao,ao_correction_amplitude, ao_correction_frac_offset, s_factor,factor)}')
				self.result = self.specific_model.at(wavelength, plots=False).data
				
				return factor*(self.result / np.nansum(self.result.data))
		
		return PSFModelFlattenedCallable()
	
	def get_psf_result_postprocessor(self): 
		def psf_result_postprocessor(params, psf_model_flattened_callable, fitted_vars, consts):
			result = params.apply_to_callable(
				psf_model_flattened_callable, 
				fitted_vars,
				consts
			)
			return result
			
		return psf_result_postprocessor

