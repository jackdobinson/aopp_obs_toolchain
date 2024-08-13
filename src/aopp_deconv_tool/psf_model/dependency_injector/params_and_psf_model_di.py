"""
Dependency injectors make communicating with the routines in `psf_data_ops.py` more managable
by providing a consistent interface. As long as the dependency injector provides parameters
and functions as specified in this file, the fitting functions should work.

"""


from typing import TypeVar, TypeVarTuple, ParamSpec, Generic, Callable, Any, NewType, Protocol, Literal
from types import GenericAlias

import numpy as np

from aopp_deconv_tool.optimise_compat import PriorParam, PriorParamSet





T = TypeVar('T')
Ts = TypeVarTuple('Ts')

# Integers that may or may not be equal
type N = TypeVar('N',bound=int)
type M = TypeVar('M',bound=int)

# A length N tuple of integers
type S[N] = GenericAlias(tuple[int], N)



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

