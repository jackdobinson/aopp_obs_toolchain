"""

"""
from typing import Sequence, TypeVar
import dataclasses as dc
from types import SimpleNamespace

import numpy as np

import cfg.logs

_lgr = cfg.logs.get_logger_at_level(__name__, 'DEBUG')

T = TypeVar('T')


class BijectiveMap:
	def __init__(self, **kwargs):
		self._imap = {}
		self._jmap = {}
		self._next_idx = []
		for k,v in kwargs.items():
			self[k] = v
	
	def __setitem__(self, key, value):
		self.__delitem__(key)
		self.__delitem__(value)
		self._imap[key] = value
		self._jmap[value] = key
	
	def __getitem__(self, key):
		return self._imap[key] if key in self._imap else self._jmap[key]
	
	def __delitem__(self, key):
		try:
			v = self.__getitem__(key)
			if v in self._imap: del self._imap[v]
			if v in self._jmap: del self._jmap[v]
		except KeyError:
			pass
		if key in self._imap: del self._imap[key]
		if key in self._jmap: del self._jmap[key]
		
	def __contains__(self, key):
		if key in self._imap or key in self._jmap:
			return True
		return False
	
	def __repr__(self):
		return repr(self._imap)
	
	def __str__(self):
		return str(self._imap)
		

@dc.dataclass
class Parameter:
	value : T # default value of the parameter
	description : str # a description of the parameter
	bounds : tuple[T|None, T|None] = (None,None) # the lower and upper bounds of the parameter, None indicates no bounds




class BaseModel:
	"""
	Interface class for some model (or any calculation really) that relies upon
	parameters and the results of other models. Intent is to make it easy to
	swap the implementation of sub-models. 
	
	E.g. the model of a PSF needs to model the atmospheric contribution, the 
	instrumental contribution, and combine the results of those models together.
	We should NOT care about the details of those underlying models, just that
	we have given the correct parameters to them. We should be able to use the
	results if we swap a simple atmospheric model for a more complex one (as an
	example).
	"""
	
	
	# Class level attributes
	required_parameters : tuple[str] = tuple() # parameters that will be available in `self.run`, via `self.p`
	required_models : tuple[str] = tuple() # model results that will be available in `self.run` via `self.m`
	provides_results : tuple[str] = tuple() # this model provides these results in an object returned by the `self.run()` method
	
	
	def __init__(self, params : dict[str, Parameter], sub_models : dict = {}) -> None:
		self.params = params
		self.sub_models = sub_models
		self.p = SimpleNamespace()
		self.m = SimpleNamespace()
		
		# consolidate all sub_model params into this model's params
		
		for m in self.sub_models.values():
			for k,v in m.params.items():
				self.add_param(k,v, replace_value=False)
		
		self.param_order = BijectiveMap()
		self.set_param_order(self.params.keys())
		_lgr.debug(f'{self.params=} {self.param_order=}')
		
	
	def add_param(self, key : str, new_param : Parameter, replace_value=True) -> None:
		"""
		Add a parameter to the model, if the parameter exists replaces the value
		of the old parameter with the new value if desired. Changes bounds to 
		the most restrictive version.
		"""
		if key in self.params:
			# switch to new value of parameter if desired
			if replace_value:
				self.params[key].value = new_param.value
				
			# but use most restrictive bounds
			old_bounds = self.params[key].bounds
			new_bounds = new_param.bounds
			self.params[key].bounds = (np.max(old_bounds[0], new_bounds[0]), np.min(old_bounds[1],new_bounds[1]))
			
			# Issue a warning if the descriptions are not the same
			_lgr.warn(f'Descriptions for repeated parameter {key} are not the same. "{self.params[key].description}" vs "{new_param.description}"')
		else:
			# otherwise, just add the parameter
			self.params[key]=new_param
	
	def set_param_order(self, param_order : Sequence[str]) -> None:
		"""
		Sets the order that parameters should be unpacked from positional arguments,
		the intent is to make `self.__call__` compatible with the `scipy.optimize`
		package.
		"""
		for i,param_name in enumerate(param_order):
			self.param_order[param_name] = i
		
		for param_name in self.params.keys():
			if param_name not in self.param_order:
				self.param_order[param_name] = len(self.param_order)
	
	
	def __call__(self, *args, **kwargs : dict[str, Parameter]):
		"""
		Runs the model with the parameter values passed according to self.param_order.
		If parameters can also be specified as keyword arguments. This should be
		compatible with the `scipy.optimize` package, so we can just pass the model
		object to the optimizers.
		"""
		_lgr.debug(dict((k,p.value) for k,p in self.params.items()))
		
		current_params = {
			**dict((k,p.value) for k,p in self.params.items()), 
			**dict((k,p.value) for k,p in kwargs.items()), 
			**dict((self.param_order[i],value) for i,value in enumerate(args))
		}
		_lgr.debug(f'{current_params=}')
		
		self.load_required_models(**current_params)
		self.load_required_parameters(**current_params)
		
		return self.run()
	
	
	def load_required_parameters(self, **kwargs):
		"""
		Add the parameters to the `self.p` object
		"""
		for param_name in self.required_parameters:
			try:
				setattr(self.p, param_name, kwargs[param_name])
			except KeyError:
				raise RuntimeError(f'{self.__class__} required parameter "{param_name}" is missing')
	
	def load_required_models(self, **kwargs):
		"""
		Calculates the result of the required models and stores the results in
		the `self.m` object.
		"""
		for model_name in self.required_models:
			try:
				self.sub_models[model_name].load_required_parameters(**kwargs)
				setattr(self.m, model_name, self.sub_models[model_name].run())
			except KeyError:
				raise RuntimeError(f'{self.__class__} required model "{model_name}" is missing')
	
	def run(self):
		"""
		Run the model, parameters are always keyword arguments, the parameters
		required by this model are attributes of the `self.p` object, sub-model
		results required by this model are attributes of the `self.m` object.
		"""
		_lgr.debug(self.p)
		raise NotImplementedError('This should be overwritten by a subclass that implements the Model protocol')
