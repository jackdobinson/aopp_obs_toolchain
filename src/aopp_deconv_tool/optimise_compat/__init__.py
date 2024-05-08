"""
Classes and routines for wrangling PSF model into a format used by various 3rd party optimisation tools.
"""
from typing import Callable, ParamSpec, Any
import dataclasses as dc
import inspect

import numpy as np

from aopp_deconv_tool.data_structures import BiDirectionalMap

import aopp_deconv_tool.cfg.logs
_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'INFO')


P = ParamSpec('P')


def linear_transform_factory(
		in_domain : tuple[float,float], 
		out_domain : tuple[float,float]
	) -> Callable[[float],float]:
	"""
	Creates a function that linearly transforms an input variable from `in_domain` to `out_domain`
	"""
	def linear_transform(x):
		x_dash = (x - in_domain[0])/(in_domain[1]-in_domain[0])
		return (x_dash * (out_domain[1]-out_domain[0])) + out_domain[0]
	return linear_transform




@dc.dataclass(slots=True)
class PriorParam:
	"""
	Class that holds parameter information.
	
	name : str
		String used to identify the parameter
	domain : tuple[float,float]
		Range of possible values the parameter can take
	is_const : bool
		Flag that signals if this parameter is a constant, constant parameters will use `const_value`
	const_value : float
		A value to use when const, and for example plots
	"""
	name : str
	domain : tuple[float,float]
	is_const : bool
	const_value : float
	
	def linear_transform_to_domain(self, in_domain=(0,1)):
		"""
		Returns a function that linearly transforms an `in_domain` to the parameter's domain
		"""
		return linear_transform_factory(in_domain, self.domain)
	
	def linear_transform_from_domain(self, out_domain=(0,1)):
		"""
		Returns a function that linearly transforms the parameter's domain to an `out_domain`
		"""
		return linear_transform_factory(self.domain, out_domain)
	
	def __repr__(self):
		return f'PriorParam(name={self.name}, domain={self.domain}, is_const={self.is_const}, const_value={self.const_value})'

	def to_dict(self):
		return {'name':self.name, 'domain':self.domain, 'is_const':self.is_const, 'const_value':self.const_value}

class PriorParamSet:
	"""
	A collection of PriorParam instances, used to define the fitting region for a fitting function.
	"""
	def __init__(self, *prior_params):
		self.prior_params = list(prior_params)
		self.param_name_index_map = BiDirectionalMap()
		self.all_param_index_to_variable_param_index_map = BiDirectionalMap()
		self.all_param_index_to_constant_param_index_map = BiDirectionalMap()
		
		i=0
		j=0
		for k, p in enumerate(self.prior_params):
			self.param_name_index_map[p.name] = k
			if not p.is_const:
				self.all_param_index_to_variable_param_index_map[k] = i
				i+=1
			else:
				self.all_param_index_to_constant_param_index_map[k] = j
				j+=1
		return
	
	def __repr__(self):
		return repr(self.prior_params)
	
	def __len__(self):
		return len(self.prior_params)
	
	def get_linear_transform_to_domain(self, param_names : tuple[str] | list[str], in_domain : tuple[float,float] | list[float,float] | np.ndarray[[2,'N'],float]):
		"""
		Gets a linear transform from some input domain to some output domain for a set of parameter names
		"""
		n = len(param_names)
		if type(in_domain) == tuple or type(in_domain)==list:
			in_domain = np.array([in_domain,]*n).T
		if in_domain.shape != (2,n):
			raise RuntimeError(f"Input domain must be either a tuple, or a numpy array with shape (2,{n})")
			
		out_domain = np.array(tuple(self[param_name].domain for param_name in param_names)).T
		return linear_transform_factory(in_domain, out_domain)
	
	@property
	def all_params(self):
		"""
		The names of all paramters
		"""
		return (*self.variable_params, *self.constant_params)
	
	@property
	def variable_params(self):
		"""
		The names of the variable parameters
		"""
		return tuple(self.prior_params[i] for i in self.all_param_index_to_variable_param_index_map.keys())
		
	@property
	def constant_params(self):
		"""
		The names of the constant parameters
		"""
		return tuple(self.prior_params[i] for i in self.all_param_index_to_constant_param_index_map.keys())
	
	@property
	def consts(self):
		"""
		Returns a dictionary of the constant parameter names and values
		"""
		return dict((p.name,p.const_value) for p in self.constant_params)
	
	def __getitem__(self, k : str | int):
		"""
		Retrieves a parameter by name or index
		"""
		if type(k) == str:
			return self.prior_params[self.param_name_index_map[k]]
		elif type(k) == int:
			return self.prior_params[k]
		else:
			raise IndexError(f'Unknown index type {type(k)} to PriorParamSet')
	
	def append(self, p : PriorParam):
		"""
		Adds a PriorParam to the set 
		"""
		i = len(self.all_param_index_to_variable_param_index_map)
		j = len(self.all_param_index_to_constant_param_index_map)
		k = len(self.prior_params)
		self.prior_params.append(p)
		self.param_name_index_map[p.name] = k
		if not p.is_const:
			self.all_param_index_to_variable_param_index_map[k] = i
		else:
			self.all_param_index_to_constant_param_index_map[k] = j
		return
	
	@staticmethod
	def get_arg_names_of_callable(
			acallable : Callable,
			arg_names : list[str,...] | None = None # names of the arguments to `acallable`, in the correct order as if from inspect.signature(acallable).parameters.keys()
		):
		"""
		Returns the argument names of the passed callable, if the callable has a "arg_names" attribute, returns that otherwise uses the inspect module.
		
		# ARGUMENTS #
			acallable : Callable
				Some callable to get the argument names of
			arg_names : None | list[str,...]
				If not None, returns this instead of doing any work. Useful if you know exactly what the argument names of the callable are.
		"""
		if arg_names is None:
			if hasattr(acallable, 'arg_names'):
				_lgr.debug(f'{acallable.arg_names=}')
				arg_names = acallable.arg_names
			else:
				sig = inspect.signature(acallable)
				_lgr.debug(f'{sig=}')
				arg_names = list(sig.parameters.keys())
		_lgr.debug(f'{arg_names=}')
		return arg_names
	
	def apply_to_callable(self,
			acallable : Callable[P, Any],
			variable_param_values : dict[str,Any],
			const_param_values : dict[str,Any],
			defaults : dict[str, Any] = {},
			not_found_value : Any = None,
			arg_names : list[str,...] | None = None # names of the arguments to `acallable`, in the correct order as if from inspect.signature(acallable).parameters.keys()
		):
		"""
		Given some callable `acallable`, that accepts `arg_names`. Call it with the values for those arguments provided in `variable_param_values`,
		`const_param_values`, `defaults`, `not_found_value`. Where that list is in order of preference (e.g. an argument in both `const_param_values` and `defaults`
		will have the value specified in `const_param_values`.
		
		If `arg_names` is None, will try and infer the argument names of `acallable` via `self.get_arg_names_of_callable()`.
		"""
		_lgr.debug(f'{defaults=}')
		_lgr.debug(f'{const_param_values=}')
		_lgr.debug(f'{variable_param_values=}')
		arg_names = self.get_arg_names_of_callable(acallable, arg_names)
		param_value_dict = {**defaults, **const_param_values, **variable_param_values} # variables overwrite consts which overwrite defaults if there's a conflict
		arg_values = tuple(param_value_dict.get(x,not_found_value) for x in arg_names)

		return acallable(*arg_values)
		
	
	
	def wrap_callable_for_scipy_parameter_order(self, 
			acallable, 
			*,
			arg_to_param_name_map : dict[str,str] = {},
			constant_params_as_defaults=True,
			arg_names : list[str,...] | None = None # names of the arguments to `acallable`, in the correct order as if from inspect.signature(acallable).parameters.keys()
		):
		"""
		Put in a callable with some arguments `callable(arg1, arg2, arg3,...)`, returns a callable 
		that packs all variable params as the first argument, and all const params as the other arguments.
		
		i.e. 
		accepts callable: 
			`callable(arg1, arg2, arg3, arg4, ...)`
		returns callable: 
			`callable((arg1,arg3,...), arg2, arg4, ...)`
		
		# RETURNS #
			new_callable
				A wrapper that accepts all variable parameters as the first argument, and all constant parameters as the other arguments
			var_params
				A list of the variable parameters in the first argument of `new_callable`, in the order they are present in the first argument
			const_params
				A list of the constant parameters that make up the rest of the arguments to `new_callable`, in the order they are present in the argument list.
		"""
		_lgr.debug(f'self.prior_params={tuple(p.name for p in self.prior_params)}')
		_lgr.debug(f'{self.param_name_index_map=}')
		_lgr.debug(f'{self.all_param_index_to_variable_param_index_map=}')
		_lgr.debug(f'{self.all_param_index_to_constant_param_index_map=}')
		
		# acallable example: some_function(carg1, varg2, carg3, varg4)
		arg_names = self.get_arg_names_of_callable(acallable, arg_names)
		"""
		if arg_names is None:
			sig = inspect.signature(acallable)
			_lgr.debug(f'{sig=}')
		
		# prior_params example: [pc3, pc1, pv4, pv2]
		# param_name_index_map: {pc3:0, pc1:1, pv4:2, pv2:3}
		# variable_param_index_map: {2:0, 3:1}
		# constant_param_index_map: {0:0, 1:1}
		
		# arg_to_param_name_map_example: {pc1 : carg1, pv2 : varg2, pc3 : carg3, pv4 : varg4}
		
		#[carg1, varg2, carg3, varg4]
		if arg_names is None:
			arg_names = list(sig.parameters.keys())
		_lgr.debug(f'{arg_names=}')
		"""
		n_args = len(arg_names)
		_lgr.debug(f'{n_args=}')
		
		# {0:1, 1:3, 2:0, 3:2}
		try:
			arg_to_param_ordering = BiDirectionalMap(dict((i,self.param_name_index_map[arg_to_param_name_map.get(arg_name,arg_name)]) for i, arg_name in enumerate(arg_names)))
		except KeyError as e:
			e.add_note(f'NOTE: Could not match up arguments of {acallable.__class__.__name__ if hasattr(acallable,"__call__") else acallable.__name__}({", ".join(arg_names)}), with parameter names {[p.name for p in self.prior_params]} subject to map {arg_to_param_name_map}')
			raise
		
		_lgr.debug(f'{arg_to_param_ordering=}')
		
		# {0:3, 1:4}
		variable_param_order_to_arg_order = dict((self.all_param_index_to_variable_param_index_map[all_param_index], arg_index) for arg_index, all_param_index in arg_to_param_ordering.items() if all_param_index in self.all_param_index_to_variable_param_index_map)
		_lgr.debug(f'{variable_param_order_to_arg_order=}')
		_lgr.debug(f'variable_param_names={tuple(self.param_name_index_map.backward[self.all_param_index_to_variable_param_index_map.backward[var_param_index]] for var_param_index, argument_index in sorted(variable_param_order_to_arg_order.items(), key=lambda x:x[0]))}')
		
		# {0:2, 1:0}
		constant_param_order_to_arg_order = dict((self.all_param_index_to_constant_param_index_map[all_param_index], arg_index) for arg_index, all_param_index in arg_to_param_ordering.items() if all_param_index in self.all_param_index_to_constant_param_index_map)
		_lgr.debug(f'{constant_param_order_to_arg_order=}')
		_lgr.debug(f'constant_param_names={tuple(self.param_name_index_map.backward[self.all_param_index_to_constant_param_index_map.backward[const_param_index]] for const_param_index, argument_index in sorted(constant_param_order_to_arg_order.items(), key=lambda x:x[0]))}')
		
		# Pack information for quick retrieval
		variable_arg_order_array = np.array(tuple(arg_index for var_param_index, arg_index in sorted(variable_param_order_to_arg_order.items(), key=lambda x:x[0])))
		const_arg_order_array = np.array(tuple(arg_index for const_param_index, arg_index in sorted(constant_param_order_to_arg_order.items(), key=lambda x:x[0])))
		
		def new_callable(one_var_arg, *one_const_arg):
			#_lgr.debug(f'{one_var_arg=}')
			#_lgr.debug(f'{one_const_arg=}')
			#_lgr.debug(f'{variable_arg_order_array=}')
			args = [None]*n_args
			for i, j in enumerate(variable_arg_order_array):
				args[j] = one_var_arg[i]
			for i,j in enumerate(const_arg_order_array):
				if i < len(one_const_arg):
					args[j] = one_const_arg[i]
				else:
					args[j] = new_callable.__defaults__[i - len(one_const_arg)]
			return acallable(*args)
		
		if constant_params_as_defaults:
			new_callable.__defaults__ = tuple(self[self.all_param_index_to_constant_param_index_map.backward[const_param_index]].const_value for const_param_index, argument_index in sorted(constant_param_order_to_arg_order.items(), key=lambda x: x[0]))
		_lgr.debug(f'{new_callable.__defaults__=}')
		
		# wrapper function, variable parameter names in order packed into wrapper first arg, constant parameter names in order of rest of wrapper args
		return (
			new_callable, 
			list(self.param_name_index_map.backward[self.all_param_index_to_variable_param_index_map.backward[k]] for k,v in sorted(variable_param_order_to_arg_order.items(), key=lambda x:x[0])), 
			list(self.param_name_index_map.backward[self.all_param_index_to_constant_param_index_map.backward[k]] for k,v in sorted(constant_param_order_to_arg_order.items(), key=lambda x:x[0]))
		)
		
	def wrap_callable_for_ultranest_parameter_order(self, 
			acallable, 
			arg_to_param_name_map : dict[str,str] = {}
		):
		"""
		Put in a callable with some arguments `callable(arg1, arg2, arg3,...)`, returns a callable 
		that packs all params as the first argument.
		
		i.e. 
		accepts callable: 
			`callable(arg1, arg2, arg3, arg4, ...)`
		
		returns callable: 
			`callable(
				(arg1,arg3, arg2, arg4, ...)
			)`
		
		# RETURNS #
			new_callable
				Wrapper that accepts arguments as a single tuple
			param_names
				Parameters names in the order they go into `new_callable`
		"""
		
		# acallable example: some_function(carg1, varg2, carg3, varg4)
		sig = inspect.signature(acallable)
		
		# prior_params example: [pc3, pc1, pv4, pv2]
		# param_name_index_map: {pc3:0, pc1:1, pv4:2, pv2:3}
		# variable_param_index_map: {2:0, 3:1}
		# constant_param_index_map: {0:0, 1:1}
		
		# arg_to_param_name_map_example: {pc1 : carg1, pv2 : varg2, pc3 : carg3, pv4 : varg4}
		
		#[carg1, varg2, carg3, varg4]
		arg_names = list(sig.parameters.keys())
		
		n_args = len(arg_names)
		
		# {1:0, 3:1, 0:2, 2:3}
		param_to_arg_ordering = dict((self.param_name_index_map[arg_to_param_name_map.get(arg_name,arg_name)], i) for i, arg_name in enumerate(arg_names))
		
		def new_callable(all_params):
			args = [None]*n_args
			for i, j in param_to_arg_ordering.items():
				args[j] = all_params[i]
			
			return acallable(*args)
		
		# wrapper function, variable parameter names in order packed into wrapper first arg, constant parameter names in order of rest of wrapper args
		return (
			new_callable, 
			tuple(self.param_name_index_map.backward[k] for k in param_to_arg_ordering.values())
		)
	
	
	
	