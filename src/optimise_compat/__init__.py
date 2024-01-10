"""
Classes and routines for wrangling PSF model into a format used by various 3rd party optimisation tools.
"""
from typing import Callable
import dataclasses as dc
import inspect

import numpy as np

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


class BiDirectionalMap:
	def __init__(self, seed_dict={}):
		self.is_forward = True
		self.forward_dict = dict((k,v) for k,v in seed_dict.items())
		self.backward_dict = dict((v,k) for k,v in seed_dict.items())
	
	def __getattr__(self, name):
		proxy = self.forward_dict if self.is_forward else self.backward_dict
		if hasattr(proxy, name):
			return getattr(proxy, name)
		else:
			raise AttributeError(f'No attribute {name} on {type(self).__name__}')
	
	@property
	def backward(self):
		self.is_forward = False
		return self
	
	def __getitem__(self, key):
		if self.is_forward:
			return self.forward_dict[key]
		else:
			self.is_forward = True
			return self.backward_dict[key]
	
	def __contains__(self, key):
		if self.is_forward:
			return key in self.forward_dict
		else:
			self.is_forward = True
			return key in self.backward_dict
	
	def __setitem__(self, key, value):
		if self.is_forward:
			self.forward_dict.__setitem__(key,value)
			self.backward_dict.__setitem__(value,key)
		else:
			self.is_forward = True
			self.forward_dict.__setitem__(value,key)
			self.backward_dict.__setitem__(key,value)
		return
	
	def update(self, *args, **kwargs):
		is_forward_flag = self.is_forward
		for k,v in dict(*args, **kwargs).items():
			self.is_forward = is_forward_flag
			self.__setitem__(k,v)
	
	def __repr__(self):
		return f'{type(self).__name__}({str(self.forward_dict)})'


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
	A collection of PriorParam instances
	"""
	def __init__(self, *prior_params):
		self.prior_params = list(prior_params)
		self.param_name_index_map = BiDirectionalMap()
		self.variable_param_index_map = BiDirectionalMap()
		self.constant_param_index_map = BiDirectionalMap()
		
		i=0
		j=0
		for k, p in enumerate(self.prior_params):
			self.param_name_index_map[p.name] = k
			if not p.is_const:
				self.variable_param_index_map[k] = i
				i+=1
			else:
				self.constant_param_index_map[k] = j
				j+=1
		return
	
	def __repr__(self):
		return repr(self.prior_params)
	
	def __len__(self):
		return len(self.prior_params)
	
	def get_linear_transform_to_domain(self, param_names, in_domain):
		n = len(param_names)
		if type(in_domain) == tuple or type(in_domain)==list:
			in_domain = np.array([in_domain,]*n).T
		if in_domain.shape != (2,n):
			raise RuntimeError(f"Input domain must be either a tuple, or a numpy array with shape (2,{n})")
			
		out_domain = np.array(tuple(self[param_name].domain for param_name in param_names)).T
		return linear_transform_factory(in_domain, out_domain)
	
	@property
	def variable_params(self):
		return tuple(self.prior_params[i] for i in self.variable_param_index_map.keys())
		
	@property
	def constant_params(self):
		return tuple(self.prior_params[i] for i in self.constant_param_index_map.keys())
	
	def __getitem__(self, k : str | int):
		if type(k) == str:
			return self.prior_params[self.param_name_index_map[k]]
		elif type(k) == int:
			return self.prior_params[k]
		else:
			raise IndexError(f'Unknown index type {type(k)} to PriorParamSet')
	
	def append(self, p : PriorParam):
		i = len(self.variable_param_index_map)
		j = len(self.constant_param_index_map)
		k = len(self.prior_params)
		self.prior_params.append(p)
		self.param_name_index_map[p.name] = k
		if not p.is_const:
			self.variable_param_index_map[k] = i
		else:
			self.constant_param_index_map[k] = j
		return
	
	def wrap_callable_for_scipy_parameter_order(self, 
			acallable, 
			arg_to_param_name_map : dict[str,str] = {},
			constant_params_as_defaults=True
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
		
		# {0:1, 1:3, 2:0, 3:2}
		arg_to_param_ordering = BiDirectionalMap(dict((i,self.param_name_index_map[arg_to_param_name_map.get(arg_name,arg_name)]) for i, arg_name in enumerate(arg_names)))
		
		# {0:3, 1:4}
		variable_param_to_arg_ordering = dict((self.variable_param_index_map[all_param_index], arg_index) for arg_index, all_param_index in arg_to_param_ordering.items() if all_param_index in self.variable_param_index_map)
		
		# {0:2, 1:0}
		constant_param_to_arg_ordering = dict((self.constant_param_index_map[all_param_index], arg_index) for arg_index, all_param_index in arg_to_param_ordering.items() if all_param_index in self.constant_param_index_map)
		
		
		def new_callable(one_var_arg, *one_const_arg):
			args = [None]*n_args
			for i, j in variable_param_to_arg_ordering.items():
				args[j] = one_var_arg[i]
			for i,j in constant_param_to_arg_ordering.items():
				if i < len(one_const_arg):
					args[j] = one_const_arg[i]
				else:
					args[j] = new_callable.__defaults__[i - len(one_const_arg)]
			
			return acallable(*args)
		
		if constant_params_as_defaults:
			new_callable.__defaults__ = tuple(self[k].const_value for k in constant_param_to_arg_ordering)
		
		# wrapper function, variable parameter names in order packed into wrapper first arg, constant parameter names in order of rest of wrapper args
		return (
			new_callable, 
			list(self.param_name_index_map.backward[self.variable_param_index_map.backward[k]] for k,v in sorted(variable_param_to_arg_ordering.items(), key=lambda x:x[0])), 
			list(self.param_name_index_map.backward[self.constant_param_index_map.backward[k]] for k,v in sorted(constant_param_to_arg_ordering.items(), key=lambda x:x[0]))
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
	
	
	
	