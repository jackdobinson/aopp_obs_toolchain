"""
Base class for all deconvolution algorithms
"""

import dataclasses as dc
import numpy as np
from typing import Callable, Any
import inspect
import datetime as dt

import aopp_deconv_tool.context as ctx
import aopp_deconv_tool.context.temp

import aopp_deconv_tool.cfg.logs
_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'DEBUG')


@dc.dataclass(slots=True)
class Base:
	"""
	Implements basic iteration of an algorithm.
	
	Algorithm parameters are set on class construction, data to operate upon is
	set when an instance is called.
	"""
	
	# Input Paramterers
	n_iter 			: int	= dc.field(default=100, 	metadata={'description':'Number of iterations'})
	
	# Hooks for callbacks at various parts of the algorithm.
	pre_init_hooks \
		: list[Callable[[Any, np.ndarray, np.ndarray],None]] \
		= dc.field(default_factory=lambda : [], init=False, repr=False, hash=False, compare=False) # callbacks before initialisation
	post_init_hooks \
		: list[Callable[[Any, np.ndarray, np.ndarray],None]] \
		= dc.field(default_factory=lambda : [], init=False, repr=False, hash=False, compare=False) # callbacks after initialisation
	pre_iter_hooks \
		: list[Callable[[Any, np.ndarray, np.ndarray],None]] \
		= dc.field(default_factory=lambda : [], init=False, repr=False, hash=False, compare=False) # callbacks at the start of each iteration
	post_iter_hooks \
		: list[Callable[[Any, np.ndarray, np.ndarray],None]] \
		= dc.field(default_factory=lambda : [], init=False, repr=False, hash=False, compare=False) # callbacks at the end of each iteration
	final_hooks \
		: list[Callable[[Any, np.ndarray, np.ndarray],None]] \
		= dc.field(default_factory=lambda : [], init=False, repr=False, hash=False, compare=False) # callbacks after the final iteration
	
	# State attributes visible from outside class
	progress_string : str 				= dc.field(default="Ended prematurely: Class not initialised.", init=False, repr=False, hash=False, compare=False) # reason that iteration was terminated
	
	# Internal attributes
	_i : int 							= dc.field(init=False, repr=False, hash=False, compare=False) # iteration counter
	_components : np.ndarray 			= dc.field(init=False, repr=False, hash=False, compare=False) # result of the deconvolution
	_residual : np.ndarray 				= dc.field(init=False, repr=False, hash=False, compare=False) # residual (obs - _components) of the deconvolution
	_last_parameters : dict[str,Any] 	= dc.field(init=False, repr=False, hash=False, compare=False) # parameters used for last call
	
	def get_parameters(self):
		return self._last_parameters
	
	def set_parameters(self, params):
		for k,v in params.items():
			if k=='parameters_recorded_at_timestamp': continue # Don't save this.
			assert k[0] != '_', f'Private attribute (beginning with "_") "{k}" cannot be set as parameters'
			assert not k.endswith('hooks'), f'Cannot set attribute "{k}" ending in "hooks", as these are lists of callables.'
			assert hasattr(k), f'{self.__class__.__name__} does not have the attribute "{k}" to set from supplied parameters.'
			assert hasattr(getattr(k), '__call__'), f'Attribute {self.__class__.__name__}.{k} is a callable, cannot set from supplied parameters.'
			setattr(self, k, v)
		self._set_last_parameters()
		return
	
	def _set_last_parameters(self):
		self._last_parameters = {}
		# Add a timestamp to the parameters
		assert not hasattr(self, 'parameters_recorded_at_timestamp'), f"'parameters_recorded_at_timestamp' is used internally and {self.__class__.__name__} cannot have it as an attribute"
		self._last_parameters["parameters_recorded_at_timestamp"] = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f%z')
		
		# Loop over all fields of this dataclass
		for k, m in self.__dataclass_fields__.items():
			# If the field doesn't begin with '_', is not a callable, and is not a list of hooks then remember it.
			if (k[0] != '_') \
					and not hasattr(m, '__call__') \
					and not k.endswith('hooks') \
				:
				self._last_parameters[k] = getattr(self, k)
	
	def __call__(self, obs : np.ndarray, psf : np.ndarray, **kwargs) -> tuple[np.ndarray, np.ndarray, int]:
		"""
		Apply algorithm to `obs` and `psf`, parameters are set at instantiation, but can
		be overwritten for a single call by passing new values via `**kwargs`. Initialises and 
		iterates the algorithm. Subclasses should overload the `_init_algorithm` and `_iter` methods.
		
		# Arguments #
			obs : np.ndarray
				Numpy array containing observation data to deconvolve
			psf : np.ndarray
				Numpy array containing the point-spread-function for the observation.
			**kwargs
				Other passed arguments are assumed to overwrite algorithm parameters during a singular invocation.
		
		# Returns #
			self._components : np.ndarray
				Components found that make up the deconvolved image, ideally these components convolved with `psf` will give `obs`
			self._residual : np.ndarray
				The residual between the convolution of `self._components with` `psf`, and `obs`
			self._i : int
				The number of iterations performed before terminating
		"""
		with ctx.temp.attributes(self, **kwargs):
			try:
				self.progress_string = f'Ended prematurely: Initialising algorithm.'
				
				self._init_algorithm(obs, psf)
				
				self.progress_string = f'Ended prematurely: Algorithm initialised.'
				
				for c in self.post_init_hooks: 
					c(self, obs, psf)
					
				self.progress_string = f'Ended prematurely: Iteration starting.'
				
				while (self._i < self.n_iter) and self._iter(obs, psf):
					for c in self.post_iter_hooks: c(self, obs, psf)
					self._i += 1
				
				for c in self.final_hooks: 
					c(self, obs, psf)
					
			except Exception as e:
				self.progress_string = f"Ended prematurely at {self._i} iterations: {str(e)}"
				raise
			finally:
				if self._i == self.n_iter:
					self.progress_string = f"Ended at {self._i} iterations: Maximum number of iterations reached."
				_lgr.info(f'{self.progress_string}')
				self._set_last_parameters()
		
			
		return(
			self.get_components(),
			self.get_residual(),
			self.get_iters()
		)
	
	def _init_algorithm(self, obs : np.ndarray, psf : np.ndarray) -> None:
		"""
		Perform any initialisation that needs to be done before the algorithm runs.
		"""
		for c in self.pre_init_hooks: c(self, obs, psf)
		self._i = 0
		self._components = np.zeros_like(obs)
		self._residual = np.array(obs)
	
	def _iter(self, obs, psf) -> bool:
		"""
		Perform a single iteration of the algorithm.
		"""
		for c in self.pre_iter_hooks: c(self, obs, psf)
		_lgr.debug(f'i={self._i}') 
		return(True)
		
		

