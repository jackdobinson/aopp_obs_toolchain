"""
Base class for all deconvolution algorithms
"""

import dataclasses as dc
import numpy as np
from typing import Callable, Any
import inspect

import context as ctx
import context.temp

import cfg.logs

_lgr = cfg.logs.get_logger_at_level(__name__, 'DEBUG')


@dc.dataclass(slots=True)
class Base:
	"""
	Implements basic iteration of an algorithm.
	
	Algorithm parameters are set on class construction, data to operate upon is
	set when an instance is called.
	"""
	
	# Input Paramterers
	n_iter : int = 10 # Number of iterations before exit
	verbosity : int = 0 # How verbose messages should be
	
	# Hooks for callbacks at various parts of the algorithm.
	pre_init_hooks \
		: list[Callable[[Any, np.ndarray, np.ndarray],None]] \
		= dc.field(default_factory=lambda : [], repr=False, hash=False, compare=False) # callbacks before initialisation
	post_init_hooks \
		: list[Callable[[Any, np.ndarray, np.ndarray],None]] \
		= dc.field(default_factory=lambda : [], repr=False, hash=False, compare=False) # callbacks after initialisation
	pre_iter_hooks \
		: list[Callable[[Any, np.ndarray, np.ndarray],None]] \
		= dc.field(default_factory=lambda : [], repr=False, hash=False, compare=False) # callbacks at the start of each iteration
	post_iter_hooks \
		: list[Callable[[Any, np.ndarray, np.ndarray],None]] \
		= dc.field(default_factory=lambda : [], repr=False, hash=False, compare=False) # callbacks at the end of each iteration
	final_hooks \
		: list[Callable[[Any, np.ndarray, np.ndarray],None]] \
		= dc.field(default_factory=lambda : [], repr=False, hash=False, compare=False) # callbacks after the final iteration
	
	# State attributes visible from outside class
	stop_reason : str = dc.field(default="unknown reason for terminating iteration", init=False, repr=False, hash=False, compare=False) # reason that iteration was terminated
	
	# Internal attributes
	_i : int = dc.field(init=False, repr=False, hash=False, compare=False) # iteration counter
	_components : np.ndarray = dc.field(init=False, repr=False, hash=False, compare=False) # result of the deconvolution
	_residual : np.ndarray = dc.field(init=False, repr=False, hash=False, compare=False) # residual (obs - _components) of the deconvolution
	
	
	def get_parameters(self):
		p = {}
		for k, m in self.__dataclass_fields__.items():
			if (k[0] != '_') \
					and not hasattr(m, '__call__') \
					and not k.endswith('hook') \
				:
				p[k] = getattr(self, k)
		return p
			
	
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
			self._init_algorithm(obs, psf)
			for c in self.post_init_hooks: c(self, obs, psf)
			while (self._i < self.n_iter) and self._iter(obs, psf):
				for c in self.post_iter_hooks: c(self, obs, psf)
				self._i += 1
			
			if self._i == self.n_iter:
				self.stop_reason = "Maximum number of iterations reached"
			
			for c in self.final_hooks: c(self, obs, psf)
		
			_lgr.info(f'Iteration terminated: {self.stop_reason}')
		return(
			self._components,
			self._residual,
			self._i
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
		
		

