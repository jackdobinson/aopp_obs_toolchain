"""
Base class for all deconvolution algorithms
"""

import dataclasses as dc
import numpy as np
from typing import Callable, Any

import context as ctx
import context.temp




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
	pre_init_hook \
		: Callable[[Any, np.ndarray, np.ndarray],None] \
		= lambda obj, obs, psf: None # callback before initialisation
	post_init_hook \
		: Callable[[Any, np.ndarray, np.ndarray],None] \
		= lambda obj, obs, psf: None # callback after initialisation
	pre_iter_hook \
		: Callable[[Any, np.ndarray, np.ndarray],None] \
		= lambda obj, obs, psf: None # callback at the start of each iteration
	post_iter_hook \
		: Callable[[Any, np.ndarray, np.ndarray],None] \
		= lambda obj, obs, psf: None # callback at the end of each iteration
	final_hook \
		: Callable[[Any, np.ndarray, np.ndarray],None] \
		= lambda obj, obs, psf: None # callback after the final iteration
	
	# Internal attributes
	_i : int = dc.field(init=False, repr=False, hash=False, compare=False) # iteration counter
	_components : np.ndarray = dc.field(init=False, repr=False, hash=False, compare=False) # result of the deconvolution
	_residual : np.ndarray = dc.field(init=False, repr=False, hash=False, compare=False) # residual (obs - _components) of the deconvolution
		
	
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
		"""
		with ctx.temp.attributes(self, **kwargs):
			self._init_algorithm(obs, psf)
			self.post_init_hook(self, obs, psf)
			while (self._i < self.n_iter) and self._iter(obs, psf):
				self.post_iter_hook(self, obs, psf)
				self._i += 1
			
			self.final_hook(self, obs, psf)
		
		return(
			self._components,
			self._residual,
			self._i
		)
	
	def _init_algorithm(self, obs : np.ndarray, psf : np.ndarray) -> None:
		"""
		Perform any initialisation that needs to be done before the algorithm runs.
		"""
		self.pre_init_hook(self, obs, psf)
		self._i = 0
		self._components = np.zeros_like(obs)
		self._residual = np.array(obs)
	
	def _iter(self, obs, psf) -> bool:
		"""
		Perform a single iteration of the algorithm.
		"""
		self.pre_iter_hook(self, obs, psf)
		print(f'TESTING: i={self._i}') 
		return(True)
		

@dc.dataclass(slots=True)
class Specific(Base):
	
	# Input parameters
	param_1 : int = 0
	param_2 : float = 7.7
	param_3 : str = "test_value"
	
	def _init_algorithm(self, obs, psf) -> None:
		super(Specific, self)._init_algorithm(obs, psf)
		print('Child class init algorithm')
	
	def _iter(self, obs, psf) -> bool:
		if not super(Specific, self)._iter(obs, psf): return(False)
		print('Child class iter')
		return(True)
		

