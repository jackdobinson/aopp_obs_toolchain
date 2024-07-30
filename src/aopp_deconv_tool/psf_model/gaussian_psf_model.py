"""
Point spread function (PSF) model that is just a gaussian
"""

from typing import Any, TypeVar, TypeVarTuple, Generic, Annotated, NewType

import numpy as np

IntVar = TypeVar('IntVar', bound=int)
T = TypeVar('T')
U = TypeVar('T')
Ts = TypeVarTuple('Ts')
class S(Generic[IntVar]): pass
class S1(Generic[IntVar]): pass
N = TypeVar('N',bound=int)

import matplotlib.pyplot as plt

class GaussianPSFModel:
	"""
	Class defines a gaussian where the shape and type are fixed at instantiaion, but the mean, standard deviation, and offset are calculated
	by the instance. This is useful for putting into an optimising function as the underlying array storage is only created once.
	"""
	def __init__(self,
			shape : S[N],
			dtype : T
		):
		self.result = np.zeros(shape, dtype=dtype)
		self.indices = np.indices(self.result.shape)
		self.working_array = np.zeros(self.indices.shape, dtype=dtype)
	
	def __call__(self,
			mean : np.ndarray[S1[N],U],
			std : np.ndarray[S1[N],U],
			const : U
		) -> np.ndarray[S[N],T]:
		
		# const from -1 -> 1, where 1 is the amplitude of the gaussian, but still want everything to sum to 1
		inverse_a = 1 - const
		
		
		self.working_array = self.indices.T - mean
		
		norm_factor = np.sqrt(((2*np.pi)**std.size)) * np.prod(std)
		self.working_array = (self.working_array**2)/(2*(std**2))
		
		self.result[...] =  inverse_a * 1/norm_factor * np.exp(-np.sum(self.working_array, axis=-1)).T + const / self.result.size
		
		return self.result