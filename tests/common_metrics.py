"""
Contains common metrics used in tests
"""
import numpy as np
import aopp_deconv_tool.stats.empirical

class FracError:
	def __init__(self, observed : np.ndarray, expected : np.ndarray):
		self.value = ((observed - expected)**2)/expected
	
	@property
	def mean(self):
		return np.mean(self.value)
	
	@property
	def min(self):
		return np.min(self.value)
	
	@property
	def max(self):
		return np.max(self.value)
	
	@property
	def pdf(self):
		return aopp_deconv_tool.stats.empirical.EmpiricalDistribution(self.value).pdf()

def frac_error(observed : np.ndarray, expected : np.ndarray) -> np.ndarray:
	return ((observed - expected)**2)/expected

def mean_frac_error(observed : np.ndarray, expected : np.ndarray) -> float:
	return np.mean(((observed - expected)**2)/expected)
