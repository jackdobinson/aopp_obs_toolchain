"""
Operations on distributions derived from empirical data
"""
import dataclasses as dc
from numbers import Number
from typing import Any

import numpy as np

import aopp_deconv_tool.cfg.logs
_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'DEBUG')


class EmpiricalDistribution:
	"""
	Represents a probability distribution found through empirical data
	"""
	def __init__(self, data : np.ndarray):
		self._data = data
	
	def whole_cdf(self) -> tuple[np.ndarray, np.ndarray]:
		sorted_data = np.sort(self._data)
		return sorted_data, self.cdf(sorted_data)
	
	def cdf(self, value : Number | np.ndarray) -> Number | np.ndarray:
		"""
		Returns the cumulative density of `value` (i.e. fraction of datapoints less than `value`), can be passed an array.
		"""
		return np.interp(
			value,
			np.take_along_axis(self._data, np.argsort(self._data, axis=None), axis=None),
			np.linspace(0,1,self._data.size),
			left=0,
			right=1
		)
	
	def ppf(self, prob : Number | np.ndarray) -> Number | np.ndarray:
		"""
		Returns the value of the distribution at cumulative probability `prob`. I.e. `ppf(0.5)` is the median, can be passed an array
		"""
		return np.interp(
			prob,
			np.linspace(0,1,self._data.size),
			np.take_along_axis(self._data, np.argsort(self._data, axis=None), axis=None),
			left=np.nan, # cannot have probabilites > 1 or < 1, so return NAN if out of bounds.
			right=np.nan
		)

	def pdf(self, nbins=100) -> Number | np.ndarray:
		"""
		Returns the probability density function of the distribution
		"""
		n_bin_centers = nbins - 1
		x = np.linspace(np.min(self._data), np.max(self._data), n_bin_centers)
		dx = np.diff(x)
		bins = np.zeros((x.shape[0]+1,*x.shape[1:]))
		bins[0] = x[0] - dx[0]/2
		bins[1:-1] = x[:-1] + dx/2
		bins[-1] = x[-1] + dx[-1]/2
				
		counts = np.zeros((n_bin_centers,), dtype=float)
		for idx in range(0, bins.shape[0]-1):
			counts[idx] = np.count_nonzero((bins[idx] < self._data) & (self._data <= bins[idx+1]))
		
		return bins, counts/self._data.size
		
		
		
		
