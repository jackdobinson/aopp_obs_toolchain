"""
Operations on distributions derived from empirical data
"""
import dataclasses as dc
from numbers import Number
from typing import Any

import numpy as np

import cfg.logs
_lgr = cfg.logs.get_logger_at_level(__name__, 'DEBUG')


class EmpiricalDistribution:
	def __init__(self, data : np.ndarray):
		self._data = data
	
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

