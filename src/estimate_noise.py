"""
Provide various ways of estimating the noise in some data
"""


import itertools as it
import numpy as np

import cfg.logs
_lgr = cfg.logs.get_logger_at_level(__name__, 'DEBUG')


def corners_standard_deviation(a : np.ndarray, corner_frac = 1/10):
	corner_mask = np.zeros_like(a, dtype=bool)
	low_corner_slices = tuple(slice(0, int(np.floor(s*corner_frac)), None) for s in a.shape) # xl, yl, zl
	high_corner_slices = tuple(slice(-int(np.floor(s*corner_frac)), None, None) for s in a.shape) # xh, yh, zh
	
	for corner in it.product(*zip(low_corner_slices, high_corner_slices)):
		corner_mask[corner] = 1
	
	return np.std(a[corner_mask])