

from typing import Callable

import numpy as np
import scipy as sp

def difference_of_scale_filters(
		data : np.ndarray, 
		lo_scale : int = 0, 
		hi_scale : int = 1, 
		filter : Callable[[np.ndarray, int], np.ndarray] = sp.ndimage.uniform_filter,
		**kwargs
	) -> np.ndarray:
	if lo_scale == 0:
		lo_data = data
	else:
		lo_data = filter(data, lo_scale, **kwargs)
	return lo_data - filter(data, hi_scale, **kwargs)
	