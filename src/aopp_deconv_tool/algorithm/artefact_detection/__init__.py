

from typing import Callable, TypeVar

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

type N = TypeVar('N',bound=int)
type N_plus_one = TypeVar('N',bound=int)


def difference_of_scale_filters(
		data : np.ndarray, 
		lo_scale : int = 0, 
		hi_scale : int = 1, 
		filter : Callable[[np.ndarray, int], np.ndarray] = sp.ndimage.uniform_filter,
		**kwargs
	) -> np.ndarray:
	print(f'{lo_scale=} {hi_scale=}')
	if lo_scale == 0:
		lo_data = data
	else:
		lo_data = filter(data, lo_scale, **kwargs)
	return lo_data - filter(data, hi_scale, **kwargs)

def wavelet_decomposition(
		data : np.ndarray['N'],
		lo_scale : int = 1,
		hi_scale : int | None = None,
		filter : Callable[[np.ndarray, float|int], np.ndarray] = sp.ndimage.uniform_filter,
		delta_function_scale = 1,
		ensure_sizes_are_odd=True,
		**kwargs
	) -> np.ndarray['N_plus_one']:
	
	hi_scale = min(int(np.floor(np.log2(s))) for s in data.shape) if hi_scale is None else hi_scale
	wavelet_planes = np.zeros((hi_scale - lo_scale+2, *data.shape), dtype=data.dtype)
	data_decumulator = np.array(data)
	
	for i, scale in enumerate(range(lo_scale, hi_scale)):
		lo_filter_is_delta_function = scale == delta_function_scale
		
		lo_size = 2**(scale-1) + (int(ensure_sizes_are_odd) if ((scale-1)>0) else 0)
		hi_size = 2**(scale) + (int(ensure_sizes_are_odd) if ((scale)>0) else 0)
		print(f'{i=} {lo_size=} {hi_size=} {lo_filter_is_delta_function=}')
		
		norm = np.nansum(data_decumulator)
		lo_filtered = data if lo_filter_is_delta_function else filter(data, lo_size)
		lo_filtered *= (np.nansum(lo_filtered)/norm)
		hi_filtered = filter(data, hi_size)
		hi_filtered *= (np.nansum(hi_filtered)/norm)
		
		wavelet_planes[i] = lo_filtered - hi_filtered
		
		data_decumulator = data - np.nansum(wavelet_planes[:i+1], axis=0)
	
	print(f'last {i=}')
	wavelet_planes[i+1] = filter(data, 2**hi_scale+int(ensure_sizes_are_odd))
	data_decumulator = data - np.nansum(wavelet_planes[:i+2], axis=0)
	wavelet_planes[i+2] = data_decumulator
	
	return wavelet_planes