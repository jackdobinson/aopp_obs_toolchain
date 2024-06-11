
import numpy as np
import scipy as sp
import scipy.ndimage

def quick_remove_nan_and_inf(data, size=5):
	mask = np.isnan(data) | np.isinf(data)
	if np.any(mask):
		#whole_median = np.nanmedian(data[~mask])
		data[mask] = 0
		#data[mask] = sp.ndimage.median_filter(data, size=size)[mask]
		data[mask] = sp.ndimage.gaussian_filter(data, sigma=size)[mask]
	return data