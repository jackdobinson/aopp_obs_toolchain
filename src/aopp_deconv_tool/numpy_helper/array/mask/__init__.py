"""
Routines that use masks to assist in array operations
"""
import numpy as np

from aopp_deconv_tool.geometry.shape import GeoShape
from aopp_deconv_tool.numpy_helper.array import N,S

def from_nan_and_inf(a : np.ndarray[[...],float]) -> np.ndarray[[...],bool]:
	"""
	Gets a mask of all nan and inf values

	Arguments:
		a : np.ndarray
			A numpy array of pixels
	
	Returns:
		A boolean numpy array that is True where `a` is NAN and INF.
	"""
	return np.isnan(a) | np.isinf(a)

def from_shape(shape : S[N], geo_shape : GeoShape, extent : np.ndarray[[N,2],float] | None = None) -> np.ndarray[S[N],bool]:
	"""
	Gets a mask of `shape` and start-end coords `extent` that are inside `poly_shape`
	"""
	if extent is None:
		# Assume we want to center the geo_shape in the region
		extent = np.array(tuple((-(s-1)/2, (s-1)/2) for s in shape))
	print(f'{extent=}')
	print(f'{extent[:,1]=}')
	stride = (extent[:,1]-extent[:,0]) / np.array(tuple(s-1 for s in shape)) # does not include end point
	print(f'{stride=}')
	#return np.vectorize(lambda p: p in geo_shape)(np.moveaxis(np.indices(shape),0,-1)*stride + extent[:,0])
	print(f'{np.moveaxis(np.moveaxis(np.indices(shape),0,-1)*stride + extent[:,0], -1,0)}')
	return geo_shape.__contains__(np.moveaxis(np.indices(shape),0,-1)*stride + extent[:,0])
