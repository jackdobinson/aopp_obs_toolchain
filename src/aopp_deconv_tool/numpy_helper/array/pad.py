"""
Padding operations
"""

import scipy as sp
import numpy as np

import aopp_deconv_tool.numpy_helper as nph
import aopp_deconv_tool.numpy_helper.slice

def with_convolution(a : np.ndarray, b : np.ndarray) -> np.ndarray:
	"""
	Pad array `a` by convolving it with array `b`, using the edges of array `b`
	and the center is array `a`
	"""
	p = sp.signal.fftconvolve(a, b, mode='full')
	p[nph.slice.around_center(p.shape,a.shape)] = a
	return(p)

def with_const(a : np.ndarray, to_shape : tuple[int,...], const = 0) -> np.ndarray:
	"""
	Padd array `a` to shape `to_shape` with the value `const`
	"""
	p = np.full(to_shape, const, dtype=a.dtype)
	p[nph.slice.around_center(to_shape, a.shape)] = a
	return p
