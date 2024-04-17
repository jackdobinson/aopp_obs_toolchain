"""
Routines for handing an array that knows it's physical size.
"""


import itertools as it
import numpy as np
from typing import Callable

from matplotlib import pyplot as plt

class GeoArray:
	"""
	A Geometric Array. An array of data, along with coordinate axes that 
	describe the position of the data.
	
	# ATTRIBUTES #
		self.data : np.ndarray
			The data the array holds
		self.axes : np.array
			The 'points in space' where `self.data` is defined. These are what changes when the GeoArray undergoes a geometric transform
	"""
	def __init__(self,
			data : np.ndarray,
			axes : np.ndarray | None,
		):
		self.data = data
		if axes is None:
			self.axes = tuple(np.linspace(-s/2, s/2, s) for s in self.data.shape())
		else:
			self.axes = axes
		
		
	@staticmethod
	def scale_to_axes(scale : tuple[float,...], shape : tuple[int,...], center : float = 0) -> np.ndarray:
		"""
		Applies a `scale` to a `shape` such that the result is centered around `center`
		"""
		return tuple(
			np.linspace(center-scale/2,center+scale/2,s) for scale, s in zip(scale,shape)
		)
	
	def copy(self):
		"""
		Copy the GeoArray
		"""
		return GeoArray(np.array(self.data), np.array(self.axes))
	
	def __array__(self):
		"""
		For compatibility with numpy routines
		"""
		return self.data
	
	def fft(self):
		"""
		Return the fast fourier transform of the GeoArray
		"""
		return GeoArray(
			np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(self.data))),
			tuple(np.fft.fftshift(np.fft.fftfreq(x.size, x[1]-x[0])) for x in self.axes),
		)
	
	def ifft(self):
		"""
		Return the inverse fast fourier transform of the GeoArray
		"""
		return GeoArray(
			np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(self.data))),
			tuple(np.fft.fftshift(np.fft.fftfreq(x.size, x[1]-x[0])) for x in self.axes),
		)
	
	@property
	def extent(self):
		"""
		Return the "extent" (i.e. min,max in each dimension) of the GeoArray
		"""
		return tuple(it.chain.from_iterable((x[0],x[-1]) for x in self.axes))

	@property
	def scale(self) -> tuple[float,...]:
		"""
		Return the scale (i.e. difference between min and max) of the GeoArray
		"""
		return tuple(x[-1]-x[0] for x in self.axes)

	@property
	def mesh(self) -> np.ndarray:
		"""
		Return a meshgrid that covers the GeoArray
		"""
		return np.array(np.meshgrid(*self.axes[::-1]))




def plot_ga(
		geo_array : GeoArray, 
		data_mutator : Callable[[np.ndarray], np.ndarray] = lambda x:x, 
		title : str = '', 
		data_units : str = '', 
		axes_units : str | tuple[str,...] = '',
		show : bool = True
	):
	"""
	Plot the GeoArray as an image. Only works for 2d arrays at present. Will call
	a mutator on the data before plotting. Returns a numpy array of the axes
	created for the plot
	"""
	if type(axes_units) is not tuple:
		axes_units = tuple(axes_units for _ in range(geo_array.data.ndim))
	plt.clf()
	f = plt.gcf()
	f.suptitle(title)
	
	if geo_array.data.ndim != 2:
		raise NotImplementedError("Geometric array plotting only works for 2d arrays at present.")
		
	a = f.subplots(2,2,squeeze=True).flatten()
	
	center_idx = tuple(s//2 for s in geo_array.data.shape)
	
	m_data = data_mutator(geo_array.data)
	m_axes = geo_array.axes
	
	a[0].set_title('array data')
	a[0].imshow(m_data, extent=geo_array.extent)
	a[0].set_xlabel(axes_units[0])
	a[0].set_ylabel(axes_units[1])
	
	a[1].set_title('x=constant centerline')
	a[1].plot(m_data[:, center_idx[1]], m_axes[0])
	a[1].set_xlabel(data_units)
	a[1].set_ylabel(axes_units[0])
	
	a[2].set_title('y=constant centerline')
	a[2].plot(m_axes[1], m_data[center_idx[0], :])
	a[2].set_xlabel(axes_units[1])
	a[2].set_ylabel(data_units)
	
	a[3].set_title('array data unmutated')
	a[3].imshow(geo_array.data if geo_array.data.dtype!=np.dtype(complex) else np.abs(geo_array.data), extent=geo_array.extent)
	a[3].set_xlabel(axes_units[0])
	a[3].set_ylabel(axes_units[1])
	
	
	if show: plt.show()
	return f, a
