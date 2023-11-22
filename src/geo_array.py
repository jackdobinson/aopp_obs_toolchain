import itertools as it
import numpy as np

class GeoArray:
	"""
	A Geometric Array. An array of data, along with coordinate axes that 
	describe the position of the data.
	"""
	def __init__(self,
			data : np.ndarray,
			axes : np.ndarray | None,
		):
		self.data = data
		if axes is None:
			self.axes = np.array([np.linspace(-s/2, s/2, s) for s in self.data.shape()])
		else:
			self.axes = axes
		
		print(f'{self.data.ndim=} {self.data.shape=} {self.axes.ndim=} {self.axes.shape=}')
	
	@classmethod
	def scale_to_axes(scale : tuple[float,...], shape : tuple[int,...], center : float = 0) -> np.ndarray:
		return np.array(
			[np.linspace(center-scale/2,center+scale/2,s) for scale, s in zip(scale,shape)]
		)
	
	def copy(self):
		return GeoArray(np.array(self.data), np.array(self.axes))
	
	def __array__(self):
		return self.data
	
	def fft(self):
		return GeoArray(
			np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(self.data))),
			np.array([np.fft.fftshift(np.fft.fftfreq(x.size, x[1]-x[0])) for x in self.axes]),
		)
	
	def ifft(self):
		return GeoArray(
			np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(self.data))),
			np.array([np.fft.fftshift(np.fft.fftfreq(x.size, x[1]-x[0])) for x in self.axes]),
		)
	
	@property
	def extent(self):
		return tuple(it.chain.from_iterable((x[0],x[-1]) for x in self.axes))

	@property
	def scale(self) -> tuple[float,...]:
		return tuple(x[-1]-x[0] for x in self.axes)

	@property
	def mesh(self) -> np.ndarray:
		return np.array(np.meshgrid(*self.axes))

