

import numpy as np


class RadialPSFModel:
	"""
	PSF model that is a radially averaged PSF (i.e. a histogram of radial bins).
	"""
	def __init__(self, data):
		self.data = data
		self.r_bins = np.zeros((1,2))
		self.r_values = np.zeros((1,))
		self._result = np.zeros_like(data)
	
	def __call__(self, x, y, nbins):
		r_max = max(*self.data.shape)
		nbins = int(nbins)
		self.r_values = np.zeros((nbins,))
		self.r_bins = np.zeros((nbins,2))
		self.r_bins[:,1] = np.logspace(0, np.log10(r_max), nbins)
		self.r_bins[1:,0] = self.r_bins[:-1,1]
		
		r = np.sqrt(np.sum((np.indices(self.data.shape).T - np.array([x,y]))**2, axis=-1)).T
		
		self._result *= 0
		for i, r_lim in enumerate(self.r_bins):
			r_mask = (r_lim[0] <= r) & (r < r_lim[1])
			self.r_values[i] = np.nanmean(self.data[r_mask])
			self._result[r_mask] = self.r_values[i]
		
		return self._result
	
	
	@property
	def centered_result(self):
		r = np.sqrt(np.sum((np.indices(self.data.shape).T - np.array([self.data.shape[0]//2, self.data.shape[1]//2]))**2, axis=-1)).T
		
		self._result *= 0
		for i, r_lim in enumerate(self.r_bins):
			r_mask = (r_lim[0] <= r) & (r < r_lim[1])
			self._result[r_mask] = self.r_values[i]
		return self._result