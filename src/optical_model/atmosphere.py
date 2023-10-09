"""
Atmospheric models, e.g. turbulence.
"""


import dataclasses as dc

import numpy as np

@dc.dataclass
class TurbulenceModel:
	"""
	The turbulence in the atmosphere at the time of the observation influences 
	the "wings" of the PSF.
	"""
	f_mag : np.ndarray
	r0 : float
	turbulence_ndim : float


	def get_phase_psd(self):
		"""
		Uses kolmogorov turbulence approximation to calculate atmospheric
		part of PSD.
		"""
		#print(f'{self.f_mag}')
		#print(f'{self.r0=}')
		#print(f'{self.turbulence_ndim=}')

		r0_pow = -5/3
		f_pow = -(2+self.turbulence_ndim*3)/3
		factor = 0.000023*10**(self.turbulence_ndim)

		# Should work out to:
		#     3D Case -> 0.023 r0^(-5/3) f^(-11/3) 
		#     2D Case -> 0.0023 r0^(-5/3) f^(-8/3) 
		#     1D Case -> 0.00023 r0^(-5/3) f^(-5/3) 
		# For kolmogorov turbulence
		
		# don't worry about "f_mag" divide by zero, we don't use that datapoint
		# anyway.
		self.psd = factor*(self.r0)**(r0_pow)*(self.f_mag)**(f_pow)
		return(self.psd) 
