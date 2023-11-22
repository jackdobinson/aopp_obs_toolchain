

from optics.function import PhasePowerSpectralDensity

import numpy as np

def phase_psd_von_karman_turbulence(
		f_axes, 
		r0, 
		turbulence_ndim,
		L0
	):
	"""
	Von Karman turbulence is the same as Kolmogorov, but with the extra "L0"
	term. L0 -> infinity gives Kolmogorov turbulence.
	"""
	f_mesh = np.array(np.meshgrid(*f_axes))
	f_sq = np.sum(f_mesh**2, axis=0)
	r0_pow = -5/3
	f_pow = -(2+turbulence_ndim*3)/6
	factor = 0.000023*10**(turbulence_ndim)
	psd = factor*(r0)**(r0_pow)*(1/L0**2 + f_sq)**(f_pow)
	center_idx = tuple(s//2 for s in psd.shape)
	psd[center_idx] = 0 # stop infinity at f==0
	return PhasePowerSpectralDensity(psd, f_axes)

