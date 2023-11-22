

from optics.function import PhasePowerSpectralDensity

import numpy as np




def moffat_function(x, alpha, beta, A):
	"""
	Used to model the phase corrections of adaptive optics systems
	
	Alpha has a similar effect to beta on the modelling side, they do different things, but they are fairly degenerate.
	"""
	return(A*((1+np.sum((x.T/alpha).T**2, axis=0))**(-beta)))


def phase_psd_fetick_2019_moffat_function(
		f_axes,
		f_ao,
		alpha : np.ndarray,
		beta : float,
		C : float,
		A : float,
	):
	"""
	Uses a moffat function to approximate the effect of AO on the
	low-frequency part of the PSD
	"""
	assert beta != 1, "beta cannot be equal to one in this model"
	f_mesh = np.array(np.meshgrid(*f_axes))
	
	if type(alpha) is float:
		alpha = np.ndarray([alpha]*2)
	part1 = (beta - 1)/(np.pi*np.prod(alpha))
	part2 = moffat_function(f_mesh, alpha, beta, A)
	part3 = (1-(1+np.prod(f_ao/alpha))**(1-beta))**(-1)
	print(f'{part1=} {part2=} {part3=}')
	psd = part1*part2*part3 + C
	return PhasePowerSpectralDensity(data=psd, axes=f_axes)
