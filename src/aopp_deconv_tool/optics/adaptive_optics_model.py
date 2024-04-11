

from aopp_deconv_tool.optics.function import PhasePowerSpectralDensity

import numpy as np




def moffat_function(x, alpha, beta):
	"""
	Used to model the phase corrections of adaptive optics systems
	
	Alpha has a similar effect to beta on the modelling side, they do different things, but they are fairly degenerate.
	"""
	return((1+np.sum((x.T/alpha).T**2, axis=0))**(-beta))


def phase_psd_fetick_2019_moffat_function(
		f_axes,
		alpha : np.ndarray | float,
		beta : float,
	):
	"""
	Uses a moffat function to approximate the effect of AO on the
	low-frequency part of the PSD
	"""
	assert beta != 1, "beta cannot be equal to one in this model"
	f_mesh = np.array(np.meshgrid(*f_axes[::-1]))
	
	if type(alpha) is float:
		alpha = np.array([alpha]*2)
	psd = moffat_function(f_mesh, alpha, beta)
	return PhasePowerSpectralDensity(data=psd, axes=f_axes)
