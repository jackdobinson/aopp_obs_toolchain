

from typing import Annotated

import numpy as np

from aopp_deconv_tool.optics.function import PhasePowerSpectralDensity

import aopp_deconv_tool.cfg.logs
_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'DEBUG')


"""
	objective                                    image plane
	-----------------------------------------------|
		|                                          |
		|                           .............. X         point on image plane, priciple ray to this point is ".../" line
		|             ............./   \           |
		|............/           theta  |          |
......../ - - - - - - - - - - - - - - - - - - - - -|         optical axis
		|                                          |
		|                                          |
		|                                          |
		|                                          |
	-----------------------------------------------|

theta - Angle between optical axis and principle ray to point X on image plane

"""

def phase_psd_von_karman_turbulence(
		f_axes : Annotated[np.ndarray, 'ModelInputData'], # f_axes, f is theta/wavelength
		wavelength : Annotated[float, 'ModelInputData'],
		r0 : Annotated[float, 'ModelParameter'],
		turbulence_ndim : Annotated[float,'ModelParameter'],
		L0 : Annotated[float,'ModelParameter']
	):
	"""
	Von Karman turbulence is the same as Kolmogorov, but with the extra "L0"
	term. L0 -> infinity gives Kolmogorov turbulence.
	"""
	wav_factor = wavelength/5E-7
	L0 = L0*wav_factor
	#_lgr.debug(f'{f_axes=} {wavelength=} {wav_factor=} {r0=} {turbulence_ndim=} {L0=}')
	
	f_mesh = np.array(np.meshgrid(*f_axes[::-1]))
	f_sq = np.sum(f_mesh**2, axis=0)
	r0_pow = -5/3
	f_pow = -(2+turbulence_ndim*3)/6
	factor = 0.000023*10**(turbulence_ndim)
	r0_wavelength_corrected = r0*wav_factor**(-1/r0_pow)
	
	psd = factor*(r0_wavelength_corrected)**(r0_pow)*(1/L0**2 + f_sq)**(f_pow)
	#center_idx = tuple(s//2 for s in psd.shape)
	#psd[center_idx] = 0 # stop infinity at f==0
	return PhasePowerSpectralDensity(psd, f_axes)

