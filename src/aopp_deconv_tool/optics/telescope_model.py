
import numpy as np

from aopp_deconv_tool.geo_array import plot_ga
from aopp_deconv_tool.optics.function import PupilFunction, PointSpreadFunction, OpticalTransferFunction 
from aopp_deconv_tool.optics.geometric.optical_component import OpticalComponentSet

import aopp_deconv_tool.cfg.logs
_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'INFO')


def pupil_function_of_optical_component_set(
		shape : tuple[int,...],
		expansion_factor : float,
		supersample_factor : float,
		ocs : OpticalComponentSet,
		scale : tuple[float,...],
	) -> PupilFunction:
	"""
	Get the pupil function of an optical component set
	"""
	if scale is None: 
		scale = ocs.get_pupil_function_scale(expansion_factor)
	return PupilFunction(
		ocs.pupil_function(shape, scale, expansion_factor, supersample_factor),
		tuple(np.linspace(-scale*expansion_factor/2,scale*expansion_factor/2,int(s*expansion_factor*supersample_factor)) for scale, s in zip(scale, shape)),
	)


def optical_transfer_function_of_optical_component_set(
		shape : tuple[int,...],
		expansion_factor : float,
		supersample_factor : float,
		ocs : OpticalComponentSet,
		scale : tuple[float,...],
	) -> OpticalTransferFunction:
	_lgr.debug(f'{shape=} {scale=}')
	pupil_function_axes = tuple(np.fft.fftshift(np.fft.fftfreq(sh, sc/sh)) for sh, sc in zip(shape,scale))
	pupil_function_scale = np.array([x[-1] - x[0] for x in pupil_function_axes])
	_lgr.debug(f'{pupil_function_scale=}')
	
	
	pupil_function = pupil_function_of_optical_component_set(
		shape,
		expansion_factor,
		supersample_factor,
		ocs,
		pupil_function_scale
	)
	# DEBUGGING
	#plot_ga(pupil_function)
	
	# Note PSF for a disk should be an Airy disk, the location of the first minimum, theta ~ 1.22*wavelength/obj_diameter
	# so theta/wavelength ~ 1.22/obj_diameter = 0.1525 for obj_diameter = 8
	# remember that this is approximate, not exact.
	psf = PointSpreadFunction.from_pupil_function(pupil_function)
	otf = OpticalTransferFunction.from_psf(psf)
	otf.data /= np.nansum(otf.data)
	return otf
