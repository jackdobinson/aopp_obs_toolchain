
import numpy as np

from optics.function import PupilFunction, PointSpreadFunction, OpticalTransferFunction 
from optics.geometric.optical_component import OpticalComponentSet

import cfg.logs
_lgr = cfg.logs.get_logger_at_level(__name__, 'DEBUG')


def pupil_function_of_optical_component_set(
		shape : tuple[int,...],
		expansion_factor : float,
		supersample_factor : float,
		ocs : OpticalComponentSet,
		scale : tuple[float,...],
	):
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
	):
	_lgr.debug(f'{shape=} {scale=}')
	pupil_function_axes = tuple(np.fft.fftshift(np.fft.fftfreq(sh, sc/sh)) for sh, sc in zip(shape,scale))
	pupil_function_scale = np.array([x[-1] - x[0] for x in pupil_function_axes])
	
	pupil_function = pupil_function_of_optical_component_set(
		shape,
		expansion_factor,
		supersample_factor,
		ocs,
		pupil_function_scale
	)
	psf = PointSpreadFunction.from_pupil_function(pupil_function)
	otf = OpticalTransferFunction.from_psf(psf)
	otf.data /= np.nansum(otf.data)
	return otf
