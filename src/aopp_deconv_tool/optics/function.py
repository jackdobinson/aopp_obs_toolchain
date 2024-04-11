"""
Contains type definitions of optical quantities. Useful for static type checking
and making sure that we are sending the correct data somewhere.
"""

import numpy as np

from aopp_deconv_tool.optics.geometric.optical_component import OpticalComponentSet
from aopp_deconv_tool.geo_array import GeoArray

import aopp_deconv_tool.cfg.logs

_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'WARN')


class PupilFunction(GeoArray):
	"""
	Represents the pupil function of an optical system
	"""

	

class PointSpreadFunction(GeoArray):
	"""
	Represents the point spread function of an optical system
	"""
	@classmethod
	def from_pupil_function(cls,
			  pupil_function : PupilFunction
		):
		"""
		Calculate the PSF from a pupil function
		"""
		pf_fft = pupil_function.fft()
		return cls(
			np.conj(pf_fft.data)*pf_fft.data,
			pf_fft.axes,
		)

	@classmethod
	def from_optical_transfer_function(cls,
			optical_transfer_function,
		):
		"""
		Calculate the PSF from the optical transfer function of an optical system
		"""
		otf_ifft = optical_transfer_function.ifft()
		
		return cls(
			otf_ifft.data,
			otf_ifft.axes,
		)
		

class OpticalTransferFunction(GeoArray):
	@classmethod
	def from_psf(cls, point_spread_function : PointSpreadFunction):
		"""
		Get the optical transfer function from a point_spread_function of an optical system
		"""
		psf_fft = point_spread_function.fft()
		return cls(
			psf_fft.data,
			psf_fft.axes,#/wavelength
		)


class PhasePowerSpectralDensity(GeoArray):
	"""
	Represents the phase power spectral density of an optical system
	"""
	


