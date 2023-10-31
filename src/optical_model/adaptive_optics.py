
import dataclasses as dc

import numpy as np
import scipy as sp

from optical_model.instrument import OpticalInstrumentModel
from optical_model.atmosphere import TurbulenceModel


# TODO: move to better location
def downsample(arr, s):
	return(sp.signal.convolve(arr, np.ones([s]*arr.ndim)/(s**arr.ndim), mode='valid')[tuple(slice(None,None,s) for ashape in arr.shape)])

# TODO: move to better location
def moffat_function(x, alpha, beta, A):
	return(A*((1+np.sum((x.T/alpha).T**2, axis=0))**(-beta)))


# NOTE:
# Want to turn this into a class where I give it the models for the instrument,
# atmosphere, and adaptive optics. And it returns something I can call in an
# optimisation routine. May need to bind model parameters to positional arguments
# somehow, possibly by using a lambda or one of the routines from the 
# `functools` module 

# TODO: 
# 1) do a better job of separating concerns in this class.
# 2) Write a test for this class, complete with plots so I can tell if something went wrong.
@dc.dataclass
class AdaptiveOpticsModel:
	"""
	Combines the instrumental OTF with the adaptive optics OTF, which is
	calculated from power spectral distribution (PSD) of the TurbulenceModel 
	and the PSD of the AO setup (usually modelled as a moffat function).
	"""
	def __init__(self,
			turbulence_model_cls : TurbulenceModel,
			instrument_model_cls : OpticalInstrumentModel,
			shape : tuple[int,int] = (201,201),
			supersample_factor : int = 1,
		):
		self.turbulence_model_cls = turbulence_model_cls
		self.instrument_model_cls = instrument_model_cls
		self.shape = shape
		self.supersample_factor = supersample_factor
		self.fov_shape = tuple(_s*self.supersample_factor for _s in self.shape)
		
		self.instrument = self.instrument_model_cls(self.shape, expansion_factor=self.supersample_factor, supersample_factor=1)
		self.f_ao = self.instrument.f_ao
		return


	def phase_psd_model(self,
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
		
		if type(alpha) is float:
			alpha = np.ndarray([alpha]*2)
		part1 = (beta - 1)/(np.pi*np.prod(alpha))
		part2 = moffat_function(self.f, alpha, beta, A)
		part3 = (1-(1+np.prod(self.f_ao/alpha))**(1-beta))**(-1)
		print(f'{part1=} {part2=} {part3=}')
		self.psd = part1*part2*part3 + C
		return(self.psd)


	def get_phase_psd(self,
			r0 : float,
			turbulence_ndim : float = 2.0,
			**kwargs
		):
		self.f_select = self.f_mag <= self.f_ao

		self.turbulence_psd = self.turbulence_model_cls(self.f_mag, r0, turbulence_ndim).get_phase_psd()
		self.ao_psd = self.phase_psd_model(**kwargs)
		
		self.psd = np.zeros(self.fov_shape)
		self.psd[self.f_select] = self.ao_psd[self.f_select]
		self.psd[~self.f_select] = self.turbulence_psd[~self.f_select]
		return(self.psd)


	def get_otf_ao(self, 
			r0, 
			turbulence_ndim=2, 
			mode : 
			str ='adjust', 
			**kwargs
		):
		self.get_phase_psd(r0, turbulence_ndim, **kwargs)
		self.phase_autocorr = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(self.psd)))

		if mode == 'classic':
			center_point = tuple(_s//2 for _s in self.phase_autocorr.shape)
			self.otf_ao = np.exp(self.phase_autocorr - self.phase_autocorr[center_point])
		elif mode ==  'adjust':
			self.otf_ao = self.phase_autocorr
		return(self.otf_ao)

	def get_otf_full(self, 
			wavelength, 
			r0, 
			turbulence_ndim=2, 
			mode='adjust', 
			**kwargs
		):
		self.rho_axes = self.instrument.rho_per_lambda_axes*wavelength
		self.rho = np.array(np.meshgrid(self.rho_axes))

		self.f_axes = np.array([np.fft.fftshift(np.fft.fftfreq(_x.size, (_x[1]-_x[0])*self.supersample_factor)) for _x in self.rho_axes])
		self.f = np.array(np.meshgrid(*self.f_axes))
		self.f_mag = np.sqrt(np.sum(self.f**2,axis=0))
		
		self.get_otf_ao(r0, turbulence_ndim, mode, **kwargs)
		self.otf_t = self.instrument.get_otf(wavelength)
		self.otf_full = self.otf_ao * self.otf_t

		return(self.otf_full)


	def get_psf(self, 
			wavelength, 
			r0, 
			turbulence_ndim=2, 
			mode='adjust', 
			**kwargs
		):

		self.get_otf_full(wavelength, r0, turbulence_ndim, mode, **kwargs)
		self.psf_t = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(self.otf_t)))
		self.psf_ao = np.fft.fftshift(np.fft.ifftn(self.otf_ao))
		self.psf_full = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(self.otf_full)))

		self.psf = downsample(self.psf_full,self.supersample_factor) if self.supersample_factor > 1 else self.psf_full
		return(self.psf/np.nansum(self.psf))
