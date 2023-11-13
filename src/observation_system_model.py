"""
Models the observation system in as much detail required to get a PSF.
"""
import itertools as it
from typing import Callable

import numpy as np
import scipy as sp
import scipy.ndimage
import matplotlib
import matplotlib.figure, matplotlib.axes

from optical_model.optical_component import OpticalComponentSet
import cfg.logs

_lgr = cfg.logs.get_logger_at_level(__name__, 'DEBUG')

def axes_to_scale(axes : np.ndarray) -> tuple[float,...]:
	return tuple(x[-1]-x[0] for x in axes)


def scale_to_axes(scale : tuple[float,...], shape : tuple[int,...]) -> np.ndarray:
	return np.array(
		[np.linspace(-scale/2,scale/2,s) for scale, s in zip(scale,shape)]
	)

def axes_to_mesh(axes : np.ndarray) -> np.ndarray:
	return np.array(np.meshgrid(*axes))

def pupil_function_axes_from_observation_params(
		obs_wavelength : float, # the wavelength of the observation
		obs_shape : tuple[int,...], # The shape of an observation (pixels)
		obs_step : np.ndarray # The angular size of a pixel (radians)
	):
	obs_scale = np.array(obs_shape)*np.array(obs_step)/obs_wavelength # the scale of the observation in rho per lambda
	obs_axes = scale_to_axes(obs_scale, obs_shape) # rho per lambda
	
	# whatever units the pupil function scale is in. Presumably meters?
	pupil_function_axes = np.array([np.fft.fftshift(np.fft.fftfreq(x.size, x[1]-x[0])) for x in obs_axes])
	return pupil_function_axes




class GeoArray:
	"""
	A Geometric Array. An array of data, along with coordinate axes that 
	describe the position of the data.
	"""
	def __init__(self,
			data : np.ndarray,
			axes : np.ndarray | None,
		):
		self.data = data
		if axes is None:
			self.axes = np.array([np.linspace(-s/2, s/2, s) for s in self.data.shape()])
		else:
			self.axes = axes
		
		print(f'{self.data.ndim=} {self.data.shape=} {self.axes.ndim=} {self.axes.shape=}')
	
	def __array__(self):
		return self.data
	
	def fft(self):
		return GeoArray(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(self.data))), np.array([np.fft.fftshift(np.fft.fftfreq(x.size, x[1]-x[0])) for x in self.axes]))
	
	def ifft(self):
		return GeoArray(
			np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(self.data))),
			np.array([np.fft.fftshift(np.fft.fftfreq(x.size, x[1]-x[0])) for x in self.axes])
		)
	
	@property
	def extent(self):
		return tuple(it.chain.from_iterable((x[0],x[-1]) for x in self.axes))
	
	def plot(self,
				fig : matplotlib.figure.Figure, 
				data_mutator : Callable[[np.ndarray], np.ndarray] = lambda x: x, 
				axes_mutator : Callable[[np.ndarray], np.ndarray] = lambda x: x,
				data_units : str | None = None,
				axes_units : tuple[str,...] | None = None,
			) -> np.ndarray[matplotlib.axes.Axes]:
		"""
		Plot the array as an image. Only works for 2d arrays at present. Will call
		a mutator on the data before plotting. Returns a numpy array of the axes
		created for the plot
		"""
		if self.data.ndim != 2:
			raise NotImplementedError("Geometric array plotting only works for 2d arrays at present.")
		
		a = fig.subplots(2,2,squeeze=True).flatten()
		a[-1].remove()
		
		center_idx = tuple(s//2 for s in self.data.shape)
		
		m_data = data_mutator(self.data)
		m_axes = axes_mutator(self.axes)
		
		a[0].set_title('array data')
		a[0].imshow(m_data, extent=self.extent)
		a[0].set_xlabel(axes_units[0])
		a[0].set_ylabel(axes_units[1])
		
		a[1].set_title('x=constant centerline')
		a[1].plot(m_data[:, center_idx[1]], m_axes[1])
		a[1].set_xlabel(data_units)
		a[1].set_ylabel(axes_units[1])
		
		a[2].set_title('y=constant centerline')
		a[2].plot(m_axes[0], m_data[center_idx[0], :])
		a[2].set_xlabel(axes_units[0])
		a[2].set_ylabel(data_units)
		
		return a
		
		
		


class PupilFunction(GeoArray):
	@classmethod
	def from_optical_component_set(cls,
			ocs : OpticalComponentSet,
			shape = (101,101),
			scale = None,
			expansion_factor = 7,
			supersample_factor = 1/7
		):
		
		if scale is None: 
			scale = ocs.get_pupil_function_scale(expansion_factor)
		_lgr.debug(f'{scale=}')
		return cls(
			ocs.pupil_function(shape, scale, expansion_factor, supersample_factor),
			np.array(
				[np.linspace(-scale*expansion_factor/2,scale*expansion_factor/2,s*expansion_factor*supersample_factor) for scale, s in zip(scale, shape)]
			)
		)

class PointSpreadFunction(GeoArray):
	"""
	self.axes is in rho/wavelength
	"""
	@classmethod
	def from_pupil_function(cls,
			  pupil_function : PupilFunction
		):
		pf_fft = pupil_function.fft()
		return cls(
			np.abs(np.conj(pf_fft.data)*pf_fft.data),
			pf_fft.axes
		)

	@classmethod
	def from_optical_transfer_function(cls,
			optical_transfer_function,
			wavelength
		):
	
		otf_ifft = optical_transfer_function.ifft()
		
		return cls(
			otf_ifft.data,
			otf_ifft.axes#*wavelength
		)
		

class OpticalTransferFunction(GeoArray):
	@classmethod
	def from_psf(cls, point_spread_function : PointSpreadFunction, wavelength : float):
		"""
		Assumes that point_spread_function.axes is in rho/wavelength
		"""
		psf_fft = point_spread_function.fft()
		return cls(
			psf_fft.data,
			psf_fft.axes#/wavelength
		)

class PhasePowerSpectralDensity(GeoArray):
	pass
	





if __name__=='__main__':
	import matplotlib.pyplot as plt
	from geometry.shape import Circle
	from optical_model.optical_component import OpticalComponentSet, Aperture, Obstruction,Refractor,LightBeam,LightBeamSet
	from optical_model.atmosphere import KolmogorovTurbulence2
	
	
	
	
	
	
	
	obj_diameter = 8 # meters
	primary_mirror_focal_length = 120 # meters
	primary_mirror_pos = 100 # meters
	primary_mirror_diameter = 8
	secondary_mirror_diameter_frac_of_primary = 2.52/18
	secondary_mirror_dist_from_primary =50 # meters
	secondary_mirror_diameter_meters = primary_mirror_diameter*secondary_mirror_diameter_frac_of_primary
	ocs = OpticalComponentSet.from_components([
		Aperture(
			0, 
			'objective aperture', 
			shape=Circle.of_radius(obj_diameter/2)
		), 
		Obstruction(
			primary_mirror_pos - secondary_mirror_dist_from_primary, 
			'secondary mirror back', 
			shape=Circle.of_radius(secondary_mirror_diameter_meters/2)
		), 
		Refractor(
			primary_mirror_pos, 
			'primary mirror', 
			shape=Circle.of_radius(primary_mirror_diameter/2), 
			focal_length=primary_mirror_focal_length
		),
	])
	
	
	# Pretend we have an observation we are fitting to
	obs_shape = (101,101)
	obs_wavelength = 5E-7 #meters
	obs_pixel_size = 0.0125 / (60*60) *np.pi/180 # 0.025 arcsec in radians
	pf_axes = pupil_function_axes_from_observation_params(obs_wavelength, obs_shape, obs_pixel_size)
	pf_scale = axes_to_scale(pf_axes)
	print(f'{pf_scale=}')
	
	
	# These are not propagated through the calcuations correctly at the moment
	expansion_factor = 3
	supersample_factor=3
	
	
	# diffration limited telescope optics
	pupil_function = PupilFunction.from_optical_component_set(
		ocs, 
		obs_shape, 
		pf_scale, 
		expansion_factor=expansion_factor, 
		supersample_factor=supersample_factor
	)
	print(pupil_function.data)
	print(f'{pupil_function.axes.shape=}')
	f = plt.gcf()
	f.suptitle('pupil_function')
	pupil_function.plot(f, data_mutator=lambda x : np.abs(x), data_units='arbitrary units', axes_units=('meters', 'meters'))
	plt.show()
	
	
	
	
	
	psf = PointSpreadFunction.from_pupil_function(pupil_function)
	print(f'{psf.axes.shape}')
	plt.clf()
	f = plt.gcf()
	f.suptitle('psf')
	psf.plot(f, data_mutator=lambda x : np.abs(x), data_units='arbitrary units', axes_units=('rho/wavelength', 'rho/wavelength'))
	plt.show()
	
	otf = OpticalTransferFunction.from_psf(psf, obs_wavelength)
	print(f'{otf.axes.shape=} {otf.axes=}')
	
	plt.clf()
	f = plt.gcf()
	f.suptitle('otf')
	otf.plot(f, data_mutator=lambda x : np.abs(x), data_units='arbitrary units', axes_units=('wavelength/rho', 'wavelength/rho'))
	plt.show()
	
	
	
	# Atmospheric blurring of phase power spectral density
	def atmosphere_psd(r0, turbulence_ndim, f_axes):
		f_mesh = axes_to_mesh(f_axes)
		f_mag = np.sqrt(np.sum(f_mesh**2, axis=0))
		r0_pow = -5/3
		f_pow = -(2+turbulence_ndim*3)/3
		factor = 0.000023*10**(turbulence_ndim)
		psd = factor*(r0)**(r0_pow)*(f_mag)**(f_pow)
		return PhasePowerSpectralDensity(psd, f_axes)
	
	atm_psd = atmosphere_psd(
		0.17, #1, 
		2, 
		otf.axes
	)
	
	plt.clf()
	f = plt.gcf()
	f.suptitle('atm_psd')
	atm_psd.plot(f, data_mutator=lambda x : np.log(x), data_units='arbitrary units', axes_units=('wavelength/rho', 'wavelength/rho'))
	plt.show()
	
	
	
	# adaptive optics corrections to atmosphere phase power spectral density
	n_actuators = 24
	f_ao = n_actuators / (2*obj_diameter)
	
	def moffat_function(x, alpha, beta, A):
		return(A*((1+np.sum((x.T/alpha).T**2, axis=0))**(-beta)))
	
	def ao_phase_psd_model(
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
		f_mesh = axes_to_mesh(f_axes)
		
		if type(alpha) is float:
			alpha = np.ndarray([alpha]*2)
		part1 = (beta - 1)/(np.pi*np.prod(alpha))
		part2 = moffat_function(f_mesh, alpha, beta, A)
		part3 = (1-(1+np.prod(f_ao/alpha))**(1-beta))**(-1)
		print(f'{part1=} {part2=} {part3=}')
		psd = part1*part2*part3 + C
		return PhasePowerSpectralDensity(data=psd, axes=f_axes)
	
	ao_model_psd = ao_phase_psd_model(
		otf.axes,
		f_ao,
		np.array([5E-2,5E-2]),
		1.6,
		2E-3,
		0.05
	)
	print(f'{f_ao=} {ao_model_psd.axes=}')
	
	plt.clf()
	f = plt.gcf()
	f.suptitle('ao_model_psd')
	ao_model_psd.plot(f, data_mutator=lambda x : np.log(x), data_units='arbitrary units', axes_units=('wavelength/rho', 'wavelength/rho'))
	plt.show()
	
	# Model adaptive optics correction to phase power spectral density
	def ao_correction(
			f_axes,
			f_ao,
			atm_psd,
			ao_model_psd
		):
	
		f_mesh = axes_to_mesh(f_axes)
		f_mag = np.sqrt(np.sum(f_mesh**2, axis=0))
		f_ao_correct = f_mag <= f_ao
		ao_corrected_psd = GeoArray(atm_psd.data, atm_psd.axes)
		ao_corrected_psd.data[f_ao_correct] = ao_model_psd.data[f_ao_correct]
		return ao_corrected_psd
	
	ao_corrected_psd = ao_correction(otf.axes, f_ao, atm_psd, ao_model_psd)
	
	plt.clf()
	f = plt.gcf()
	f.suptitle('ao_corrected_psd')
	ao_corrected_psd.plot(f, data_mutator=lambda x : np.log(x), data_units='arbitrary units', axes_units=('wavelength/rho', 'wavelength/rho'))
	plt.show()
	
	# The axes units are a bit funky here. I am not sure I have got them correct.
	phase_autocorr = ao_corrected_psd.ifft()
	#phase_autocorr.data = np.abs(phase_autocorr.data)
	
	mode = "classic"
	match mode:
		case "classic":
			center_point = tuple(_s//2 for _s in phase_autocorr.data.shape)
			otf_atm_ao_corrected = GeoArray(np.exp(phase_autocorr.data - phase_autocorr.data[center_point]), phase_autocorr.axes)

			print(f'{otf_atm_ao_corrected.data[center_point]=}')
			
			"""
			ndim = otf_atm_ao_corrected.data.ndim
			counter = 0
			accumulator = 0
			for i in range(ndim):
				for j in (-1,1):
					delta = tuple(0 if i!=k else j for k in range(ndim))
					accumulator += otf_atm_ao_corrected.data[tuple(c+d for c,d in zip(center_point, delta))]
					counter += 1
			print(f'{ndim=} {counter=} {accumulator=}')
			otf_atm_ao_corrected.data[center_point] = accumulator / counter
			"""
			
			#otf_atm_ao_corrected.data = sp.ndimage.gaussian_filter(otf_atm_ao_corrected.data, sigma=1)
			#otf_atm_ao_corrected.data = sp.ndimage.minimum_filter(np.abs(otf_atm_ao_corrected.data), size=3)
			#otf_atm_ao_corrected.data[center_point] = 0
			print(f'{otf_atm_ao_corrected.data[center_point]=}')
		case "adjust":
			otf_atm_ao_corrected = GeoArray(np.abs(phase_autocorr.data), phase_autocorr.axes)
	
	plt.clf()
	f = plt.gcf()
	f.suptitle('otf_atm_ao_corrected')
	otf_atm_ao_corrected.plot(f, data_mutator=lambda x : np.log(np.abs(x)), data_units='arbitrary units', axes_units=('wavelength/rho', 'wavelength/rho'))
	plt.show()
	
	
	
	# Combination of diffraction-limited optics, atmospheric effects, AO correction to atmospheric effects
	otf_full = GeoArray(otf_atm_ao_corrected.data * otf.data, otf.axes)
	
	plt.clf()
	f = plt.gcf()
	f.suptitle('otf_full')
	otf_full.plot(f,data_mutator=lambda x : np.log(np.abs(x)), data_units='arbitrary units', axes_units=('wavelength/rho', 'wavelength/rho'))
	plt.show()
	
	
	
	
	psf_full = otf_full.ifft() # in units of rho/lambda
	
	plt.clf()
	f = plt.gcf()
	f.suptitle('psf_full')
	psf_full.plot(f, data_mutator=lambda x : np.log(np.abs(x)), data_units='arbitrary units', axes_units=('rho/wavelength', 'rho/wavelength'))
	plt.show()
	
	
	#psf_full_filtered = GeoArray(sp.ndimage.gaussian_filter(np.abs(psf_full.data), sigma=9), psf_full.axes)
	psf_full_min_filtered = GeoArray(sp.ndimage.minimum_filter(np.abs(psf_full.data), size=9), psf_full.axes)
	
	plt.clf()
	f = plt.gcf()
	f.suptitle('psf_full_min_filtered')
	psf_full_min_filtered.plot(f, data_mutator=lambda x : np.log(np.abs(x)), data_units='arbitrary units', axes_units=('rho/wavelength', 'rho/wavelength'))
	plt.show()
	
	psf_full_max_filtered = GeoArray(sp.ndimage.maximum_filter(np.abs(psf_full.data), size=9), psf_full.axes)
	
	plt.clf()
	f = plt.gcf()
	f.suptitle('psf_full_max_filtered')
	psf_full_max_filtered.plot(f, data_mutator=lambda x : np.log(np.abs(x)), data_units='arbitrary units', axes_units=('rho/wavelength', 'rho/wavelength'))
	plt.show()
	
	psf_full_uni_filtered = GeoArray(sp.ndimage.uniform_filter(np.abs(psf_full.data), size=9), psf_full.axes)
	
	plt.clf()
	f = plt.gcf()
	f.suptitle('psf_full_uni_filtered')
	psf_full_uni_filtered.plot(f, data_mutator=lambda x : np.log(np.abs(x)), data_units='arbitrary units', axes_units=('rho/wavelength', 'rho/wavelength'))
	plt.show()
