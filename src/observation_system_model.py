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
import numpy_helper as nph
import numpy_helper.array

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

def moffat_function(x, alpha, beta, A):
	"""
	Used to model the phase corrections of adaptive optics systems
	
	Alpha has a similar effect to beta on the modelling side, they do different things, but they are fairly degenerate.
	"""
	return(A*((1+np.sum((x.T/alpha).T**2, axis=0))**(-beta)))


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
	
	def copy(self):
		return GeoArray(np.array(self.data), np.array(self.axes))
	
	def __array__(self):
		return self.data
	
	def fft(self):
		return GeoArray(
			np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(self.data))),
			np.array([np.fft.fftshift(np.fft.fftfreq(x.size, x[1]-x[0])) for x in self.axes]),
		)
	
	def ifft(self):
		return GeoArray(
			np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(self.data))),
			np.array([np.fft.fftshift(np.fft.fftfreq(x.size, x[1]-x[0])) for x in self.axes]),
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
		
		a[3].set_title('array data unmutated')
		a[3].imshow(self.data if self.data.dtype!=np.dtype(complex) else np.abs(self.data), extent=self.extent)
		a[3].set_xlabel(axes_units[0])
		a[3].set_ylabel(axes_units[1])
		
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
				[np.linspace(-scale*expansion_factor/2,scale*expansion_factor/2,int(s*expansion_factor*supersample_factor)) for scale, s in zip(scale, shape)]
			),
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
			np.conj(pf_fft.data)*pf_fft.data,
			pf_fft.axes,
		)

	@classmethod
	def from_optical_transfer_function(cls,
			optical_transfer_function,
			wavelength
		):
	
		otf_ifft = optical_transfer_function.ifft()
		
		return cls(
			otf_ifft.data,
			otf_ifft.axes,#*wavelength
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
			psf_fft.axes,#/wavelength
		)

class PhasePowerSpectralDensity(GeoArray):
	@classmethod
	def of_von_karman_turbulence(cls,
			f_axes, 
			r0, 
			turbulence_ndim,
			L0
		):
		"""
		Von Karman turbulence is the same as Kolmogorov, but with the extra "L0"
		term. L0 -> infinity gives Kolmogorov turbulence.
		"""
		f_mesh = axes_to_mesh(f_axes)
		f_sq = np.sum(f_mesh**2, axis=0)
		r0_pow = -5/3
		f_pow = -(2+turbulence_ndim*3)/6
		factor = 0.000023*10**(turbulence_ndim)
		psd = factor*(r0)**(r0_pow)*(1/L0**2 + f_sq)**(f_pow)
		center_idx = tuple(s//2 for s in psd.shape)
		psd[center_idx] = 0 # stop infinity at f==0
		return cls(psd, f_axes)
	
	@classmethod
	def of_adaptive_optics_moffat_function_model(cls,
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
		return cls(data=psd, axes=f_axes)
	
	@classmethod
	def from_atmospheric_phase_psd_with_adaptive_optics_corrections(cls,
			f_axes : np.ndarray, # frequency axes over which to evaluate the corrections.
			f_ao : float, # highest frequency that the adaptive optics can adjust for, below this value the corrected PSD will be from adaptive optics, above this value the PSD will from the atmosphere.
			atm_psd : GeoArray, # power spectral density of the atmosphere's influence on the phase of light going into the telescope
			ao_model_psd : GeoArray # power spectral density of the corrections the adaptive optics system performs
		):
		"""
		Note: if have differing axes for atm_psd and ao_model_psd, will need
		to interpolate to a common f_axes.
		"""
		if ao_model_psd is None: return atm_psd.copy()
		f_mesh = axes_to_mesh(f_axes)
		f_mag = np.sqrt(np.sum(f_mesh**2, axis=0))
		f_ao_correct = f_mag <= f_ao
		ao_corrected_psd = np.array(atm_psd.data)
		ao_corrected_psd[f_ao_correct] = ao_model_psd.data[f_ao_correct]
		return cls(ao_corrected_psd, f_axes)





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
	obs_shape = (201,201)
	obs_wavelength = 5E-7 #meters
	obs_pixel_size = 0.0125 / (60*60) *np.pi/180 # 0.025 arcsec in radians
	pf_axes = pupil_function_axes_from_observation_params(obs_wavelength, obs_shape, obs_pixel_size)
	pf_scale = axes_to_scale(pf_axes)
	print(f'{pf_scale=}')
	
	
	# These are not propagated through the calcuations correctly at the moment
	expansion_factor = 3 # increasing this one gives best results normally
	supersample_factor=1
	
	
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
	psf.plot(f, data_mutator=lambda x : np.log(np.abs(x)), data_units='arbitrary units', axes_units=('rho/wavelength', 'rho/wavelength'))
	plt.show()
	
	f = plt.gcf()
	f.suptitle('psf centering investigation')
	plt.plot(psf.axes[0], psf.data[:,psf.data.shape[1]//2])
	x = np.zeros_like(psf.data)
	x[tuple(s//2 for s in x.shape)] = 1
	x *= np.max(psf.data) - np.min(psf.data)
	x += np.min(psf.data)
	plt.plot(psf.axes[0], psf.data[:,psf.data.shape[1]//2])
	plt.plot(psf.axes[0], x[:,x.shape[1]//2])
	plt.show()
	
	
	
	
	
	otf = OpticalTransferFunction.from_psf(psf, obs_wavelength)
	otf.data /= np.nansum(otf.data)
	print(f'{otf.axes.shape=} {otf.axes=}')
	
	plt.clf()
	f = plt.gcf()
	f.suptitle('otf')
	otf.plot(f, data_mutator=lambda x : np.abs(x), data_units='arbitrary units', axes_units=('wavelength/rho', 'wavelength/rho'))
	plt.show()
	
	
	otf_copy = GeoArray(otf.data, otf.axes)
	psf_copy = otf_copy.ifft()
	plt.clf()
	f = plt.gcf()
	f.suptitle('psf_from_otf')
	psf_copy.plot(f, data_mutator=lambda x : np.log(np.abs(x)), data_units='arbitrary units', axes_units=('rho/wavelength', 'rho/wavelength'))
	plt.show()
	
	
	
	# Phase power spectral density axes
	f_axes = psf.axes
	
	# Atmospheric blurring of phase power spectral density
	atm_psd = PhasePowerSpectralDensity.of_von_karman_turbulence(
		f_axes,
		0.17, 
		2,
		8
	)
	
	plt.clf()
	f = plt.gcf()
	f.suptitle('atm_psd')
	atm_psd.plot(f, data_mutator=lambda x : np.log(x), data_units='arbitrary units', axes_units=('wavelength/rho', 'wavelength/rho'))
	plt.show()
	
	
	
	# adaptive optics corrections to atmosphere phase power spectral density
	n_actuators = 24
	f_ao = n_actuators / (2*obj_diameter)
	
	ao_model_psd = PhasePowerSpectralDensity.of_adaptive_optics_moffat_function_model(
		f_axes,
		f_ao,
		np.array([5E-2,5E-2]),#np.array([5E-2,5E-2]),
		1.6,#1.6
		2E-3,#2E-3
		0.05#0.05
	)
	
	print(f'{f_ao=} {ao_model_psd.axes=}')
	
	plt.clf()
	f = plt.gcf()
	f.suptitle('ao_model_psd')
	ao_model_psd.plot(f, data_mutator=lambda x : np.log(x), data_units='arbitrary units', axes_units=('rho/wavelength', 'rho/wavelength'))
	plt.show()
	
	
	ao_corrected_psd = PhasePowerSpectralDensity.from_atmospheric_phase_psd_with_adaptive_optics_corrections(
		f_axes, 
		f_ao, 
		atm_psd, 
		ao_model_psd
	)
	
	plt.clf()
	f = plt.gcf()
	f.suptitle('ao_corrected_psd')
	ao_corrected_psd.plot(f, data_mutator=lambda x : np.log(x), data_units='arbitrary units', axes_units=('rho/wavelength', 'rho/wavelength'))
	plt.show()
	
	# Get the autocorrelation of the phase adjustements from the atmosphere + adaptive optics corrections
	phase_autocorr = ao_corrected_psd.ifft()
	_lgr.debug(f'{phase_autocorr.data=}')
	phase_autocorr.data = phase_autocorr.data.real
	plt.clf()
	f = plt.gcf()
	f.suptitle('phase_autocorr 1')
	phase_autocorr.plot(f, data_mutator=lambda x : np.log(x), data_units='arbitrary units', axes_units=('wavelength/rho', 'wavelength/rho'))
	plt.show()
	#phase_autocorr.data /= np.nansum(phase_autocorr.data)
	
	plt.clf()
	f = plt.gcf()
	f.suptitle('phase_autocorr 2')
	phase_autocorr.plot(f, data_mutator=lambda x : np.log(x), data_units='arbitrary units', axes_units=('wavelength/rho', 'wavelength/rho'))
	plt.show()
	
	mode = "classic"
	match mode:
		case "classic":
			center_point = tuple(_s//2 for _s in phase_autocorr.data.shape)
			otf_atm_ao_corrected_data = np.exp(phase_autocorr.data - phase_autocorr.data[center_point])
			
			# This attenuates the 'spike' that results in an airy disk pattern.
			s_factor = 0#1E-3
			
			# Work out the offset from zero of the OTF
			center_idx_offsets = nph.array.offsets_from_point(otf_atm_ao_corrected_data.shape)
			center_idx_dist = np.sqrt(np.sum(center_idx_offsets**2, axis=0))
			outer_region_mask = center_idx_dist > (center_idx_dist.shape[0]//2)*0.9
			otf_atm_ao_corrected_data_offset_from_zero = np.sum(otf_atm_ao_corrected_data[outer_region_mask])/np.count_nonzero(outer_region_mask)
			
			_lgr.debug(f'{otf_atm_ao_corrected_data_offset_from_zero=}')
			# Subtract the offset to remove the delta-function spike
			otf_atm_ao_corrected_data -= otf_atm_ao_corrected_data_offset_from_zero 
			# If required, add back on a fraction of the maximum to add part of the spike back in.
			otf_atm_ao_corrected_data += s_factor*np.max(otf_atm_ao_corrected_data)
			
			otf_atm_ao_corrected = GeoArray(otf_atm_ao_corrected_data, phase_autocorr.axes)

		case "adjust":
			otf_atm_ao_corrected = GeoArray(np.abs(phase_autocorr.data), phase_autocorr.axes)
	
	plt.clf()
	f = plt.gcf()
	f.suptitle('otf_atm_ao_corrected')
	otf_atm_ao_corrected.plot(f, data_mutator=lambda x : np.log(np.abs(x)), data_units='arbitrary units', axes_units=('wavelength/rho', 'wavelength/rho'))
	plt.show()
	
	# plot ifft of otf_atm_ao_corrected to diagnose problems
	otf_atm_ao_corrected_ifft = otf_atm_ao_corrected.ifft()

	center_idx = tuple(s//2 for s in otf_atm_ao_corrected_ifft.data.shape)
	d = 5
	aslice = tuple(slice(i-d , i+d+ s%2) for i, s in zip(center_idx, otf_atm_ao_corrected_ifft.data.shape))
	center_offsets = nph.array.offsets_from_point(otf_atm_ao_corrected_ifft.data.shape)
	mask = np.sum(center_offsets**2, axis=0) < 0
	otf_atm_ao_corrected_ifft.data[mask]=0

	otf_atm_ao_corrected_ifft.data = np.array(np.abs(otf_atm_ao_corrected_ifft.data), dtype=float)
	otf_atm_ao_corrected_ifft.data = otf_atm_ao_corrected_ifft.data - np.min(otf_atm_ao_corrected_ifft.data)
	otf_atm_ao_corrected_ifft.data /= np.max(otf_atm_ao_corrected_ifft.data)
	otf_atm_ao_corrected_ifft.data[mask]=1
	#otf_atm_ao_corrected_ifft.data += 1
	_lgr.debug(f'{np.min(otf_atm_ao_corrected_ifft.data)=}')
	_lgr.debug(f'{np.max(otf_atm_ao_corrected_ifft.data)=}')
	
	plt.clf()
	f = plt.gcf()
	f.suptitle('otf_atm_ao_corrected_ifft')
	otf_atm_ao_corrected_ifft.plot(f, data_mutator=lambda x : np.log(x), data_units='arbitrary units', axes_units=('rho/wavelength', 'rho/wavelength'))
	plt.show()
	
	
	# Can I use a non-fft way to get the correct OTF?
	# Need to put both lines on the same graph, it's hard to compare this way
	_lgr.debug(f'{np.min(ao_corrected_psd.data)=}')
	ao_corrected_psd.data = (ao_corrected_psd.data)**1
	ao_corrected_psd.data = ao_corrected_psd.data - np.min(ao_corrected_psd.data)
	ao_corrected_psd.data /= np.max(ao_corrected_psd.data)
	_lgr.debug(f'{np.min(ao_corrected_psd.data)=}')
	_lgr.debug(f'{np.max(ao_corrected_psd.data)=}')
	plt.clf()
	f = plt.gcf()
	f.suptitle('ao_corrected_psd')
	ao_corrected_psd.plot(f, data_mutator=lambda x : np.log(x), data_units='arbitrary units', axes_units=('rho/wavelength', 'rho/wavelength'))
	plt.show()
	
	plt.clf()
	f.suptitle('otf_atm_ao_corrected_ifft vs ao_corrected_psd')
	plt.plot(otf_atm_ao_corrected_ifft.axes[0], np.log(otf_atm_ao_corrected_ifft.data[:,otf_atm_ao_corrected_ifft.data.shape[1]//2]))
	plt.plot(ao_corrected_psd.axes[0], np.log(ao_corrected_psd.data[:,ao_corrected_psd.data.shape[1]//2]))
	plt.show()
	
	
	#otf_atm_ao_corrected = otf_atm_ao_corrected_ifft.fft()
	
	# Combination of diffraction-limited optics, atmospheric effects, AO correction to atmospheric effects
	otf_full = GeoArray(otf_atm_ao_corrected.data * otf.data, otf.axes)
	
	plt.clf()
	f = plt.gcf()
	ax = f.subplots(3,3, squeeze=True).flatten()
	ax[0].imshow(np.log(otf.data.real))
	ax[1].imshow(np.log(otf.data.imag))
	ax[2].imshow(np.log(np.abs(otf.data)))
	
	ax[3].imshow(np.log(otf_atm_ao_corrected.data.real))
	ax[4].imshow(np.log(otf_atm_ao_corrected.data.imag))
	ax[5].imshow(np.log(np.abs(otf_atm_ao_corrected.data)))
	
	ax[6].imshow(np.log((otf_full.data).real))
	ax[7].imshow(np.log((otf_full.data).imag))
	ax[8].imshow(np.log(np.abs(otf_full.data)))
	
	plt.show()
	
	
	
	
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
	
	lump = int(expansion_factor*supersample_factor)
	psf_full_uni_filtered = GeoArray(sp.ndimage.uniform_filter(np.abs(psf_full.data), size=lump)[lump//2::lump,lump//2::lump], psf_full.axes[:,lump//2::lump])
	
	plt.clf()
	f = plt.gcf()
	f.suptitle('psf_full_uni_filtered')
	psf_full_uni_filtered.plot(f, data_mutator=lambda x : np.log(np.abs(x)), data_units='arbitrary units', axes_units=('rho/wavelength', 'rho/wavelength'))
	plt.show()
