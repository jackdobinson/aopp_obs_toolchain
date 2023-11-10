"""
Models the observation system in as much detail required to get a PSF.
"""
import itertools as it
import numpy as np

from optical_model.optical_component import OpticalComponentSet


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


class PupilFunction(GeoArray):
	@classmethod
	def from_optical_component_set(cls,
			ocs : OpticalComponentSet,
			shape = (101,101),
			scale = None,
			expansion_factor = 7,
			supersample_factor = 1/7
		):
		
		if scale is None: scale = ocs.get_pupil_function_scale(expansion_factor)
		return cls(
			ocs.pupil_function(shape, scale, expansion_factor, supersample_factor),
			np.array(
				[np.linspace(-scale/2,scale/2,s*expansion_factor*supersample_factor) for scale, s in zip(scale, shape)]
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
	
	
	
	
	# diffration limited telescope optics
	pupil_function = PupilFunction.from_optical_component_set(ocs, obs_shape, pf_scale, expansion_factor=7, supersample_factor=1)
	print(pupil_function.data)
	print(f'{pupil_function.axes.shape=}')
	plt.title('pupil_function')
	plt.imshow(pupil_function, extent=pupil_function.extent)
	plt.show()
	
	
	"""
	geo_array = GeoArray(pupil_function.data, pupil_function.axes)
	print(f'{geo_array.axes}')
	print(f'{geo_array.extent=}')
	plt.title('geo_array')
	plt.imshow(geo_array, extent=geo_array.extent)
	plt.show()
	
	print(f'{geo_array.fft().axes}')
	plt.title('geo_array.fft()')
	plt.imshow(np.abs(geo_array.fft()), extent=geo_array.fft().extent)
	plt.show()
	
	print(f'{geo_array.fft().ifft().axes}')
	plt.title('geo_array.fft().ifft()')
	plt.imshow(np.abs(geo_array.fft().ifft()), extent=geo_array.fft().ifft().extent)
	plt.show()
	"""
	
	
	psf = PointSpreadFunction.from_pupil_function(pupil_function)
	print(f'{psf.axes.shape}')
	plt.title('psf')
	plt.imshow(psf, extent=psf.extent)
	plt.show()
	
	otf = OpticalTransferFunction.from_psf(psf, obs_wavelength)
	print(f'{otf.axes.shape=} {otf.axes=}')
	plt.title('otf')
	plt.imshow(np.abs(otf), extent=otf.extent)
	plt.show()
	
	
	# Atmospheric blurring
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
	plt.title('atm_psd')
	plt.imshow(np.log(atm_psd), extent=atm_psd.extent)
	plt.show()
	
	plt.plot(atm_psd.axes[1], np.log(atm_psd.data[atm_psd.data.shape[0]//2, :]))
	plt.show()
	
	
	
	# adaptive optics corrections
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
	plt.title('ao_model_psd')
	plt.imshow(np.log(ao_model_psd), extent=ao_model_psd.extent)
	plt.show()
	
	plt.plot(ao_model_psd.axes[1], np.log(ao_model_psd.data[ao_model_psd.data.shape[0]//2, :]))
	ax = plt.gca()
	ylim = ax.get_ylim()
	plt.plot(atm_psd.axes[1], np.log(atm_psd.data[atm_psd.data.shape[0]//2, :]))
	ax.set_ylim(*ylim)
	plt.show()
	
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
	plt.title('ao_corrected_psd')
	plt.imshow(np.log(ao_corrected_psd), extent=ao_corrected_psd.extent)
	plt.show()
	
	
	plt.plot(ao_corrected_psd.axes[1], np.log(ao_corrected_psd.data[ao_corrected_psd.data.shape[0]//2, :]))
	plt.show()
	
	
	"""
	self.get_phase_psd(r0, turbulence_ndim, **kwargs)
		self.phase_autocorr = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(self.psd)))

		if mode == 'classic':
			center_point = tuple(_s//2 for _s in self.phase_autocorr.shape)
			self.otf_ao = np.exp(self.phase_autocorr - self.phase_autocorr[center_point])
		elif mode ==  'adjust':
			self.otf_ao = self.phase_autocorr
		return(self.otf_ao)
	"""
	
	phase_autocorr = ao_corrected_psd.ifft()
	
	mode = "adjust"
	match mode:
		case "classic":
			center_point = tuple(_s//2 for _s in phase_autocorr.data.shape)
			otf_atm_ao_corrected = GeoArray(np.exp(phase_autocorr.data - phase_autocorr.data[center_point]), phase_autocorr.axes)
		case "adjust":
			otf_atm_ao_corrected = GeoArray(np.abs(phase_autocorr.data), phase_autocorr.axes)
	
	plt.title('otf_atm_ao_corrected')
	plt.imshow(np.log(np.abs(otf_atm_ao_corrected)), extent=otf_atm_ao_corrected.extent)
	plt.show()
	
	
	plt.plot(otf_atm_ao_corrected.axes[1], np.log(np.abs(otf_atm_ao_corrected.data)[otf_atm_ao_corrected.data.shape[0]//2, :]))
	plt.show()
	
	
	
	otf_full = GeoArray(otf_atm_ao_corrected.data * otf.data, otf.axes)
	
	plt.title('otf_full')
	plt.imshow(np.log(np.abs(otf_full)), extent=otf_full.extent)
	plt.show()
	
	plt.plot(otf_full.axes[1], np.log(np.abs(otf_full.data)[otf_full.data.shape[0]//2, :]))
	plt.show()
	
	
	
	
	psf_full = otf_full.ifft() # in units of rho/lambda
	
	plt.title('psf_full')
	#plt.imshow(np.abs(psf_full), extent=psf_full.extent)
	plt.imshow(np.log(np.abs(psf_full)), extent=psf_full.extent)
	plt.show()
	
	#plt.plot(psf_full.axes[1], np.abs(psf_full.data)[psf_full.data.shape[0]//2, :])
	plt.plot(psf_full.axes[1], np.log(np.abs(psf_full.data)[psf_full.data.shape[0]//2, :]))
	plt.show()
