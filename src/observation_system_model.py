"""
Models the observation system in as much detail required to get a PSF.
"""

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




class PupilFunction:
	def __init__(self,
			data : np.ndarray,
			axes : np.ndarray
		):
		self.data = data # transparancy of the pupil
		self.axes = axes # physical size of the pupil (normally the same as objective lens)
	
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

class PointSpreadFunction:
	"""
	self.axes is in rho/wavelength
	"""
	
	def __init__(self,
			data,
			axes
		):
		self.data = data/np.nansum(data) # response of system to a point source
		self.axes = axes # angle from center of field divided by wavelength of light (rho/wavelength)
	
	@classmethod
	def from_pupil_function(cls,
			  pupil_function : PupilFunction
		):
		pf_fft = np.fft.fftshift(np.fft.fftn(pupil_function.data))
		return cls(
			np.abs(np.conj(pf_fft)*pf_fft),
			np.array([np.fft.fftshift(np.fft.fftfreq(x.size, x[1]-x[0])) for x in pupil_function.axes])
		)

	@classmethod
	def from_optical_transfer_function(cls,
			optical_transfer_function,
			wavelength
		):
		return cls(
			np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(optical_transfer_function.data))),
			np.array([np.fft.fftshift(np.fft.fftfreq(x.size, (x[1]-x[0])))/wavelength for x in optical_transfer_function.axes])
		)
		

class OpticalTransferFunction:
	def __init__(self,
			data,
			axes
		):
		self.data = data # frequency response of system
		self.axes = axes # units of spatial frequency (f) as wavelength has been removed at this point
	
	@classmethod
	def from_psf(cls, point_spread_function : PointSpreadFunction, wavelength : float):
		"""
		Assumes that point_spread_function.axes is in rho/wavelength
		"""
		return cls(
			np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(point_spread_function.data))),
			np.array([np.fft.fftshift(np.fft.fftfreq(x.size, (x[1]-x[0])*wavelength)) for x in point_spread_function.axes])
		)

class PhasePowerSpectralDensity:
	def __init__(self,
			data,
			axes
		):
		self.data = data # power per frequency component
		self.axes = axes # Units of frequency (f)
	





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
	plt.imshow(pupil_function.data)
	plt.show()
	
	psf = PointSpreadFunction.from_pupil_function(pupil_function)
	print(f'{psf.axes.shape}')
	plt.imshow(psf.data)
	plt.show()
	
	otf = OpticalTransferFunction.from_psf(psf, obs_wavelength)
	print(f'{otf.axes.shape=} {otf.axes=}')
	plt.imshow(otf.data.real)
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
	
	atm_psd = atmosphere_psd(0.1, 2, otf.axes)
	plt.imshow(atm_psd.data)
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
		np.array([1,1]),
		2,
		1,
		1
	)
	print(f'{f_ao=} {ao_model_psd.axes=}')
	plt.imshow(ao_model_psd.data)
	plt.show()
