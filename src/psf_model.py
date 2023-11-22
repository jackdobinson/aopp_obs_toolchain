"""
Models the observation system in as much detail required to get a PSF.
"""
import itertools as it
from typing import Callable

import numpy as np
import scipy as sp
import scipy.ndimage
import scipy.interpolate
import matplotlib
import matplotlib.figure, matplotlib.axes

from optical_model.optical_component import OpticalComponentSet
import cfg.logs
import numpy_helper as nph
import numpy_helper.array

from geo_array import GeoArray

_lgr = cfg.logs.get_logger_at_level(__name__, 'DEBUG')



class PSFModel:
	"""
	Class that calculates a model point spread function.
	"""
	def __init__(self,
			diffraction_limited_otf_model : Callable[[tuple[int,...], float,float,...],GeoArray], # axes in units of wavelength/rho
			atmospheric_turbulence_psd_model : Callable[[np.ndarray,...],GeoArray], # takes in spatial scale axis, returns phase power spectrum distribution
			adaptive_optics_psd_model : Callable[[np.ndarray, ...], GeoArray], # takes in spatial scale axis, returns phase PSD
		):
		self.diffraction_limited_otf_model = diffraction_limited_otf_model
		self.atmospheric_turbulence_psd_model = atmospheric_turbulence_psd_model
		self.adaptive_optics_psd_model = adaptive_optics_psd_model
	
	
	def ao_corrections_to_phase_psd(self, phase_psd, ao_phase_psd, f_ao):
		"""
		Apply adaptive optics corrections to the phase power spectrum distribution of the atmosphere
		"""
		if ao_phase_psd is None: 
			return phase_psd.copy()
		f_mesh = phase_psd.mesh
		f_mag = np.sqrt(np.sum(f_mesh**2, axis=0))
		f_ao_correct = f_mag <= f_ao
		ao_corrected_psd = np.array(phase_psd.data)
		ao_corrected_psd[f_ao_correct] = ao_phase_psd.data[f_ao_correct]
		return PhasePowerSpectralDensity(ao_corrected_psd, phase_psd.axes)
	
	
	def optical_transfer_fuction_from_phase_psd(self, phase_psd, mode='classic', s_factor=0):
		"""
		From a phase power spectral distribution, derive an optical transfer function.
		"""
		phase_autocorr = phase_psd.ifft()
		phase_autocorr.data = phase_autocorr.data.real
		
		match mode:
			case "classic":
				center_point = tuple(_s//2 for _s in phase_autocorr.data.shape)
				otf_data = np.exp(phase_autocorr.data - phase_autocorr.data[center_point])
				
				# Work out the offset from zero of the OTF
				center_idx_offsets = nph.array.offsets_from_point(otf_data.shape)
				center_idx_dist = np.sqrt(np.sum(center_idx_offsets**2, axis=0))
				outer_region_mask = center_idx_dist > (center_idx_dist.shape[0]//2)*0.9
				otf_data_offset_from_zero = np.sum(otf_data[outer_region_mask])/np.count_nonzero(outer_region_mask)
				
				_lgr.debug(f'{otf_data_offset_from_zero=}')
				# Subtract the offset to remove the delta-function spike
				otf_data -= otf_data_offset_from_zero 
				# If required, add back on a fraction of the maximum to add part of the spike back in.
				otf_data += s_factor*np.max(otf_data)
				
				otf = GeoArray(otf_data, phase_autocorr.axes)
	
			case "adjust":
				otf = GeoArray(np.abs(phase_autocorr.data), phase_autocorr.axes)
		return otf


	def __call__(self,
			shape,
			expansion_factor,
			supersample_factor,
			f_ao,
			diffraction_limited_otf_model_args,
			atmospheric_turbulence_psd_model_args,
			adaptive_optics_psd_model_args,
			s_factor=0,
			mode='classic',
			plots=True
		):
		"""
		Calculate psf in terms of rho/wavelength for given parameters.
		"""
		self.shape = shape
		self.expansion_factor = expansion_factor
		self.supersample_factor = supersample_factor
		
		dl_otf = self.diffraction_limited_otf_model(shape, expansion_factor, supersample_factor, *diffraction_limited_otf_model_args)
		if plots: plot_ga(dl_otf, lambda x: np.log(np.abs(x)), 'diffraction limited otf', 'arbitrary units', 'wavelength/rho')
		if plots: plot_ga(dl_otf.ifft(), lambda x: np.log(np.abs(x)), 'diffraction limited psf', 'arbitrary units', 'rho/wavelength')
		
		f_axes = dl_otf.ifft().axes # dl_psf.axes
		
		atm_phase_psd = self.atmospheric_turbulence_psd_model(f_axes, *atmospheric_turbulence_psd_model_args)
		if plots: plot_ga(atm_phase_psd, lambda x: np.log(np.abs(x)), 'atm_phase_psd', 'arbitrary units', 'rho/wavelength')
		
		ao_phase_psd = self.adaptive_optics_psd_model(f_axes, *adaptive_optics_psd_model_args)
		if plots: plot_ga(ao_phase_psd, lambda x: np.log(np.abs(x)), 'ao_phase_psd', 'arbitrary units', 'rho/wavelength')
		
		ao_corrected_atm_phase_psd = self.ao_corrections_to_phase_psd(atm_phase_psd, ao_phase_psd, f_ao)
		if plots: plot_ga(ao_corrected_atm_phase_psd, lambda x: np.log(np.abs(x)), 'ao_corrected_atm_phase_psd', 'arbitrary units', 'rho/wavelength')
		
		ao_corrected_otf = self.optical_transfer_fuction_from_phase_psd(ao_corrected_atm_phase_psd, mode, s_factor)
		
		# Combination of diffraction-limited optics, atmospheric effects, AO correction to atmospheric effects
		otf_full = GeoArray(ao_corrected_otf.data * dl_otf.data, dl_otf.axes)
		if plots: plot_ga(otf_full, lambda x: np.log(np.abs(x)), 'otf full', 'arbitrary units', 'wavelength/rho')
		
		psf_full = otf_full.ifft()
		if plots: plot_ga(psf_full, lambda x: np.log(x), 'psf full', 'arbitrary units', 'rho/wavelength')
		
		self.psf_full = psf_full
		
		return psf_full
	
	def at(self, scale, wavelength, plots=True):
		"""
		Calculate psf for a given angular scale and wavelength.
		"""
		output_axes = np.array([np.linspace(-z/2,z/2,s) for z,s in zip(scale,self.shape)])
		_lgr.debug(f'{output_axes=}')
		
		rho_axes = tuple(a*wavelength for a in self.psf_full.axes)
		_lgr.debug(f'{rho_axes=}')
		interp = sp.interpolate.RegularGridInterpolator(rho_axes, self.psf_full.data,method='linear', bounds_error=False, fill_value=0)
		
		points = np.swapaxes(np.array(np.meshgrid(*output_axes)), 0,-1)
		
		result = GeoArray(np.array(interp(points)), output_axes)
		_lgr.debug(f'{result.data.shape=}')
		
		if plots: plot_ga(result, lambda x: np.log(np.abs(x)), f'psf at {wavelength=}', 'arbitrary units', 'radians')
		
		return result



def plot_ga(
		geo_array : GeoArray, 
		data_mutator : Callable[[np.ndarray], np.ndarray] = lambda x:x, 
		title : str = '', 
		data_units : str = '', 
		axes_units : str | tuple[str,...] = '',
		show : bool = True
	):
	"""
	Plot the array as an image. Only works for 2d arrays at present. Will call
	a mutator on the data before plotting. Returns a numpy array of the axes
	created for the plot
	"""
	if type(axes_units) is not tuple:
		axes_units = tuple(axes_units for _ in range(geo_array.data.ndim))
	plt.clf()
	f = plt.gcf()
	f.suptitle(title)
	
	if geo_array.data.ndim != 2:
		raise NotImplementedError("Geometric array plotting only works for 2d arrays at present.")
		
	a = f.subplots(2,2,squeeze=True).flatten()
	
	center_idx = tuple(s//2 for s in geo_array.data.shape)
	
	m_data = data_mutator(geo_array.data)
	m_axes = geo_array.axes
	
	a[0].set_title('array data')
	a[0].imshow(m_data, extent=geo_array.extent)
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
	a[3].imshow(geo_array.data if geo_array.data.dtype!=np.dtype(complex) else np.abs(geo_array.data), extent=geo_array.extent)
	a[3].set_xlabel(axes_units[0])
	a[3].set_ylabel(axes_units[1])
	
	
	if show: plt.show()
	return f, a



def moffat_function(x, alpha, beta, A):
	"""
	Used to model the phase corrections of adaptive optics systems
	
	Alpha has a similar effect to beta on the modelling side, they do different things, but they are fairly degenerate.
	"""
	return(A*((1+np.sum((x.T/alpha).T**2, axis=0))**(-beta)))



class PupilFunction(GeoArray):
	"""
	Represents the pupil function of a telescope
	"""
	
	@classmethod
	def from_optical_component_set(cls,
			ocs : OpticalComponentSet,
			shape = (101,101),
			scale = None,
			expansion_factor = 7,
			supersample_factor = 1/7
		):
		"""
		Get the pupil function of an optical component set
		"""
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
			wavelength
		):
		"""
		Calculate the PSF from an optical transfer function
		"""
		otf_ifft = optical_transfer_function.ifft()
		
		return cls(
			otf_ifft.data,
			otf_ifft.axes,#*wavelength
		)
		

class OpticalTransferFunction(GeoArray):
	@classmethod
	def from_psf(cls, point_spread_function : PointSpreadFunction):
		"""
		Get the optical transfer function from a point_spread_function
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
		f_mesh = np.array(np.meshgrid(*f_axes))
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
		f_mesh = np.array(np.meshgrid(*f_axes))
		
		if type(alpha) is float:
			alpha = np.ndarray([alpha]*2)
		part1 = (beta - 1)/(np.pi*np.prod(alpha))
		part2 = moffat_function(f_mesh, alpha, beta, A)
		part3 = (1-(1+np.prod(f_ao/alpha))**(1-beta))**(-1)
		print(f'{part1=} {part2=} {part3=}')
		psd = part1*part2*part3 + C
		return cls(data=psd, axes=f_axes)






		
	
	
def diffraction_limited_otf_model(shape, expansion_factor, supersample_factor, ocs, pf_scale):
	pupil_function = PupilFunction.from_optical_component_set(
		ocs, 
		obs_shape, 
		pf_scale, 
		expansion_factor=expansion_factor, 
		supersample_factor=supersample_factor
	)
	diffraction_limited_psf = PointSpreadFunction.from_pupil_function(pupil_function)
	dl_otf = OpticalTransferFunction.from_psf(diffraction_limited_psf)
	dl_otf.data /= np.nansum(dl_otf.data)
	return dl_otf
	
		
		



if __name__=='__main__':
	import matplotlib.pyplot as plt
	from geometry.shape import Circle
	from optical_model.optical_component import OpticalComponentSet, Aperture, Obstruction,Refractor,LightBeam,LightBeamSet
	from optical_model.atmosphere import KolmogorovTurbulence2
	
	
	
	
	
	
	# Define parameters of the instrument
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
	
	
	
	obs_scale = np.array(obs_shape)*np.array(obs_pixel_size)/obs_wavelength
	pupil_function_axes = np.array([np.fft.fftshift(np.fft.fftfreq(shape, scale/shape)) for shape, scale in zip(obs_shape,obs_scale)])
	pupil_function_scale = np.array([x[-1] - x[0] for x in pupil_function_axes])
	
	_lgr.debug(f'{pupil_function_scale=}')
	
	
	
	# Parameters that influence the size and scale of the calculated PSF
	expansion_factor = 3 # increasing this one gives best results normally
	supersample_factor=1
	
	
	
	
	psf_model = PSFModel(
		diffraction_limited_otf_model,
		PhasePowerSpectralDensity.of_von_karman_turbulence,
		PhasePowerSpectralDensity.of_adaptive_optics_moffat_function_model,
	)
	
	
	
	
	n_actuators = 24
	
	psf = psf_model(
		(201,201), 
		3, 
		1, 
		n_actuators / (2*obj_diameter), 
		(	ocs, 
			pupil_function_scale
		), 
		(	0.17, 
			2, 
			8
		), 
		(	24/(2*8),
			np.array([5E-2,5E-2]),#np.array([5E-2,5E-2]),
			1.6,#1.6
			2E-2,#2E-3
			0.05,#0.05
		),
		plots=False
	)
	
	psf_model.at(tuple(101*x for x in (1.212E-7, 1.212E-7)), 5E-7)
	psf_model.at(tuple(101*x for x in (1.212E-7, 1.212E-7)), 6E-7)
	psf_model.at(tuple(101*x for x in (1.212E-7, 1.212E-7)), 7E-7)
	psf_model.at(tuple(101*x for x in (1.212E-7, 1.212E-7)), 8E-7)
	
