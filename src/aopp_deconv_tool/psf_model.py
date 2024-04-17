"""
Models an adaptive optics observation system in as much detail required to get a PSF.
"""
import sys

import typing
from typing import Callable

import numpy as np
import scipy as sp
import scipy.ndimage
import scipy.interpolate
import scipy.signal
import inspect

import matplotlib.pyplot as plt

import aopp_deconv_tool.cfg.logs
import aopp_deconv_tool.numpy_helper as nph
import aopp_deconv_tool.numpy_helper.array

from aopp_deconv_tool.geo_array import GeoArray, plot_ga
from aopp_deconv_tool.optics.function import PointSpreadFunction, OpticalTransferFunction, PhasePowerSpectralDensity
from aopp_deconv_tool.instrument_model.instrument_base import InstrumentBase


_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'INFO')


def downsample(a, s):
	return sp.signal.convolve(a, np.ones([s]*a.ndim)/(s**a.ndim), mode='valid')[tuple(slice(None,None,s) for _ in a.shape)]





class PSFModel:
	"""
	Class that calculates a model point spread function.
	"""
	def __init__(self,
			telescope_otf_model : 
				Callable[[tuple[int,...], float,float,...],OpticalTransferFunction]
				| OpticalTransferFunction, # axes in units of wavelength/rho
			atmospheric_turbulence_psd_model : 
				Callable[[np.ndarray,...],PhasePowerSpectralDensity]
				| PhasePowerSpectralDensity, # takes in spatial scale axis, returns phase power spectrum distribution
			adaptive_optics_psd_model : 
				Callable[[np.ndarray, ...], PhasePowerSpectralDensity]
				| PhasePowerSpectralDensity, # takes in spatial scale axis, returns phase PSD
			instrument : InstrumentBase # instrument used to take observation, defines certain scale parameters
		):
	
		
		if callable(telescope_otf_model):
			self.telescope_otf_model = telescope_otf_model
			self.telescope_otf = None
		else:
			self.telescope_otf_model = None
			self.telescope_otf = telescope_otf_model
		
		if callable(atmospheric_turbulence_psd_model):
			self.atmospheric_turbulence_psd_model = atmospheric_turbulence_psd_model
			self.atmospheric_turbulence_psd = None
		else:
			self.atmospheric_turbulence_psd_model = None
			self.atmospheric_turbulence_psd = atmospheric_turbulence_psd_model
	
		if callable(adaptive_optics_psd_model):
			self.adaptive_optics_psd_model = adaptive_optics_psd_model
			self.adaptive_optics_psd = None
		else:
			self.adaptive_optics_psd_model=None
			self.adaptive_optics_psd = adaptive_optics_psd_model
		
		
		self.instrument = instrument
		self.specific_model_ready = False
		
		
	def ao_corrections_to_phase_psd(self, 
			phase_psd, 
			ao_phase_psd, 
			f_ao, 
			ao_correction_amplitude=1, 
			ao_correction_frac_offset=0
		):
		"""
		Apply adaptive optics corrections to the phase power spectrum distribution of the atmosphere
		"""
		if ao_phase_psd is None: 
			return phase_psd.copy()
		
		
		f_delta = 1.5
		
		_lgr.debug(f'{phase_psd.data=} {ao_phase_psd.data=}')
		_lgr.debug(f'{f_ao=} {ao_correction_amplitude=} {ao_correction_frac_offset=}')
		
		f_mesh = phase_psd.mesh
		f_mag = np.sqrt(np.sum(f_mesh**2, axis=0))
		
		_lgr.debug(f'{np.max(f_mag)=}')
		_lgr.debug(f'{np.count_nonzero(np.isnan(f_mesh))=} {np.count_nonzero(np.isnan(f_mag))=}')
		
		f_ao_correct = f_mag <= f_ao
		f_ao_continuity_region = ((f_ao-f_delta) <= f_mag) & (f_mag < (f_ao+f_delta))
		
		_lgr.debug(f'{np.count_nonzero(f_ao_continuity_region)=} {np.sum(f_ao_continuity_region)=}')
		
		f_ao_continuity_factor = np.nanmean(phase_psd.data[f_ao_continuity_region])
		f_ao_correction_offset = np.nanmean(ao_phase_psd.data[f_ao_continuity_region])
		
		_lgr.debug(f'{f_ao_continuity_factor=} {f_ao_correction_offset=}')
		
		
		ao_corrected_psd = np.array(phase_psd.data)
		ao_corrected_psd[f_ao_correct] = (
			np.exp(ao_correction_amplitude)*(ao_phase_psd.data[f_ao_correct] - f_ao_correction_offset)
			+ np.exp(ao_correction_frac_offset)
		)*f_ao_continuity_factor
		
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
			telescope_otf_model_args,
			atmospheric_turbulence_psd_model_args,
			adaptive_optics_psd_model_args,
			f_ao,
			ao_correction_amplitude=1,
			ao_correction_frac_offset=0,
			s_factor=0,
			mode='adjust'
		):
		"""
		Calculate psf in terms of rho/wavelength for given parameters. 
		We are making an implicit assumption that all of the calculations can be
		done in rho/wavelength units.
		"""
		
		_lgr.debug(f'{telescope_otf_model_args=}')
		_lgr.debug(f'{atmospheric_turbulence_psd_model_args=}')
		_lgr.debug(f'{adaptive_optics_psd_model_args=}')
		_lgr.debug(f'{f_ao=}')
		_lgr.debug(f'{ao_correction_amplitude=}')
		_lgr.debug(f'{ao_correction_frac_offset=}')
		_lgr.debug(f'{s_factor=}')
		_lgr.debug(f'{mode=}')
		
		self.shape = self.instrument.obs_shape
		self.expansion_factor = self.instrument.expansion_factor
		self.supersample_factor = self.instrument.supersample_factor
		self.adaptive_optics_psd_model_args = adaptive_optics_psd_model_args
		
		
		self.telescope_otf_model_args = telescope_otf_model_args
		self.atmospheric_turbulence_psd_model_args = atmospheric_turbulence_psd_model_args
		self.f_ao = f_ao
		self.ao_correction_amplitude = ao_correction_amplitude
		self.ao_correction_frac_offset = ao_correction_frac_offset
		self.mode= mode
		self.s_factor = s_factor
		
		self.specific_model_ready = True
		return self
	
	
	
	def at(self, wavelength, plots=True):
		"""
		Calculate psf for a given angular scale and wavelength, i.e. convert
		from rho/wavelength units to rho units.
		"""
		if not self.specific_model_ready:
			raise RuntimeError(f'{type(self).__name__}.at({wavelength}) cannot be called yet. You need to specify model parameters with {type(self).__name__}.__call__(...) first.')
		
		wav_factor = (wavelength/self.instrument.ref_wavelength)
		ang_wav_factor = (self.instrument.obs_pixel_size/wavelength)
		_lgr.debug(f'{wav_factor=} {ang_wav_factor=}')
		
		# Get the telescope optical transfer function
		if self.telescope_otf_model is not None:
			self.telescope_otf = self.telescope_otf_model(self.shape, self.expansion_factor, self.supersample_factor, *self.telescope_otf_model_args)
		
		telescope_otf = self.telescope_otf#GeoArray(self.telescope_otf.data, tuple(a*ang_wav_factor for a in self.telescope_otf.axes))
				
		_lgr.debug(f'{telescope_otf.data.shape=} {tuple(x.size for x in telescope_otf.axes)=}')
		_lgr.debug(f'{telescope_otf.axes=}')
		
		if plots: plot_ga(telescope_otf, lambda x: np.log(np.abs(x)), 'diffraction limited otf', 'arbitrary units', 'wavelength/rho')
		if plots: plot_ga(telescope_otf.ifft(), lambda x: np.log(np.abs(x)), 'diffraction limited psf', 'arbitrary units', 'rho/wavelength')
		
		_lgr.debug(f'{self.instrument.obs_scale=} {self.instrument.obs_shape=}')
		
		#f_axes = telescope_otf.ifft().axes
		f_axes = tuple(x/wav_factor for x in telescope_otf.ifft().axes) # TEST THIS
		#f_axes = tuple(a/self.instrument.obs_pixel_size for a,sc,sh in zip(self.telescope_otf.ifft().axes,self.instrument.obs_scale,self.instrument.obs_shape))
		
		_lgr.debug(f'{tuple(x.size for x in f_axes)=}')
		_lgr.debug(f'{f_axes=}')
		
		# Get the atmospheric phase power spectral density
		if self.atmospheric_turbulence_psd_model is not None:
			self.atmospheric_turbulence_psd = self.atmospheric_turbulence_psd_model(
				f_axes,
				wavelength,
				*self.atmospheric_turbulence_psd_model_args
			)
		if plots: plot_ga(self.atmospheric_turbulence_psd, lambda x: np.log(np.abs(x)), 'atmospheric_turbulence_psd', 'arbitrary units', 'rho/wavelength')
		
		# Get the adaptive optics phase power spectral density
		if self.adaptive_optics_psd_model is not None:
			self.adaptive_optics_psd = self.adaptive_optics_psd_model(
				f_axes, 
				*self.adaptive_optics_psd_model_args
			)
		if plots: plot_ga(self.adaptive_optics_psd, lambda x: np.log(np.abs(x)), 'adaptive_optics_psd', 'arbitrary units', 'rho/wavelength')
		_lgr.debug(f'{self.atmospheric_turbulence_psd.data=}')
		_lgr.debug(f'{self.adaptive_optics_psd.data=}')
		
		
		
		
		
		# Apply the adapative optics phase power spectral density corrections to the atmospheric phase power spectral density
		ao_corrected_atm_phase_psd = self.ao_corrections_to_phase_psd(
			self.atmospheric_turbulence_psd, 
			self.adaptive_optics_psd, 
			self.f_ao,#*wav_factor, # TEST THIS
			self.ao_correction_amplitude,#*wav_factor, # TEST THIS
			self.ao_correction_frac_offset
		)
		if plots: plot_ga(ao_corrected_atm_phase_psd, lambda x: np.log(np.abs(x)), 'ao_corrected_atm_phase_psd', 'arbitrary units', 'rho/wavelength')
		
		ao_corrected_otf = self.optical_transfer_fuction_from_phase_psd(ao_corrected_atm_phase_psd, self.mode, self.s_factor)
		if plots: plot_ga(ao_corrected_otf, lambda x: np.log(np.abs(x)), 'ao_corrected_otf', 'arbitrary units', 'wavelength/rho')
		if plots: plot_ga(ao_corrected_otf.ifft(), lambda x: np.log(np.abs(x)), 'ao_corrected_psf', 'arbitrary units', 'rho/wavelength')
		
		_lgr.debug(f'{ao_corrected_otf.data.shape=} {telescope_otf.data.shape=}')
		_lgr.debug(f'{ao_corrected_otf.data=}')
		_lgr.debug(f'{telescope_otf.data=}')
		
		"""
		t_otf_wav_axes = tuple(a/ang_wav_factor for a in self.telescope_otf.axes)
		interp = sp.interpolate.RegularGridInterpolator(
			self.telescope_otf.axes, 
			self.telescope_otf.data,method='linear',
			bounds_error=False, 
			fill_value=np.min(self.telescope_otf.data)
		)
		points = np.swapaxes(np.array(np.meshgrid(*t_otf_wav_axes)), 0,-1)
		t_otf_wav = GeoArray(np.array(interp(points)), t_otf_wav_axes)
		if plots: plot_ga(t_otf_wav, lambda x: np.log(np.abs(x)), 't_otf_wav', 'arbitrary units', '1/rho')
		if plots: plot_ga(t_otf_wav.ifft(), lambda x: np.log(np.abs(x)), 't_psf_wav', 'arbitrary units', 'rho')
		"""
		
		
		# Combination of diffraction-limited optics, atmospheric effects, AO correction to atmospheric effects
		otf_full = GeoArray(
			ao_corrected_otf.data * telescope_otf.data, 
			#(ao_corrected_otf.data/np.nansum(ao_corrected_otf.data)) * (self.telescope_otf.data/np.nansum(self.telescope_otf.data)), 
			telescope_otf.axes
		)
		if plots: plot_ga(otf_full, lambda x: np.log(np.abs(x)), 'otf full', 'arbitrary units', 'wavelength/rho')
		
		psf_full = otf_full.ifft()
		if plots: plot_ga(psf_full, lambda x: np.log(np.abs(x)), 'psf full', 'arbitrary units', 'rho/wavelength')
		
		psf_full = PointSpreadFunction(np.abs(psf_full.data), psf_full.axes)
		_lgr.debug(f'{psf_full.data=}')
		

		
		output_axes = tuple(np.linspace(-s/2*ang_wav_factor,s/2*ang_wav_factor,s*self.instrument.supersample_factor) for s in self.shape)
		
		#output_axes = tuple(np.linspace(-z/2,z/2,s) for z,s in zip(self.instrument.obs_scale*self.instrument.ref_wavelength,self.psf_full.data.shape))
		_lgr.debug(f'{output_axes=}')
		
		rho_axes = psf_full.axes#tuple(a*wavelength for a in self.psf_full.axes)
		_lgr.debug(f'{rho_axes=}')
		
		# swap output and rho axes to see if it makes a difference
		#temp = output_axes
		#output_axes = rho_axes
		#rho_axes = temp
				
		interp = sp.interpolate.RegularGridInterpolator(rho_axes, psf_full.data,method='linear', bounds_error=False, fill_value=np.min(psf_full.data))
		
		points = np.swapaxes(np.array(np.meshgrid(*output_axes)), 0,-1)
		
		
		
		sample_size = tuple(s1/(self.instrument.expansion_factor*s2) for s1,s2 in zip(psf_full.data.shape, self.shape))
		assert all([s==int(sample_size[0]) for s in sample_size]), "Should have a constant integer factor"
		sample_size = int(sample_size[0])
		_lgr.debug(f'{sample_size=}')
		
		result = GeoArray(downsample(np.array(interp(points)), sample_size), tuple(a[::sample_size] for a in output_axes))
		
		#result = GeoArray(downsample(self.psf_full.data, sample_size), tuple(wav_factor*a[::sample_size] for a in self.psf_full.axes))
		_lgr.debug(f'{result.data.shape=}')
		
		_lgr.debug(f'{result.data=}')
		
		if plots: plot_ga(result, lambda x: np.log(np.abs(x)), f'psf at {wavelength=}', 'arbitrary units', 'radians')
		
		return result









if __name__=='__main__':
	from geometry.shape import Circle
	from optics.geometric.optical_component import Aperture, Obstruction,Refractor
	from optics.telescope_model import optical_transfer_function_of_optical_component_set
	from optics.turbulence_model import phase_psd_von_karman_turbulence
	from optics.adaptive_optics_model import phase_psd_fetick_2019_moffat_function
	from instrument_model.vlt import VLT
	
	
	
	instrument = VLT.muse()
	
	psf_model = PSFModel(
		instrument.optical_transfer_function(3,1),
		phase_psd_von_karman_turbulence,
		phase_psd_fetick_2019_moffat_function,
	)
	
	
	psf = psf_model(
		instrument.obs_shape, 
		instrument.expansion_factor, 
		instrument.supersample_factor, 
		instrument.f_ao, 
		None, 
		(	0.17, 
			2, 
			8
		), 
		(	instrument.f_ao,
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
	
