

from aopp_deconv_tool.psf_model.dependency_injector.params_and_psf_model_di import *#ParamsAndPsfModelDependencyInjector



from aopp_deconv_tool.optics.turbulence_model import phase_psd_von_karman_turbulence as turbulence_model
from aopp_deconv_tool.optics.turbulence_model import phase_psd_von_karman_turbulence
from aopp_deconv_tool.optics.adaptive_optics_model import phase_psd_fetick_2019_moffat_function

from aopp_deconv_tool.instrument_model.vlt import VLT



class MUSEAdaptiveOpticsPSFModelDependencyInjector(ParamsAndPsfModelDependencyInjector):
	"""
	Models the PSF using a model of the adaptive optics MUSE instrument on the VLT telescope.
	"""
	
	def __init__(self, 
			psf_data, 
			var_params : list[str] | tuple[str] = [], 
			const_params : list [str] | tuple [str] = [],
			initial_values : dict[str,float] = {},
			range_values : dict[str,tuple[float,float]] = {}
		):
		super().__init__(psf_data)
		
		instrument = VLT.muse(
			expansion_factor = 3,
			supersample_factor = 2,
			obs_shape=psf_data.shape[-2:]
		)
		
		self._psf_model = AOInstrumentPSFModel(
			instrument.optical_transfer_function(),
			phase_psd_von_karman_turbulence,
			phase_psd_fetick_2019_moffat_function,
			instrument
		)
		
		self._params = params = PriorParamSet(
			PriorParam(
				*prior_param_args_from_param_spec('wavelength', True, 750E-9, (0,np.inf), var_params, const_params, initial_values, range_values),
				"Wavelength (meters) that properties are calculated at, will be automatically set if spectral information is present in the FITS cube"
			),
			PriorParam(
				*prior_param_args_from_param_spec('r0', True, 0.15, (0.01,10), var_params, const_params, initial_values, range_values),
				"Fried parameter"
			),
			PriorParam(
				*prior_param_args_from_param_spec('turb_ndim', True, 1.3, (1,2), var_params, const_params, initial_values, range_values),
				"number of dimensions the turbulence has"
			),
			PriorParam(
				*prior_param_args_from_param_spec('L0', True, 1.5, (0,10), var_params, const_params, initial_values, range_values),
				"von-karman turbulence parameter"
			),
			PriorParam(
				*prior_param_args_from_param_spec('alpha', False, 0.4, (0.1,3), var_params, const_params, initial_values, range_values) ,
				"shape parameter of moffat distribution, equivalent to standard deviation of a gaussian"
			),
			PriorParam(
				*prior_param_args_from_param_spec('beta', True, 1.6, (1.1, 10), var_params, const_params, initial_values, range_values),
				"shape parameter of moffat distribution, controls pointy-vs-spreadiness of the distribution."
			),
			PriorParam(
				*prior_param_args_from_param_spec('ao_correction_frac_offset', False, 0, (-1,1), var_params, const_params, initial_values, range_values),
				"how much of an offset the adaptive optics correction has as a fraction of the maximum. I.e., models a discontinuity where the adaptive optics corrections stop and the high-frequency turbulent psf begins"
			),
			PriorParam(
				*prior_param_args_from_param_spec('ao_correction_amplitude', False, 2.2, (0,5), var_params, const_params, initial_values, range_values),
				"scaling of the adaptive optics correction. I.e., increases/decreases how large the AO correction bump is w.r.t the halo."
			),
			PriorParam(
				*prior_param_args_from_param_spec('factor', False, 1, (0.7,1.3), var_params, const_params, initial_values, range_values),
				"overall scaling factor to account for truncated PSFs which do not sum to 1"
			),
			PriorParam(
				*prior_param_args_from_param_spec('s_factor', True, 0, (0,10), var_params, const_params, initial_values, range_values),
				"'spike factor', how much of a spike (delta-function like part) there should be in the PSF. Not often used, generally constant and set to zero"
			),
			PriorParam(
				*prior_param_args_from_param_spec('f_ao', True, instrument.f_ao, (24.0/(2*instrument.obj_diameter),52.0/(2*instrument.obj_diameter)), var_params, const_params, initial_values, range_values),
				"frequency cutoff between adaptive optics corrections and high-frequency turbulence. Alters the position of the halo-glow effect."
			)
		)
		
		
		
	def get_parameters(self):
		return self._params
	
	def get_psf_model_flattened_callable(self):
		parent = self
		class PSFModelFlattenedCallable:
			def __init__(self):
				self.specific_model = None
				self.result = None
				self.cached_args = np.zeros((10,))
				
			
			def __call__(
					self,
					wavelength,
					r0, 
					turb_ndim, 
					L0, 
					alpha, 
					beta,
					f_ao,
					ao_correction_amplitude, 
					ao_correction_frac_offset, 
					s_factor,
					factor
				):

				# if we should recalculate everything, do so. Otherwise use the saved model
				if np.any(np.abs(np.array((r0, turb_ndim, L0, alpha, beta, f_ao,ao_correction_amplitude, ao_correction_frac_offset, s_factor,factor)) - self.cached_args) > 1E-5):
					self.specific_model = parent._psf_model(
						None,
						(r0, turb_ndim, L0),
						(alpha, beta),
						f_ao,
						ao_correction_amplitude,
						ao_correction_frac_offset,
						s_factor
					)
				
				
				#_lgr.debug(f'{(wavelength,r0, turb_ndim, L0, alpha, beta, f_ao,ao_correction_amplitude, ao_correction_frac_offset, s_factor,factor)}')
				self.result = self.specific_model.at(wavelength, plots=False).data
				
				return factor*(self.result / np.nansum(self.result.data))
		
		return PSFModelFlattenedCallable()
	
	def get_psf_result_postprocessor(self): 
		def psf_result_postprocessor(params, psf_model_flattened_callable, fitted_vars, consts):
			result = params.apply_to_callable(
				psf_model_flattened_callable, 
				fitted_vars,
				consts
			)
			return result
			
		return psf_result_postprocessor

