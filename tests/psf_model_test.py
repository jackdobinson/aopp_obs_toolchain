

import cfg.logs
_lgr = cfg.logs.get_logger_at_level(__name__, 'DEBUG')

import psf_model

def test_psf_model_produces_plots():
	import numpy as np
	from geometry.shape import Circle
	from optics.geometric.optical_component import OpticalComponentSet, Aperture, Obstruction,Refractor
	from optics.telescope_model import optical_transfer_function_of_optical_component_set
	from optics.turbulence_model import phase_psd_von_karman_turbulence
	from optics.adaptive_optics_model import phase_psd_fetick_2019_moffat_function
	from instrument_model.vlt import VLT
	
	
	
	instrument = VLT.muse()
	
	test_psf_model = psf_model.PSFModel(
		instrument.optical_transfer_function(3,1),
		phase_psd_von_karman_turbulence,
		phase_psd_fetick_2019_moffat_function,
	)
	
	
	test_psf_model(
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
	
	test_psf_model.at(tuple(101*x for x in (1.212E-7, 1.212E-7)), 5E-7)
	test_psf_model.at(tuple(101*x for x in (1.212E-7, 1.212E-7)), 6E-7)
	test_psf_model.at(tuple(101*x for x in (1.212E-7, 1.212E-7)), 7E-7)
	test_psf_model.at(tuple(101*x for x in (1.212E-7, 1.212E-7)), 8E-7)
	


if __name__=='__main__':
	test_psf_model_produces_plots()
