
# TODO: Alter this to work with "fits_psf_model.py::run()"



import numpy as np

from aopp_deconv_tool.geometry.shape import Circle
from aopp_deconv_tool.optics.geometric.optical_component import OpticalComponentSet, Aperture, Obstruction,Refractor
from aopp_deconv_tool.optics.telescope_model import optical_transfer_function_of_optical_component_set
from aopp_deconv_tool.optics.turbulence_model import phase_psd_von_karman_turbulence
from aopp_deconv_tool.optics.adaptive_optics_model import phase_psd_fetick_2019_moffat_function
from aopp_deconv_tool.instrument_model.vlt import VLT
import aopp_deconv_tool.psf_model as psf_model

import aopp_deconv_tool.cfg.logs
_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'WARN')


def test_psf_model_produces_plots():
	
	
	
	
	instrument = VLT.muse(
		expansion_factor=3,
		supersample_factor=2
	)
	
	
	test_psf_model = psf_model.PSFModel(
		instrument.optical_transfer_function(),
		phase_psd_von_karman_turbulence,
		phase_psd_fetick_2019_moffat_function,
		instrument
	)
	
	
	specific_test_psf_model = test_psf_model(
		None, # telescope_otf_model_args,
		(0.17, 2, 8), # atmospheric_turbulence_psd_model_args,
		(5E-2, 1.6), # adaptive_optics_psd_model_args,
		42, #f_ao,
		2, # ao_correction_amplitude=1,
		0, # ao_correction_frac_offset=0,
		0 # s_factor=0
	)
	
	
	r1 = specific_test_psf_model.at(5E-7)
	r2 = specific_test_psf_model.at(6E-7)
	r3 = specific_test_psf_model.at(7E-7)
	r4 = specific_test_psf_model.at(8E-7)
	
	_lgr.debug(f'{r1=} {r2=} {r3=} {r4=}')
	


if __name__=='__main__':
	test_psf_model_produces_plots()
