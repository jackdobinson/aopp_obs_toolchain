"""
Models the observation system in as much detail required to get a PSF.
"""


from optical_model.base import BaseModel, Parameter

# TODO: Tidy this up and get a working example. May need to re-think ordering of
# calculations. Want to break into multiple independent bits

# TODO: Move this to correct location
class KolmogorovTurbulenceModel(BaseModel):
	required_parameters=('r0', 'turbulence_ndim', 'f_mag')
	required_models = tuple()


# TODO: Move this to correct location
class FetickAdaptiveOpticsModel(BaseModel):
	required_parameters=('f_mag', 'f_ao','alpha','beta','A','C')
	required_models=('atmosphere_turbulence_model')


# TODO: Move this to correct location
class InstrumentPSFModel(BaseModel):
	required_parameters = ('optical_component_set', 'shape', 'expansion_factor', 'supersample_factor')
	required_models = tuple()
	provides_results = ('instrument_psf_rho_per_lambda', 'rho_per_lambda_axes')

# TODO: Move this to correct location
class OTF(BaseModel):
	required_parameters = ('wavelength',)
	required_models = ('psf',)
	provides_results = ('otf', 'f_axes')

class ObservationPSFModel(BaseModel):
	required_parameters = tuple('wavelength')
	required_models = ('adaptive_optics_model', 'instrument_model')
