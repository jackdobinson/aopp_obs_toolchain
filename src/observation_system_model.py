"""
Models the observation system in as much detail required to get a PSF.
"""


from typing import Protocol
import dataclasses as dc

import optical_model
import optical_model.adaptive_optics
import optical_model.instrument
import optical_model.atmosphere


class AOModelProtocol(Protocol):
	def get_otf(self, f_ao : float, wavelength : float):
		...
	
class InstrumentModelProtocol(Protocol):
	f_ao : float
	
	def get_otf(self, wavelength : float):
		...

class AtmosphereModelProtocol(Protocol):
	def get_phase_psd(self):
		...




@dc.dataclass(slots=True)
class ObservationSystemModel:
	ao_model : AOModelProtocol
	instrument_model : InstrumentModelProtocol
	atmosphere_model : AtmosphereModelProtocol
	
	def get_otf(self, wavelength):
		_, otf_t = self.instrument_model.get_otf(wavelength)
		_, otf_ao = self.ao_model.get_otf(self.instrument_model.f_ao, self.atmosphere_model, wavelength)
		
		return otf_t * otf_ao
