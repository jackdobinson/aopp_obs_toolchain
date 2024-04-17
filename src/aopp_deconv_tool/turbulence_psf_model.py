
from typing import Any, TypeVar, TypeVarTuple, ParamSpec, Generic, Annotated, NewType, Callable
import inspect

import numpy as np

IntVar = TypeVar('IntVar', bound=int)
T = TypeVar('T')
U = TypeVar('T')
Ts = TypeVarTuple('Ts')
class S(Generic[IntVar]): pass
class S1(Generic[IntVar]): pass
N = TypeVar('N',bound=int)
M = TypeVar('N',bound=int)
P = ParamSpec('P')

import matplotlib.pyplot as plt

from aopp_deconv_tool.optics.function import PhasePowerSpectralDensity

import aopp_deconv_tool.cfg.logs

_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'INFO')

class CCDSensor:
	"""
	Class that represents a CCD Sensor
	"""
	def __init__(self,
			shape : S[N],
			px_transform : np.ndarray[[M,M]] # M = N + 1
		):
		self.shape = shape
		self.px_transform = px_transform
	
	@classmethod
	def from_shape_and_pixel_size(cls, shape, px_size):
		n = len(shape)
		px_transform = np.eye(n+1,n+1)
		px_transform[:n,:n] *= px_size
		px_transform[:n, n] = np.array([-s/2 for s in shape])
		return cls(shape, px_transform)

	def px_axis(self):
		return tuple(np.linspace(0+t, s+t, s) for s, t in zip(self.shape, self.px_transform[:-1,-1]))
		
class SimpleTelescope:
	"""
	Class that represents a simple telescope
	"""
	def __init__(self,
			objective_diameter : float,
			effective_focal_length : float,
			ccd_sensor : CCDSensor
		):
		self.objective_diameter = objective_diameter
		self.effective_focal_length = effective_focal_length
		self.ccd_sensor = ccd_sensor
	
	
	def f_axis(self, wavelength):
		_lgr.debug(f'{self.ccd_sensor.px_transform=}')
		_lgr.debug(f'{self.ccd_sensor.px_axis()=}')
		return tuple(ax*t/(self.effective_focal_length*wavelength) for ax,t in zip(self.ccd_sensor.px_axis(), np.diag(self.ccd_sensor.px_transform[:-1,:-1])))
	
	def f_mesh(self, wavelength):
		return (self.ccd_sensor.px_transform @ np.indices(self.ccd_sensor.shape))/(self.effective_focal_length * wavelength) 
		

class TurbulencePSFModel:
	"""
	Class that represents a PSF as only dependent on turbulence and a simple telescope model
	"""
	def __init__(self,
			telescope_model : SimpleTelescope,
			turbulence_model : Callable[P, PhasePowerSpectralDensity],
		):
		self.turbulence_model = turbulence_model
		self.telescope_model = telescope_model
		
		# Ensure we have the argument names from the turbulence model
		# remove `f_axes`
		# ensure `wavelength` is first
		self.arg_names = list(inspect.signature(self.turbulence_model).parameters.keys())
		self.arg_names.remove('f_axes')
		assert self.arg_names[0] == 'wavelength'
		
	def __call__(self, wavelength, *args):
		f_axes = self.telescope_model.f_axis(wavelength)
		_lgr.debug(f'{f_axes=}')
		_lgr.debug(f'{args=}')
		result = self.turbulence_model(f_axes, wavelength, *args).data
		return result / np.nansum(result)
	
	
	