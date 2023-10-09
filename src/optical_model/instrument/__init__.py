"""
Models of optical instruments should be stored here
"""
import numpy as np


class OpticalInstrumentModel:
	"""
	Interface that ensures class implementing it returns an Optical Transfer
	Function.
	"""
	def get_otf(self, wavelengths : np.ndarray):
		"""
		Returns optical transfer function for an array of wavelengths
		"""
		raise NotImplementedError
