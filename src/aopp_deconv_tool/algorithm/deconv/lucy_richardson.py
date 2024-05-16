

from typing import Optional, Literal, Any
import dataclasses as dc

import numpy as np
import scipy as sp

from .base import Base
import aopp_deconv_tool.numpy_helper as nph
import aopp_deconv_tool.numpy_helper.array.pad
import aopp_deconv_tool.numpy_helper.slice
import aopp_deconv_tool.cfg.logs
_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'DEBUG')

@dc.dataclass(slots=True, repr=False, eq=False,)
class LucyRichardson(Base):
	"""
	Implementation of the Lucy-Richardson algorithm
	"""
	
	# Public attributes
	
	nudge_factor 	: float = dc.field(default=1E-2, 	metadata={'description':'Fraction of maximum brightness to add to numerator and denominator to try and avoid numerical instability. This value should be in the range [0,1), and will usually be small. Larger values require more steps to give a solution, but suffer less numerical instability.'})
	strength 		: float = dc.field(default=1E-1,	metadata={'description':'Multiplier to the correction factors, if numerical insability is encountered decrease this. A more crude method of avoiding instability than `nudge_factor`, should be in the range (0,1].'})
	cf_negative_fix : bool 	= dc.field(default=True, 	metadata={'description':'Should we change negative correction factors to close-to-zero correction factors? Usually we should as we don\'t want any negative correction factors to flip-flop the end result.'})
	cf_limit 		: float = dc.field(default=np.inf,	metadata={'description':'End iteration if the correction factors are larger than this limit. Large correction factors are a symptom of numerical instability.'})
	cf_uclip 		: float = dc.field(default=np.inf,	metadata={'description':'Clip the correction factors to be no larger than this value. A crude method to control numerical instability.'})
	cf_lclip 		: float = dc.field(default=-np.inf,	metadata={'description':'Clip the correction factors to be no smaller than this value. A crude method to control numerical instability.'})
	offset_obs 		: bool 	= dc.field(default=False,	metadata={'description':'Should we offset the observation so there are no negative pixels? Enables the algorithm to find -ve values as the offset is reversed at the end.'})
	threshold : \
		Optional[float]     = dc.field(default=None,	metadata={'description':'Below this value LR will not be applied to pixels. This is useful as at low brightness LR has a tendency to fit itself to noise. If -ve will use |threshold|*brightest_pixel as threshold each step. If zero will use mean and standard deviation to work out a threshold, if None will not be used.'})
	pad_observation : bool  = dc.field(default=True,	metadata={'description':'Should we pad the input data with extra space to avoid edge effects? Padding will take the form of convolving the observation with the psf, but only keeping the edges of the convolved data. This will hopefully cause a smooth-ish drop-off at the edges instead of a hard cutoff, thus reducing insability.'})
	
	# Private attributes
	#_example : np.ndarray = dc.field(init=False, repr=False, hash=False, compare=False)
	_nudge : float = dc.field(init=False, repr=False, hash=False, compare=False)
	_obs_shape : tuple[int,...] = dc.field(init=False, repr=False, hash=False, compare=False)
	_dirty_img : float = dc.field(init=False, repr=False, hash=False, compare=False)
	_offset : float = dc.field(init=False, repr=False, hash=False, compare=False)
	_psf_reversed : np.ndarray = dc.field(init=False, repr=False, hash=False, compare=False)
	_cf : np.ndarray = dc.field(init=False, repr=False, hash=False, compare=False)
	_blurred_est : np.ndarray = dc.field(init=False, repr=False, hash=False, compare=False)
	_obs_per_est : np.ndarray = dc.field(init=False, repr=False, hash=False, compare=False)
	_out_of_bounds_mask : np.ndarray = dc.field(init=False, repr=False, hash=False, compare=False)
	
	

	def _init_algorithm(self, obs, psf):
		super(LucyRichardson, self)._init_algorithm(obs, psf)
		self._nudge = self.nudge_factor*np.max(obs) if self.nudge_factor > 0 else -self.nudge_factor

		# Pad observation to avoid edge effects
		self._obs_shape = obs.shape
		self._dirty_img = nph.array.pad.with_convolution(obs, psf) if self.pad_observation else np.array(obs)
		
		# offset dirty_img if we want to allow -ve values in our results
		# increase the value from 0 to allow more -ve values
		self._offset = 0-np.nanmin(self._dirty_img) if self.offset_obs else 0
		self._dirty_img += self._offset

		# initialise common arrays
		self._components = np.ones_like(self._dirty_img)*np.mean(self._dirty_img)
		self._residual = np.array(self._dirty_img)

		# initialise special arrays
		self._psf_reversed = np.array(psf[::-1,::-1])
		self._cf = np.ones_like(self._dirty_img) # correction factors
		self._blurred_est = np.zeros_like(self._components)
		self._obs_per_est = np.zeros_like(self._components)
		self._out_of_bounds_mask = np.ones_like(self._components, dtype=bool)
	
		# set array values that can't be done during initalisation	
		self._out_of_bounds_mask[nph.slice.around_center(self._out_of_bounds_mask.shape, obs.shape)] = 0

	
		return

	def get_components(self):
		if self.pad_observation:
			return(self._components[nph.slice.around_center(self._dirty_img.shape, self._obs_shape)] - self._offset)
		else:
			return(self._components - self._offset) 
	
	def get_residual(self):
		if self.pad_observation:
			return(self._residual[nph.slice.around_center(self._dirty_img.shape, self._obs_shape)])
		else:
			return(self._residual) 
	
	def get_iters(self):
		return self._i
	
	def _iter(self, obs, psf):
		"""
		Perform an iteration of the algorithm, return True if iteration was successful, False otherwise
		"""
		if not super(LucyRichardson, self)._iter(obs, psf): return(False)
		self._blurred_est[...] = sp.signal.fftconvolve(self._components, psf, mode='same')
		#self.blurred_est[self.blurred_est==0] = np.min(np.fabs(self.blurred_est))

		self._obs_per_est[...] = (self._dirty_img + self._nudge)/(self._blurred_est + self._nudge)

		self._cf[...] = sp.signal.fftconvolve(self._obs_per_est, self._psf_reversed, mode='same')
		self._cf[...] = self.strength*(self._cf - 1) + 1 # correction factors should always be centered around one

		# if we have a threshold, apply it
		if self.threshold is not None:
			if self.threshold < 0:
				threshold_value = np.nanmax(self._residual)*abs(self.threshold)
			elif self.threshold == 0:
				threshold_value = np.nanmean(self._residual)+1*np.nanstd(self._residual)
			else:
				threshold_value = self.threshold
			self._cf[self._residual < threshold_value] = 1

		# once these get large the result becomes unstable, clip the upper bound if desired
		if (self.cf_uclip != np.inf) and np.any(self._cf > self.cf_uclip): 
			self._cf[self._cf>self.cf_uclip] = self.cf_uclip
		
		# anything close to zero can just be zero, clip the lower bound if desired
		if (self.cf_lclip != -np.inf) and np.any(self._cf < self.cf_lclip):
			self._cf[self.cf<self.cf_lclip] = 0
			
		# we probably shouldn't even have -ve correction factors, turn them into a close-to-zero factor instead
		if (self.cf_negative_fix and np.any(self._cf < 0)):
			cf_negative = self._cf < 0
			cf_positive = self._cf > 0
			if not np.any(cf_positive):
				raise ValueError('All correction factors in Lucy-Richardson deconvolution have become negative, exiting...')
			self._cf[cf_negative] = np.min(self._cf[cf_positive])*np.exp(self._cf[cf_negative])
		
		self._components *= self._cf
		self._components[self._out_of_bounds_mask] = self._offset
 
		# could make this faster by doing an inexact version
		self._residual[...] = self._dirty_img - sp.signal.fftconvolve(self._components, psf, mode='same')
		#self.resigual = self.dirty_img - self.blurred_est
	
		# If any of our exit conditions trip, exit the loop
		if np.nanmax(np.fabs(self._cf)) > self.cf_limit:
			print('WARNING: Correction factors getting large, stopping iteration')
			return(False)
		if np.all(np.isnan(self._cf)):
			print('ERROR: Correction factors have all become NAN, stopping iteration')
			return(False)
		return(True)
