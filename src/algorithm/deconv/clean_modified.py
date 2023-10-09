from typing import Any
import dataclasses as dc

import numpy as np
import scipy as sp
import scipy.signal

from .base import Base

from utilities.classes import Pointer
import fitscube.deconvolve.helpers
import image_processing as im_proc
import image_processing.otsu_thresholding




@dc.dataclass(slots=True,repr=False)
class CleanModified(Base):
	"""
	A modified verison of the CLEAN algorithm, designed to account for non-point objects better
	"""
	n_iter 				: int 	= 1000	# Number of iterations
	loop_gain 			: float = 0.02	# Fraction of emission that could be accounted for by a PSF added to components each iteration. Higher values are faster, but unstable.
	threshold 			: float = 0.3	# Fraction of maximum brightness of residual above which pixels will be included in CLEAN step, if negative will use the maximum fractional difference otsu threshold. 0.3 is a  good default value, if stippling becomes an issue, reduce or set to a negative value. Lower positive numbers will require more iterations, but give a more "accurate" result.
	n_positive_iter 	: int 	= 0		# Number of iterations to do that only "adds" emission, before switching to "adding and subtracting" emission
	noise_std 			: float = 1E-2	# Estimate of the deviation of the noise present in the observation
	rms_frac_threshold 	: float = 1E-1	# Fraction of original RMS of residual at which iteration is stopped, lower values continue iteration for longer.
	fabs_frac_threshold : float = 1E-1	# Fraction of original Absolute Brightest Pixel of residual at which iteration is stopped, lower values continue iteration for longer.
	
	# private attributes
	_residual_copy : np.ndarray = dc.field(init=False, repr=False, hash=False, compare=False)
	_iter_stat_record : Any = dc.field(init=False, repr=False, hash=False, compare=False)
	_iter_stat_names : tuple[str,...] = dc.field(init=False, repr=False, hash=False, compare=False)
	_selected_px : np.ndarray = dc.field(init=False, repr=False, hash=False, compare=False)
	_selected_map : np.ndarray = dc.field(init=False, repr=False, hash=False, compare=False)
	_accumulator : np.ndarray = dc.field(init=False, repr=False, hash=False, compare=False)
	_current_convolved : np.ndarray = dc.field(init=False, repr=False, hash=False, compare=False)
	_current_cleaned : np.ndarray = dc.field(init=False, repr=False, hash=False, compare=False)
	_tmp_r : np.ndarray = dc.field(init=False, repr=False, hash=False, compare=False)
	_px_choice_img_ptr : Pointer = dc.field(init=False, repr=False, hash=False, compare=False)
	_loop_gain_correction_factor : Any = dc.field(init=False, repr=False, hash=False, compare=False)
	_flux_correction_factor : Any = dc.field(init=False, repr=False, hash=False, compare=False)
	_choose_update_pixels : Any = dc.field(init=False, repr=False, hash=False, compare=False)
	_pixel_threshold : Any = dc.field(init=False, repr=False, hash=False, compare=False)
	_get_pixel_threshold : Any = dc.field(init=False, repr=False, hash=False, compare=False)
	_fabs_threshold : float = dc.field(init=False, repr=False, hash=False, compare=False)
	_rms_threshold : float = dc.field(init=False, repr=False, hash=False, compare=False)

	def get_components(self) -> np.ndarray:
		return(self._components)
	def get_residual(self) -> np.ndarray:
		return(self._residual)

	def _init_algorithm(self, obs, psf) -> None:
		super(CleanModified, self)._init_algorithm(obs, psf)
		# initialise arrays
		self._components = np.zeros_like(obs)
		self._residual = np.array(obs)
		self._residual_copy = np.array(self._residual)
		self._iter_stat_record = np.full((self.n_iter, 8), fill_value=np.nan)
		self._iter_stat_names = ('fabs', 'rms', 'UNUSED', 'UNUSED', 'UNUSED','generalised_least_squares', 'UNUSED', 'UNUSED')
		self._selected_px = np.zeros_like(obs)
		self._selected_map = np.zeros_like(obs, dtype=bool)
		self._current_cleaned = np.zeros_like(obs)
		self._accumulator = np.zeros_like(obs)
		self._current_convolved = np.zeros_like(obs)
		self._current_cleaned = np.zeros_like(obs)
		self._tmp_r = np.zeros_like(obs)
		
		# this variable should only ever reference another array variable
		# and not have a value of it's own. We use it to change how we
		# pick the pixels to be adjusted without changing later code
		self._px_choice_img_ptr = Pointer(self._tmp_r)
		#self.testing_ptr = Pointer(np.zeros_like(obs))
		
		self._loop_gain_correction_factor = 1.0/np.nanmax(psf)
		self._flux_correction_factor = np.nansum(psf)
		
		self._choose_update_pixels = self.choose_residual_extrema
		#self.choose_update_pixels = self.choose_regularised_least_squares
		#self.choose_update_pixels = self.choose_generalised_least_squares
		#self.choose_update_pixels = self.choose_residual_extrema_reject_entropy
		
		# set variable values
		self._fabs_threshold = np.nanmax(np.fabs(self._residual))*self.fabs_frac_threshold
		self._rms_threshold = np.sqrt(np.nansum(self._residual**2)/self._residual.size)*self.rms_frac_threshold

		# ensure that PSF is centered and an odd number in shape
		#slices = tuple(slice(s-s%2) for s in psf.shape)
		#psf = psf[slices]
		#bp = np.unravel_index(np.nanargmax(psf), psf.shape)
		#ut.np.center_on(psf, np.array(bp))
		self._pixel_threshold = 0


		if self.threshold < 0:
			#self.get_pixel_threshold = lambda : im_proc.otsu_thresholding.n_thresholds(self.residual_copy, 4)[-1]
			self._get_pixel_threshold = lambda : im_proc.otsu_thresholding.max_frac_diff_threshold(self._residual_copy)
		else:
			self._get_pixel_threshold = lambda : self.threshold*np.nanmax(self._px_choice_img_ptr.val)

		return
	
	def _iter(self, obs, psf) -> bool:
		if not super(CleanModified, self)._iter(obs, psf): return(False)
		self._selected_px[...] *= 0 # reset selected pixels at start of iteration
		
		self._choose_update_pixels() # sets "self.choice_img_ptr" to an array that holds the values we should use for choosing pixels
		self._pixel_threshold = self._get_pixel_threshold()
		self._selected_map[...] = (self._px_choice_img_ptr.val > self._pixel_threshold)
		
		self._selected_px[self._selected_map] = self._residual[self._selected_map]*self.loop_gain
		self._accumulator[self._selected_map] += 1
		
		# convolve selected pixels with PSF and adjust so that flux is conserved
		self._current_convolved[...] = sp.signal.fftconvolve(self._selected_px, psf, mode='same')/self._flux_correction_factor
		
		# update residual
		self._residual[...] -= self._current_convolved
		self._residual_copy[...] = self._residual
		
		# add new components to component map
		self._components[...] += self._selected_px
		
		# update the current cleaned map
		self._current_cleaned[...] += self._current_convolved
		
		# update statistics that tell us how it's going
		self._iter_stat_record[self._i] = (	
			np.nanmax(np.fabs(self._residual)), # fabs
			np.sqrt(np.nansum(self._residual**2)/self._residual.size), # rms
			np.nan, #unused slot
			np.nan, #unused slot
			np.nan, #unused slot
			np.nansum(
				0.5*fitscube.deconvolve.helpers.generalised_least_squares(
					self._components,
					obs, self.noise_std, psf
				)
			),
			np.nan, #unused slot
			np.nan, #unused slot
		)
		
		if (self._iter_stat_record[self._i,1] < self._rms_threshold)  or (self._iter_stat_record[self._i,0] < self._fabs_threshold):
			return(False)
		return(True)


	def choose_residual_extrema(self):
		if (self._i < self.n_positive_iter):
			self._px_choice_img_ptr.val = self._residual
		else:
			self._tmp_r[...] = np.fabs(self._residual)
			self._px_choice_img_ptr.val = self._tmp_r
		return
