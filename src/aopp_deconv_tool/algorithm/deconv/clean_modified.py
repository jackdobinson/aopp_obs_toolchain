from typing import Any
import dataclasses as dc

import numpy as np
import scipy as sp
import scipy.signal

from .base import Base

from aopp_deconv_tool.data_structures.pointer import Pointer
import aopp_deconv_tool.image_processing as im_proc
import aopp_deconv_tool.image_processing.otsu_thresholding
import aopp_deconv_tool.mfunc as mfunc

import aopp_deconv_tool.cfg.logs
_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'DEBUG')


@dc.dataclass(slots=True,repr=False)
class CleanModified(Base):
	"""
	A modified verison of the CLEAN algorithm, designed to account for extended emission.
	
	# ARGUMENTS #
		n_iter : int = 1000
			Number of iterations to perform
		loop_gain : float = 0.02
			Fraction of emission that could be accounted for by a PSF added to components with each iteration.
			Higher values are faster but unstable
		threshold : float = 0.3
			Fraction of maximum brightness of residual above which pixels will be included in CLEAN step, if 
			negative will use the "maximum fractional difference otsu threshold". 0.3 is a  good default value, 
			if stippling becomes an issue, reduce or set to a negative value. Lower positive numbers will 
			require more iterations, but give a more "accurate" result.
		n_positive_iter : int = 0
			Number of iterations to do that only "adds" emission, before switching to "adding and subtracting" 
			emission.
		noise_std : float = 1E-1
			Estimate of the deviation of the noise present in the observation
		rms_frac_threshold : float = 1E-3
			Fraction of original RMS of residual at which iteration is stopped, lower values continue iteration 
			for longer.
		fabs_frac_threshold : float = 1E-3
			Fraction of original Absolute Brightest Pixel of residual at which iteration is stopped, lower 
			values continue iteration for longer.
		max_stat_increase : float = np.inf
			Maximum fractional increase of a statistic before terminating
		min_frac_stat_delta : float = 1E-3
			Minimum fractional standard deviation of statistics before assuming no progress is being made and 
			terminating iteration.
		give_best_result : bool = True
			If True, will return the best (measured by statistics) result instead of final result.
	
	# RETURNS #
		A "CleanModified" instance. Run the model using the __call__(...) method.
		e.g.
		```
		from aopp_deconv_tool.algorithm.deconv.clean_modified import CleanModified
		
		deconvolver = CleanModified()
		...
		deconv_components, deconv_residual, deconv_iters = deconvolver(processed_obs, normed_psf)
		
		```
		See `aopp_deconv_tool.deconvolve` for a full example.
	"""
	n_iter 				: int 	= dc.field(default=1000, 	metadata={'description':'Number of iterations'})
	loop_gain 			: float = dc.field(default=0.02, 	metadata={'description':'Fraction of emission that could be accounted for by a PSF added to components each iteration. Higher values are faster, but unstable.'})
	threshold 			: float = dc.field(default=0.3, 	metadata={'description':'Fraction of maximum brightness of residual above which pixels will be included in CLEAN step, if negative will use the maximum fractional difference otsu threshold. 0.3 is a  good default value, if stippling becomes an issue, reduce or set to a negative value. Lower positive numbers will require more iterations, but give a more "accurate" result.'})
	n_positive_iter 	: int 	= dc.field(default=0, 		metadata={'description':'Number of iterations to do that only "adds" emission, before switching to "adding and subtracting" emission'})
	noise_std 			: float = dc.field(default=1E-1, 	metadata={'description':'Estimate of the deviation of the noise present in the observation'})
	rms_frac_threshold 	: float = dc.field(default=1E-2, 	metadata={'description':'Fraction of original RMS of residual at which iteration is stopped, lower values continue iteration for longer.'})
	fabs_frac_threshold : float = dc.field(default=1E-2, 	metadata={'description':'Fraction of original Absolute Brightest Pixel of residual at which iteration is stopped, lower values continue iteration for longer.'})
	max_stat_increase	: float = dc.field(default=np.inf, 	metadata={'description':'Maximum fractional increase of a statistic before terminating'})
	min_frac_stat_delta	: float = dc.field(default=1E-3, 	metadata={'description':'Minimum fractional standard deviation of statistics before assuming no progress is being made and terminating iteration'})
	give_best_result	: bool  = dc.field(default=True, 	metadata={'description':'If True, will return the best (measured by statistics) result instead of final result.'})
	clean_beam_sigma	: float = dc.field(default=0, 		metadata={'description':'If not zero, will convolve the components with a gaussian with this sigma, useful for regularising the result'})
	
	# private attributes
	_obs                         : np.ndarray     = dc.field(init=False, repr=False, hash=False, compare=False)
	_psf                         : np.ndarray     = dc.field(init=False, repr=False, hash=False, compare=False)
	_residual_copy               : np.ndarray     = dc.field(init=False, repr=False, hash=False, compare=False)
	_iter_stat_record            : Any            = dc.field(init=False, repr=False, hash=False, compare=False)
	_iter_stat_names             : tuple[str,...] = dc.field(init=False, repr=False, hash=False, compare=False)
	_selected_map                : np.ndarray     = dc.field(init=False, repr=False, hash=False, compare=False)
	_accumulator                 : np.ndarray     = dc.field(init=False, repr=False, hash=False, compare=False)
	_current_convolved           : np.ndarray     = dc.field(init=False, repr=False, hash=False, compare=False)
	_current_cleaned             : np.ndarray     = dc.field(init=False, repr=False, hash=False, compare=False)
	_selected_px                 : np.ndarray     = dc.field(init=False, repr=False, hash=False, compare=False)
	_tmp_r                       : np.ndarray     = dc.field(init=False, repr=False, hash=False, compare=False)
	_px_choice_img_ptr           : Pointer        = dc.field(init=False, repr=False, hash=False, compare=False)
	_loop_gain_correction_factor : Any            = dc.field(init=False, repr=False, hash=False, compare=False)
	_flux_correction_factor      : Any            = dc.field(init=False, repr=False, hash=False, compare=False)
	_choose_update_pixels        : Any            = dc.field(init=False, repr=False, hash=False, compare=False)
	_pixel_threshold             : Any            = dc.field(init=False, repr=False, hash=False, compare=False)
	_get_pixel_threshold         : Any            = dc.field(init=False, repr=False, hash=False, compare=False)
	_fabs_threshold              : float          = dc.field(init=False, repr=False, hash=False, compare=False)
	_rms_threshold               : float          = dc.field(init=False, repr=False, hash=False, compare=False)
	_components_best             : np.ndarray     = dc.field(init=False, repr=False, hash=False, compare=False)
	_stats_best                  : np.ndarray     = dc.field(init=False, repr=False, hash=False, compare=False)
	_i_best                      : int            = dc.field(init=False, repr=False, hash=False, compare=False)
	_stats_delta                 : np.ndarray     = dc.field(init=False, repr=False, hash=False, compare=False)

	def get_components(self) -> np.ndarray:
		#return(self._components)
		if self.clean_beam_sigma == 0:
			return(self._components_best if self.give_best_result else self._components)
		else:
			return sp.ndimage.gaussian_filter(
				(self._components_best if self.give_best_result else self._components),
				self.clean_beam_sigma
			)
	def get_residual(self) -> np.ndarray:
		#return(self._residual)
		return self._obs - sp.signal.fftconvolve(self.get_components(), self._psf, mode='same')/self._flux_correction_factor
	def get_iters(self):
		return(self._i_best if self.give_best_result else self._i)
	
	
	def _init_algorithm(self, obs, psf) -> None:
		super(CleanModified, self)._init_algorithm(obs, psf)
		# initialise arrays
		self._obs = obs
		self._psf = psf
		self._components = np.zeros_like(obs)
		self._components_best = np.zeros_like(obs)
		
		self._i_best = 0
		self._residual = np.array(obs)
		self._residual_copy = np.array(self._residual)
		
		#self._iter_stat_names = ('fabs', 'rms', 'UNUSED', 'UNUSED', 'UNUSED','generalised_least_squares', 'UNUSED', 'UNUSED')
		#self._iter_stat_record = np.full((self.n_iter, 8), fill_value=np.nan)
		
		self._iter_stat_names = ('fabs', 'rms', 'generalised_least_squares')
		self._iter_stat_record = np.full((self.n_iter, len(self._iter_stat_names)), fill_value=np.nan)
		self._stats_best = np.full((len(self._iter_stat_names),),np.inf)
		
		self._selected_px = np.zeros_like(obs)
		self._selected_map = np.zeros_like(obs, dtype=bool)
		self._current_cleaned = np.zeros_like(obs)
		self._accumulator = np.zeros_like(obs)
		self._current_convolved = np.zeros_like(obs)
		self._current_cleaned = np.zeros_like(obs)
		self._tmp_r = np.zeros_like(obs)
		
		self._stats_delta = np.zeros((len(self._iter_stat_names),))
		#self._recency_weighted_rms_delta = 0
		#self._recency_weighted_fabs_delta = 0
		
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
		fabs = np.nanmax(np.fabs(self._residual))
		rms = np.sqrt(np.nansum(self._residual**2)/self._residual.size)
		self._fabs_threshold = fabs*self.fabs_frac_threshold
		self._rms_threshold = rms*self.rms_frac_threshold
		#self._fabs_min_delta = fabs*self.fabs_min_delta
		#self._rms_min_delta = rms*self.rms_min_delta

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
			#self._get_pixel_threshold = lambda : self.threshold*np.nanmax(sp.ndimage.gaussian_filter(self._px_choice_img_ptr.val, sigma=2))

		return
	
	
	
	def _stopping_criteria(self):
	
		# Check thresholds
		_lgr.debug(f'{self._rms_threshold=:0.3g} rms = {self._iter_stat_record[self._i,1]:0.3g}')
		_lgr.debug(f'{self._fabs_threshold=:0.3g} fabs = {self._iter_stat_record[self._i,0]:0.3g}')
		if (self._iter_stat_record[self._i,1] < self._rms_threshold)  or (self._iter_stat_record[self._i,0] < self._fabs_threshold):
			self.progress_string = f"Ended at {self._i} iterations: Absolute brightest pixel, or root mean squared statistic have dropped below set threshold."
			return(False)
	
	
		# Check each statistic is better than the previously saved values
		new_best = True
		for name, this_stat, best_stat in zip(self._iter_stat_names, self._iter_stat_record[self._i], self._stats_best):
			if name != 'UNUSED':
				_lgr.debug(f'{new_best=} {this_stat=} {best_stat=}')
				new_best &= this_stat <= best_stat
		# If saved statistics are better, save the current state. 
		# Else, work out the maximum fractional increase . If it's 
		# larger than a threshold terminate iteration and return the best saved result
		if new_best:
			self._components_best[...] = self._components
			self._i_best = self._i
			self._stats_best = self._iter_stat_record[self._i]
		else:
			max_increase_frac = np.nanmax((self._iter_stat_record[self._i] - self._stats_best)/self._stats_best)
			_lgr.debug(f'{max_increase_frac=}')
			if max_increase_frac >= self.max_stat_increase:
				self.progress_string = f"Ended at {self._i} iterations: A statistic has increased from the best seen statistic beyond set threshold."
				return False
		
		
		
		# Check the stability of the statistics. If they have all stopped evolving, terminate iteration
		n_lookback = 10
		if self._i >=n_lookback :
			for j, name in enumerate(self._iter_stat_names):
				if name != 'UNUSED':
					self._stats_delta[j] = np.std(self._iter_stat_record[self._i-n_lookback:self._i,j])/self._iter_stat_record[self._i,j]
					_lgr.debug(f'{name} fractional standard deviation {self._stats_delta[j]}')
			_lgr.debug(f'{self.min_frac_stat_delta=}')
			if all([_std < self.min_frac_stat_delta for j, _std in enumerate(self._stats_delta) if self._iter_stat_names[j] != 'UNUSED']):
				self.progress_string = f"Ended at {self._i} iterations: Standard deviation of statistics in last {n_lookback} steps are all below minimum fraction."
				return False
				
		return True
	
	def _iter(self, obs, psf) -> bool:
		if not super(CleanModified, self)._iter(obs, psf): return(False)
		self._selected_px[...] *= 0 # reset selected pixels at start of iteration
		
		self._choose_update_pixels() # sets "self.choice_img_ptr" to an array that holds the values we should use for choosing pixels
		self._pixel_threshold = self._get_pixel_threshold()
		self._selected_map[...] = (self._px_choice_img_ptr.val > self._pixel_threshold)
		
		#min_region_size = 5
		#self._selected_map[...] = (sp.ndimage.grey_closing(self._px_choice_img_ptr.val, size=(min_region_size,min_region_size)) > self._pixel_threshold)
		#self._selected_map[...] = (sp.ndimage.gaussian_filter(self._px_choice_img_ptr.val, sigma=2) > self._pixel_threshold)
		
		
		# erode then dilate chosen pixels to remove single pixels, algorithm must have regions
		# of at least the size of the PSF to operate upon
		#for i in range(min_region_size):
		#	self._selected_map = sp.ndimage.binary_closing(self._selected_map, iterations=min_region_size//2)
		#self._selected_map = sp.ndimage.binary_dilation(self._selected_map, iterations=min_region_size//2)
		#self._selected_map = sp.ndimage.binary_erosion(self._selected_map, iterations=min_region_size)
		#self._selected_map = sp.ndimage.binary_dilation(self._selected_map, iterations=min_region_size)
		#self._selected_map = sp.ndimage.gaussian_filter(self._selected_map.astype(float), sigma=2) > 0.01
		
		"""
		label_map, n_labels = sp.ndimage.label(self._selected_map)
		# order labels by number of pixels they contain
						
		ordered_labels = list(range(1,n_labels+1)) # label 0 is background
		ordered_labels.sort(key = lambda x: np.count_nonzero(label_map == x), reverse=True)
		
		self._selected_map *= False
		for i in range(n_labels):
			if np.count_nonzero(label_map == ordered_labels[i]) >= min_region_size:
				self._selected_map |= (label_map == ordered_labels[i])
		"""
		
		
		
		self._selected_px[self._selected_map] = self._residual[self._selected_map]*self.loop_gain
		
		#self._selected_px[...] = sp.ndimage.median_filter(self._selected_px, size=5)
		#self._selected_px[...] = sp.ndimage.gaussian_filter(self._selected_px, sigma=1)
		
		# TESTING DIFFERENT STRATEGIES
		# This one may have better convergence statistics, I should check it
		#rma = np.nanmax(np.abs(self._residual))
		#r2 = (np.abs(self._residual)/rma)**2
		#self._selected_px[self._selected_map] = (np.sign(self._residual)*(r2*rma))[self._selected_map]*self.loop_gain
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
			np.nansum( # generalised_least_squares
				0.5*mfunc.generalised_least_squares(
					self._components,
					obs, self.noise_std, psf
				)
			),
		)
		
		return self._stopping_criteria()


	def choose_residual_extrema(self):
		if (self._i < self.n_positive_iter):
			self._px_choice_img_ptr.val = self._residual
		else:
			self._tmp_r[...] = np.fabs(self._residual)
			self._px_choice_img_ptr.val = self._tmp_r
		
		return
