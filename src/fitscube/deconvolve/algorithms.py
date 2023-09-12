#!/usr/bin/python3
"""
Deconvolution algorithms that all share the same interface

"""
import utilities.logging_setup
logging, _lgr = utilities.logging_setup.getLoggers(__name__, 'DEBUG')


import dataclasses as dc
import collections
import typing

import numpy as np
import scipy as sp
import scipy.signal
import matplotlib as mpl
import matplotlib.pyplot as plt

import utilities as ut
import utilities.sp
import utilities.np
import utilities.plt
from utilities.classes import Pointer

import fitscube
import fitscube.deconvolve
import fitscube.deconvolve.helpers

import image_processing as im_proc
import image_processing.otsu_thresholding

@dc.dataclass
class BaseAlgorithm():
	show_plots : bool = False # should plots be displayed interactively when running?
	save_plots : bool = False # should plots be saved to a file?
	plot_dir : str = "./plots" # folder to save plots to (if they should be saved), relative to current working directory
	verbose : int = 1 # verbosity of messages from algorithm

	def _init_algorithm(self, obs, psf, **kwargs):
		self.obs = obs
		self.psf = psf
		for k,v in kwargs.items():
			if hasattr(self, k): setattr(self, k, v)
		return

	@classmethod
	def get_default_params(cls):
		return({field.name:field.default if field.default is not dc.MISSING else field.default_factory() for field in dc.fields(cls)})

	@classmethod
	def get_param_names(cls):
		return(tuple(field.name for field in dc.fields(cls)))
	
	def get_params(self):
		return({field.name:self.__dict__[field.name] for field in dc.fields(self)})	

	def __call__(self):
		raise NotImplementedError

	def get_components(self):
		return(self.components)

	def _iter(self, i):
		raise NotImplementedError

	def _init_plots(self):
		raise NotImplementedError

	def _update_plots(self):
		raise NotImplementedError
# END class BaseAlgorithm

@dc.dataclass
class BaseIterableAlgorithm(BaseAlgorithm):
	n_iter : int =  200 # Maximum number of iterations to perform before returning.
	def _init_algorithm(self, *args, **kwargs):
		super()._init_algorithm(*args, **kwargs)
		self._i = 0
		return
# END class BaseIterableAlgorithm

@dc.dataclass
class LucyRichardson(BaseIterableAlgorithm):
	"""
	Implementation of the Lucy-Richardson algorithm
	"""
	nudge_factor 	: float = 1E-2 		# Fraction of maximum brightness to add to numerator and denominator to try and avoid numerical instability.
	strength 		: float = 1E-1 		# Multiplier to the correction factors, if numerical insability is encountered decrease this.
	cf_negative_fix : bool 	= True 		# Should we change negative correction factors in to close-to-zero correction factors?
	cf_limit 		: float = np.inf 	# End iteration if the correction factors are larger than this limit
	cf_uclip 		: float = np.inf 	# Clip the correction factors to be no larger than this value
	cf_lclip 		: float = -np.inf 	# Clip the correction factors to be no smaller than this value
	offset_obs 		: bool 	= False 	# Should we offset the observation so there are no negative pixels?
										# Enables the algorithm to find -ve values (offset is reversed at the end)
	threshold : \
		typing.Optional[float] = None	# Below this value LR will not be applied to pixels. This is useful as at low brightness LR has a tendency
										# to fit itself to noise. If -ve will use |threshold|*brightest_pixel as threshold each step,
										# If zero will use mean and standard deviation to work out a threshold, if None will not be used.
	fix_nans : \
		typing.Literal['interpolate','zero'] \
		= 'interpolate' 				# How should we treat NANs in the input data?
	pad_observation : bool = True 		# Should we pad the input data with extra space to avoid edge effects?
	
	def __call__(self, obs, psf, **kwargs):
		self.nudge = self.nudge_factor*np.max(obs)
		super()._init_algorithm(obs, psf, **kwargs)
		self._init_algorithm(**kwargs)
		#_lgr.INFO(f'{self.show_plots=} {self.save_plots=} {self.plot_dir=} {self.verbose=}')
		while (self._i < self.n_iter) and self._iter():
			_lgr.INFO(f'Iteration {self._i}/{self.n_iter}') 
			self.update_plots() # DEBUGGING
			self._i += 1

		self.close_plots()
		return(	self.components[ut.np.slice_center(self.components.shape, self.obs.shape)] - self.offset,  
				self.residual[ut.np.slice_center(self.residual.shape, self.obs.shape)], 
				self.iter_stat_record, 
				self.iter_stat_names, 
				self._i
				)


	def _init_algorithm(self, **kwargs):
		_lgr.INFO('\n'.join(['# ARGUMENTS #']+[f'\t{k} {v}' for k, v in self.get_params().items()]+['#############']))
		if self.fix_nans == 'interpolate':
			self.obs = ut.sp.interpolate_at_mask(self.obs, np.isnan(self.obs))
		elif self.fix_nans == 'zero':
			self.obs[np.isnan(obs)] = 0
		elif not self.fix_nans:
			pass
		else:
			raise ValueError(f'In "LucyRichardson.post_init()", unknown value "{fix_nans}" passed to argument "fix_nans"')


		# normalise PSF
		self.psf /= np.nansum(self.psf)

		# Pad observation to avoid edge effects
		self.dirty_img = self.pad_obs() if self.pad_observation else np.array(self.obs)
		
		# offset dirty_img if we want to allow -ve values in our results
		# increase the value from 0 to allow more -ve values
		self.offset = 0-np.nanmin(self.dirty_img) if self.offset_obs else 0
		self.dirty_img += self.offset

		# initialise common arrays
		self.components = np.ones_like(self.dirty_img)*np.mean(self.dirty_img)
		self.residual = np.array(self.dirty_img)
		self.iter_stat_record = np.full((self.n_iter,8), fill_value=np.nan)
		self.iter_stat_names = ('fabs', 'rms', 'correction factor min', 'correction factor max', 
			'correction factor median','chi_sq', 'entropy','chi_sq - 1.0*entropy'
		)

		# initialise special arrays
		self.psf_reversed = np.array(self.psf[::-1,::-1])
		self.cf = np.ones_like(self.dirty_img) # correction factors
		self.blurred_est = np.zeros_like(self.components)
		self.obs_per_est = np.zeros_like(self.components)
		self.out_of_bounds_mask = np.ones_like(self.components, dtype=bool)
	
		# set array values that can't be done during initalisation	
		self.out_of_bounds_mask[ut.np.slice_center(self.out_of_bounds_mask.shape, self.obs.shape)] = 0

		#""" # DEBUGGING
		self.f1 = None
		if self.show_plots or self.save_plots:
			self.f1, self.a1 = ut.plt.create_figure_with_subplots(2, 4)
			self.d1 = []
			self.get_plt_schema()
			for ax_idx, data, init_func, update_func, lim_func, title_str in self.plt_schema:
				self.d1.append(init_func(ax_idx, data))
		#"""
		return

	def get_components(self):
		if self.pad_observation:
			return(self.components[ut.np.slice_center(self.components, self.obs)] - self.offset)
		else:
			return(self.components - self.offset)
	
	def close_plots(self):
		if self.f1 is not None:
			plt.close(self.f1)


	def get_plt_schema(self):
		import copy
		cmap = copy.copy(mpl.cm.get_cmap('bwr'))
		cmap.set_over('magenta')
		cmap.set_under('green')
		cmap.set_bad('black')
		def init_imshow(ax_idx, data):
			ax = self.a1[ax_idx]
			im = ax.imshow(data, origin='lower', cmap=cmap)
			ut.plt.remove_axes_ticks_and_labels(ax)
			return(im)
		def update_imshow(ax_idx, i, data, title, lim_func):
			ax = self.a1[ax_idx]
			im = self.d1[i]
			im.set_data(data)
			im.set_clim(lim_func(data))
			ax.set_title(title.format(*lim_func(data), np.nansum(data)))
			return

		def init_lineplot(ax_idx, data):
			ax=self.a1[ax_idx]
			ls = []
			for j in range(data.shape[1]):
				l = ax.plot(data[:,j])
				ls += l
			ax.set_xlim(0, self.n_iter)
			return(ls)
		def update_lineplot(ax_idx, i, data, title, lim_func):
			ax = self.a1[ax_idx]
			ls = self.d1[i]
			for j in range(data.shape[1]):
				ls[j].set_ydata(data[:,j])
			ax.set_ylim(lim_func(data))
			ax.set_title(title.format(*[data[self._i,j] for j in range(data.shape[1])]))
			return

		self.plt_schema = [	[(0,0), self.dirty_img, 					init_imshow, 	update_imshow, 	lambda x: ut.plt.lim_sym_around_value(x, value=self.offset),	'dirty_img\nlims [{:07.2E}, {:07.2E}]\nsum {:07.2E}'],
							[(0,1), self.psf, 							init_imshow, 	update_imshow, 	lambda x: ut.plt.lim_sym_around_value(x, value=0),				'psf\nlims [{:07.2E}, {:07.2E}]\nsum {:07.2E}'],
							[(0,2), self.components,					init_imshow, 	update_imshow, 	lambda x: ut.plt.lim_sym_around_value(x, value=self.offset),	'componets\nlims [{:07.2E}, {:07.2E}]\nsum {:07.2E}'],
							[(0,3), self.blurred_est,					init_imshow, 	update_imshow, 	lambda x: ut.plt.lim_sym_around_value(x, value=self.offset),	'blurred_est\nlims [{:07.2E}, {:07.2E}]\nsum {:07.2E}'],
							[(1,0), self.obs_per_est,					init_imshow, 	update_imshow, 	lambda x: ut.plt.lim_sym_around_value(x, value=1),	'obs_per_est\nlims [{:07.2E}, {:07.2E}]\nsum {:07.2E}'],
							[(1,1), self.cf,							init_imshow, 	update_imshow, 	lambda x: ut.plt.lim_sym_around_value(x, value=1),	'correction_factors\nlims [{:07.2E}, {:07.2E}]\nsum {:07.2E}'],
							[(1,2), self.residual,						init_imshow, 	update_imshow, 	lambda x: ut.plt.lim_sym_around_value(x, value=0),	'residual\nlims [{:07.2E}, {:07.2E}]\nsum {:07.2E}'],
							[(1,3), self.iter_stat_record[:,2:5].view(),	init_lineplot,	update_lineplot,lambda x: ut.plt.lim_around_extrema(x, factor=0.1),	'cf [min max median]\nlims [{:07.2E}, {:07.2E}, {:07.2E}]']
							]
		return
		
	def update_plots(self):
		#print(f'DEBUG: {self.show_plots=} {self.save_plots=}')
		if not self.show_plots and not self.save_plots:
			return
		self.f1.suptitle(f'Iteration {self._i}/{self.n_iter}')
		for i, (ax_idx, data, init_func, update_func, lim_func, title_str) in enumerate(self.plt_schema):
			update_func(ax_idx, i, data, title_str, lim_func) 
		mpl.pyplot.pause(0.001)
		return

	def pad_obs(self):
		a = sp.signal.fftconvolve(self.obs, self.psf, mode='full')
		#_lgr.DEBUG(f"{a.shape=} {self.obs.shape=}")
		a[ut.np.slice_center(a,self.obs)] = self.obs
		return(a)


	def _iter(self):
		"""
		Perform an iteration of the algorithm, return True if iteration was successful, False otherwise
		"""
		self.blurred_est[...] = sp.signal.fftconvolve(self.components, self.psf, mode='same')
		#self.blurred_est[self.blurred_est==0] = np.min(np.fabs(self.blurred_est))

		self.obs_per_est[...] = (self.dirty_img + self.nudge)/(self.blurred_est + self.nudge)

		self.cf[...] = sp.signal.fftconvolve(self.obs_per_est, self.psf_reversed, mode='same')
		self.cf[...] = self.strength*(self.cf - 1) + 1 # correction factors should always be centered around one

		# if we have a threshold, apply it
		if self.threshold is not None:
			if self.threshold < 0:
				threshold_value = np.nanmax(self.residual)*abs(self.threshold)
			elif self.threshold == 0:
				threshold_value = np.nanmean(self.residual)+1*np.nanstd(self.residual)
			else:
				threshold_value = self.threshold
			self.cf[self.residual < threshold_value] = 1

		# once these get large the result becomes unstable, clip the upper bound if desired
		if (self.cf_uclip != np.inf) and np.any(self.cf > self.cf_uclip): 
			self.cf[self.cf>self.cf_uclip] = self.cf_uclip
		
		# anything close to zero can just be zero, clip the lower bound if desired
		if (self.cf_lclip != -np.inf) and np.any(self.cf < self.cf_lclip):
			self.cf[self.cf<self.cf__lclip] = 0
			
		# we probably shouldn't even have -ve correction factors, turn them into a close-to-zero factor instead
		if (self.cf_negative_fix and np.any(self.cf < 0)):
			cf_negative = self.cf < 0
			cf_positive = self.cf > 0
			if not np.any(cf_positive):
				raise ValueError('All correction factors in Lucy-Richardson deconvolution have become negative, exiting...')
			self.cf[cf_negative] = np.min(self.cf[cf_positive])*np.exp(self.cf[cf_negative])
		
		self.components *= self.cf
		self.components[self.out_of_bounds_mask] = self.offset
 
		# could make this faster by doing an inexact version
		self.residual[...] = self.dirty_img - sp.signal.fftconvolve(self.components, self.psf, mode='same')
		#self.resigual = self.dirty_img - self.blurred_est
	
		self.iter_stat_record[self._i] = (	np.nanmax(np.fabs(self.residual)),
											np.sqrt(np.nansum(self.residual**2)/self.residual.size),
											np.min(self.cf),
											np.max(self.cf),
											np.median(self.cf),
											np.nansum(0.5*fitscube.deconvolve.helpers.generalised_least_squares(self.components,
																							self.dirty_img, 1E-1, self.psf
																							)
																						),
											np.nansum(fitscube.deconvolve.helpers.entropy_adj(self.components, 1E1)),
											np.nansum(fitscube.deconvolve.helpers.regularised_least_squares(self.components, 
																							self.dirty_img, 
																							1E1, 
																							1, 
																							self.psf,
																							regularising_func=fitscube.deconvolve.helpers.entropy_adj
																							))
										)

		# If any of our exit conditions trip, exit the loop
		if np.nanmax(np.fabs(self.cf)) > self.cf_limit:
			print('WARNING: Correction factors getting large, stopping iteration')
			return(False)
		if np.all(np.isnan(self.cf)):
			print('ERROR: Correction factors have all become NAN, stopping iteration')
			return(False)
		return(True)
# END class LucyRichardson
	
@dc.dataclass
class CleanModified(BaseIterableAlgorithm):
	"""
	A modified verison of the CLEAN algorithm, designed to account for non-point objects better
	"""
	n_iter 				: int 	= 1000	# Number of iterations
	loop_gain 			: float = 0.02	# Fraction of emission that could be accounted for by a PSF added to components each iteration. Higher values are faster, but unstable.
	threshold 			: float = 0.3	# Fraction of maximum brightness above which pixels will be included in CLEAN step
	n_positive_iter 	: int 	= 0		# Number of iterations to do that only "adds" emission, before switching to "adding and subtracting" emission
	noise_std 			: float = 1E-2	# Estimate of the deviation of the noise present in the observation
	rms_frac_threshold 	: float = 1E-1	# Fraction of original RMS of residual at which iteration is stopped
	fabs_frac_threshold : float = 1E-1	# Fraction of original Absolute Brightest Pixel of residual at which iteration is stopped
		
	def __call__(self, obs, psf, **kwargs):
		super()._init_algorithm(obs, psf, **kwargs)
		self._init_algorithm()
		self.init_plots()
		while (self._i < self.n_iter) and self._iter():
			_lgr.INFO(f'Iteration {self._i}/{self.n_iter}') 
			self.update_plots()
			self._i += 1
		
		self.close_plots()
		return(	self.components[ut.np.slice_center(self.components.shape, self.obs.shape)],  
				self.residual[ut.np.slice_center(self.residual.shape, self.obs.shape)], 
				self.iter_stat_record, 
				self.iter_stat_names, 
				self._i
				)


	def _init_algorithm(self):
		
		# initialise arrays
		self.components = np.zeros_like(self.obs)
		self.residual = np.array(self.obs)
		self.residual_copy = np.array(self.residual)
		self.iter_stat_record = np.full((self.n_iter, 8), fill_value=np.nan)
		self.iter_stat_names = ('fabs', 'rms', 'UNUSED', 'UNUSED', 'UNUSED','generalised_least_squares', 'UNUSED', 'UNUSED')
		self.selected_px = np.zeros_like(self.obs)
		self.selected_map = np.zeros_like(self.obs, dtype=bool)
		self.current_cleaned = np.zeros_like(self.obs)
		self.accumulator = np.zeros_like(self.obs)
		self.current_convolved = np.zeros_like(self.obs)
		self.current_cleaned = np.zeros_like(self.obs)
		self.tmp_r = np.zeros_like(self.obs)
		
		# this variable should only ever reference another array variable
		# and not have a value of it's own. We use it to change how we
		# pick the pixels to be adjusted without changing later code
		self.px_choice_img_ptr = Pointer(self.tmp_r)
		#self.testing_ptr = Pointer(np.zeros_like(self.obs))
		
		self.loop_gain_correction_factor = 1.0/np.nanmax(self.psf)
		self.flux_correction_factor = np.nansum(self.psf)
		
		self.choose_update_pixels = self.choose_residual_extrema
		#self.choose_update_pixels = self.choose_regularised_least_squares
		#self.choose_update_pixels = self.choose_generalised_least_squares
		#self.choose_update_pixels = self.choose_residual_extrema_reject_entropy
		
		# set variable values
		self.fabs_threshold = np.nanmax(np.fabs(self.residual))*self.fabs_frac_threshold
		self.rms_threshold = np.sqrt(np.nansum(self.residual**2)/self.residual.size)*self.rms_frac_threshold

		# ensure that PSF is centered and an odd number in shape
		#slices = tuple(slice(s-s%2) for s in self.psf.shape)
		#self.psf = self.psf[slices]
		#bp = np.unravel_index(np.nanargmax(self.psf), self.psf.shape)
		#ut.np.center_on(self.psf, np.array(bp))
		self.pixel_threshold = 0


		if self.threshold < 0:
			#self.get_pixel_threshold = lambda : im_proc.otsu_thresholding.n_thresholds(self.residual_copy, 4)[-1]
			self.get_pixel_threshold = lambda : im_proc.otsu_thresholding.max_frac_diff_threshold(self.residual_copy)
		else:
			self.get_pixel_threshold = lambda : self.threshold*np.nanmax(self.px_choice_img_ptr.val)

		return
	
	


	def _iter(self):
		self.selected_px[...] *= 0 # reset selected pixels at start of iteration
		
		self.choose_update_pixels() # sets "self.choice_img_ptr" to an array that holds the values we should use for choosing pixels
		self.pixel_threshold = self.get_pixel_threshold()
		self.selected_map[...] = (self.px_choice_img_ptr.val > self.pixel_threshold)
		
		self.selected_px[self.selected_map] = self.residual[self.selected_map]*self.loop_gain
		self.accumulator[self.selected_map] += 1
		
		# convolve selected pixels with PSF and adjust so that flux is conserved
		self.current_convolved[...] = sp.signal.fftconvolve(self.selected_px, self.psf, mode='same')/self.flux_correction_factor
		
		# update residual
		self.residual[...] -= self.current_convolved
		self.residual_copy[...] = self.residual
		
		# add new components to component map
		self.components[...] += self.selected_px
		
		# update the current cleaned map
		self.current_cleaned[...] += self.current_convolved
		
		# update statistics that tell us how it's going
		self.iter_stat_record[self._i] = (	
			np.nanmax(np.fabs(self.residual)), # fabs
			np.sqrt(np.nansum(self.residual**2)/self.residual.size), # rms
			np.nan, #unused slot
			np.nan, #unused slot
			np.nan, #unused slot
			np.nansum(
				0.5*fitscube.deconvolve.helpers.generalised_least_squares(
					self.components,
					self.obs, self.noise_std, self.psf
				)
			),
			np.nan, #unused slot
			np.nan, #unused slot
		)
		
		if (self.iter_stat_record[self._i,1] < self.rms_threshold)  or (self.iter_stat_record[self._i,0] < self.fabs_threshold):
			return(False)
		return(True)


	"""
	def get_pixel_threshold(self):
		# simple version
		#return(self.threshold*np.nanmax(self.px_choice_img_ptr.val))
		#return(im_proc.otsu_thresholding.max_frac_per_fpix_threshold(self.residual_copy))
		#return(im_proc.otsu_thresholding.frac_per_fpix_threshold(self.residual_copy, 20))
		return(im_proc.otsu_thresholding.n_thresholds(self.residual_copy, 4)[-1])
	"""



	def choose_residual_extrema(self):
		if (self._i < self.n_positive_iter):
			self.px_choice_img_ptr.val = self.residual
		else:
			self.tmp_r[...] = np.fabs(self.residual)
			self.px_choice_img_ptr.val = self.tmp_r
		return
	
	
	def choose_generalised_least_squares(self):
		self.tmp_r[...] = fitscube.deconvolve.helpers.generalised_least_squares(
			self.components,
			self.obs,
			self.noise_std,
			self.psf)
		self.px_choice_img_ptr.val = self.tmp_r
		return


	def update_plots(self):
		if not (self.show_plots or self.save_plots):
			return
		self.f1.suptitle(f'Iteration {self._i}/{self.n_iter}')
		for k in self.h1.keys():
			h = self.h1[k]['h']
			ax= self.h1[k]['ax']
			v = self.h1[k]['v'] if type(self.h1[k]['v']) is not Pointer else self.h1[k]['v'].val # de-reference the pointer if we should
			h.set_data(v)
			lims = ut.plt.lim_sym_around_value(v, value=0)
			h.set_clim(lims)
			ax.set_title(f'{k} \nlimits [{lims[0]:07.2E}, {lims[1]:07.2E}]\nsum {np.nansum(v):08.2E}')
		mpl.pyplot.pause(0.001)
		return
	
	def close_plots(self):
		if self.f1 is not None:
			plt.close(self.f1)

	def init_plots(self):
		import copy
		self.f1 = None
		if not (self.show_plots or self.save_plots):
			return
		cmap = copy.copy(mpl.cm.get_cmap('bwr'))
		cmap.set_over('magenta')
		cmap.set_under('green')
		cmap.set_bad('black')
		mpl.cm.register_cmap(name='user_cmap', cmap=cmap)
		mpl.rcParams['image.cmap'] = 'user_cmap'
		
		(nr, nc, s) = (2,6,6)
		self.f1 = plt.figure(figsize=[x*s for x in (nc,nr)])
		self.a1 = self.f1.subplots(nr, nc, squeeze=False).ravel()
		self.h1 = collections.OrderedDict()
		
		
		for ax in self.a1:
			ut.plt.remove_axes_ticks_and_labels(ax)
		
		# Setting up an iterator so that order of plots can be changed by
		# changing the order they are defined in.
		axes_iter = iter(self.a1)
		
		ax = next(axes_iter)
		lims = ut.plt.lim_sym_around_value(self.obs, value=0)
		ax.set_title(f'Observation \n [{lims[0]:07.2E}, {lims[1]:07.2E}]\nsum {np.nansum(self.obs):08.2E}')
		h = ax.imshow(self.obs, origin='lower')
		h.set_clim(lims)
		
		ax = next(axes_iter)
		lims = ut.plt.lim_sym_around_value(self.psf, value=0)
		ax.set_title(f'PSF \n [{lims[0]:07.2E}, {lims[1]:07.2E}]\nsum {np.nansum(self.psf):08.2E}')
		h = ax.imshow(self.psf, origin='lower')
		h.set_clim(lims)
		
		ax = next(axes_iter)
		self.h1['current_convolved'] = dict(
			h=ax.imshow(self.current_convolved, origin='lower'),
			ax=ax,
			v=self.current_convolved
		)
		
		ax = next(axes_iter)
		self.h1['selected_px'] = dict(	
			h=ax.imshow(self.selected_px, origin='lower'),
			ax=ax,
			v=self.selected_px
		)
		
		ax = next(axes_iter)
		self.h1['pixel_choice_metric'] = dict(	
			h = ax.imshow(self.px_choice_img_ptr.val, origin='lower'),
			ax=ax,
			v=self.px_choice_img_ptr
		)

		ax = next(axes_iter)
		self.h1['current_cleaned'] = dict( 
			h=ax.imshow(self.current_cleaned, origin='lower'),
			ax=ax,
			v=self.current_cleaned
		)

		ax = next(axes_iter)
		self.h1['residual'] = dict(
			h=ax.imshow(self.residual, origin='lower'),
			ax=ax,
			v=self.residual
		)

		ax = next(axes_iter)
		self.h1['components'] = dict(
			h=ax.imshow(self.components, origin='lower'),
			ax=ax,
			v=self.components
		)

		@dc.dataclass
		class histogram:
			data : np.ndarray
			nbins : int
			ax : typing.Any
			parent : typing.Any

			def __post_init__(self):
				self._hist, self._bins = np.histogram(self.data, bins=self.nbins)
				#self._h = self.ax.hist(self.data.flatten(), bins=self.bins)
				self._lines = self.ax.step(self._bins[1:], self._hist)

				self.ax2 = self.ax.twinx()
				self._icv= im_proc.otsu_thresholding.calc(self._hist, self._bins)
				self._lines += self.ax2.plot(self._bins, self._icv, color='red')
				self.ax.yaxis.set_visible(True)
				self.ax.xaxis.set_visible(True)

				self.vline = self.ax.axvline(self.parent.pixel_threshold, color='red')

			def set_data(self, data):
				#self.ax.clear()
				#self.data = data
				#self.__post_init__()
				self._hist, self._bins = np.histogram(data, bins=self.nbins)
				self._icv= im_proc.otsu_thresholding.calc(self._hist, self._bins)
				self._lines[0].set_data(self._bins[1:], self._hist)
				self._lines[1].set_data(self._bins, self._icv)
				
				self.vline.remove()
				self.vline = self.ax.axvline(self.parent.pixel_threshold, color='red')

				#self.vline = self.ax.axvline(self.obins[np.nanargmin(self.ovals[1:])+1], color='red')
				return

			def set_clim(self, lims):
				self.ax.set_xlim(np.min(self._bins), np.max(self._bins))
				self.ax.set_ylim(np.min(self._hist), np.max(self._hist))
				self.ax2.set_xlim(np.min(self._bins), np.max(self._bins))
				self.ax2.set_ylim(np.nanmin(self._icv), np.nanmax(self._icv))
				return
				

		ax = next(axes_iter)
		self.h1['residual_histogram'] = dict(
			h = histogram(self.residual, int(np.sqrt(self.residual.size)), ax, self),
			ax=ax,
			v=self.residual,
		)

		ax = next(axes_iter)
		self.h1['pixel_choice_metric_histogram'] = dict(
			h = histogram(self.px_choice_img_ptr.val, int(np.sqrt(self.px_choice_img_ptr.val.size)), ax, self),
			ax=ax,
			v=self.px_choice_img_ptr,
		)

		"""
		@dc.dataclass
		class otsu2d_threshold_plot:
			plot_func : typing.Callable
			data : np.ndarray
			ax : typing.Any

			def __post_init__(self):
				self._h = self.ax.imshow(self.plot_func(self.data)[1])
			
			def set_data(self, data):
				self._h.set_data(self.plot_func(data)[1])

			def set_clim(self, lims):
				self._h.set_clim(np.nanmin(self._h.get_data()), np.nanmax(self._h.get_data()))
				return

		ax = next(axes_iter)
		self.h1['otsu2d_threshold_value'] = dict(
			h = otsu2d_threshold_plot(self.otsu2d_vals, self.residual, ax),
			ax=ax,
			v=self.residual
		)
		"""
		"""
		@dc.dataclass
		class otsu_threshold_plot:
			plot_func : typing.Callable
			data : np.ndarray
			ax : typing.Any
			parent : typing.Any

			def __post_init__(self):
				self._get_data()
				self._h = self.ax.step(self._bin_edges[1:], self._counts)

			def _get_data(self):
				self._bin_edges = np.linspace(np.nanmin(self.data), np.nanmax(self.data), int(np.sqrt(~np.nan(self.data))))
				self._counts, self._bin_edges = np.histogram(self.data, self._bin_edges, density=True)
				self._icv= im_proc.otsu_thresholding.calc(self.data)

			def set_data(self, data):
				self.data = data
				self._get_data()
				for line in self._h:
					line.set_data(self._bin_edges[1:], self._counts)

			def set_clim(self, lims):
				self.ax.set_xlim(np.nanmin(self._x), np.nanmax(self._x))
				self.ax.set_ylim(np.nanmin(self._y), np.nanmax(self._y))

		ax = next(axes_iter)
		self.h1['otsu_threshold_value'] = dict(
			h = otsu_threshold_plot(self.otsu_vals, self.residual, ax, self),
			ax=ax,
			v=self.residual
		)
		"""

		return		
# END class CleanModified

@dc.dataclass
class MaximumEntropy(BaseIterableAlgorithm):
	"""
	Implementation of the Maximum Entropy Method. Gradient decent is stochastic for speed.
	"""
	alpha 					: float = 1 	# balance between regularising function and least squares 
	rms_frac_threshold 		: float = 1E-1	# If the RMS of the residual goes below this 
											# fraction of it's original value, terminate iteration
	fabs_frac_threshold 	: float = 1E-1	# If the absolute brightest pixel of the residual goes 
											# below this fraction of it's original value, terminate iteration.
	grad_step_size 			: float = 1E-6 	# how large the steps are during gradient descent
	grad_step_multiplier 	: float = 0.01 	# maximum step size as a multiple of magnitude of residual
	objf_tol 				: float = 5E-3 	# tolerance for objective function 
	stochastic_decent_n 	: int 	= 100 	# number of points to use in stochastic gradient decent.
	noise_std 				: typing.Union[np.ndarray, float] 		= 2E-2	# Standard deviation of 
																			# observation noise, 
																			# either an array or a float
	model 					: typing.Union[np.ndarray, float, None] = None 	# A 'guess' as to the underlying 
																			# "real" image. Should be either 
																			# a physically informed model 
																			# (e.g. synthetic image) or values 
																			# expected from an empty field 
																			# (e.g. RMS noise values)
	regularising_function 	: \
		typing.Literal['fitscube.deconvolve.helpers.entropy_pos_neg'] \
		= 'fitscube.deconvolve.helpers.entropy_pos_neg'	# Function that computes the "regularising" factor
														# this factor is present to avoid over-fitting, and is
														# usually some sort of entropy-based function (hence
														# "maximum entropy method".

	def __post_init__(self):
		self.regularising_function = eval(self.regularising_function)

	def __call__(self, obs, psf, **kwargs):
		super()._init_algorithm(obs, psf, **kwargs)
		self._init_algorithm()
		self.init_plots()
		_lgr.INFO(f'Starting MaximumEntropy iteration...')
		while (self._i < self.n_iter) and self._iter():
			_lgr.INFO(f'Iteration {self._i}/{self.n_iter}') 
			self.update_plots()
			self._i += 1
		
		self.close_plots()

		return(	self.components[ut.np.slice_center(self.components.shape, self.obs.shape)],  
				self.residual[ut.np.slice_center(self.residual.shape, self.obs.shape)], 
				self.iter_stat_record, 
				self.iter_stat_names, 
				self._i
				)
	
	def _init_algorithm(self):
		# adjust arguments if needed
		if self.model is None:
			self.model = self.noise_std 
			
		# create needed arrays
		self.residual = np.array(self.obs)
		self.components = np.zeros_like(self.obs)
		self.current_convolved = np.zeros_like(self.obs)

		self.iter_stat_record = np.full((self.n_iter, 8), fill_value=np.nan)
		self.iter_stat_names = ('fabs', 'rms', 'UNUSED', 'UNUSED', 'UNUSED','(1/2)*generalised_least_squares', 'regularising_function', f'GLS(h) - {self.alpha}*F(h)')
		
		self.px_decent_dir = np.zeros_like(self.obs)
		self.px_decent_delta = np.zeros_like(self.obs)
		self.grad_along_px_decent_delta = np.zeros_like(self.obs)

		self.temp_img = np.zeros_like(self.obs)
		
		# testing members
		self.chosen_idxs_ptr = Pointer(np.zeros((2,0)))
		self.n_steps_taken_ptr = Pointer(np.zeros((self.n_iter,)))
		

		# TESTING
		self.rls_0, self.grad_0, self.hess_0 = self.grad_via_simultanious_perturbation(ck=1E-4*np.nanmax(self.residual), n=10)
		self.ak_dash_0_fac = (1+np.abs(self.rls_0))/(1+np.abs(self.hess_0))
		return
	
	
	def _iter(self):
		# reset decent array
		self.px_decent_dir *= 0
		
		# get decent direction
		#self.find_decent_dir_stochastic()
		self.find_decent_dir_test1()

		#_lgr.WARN('Testing interpolating decent direction.')
		#self.px_decent_dir[...] = ut.sp.interpolate_at_mask(self.px_decent_dir, self.px_decent_dir==0, edges='convolution', method='linear')
		
		# take a step in that direction
		self.components += self.px_decent_dir
		
		# find the new predicted image
		self.current_convolved[...] = sp.signal.fftconvolve(self.components,self.psf, mode='same')
		
		# get the residual
		self.residual[...] = self.obs - self.current_convolved
		
		# update statistics that tell us how it's going
		self.iter_stat_record[self._i] = (	
			np.nanmax(np.fabs(self.residual)), # fabs
			np.sqrt(np.nansum(self.residual**2)/self.residual.size), # rms
			np.nan, #unused slot
			np.nan, #unused slot
			np.nan, #unused slot
			np.nansum(
				0.5*fitscube.deconvolve.helpers.generalised_least_squares(
					self.components,
					self.obs, self.noise_std**2, self.psf
				)
			),
			np.nansum(
				self.regularising_function(
					self.components, 
					self.model
				)
			),
			np.nansum(self.regularised_least_squares())
		)
		
		# DEBUGGING turn off for now
		if False and (self._i > 1) and (self.iter_stat_record[self._i][7] - self.iter_stat_record[self._i-1][7] > self.objf_tol*self.iter_stat_record[self._i][7]):
			print(f'WARNING: This iteration increased the value of RLS function from {self.iter_stat_record[self._i-1][7]:07.2E} to {self.iter_stat_record[self._i][7]:07.2E}. Ending iterations...')
			return(False)
		
		return(True)

	
	def find_decent_dir_test1(self):
		#import scipy.optimize
		# want a symmetic bernouli (-1,+1) distribution,
		# need to have random direction vector that is identically and independently distributed around zero,
		# with finite inverse moments. See <https://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_An_Overview.PDF>, page 7.
		#step_max = np.nanmax(self.residual)
		step_max = np.sqrt(np.nansum(self.residual**2))

		#ak = 0.5*step_max/(self._i+1)
		ak = 1E-2*step_max
		#ck = 0.5*0.5*step_max/((self._i+1)**0.5)
		ck = 1E-4*step_max
		print(f'{ak = }')	
		print(f'{ck = }')	
		"""
		n_grads = 3
		self.grad_along_px_decent_delta *= 0
		hess = 0
		rls = self.regularised_least_squares(self.components)
		for i in range(0,n_grads):
			_g, _h = self.grad_via_simultanious_perturbation(ck, rls)
			self.grad_along_px_decent_delta += _g
			hess += _h
		"""	
		rls, self.grad_along_px_decent_delta, hess = self.grad_via_simultanious_perturbation(ck, n=10)

		print(f'{hess=}')
		print(f'{self.hess_0=}')
		print(f'{hess/self.hess_0=}')
		print(f'{np.sqrt(np.sum(self.grad_along_px_decent_delta**2))=}')
		print(f'{rls=}')
		print(f'{self.rls_0=}')
		print(f'{rls/self.rls_0=}')
		#ak_dash = ak/np.sqrt(np.sum(self.grad_along_px_decent_delta**2))
		ak_dash = ak*np.sqrt((np.abs(rls/self.rls_0))/(1+np.abs(hess/self.hess_0)))
		print(f'{ak_dash = }')

		#print(f'{self.grad_along_px_decent_delta = }')	
		self.px_decent_dir[...] = -ak_dash*(self.grad_along_px_decent_delta/np.sqrt(np.sum(self.grad_along_px_decent_delta**2)))
		_lgr.INFO(f'{self.px_decent_dir.shape = }')

	def get_random_dir(self, i):
		"""
		if i ==1 :
			self.px_decent_delta = np.ones_like(self.components)
			return
		if i==2:
			self.px_decent_delta = -1*np.ones_like(self.components)
			return
		"""
		self.px_decent_delta[...] = np.random.random(self.components.shape)-0.5
		self.px_decent_delta[self.px_decent_delta<0] = -1
		self.px_decent_delta[self.px_decent_delta>=0] = 1

	def grad_via_simultanious_perturbation(self, ck, n=1):
		rls = self.regularised_least_squares(self.components)
		hess = 0
		grad = np.zeros_like(self.px_decent_delta)
		weights = np.sign(self.residual)*np.abs(self.residual)**3
		print(f'{weights.shape=}')
		for i in range(0,n):
			self.get_random_dir(i)
			self.px_decent_delta *= weights 
			self.px_decent_delta /= np.sqrt(np.sum(self.px_decent_delta**2))

			rls_plus = self.regularised_least_squares(self.components + ck*self.px_decent_delta)
			rls_minus = self.regularised_least_squares(self.components - ck*self.px_decent_delta)

			grad += (((rls_plus - rls_minus)/(2*ck)) * self.px_decent_delta)
			hess += (rls_plus - 2*rls + rls_minus)/(ck**2)
		grad /= n
		hess /= n
		return(rls, grad, hess)
			


	def grad_via_simultanious_perturbation_old(self, ck, rls):
		self.get_random_dir()
		self.px_decent_delta *= np.abs(self.residual/np.nanmax(self.residual))

		self.temp_img[...] = self.components + ck*self.px_decent_delta
		rls_plus_dk = self.regularised_least_squares(self.temp_img)
		self.temp_img[...] = self.components - ck*self.px_decent_delta
		rls_minus_dk = self.regularised_least_squares(self.temp_img)
		grad_along_dk = (rls_plus_dk - rls_minus_dk)/(2*ck)
		hess_along_dk = (rls_plus_dk - 2*rls + rls_minus_dk)/(ck**2)
		return(grad_along_dk*self.px_decent_delta, hess_along_dk)
		
	

	def find_decent_dir_stochastic(self):
		# this version of rand_idxs seems to not find a solution easily
		#rand_idxs = tuple(zip(*tuple([np.random.randint(0,s,n) for s in self.px_decent_delta.shape])))
		
		# try getting indexes weighted by the residuals
		#print(ut.np.idx_grid(self.px_decent_delta))
		#print(self.px_decent_delta.size, len(self.px_decent_delta.shape))
		idx_array = np.reshape(ut.np.idx_grid(self.px_decent_delta), (len(self.px_decent_delta.shape), self.px_decent_delta.size)).T
		#print(idx_array)
		#sys.exit()
		p = np.nan_to_num(self.residual**2).ravel()/(self.noise_std**2) # preferentially choose values with a high GLS statistic
		rand_idxs = np.random.default_rng().choice(
			idx_array,
			size = self.stochastic_decent_n,
			replace=False,
			#p = p/np.sum(p) #TESTING UNIFORM RANDOM CHOICE
		)
		self.chosen_idxs_ptr.val = rand_idxs
		
		
		self.px_decent_dir *= 0 #reset direction vector
		for idx in rand_idxs:
			idx = tuple(idx) # have to use tuples because numpy only accepts tuples as indexes to arrays
			#print(f'DEBUGGING: {idx}') # DEBUGGING
			self.px_decent_delta[idx] = self.grad_step_size # TODO: make this better
			
			# get gradient and hessian matricies
			rls_grad, rls_hess = self.grad_hess_along_delta()
			
			self.px_decent_delta[idx] = 0 # reset the delta vector for next iteration
						
			# have found the gradient and hessian for adjusting our solution in the direction of self.px_decent_delta
			# now need to take a step along that direction. we should take a step in the direction of -grad 
			# only if hess is not positive.
			# should take a step of size propotional to inverse of gradient, i.e. large step on flatter regions,
			# small step on steeper regions
			#print(f'DEBUGGING: G {rls_grad:07.2E} H {rls_hess:07.2E}') # DEBUGGING
			if True:#rls_hess < 0: 
				#step = self.grad_step_multiplier*rls_grad
				step = self.find_opt_stepsize(self.grad_step_multiplier*np.fabs(self.residual[idx]))
				#print(f'DEBUGGING: taking step {step:07.2E}') # DEBUGGING
				self.px_decent_dir[idx] = step
				self.n_steps_taken_ptr.val[self._i] += 1
			else:
				#print(f'DEBUGGING: Not taking a step')
				pass
			# if we're not taking a step, then move to next index
		# once the loop is complete, self.px_decent_dir should hold a valid decent direction.
		#sys.exit()
		return
	
	def find_opt_stepsize(self, max_step_size):
		near_est = sp.signal.fftconvolve(self.px_decent_dir + self.components, self.psf, mode='same')
		delta_est = sp.signal.fftconvolve(self.px_decent_delta, self.psf, mode='same')
		near_comps = self.components+self.px_decent_dir
		def rls_in_delta_dir(step):
			es = near_est + step*delta_est
			delta_comps = near_comps + self.px_decent_delta*step 
			return(np.nansum(0.5*fitscube.deconvolve.helpers.generalised_least_squares_preconv(es, self.obs, self.noise_std**2) \
											- self.alpha*self.regularising_function(delta_comps, self.model)))
	
		#res = sp.optimize.minimize_scalar(rls_in_delta_dir, method='Bounded', bounds=(0,max_step_size), options=dict(disp=True))
		res = sp.optimize.minimize_scalar(rls_in_delta_dir, bracket=(-max_step_size,max_step_size))
		#print(res)
		return(res.x)
	
	def grad_hess_along_delta(self):
		#get value of RLS at current state
		cur_val = self.regularised_least_squares(self.px_decent_dir+self.components)
		# get value of rls at adjusted state
		pos_val = self.regularised_least_squares(self.px_decent_delta + self.px_decent_dir+self.components)
		# get value of rls at negative of adjusted state
		neg_val = self.regularised_least_squares(-self.px_decent_delta + self.px_decent_dir+self.components)
		# the difference between neg_val and pos_val is the 3 point gradient,
		# the difference between neg_val - cur_val and cur_val - pos_val is the 2 point hessian = neg_val + pos_val -2*cur_val
		#print(f'DEBUGGING: {neg_val:07.2E} {cur_val:07.2E} {pos_val:07.2E}') # DEBUGGING
		return(neg_val - pos_val, neg_val + pos_val - 2*cur_val)
		
		
	def find_decent_dir_domain_decomp(self):
		
		def nudge_img_region(aslice, n=10):
			rls_test_points = np.zeros((n,))
			#self.px_decent_delta *= 0
			for k in range(-n,n):
				#print(k) # DEBUGGING
				self.px_decent_delta[aslice] = np.max(np.fabs(self.residual[aslice]))*(k/n)
				# test what RLS is and record value
				rls_test_points[k] = self.regularised_least_squares(self.px_decent_delta+self.px_decent_dir+self.components)
			self.px_decent_delta[aslice] = np.max(np.fabs(self.residual[aslice]))*(np.argmin(rls_test_points)/n)
			return
		
		ss_list = [(0, self.px_decent_delta.shape[0], 0, self.px_decent_delta.shape[1])]
		a = [(slice(0, self.px_decent_delta.shape[0]),slice( 0, self.px_decent_delta.shape[1]))]
		for j, (start1,stop1,start2,stop2) in enumerate(ss_list):
			#print(f'DEBUGGING: Decomposition layer {j} grid size {stop1-start1}x{stop2-start2}')
			if start1==stop1 or start2==stop2:
				break
			x1, x2, x3, y1, y2, y3 = start1, (stop1+start1)//2, stop1, start2, (stop2+start2)//2, stop2
			a = []
			a.append((slice(x1, x2), slice(y1, y2)))
			a.append((slice(x2, x3),    slice(y1,  y2)))
			a.append((slice(x1,  x2), slice(y2,y3)))
			a.append((slice(x2,x3),    slice(y2,y3)))
			ss_list.append((x1,    x2, y1,    y2))
			ss_list.append((x2,  x3,    y1,    y2))
			ss_list.append((x1,    x2, y2,  y3))
			ss_list.append((x2, x3,    y2,  y3))
			for dd_slice in a:
				#print(dd_slice) # DEBUGGING
				nudge_img_region(dd_slice)
				self.px_decent_dir += self.px_decent_delta
			
		return
	def find_decent_dir(self):
		# this version is too slow
		small_step_fac = 0.1
		
		ij_from_k = lambda k: ( k // self.px_decent_dir.shape[0], k % self.px_decent_dir.shape[1] )
		
		k_ignore = []
		for l in range(self.px_decent_dir.size):
			rls_test_points = np.zeros((self.px_decent_dir.size-len(k_ignore),))
			for k in range(self.px_decent_dir.size):
				#print(f'DEBUG: {l} {k}') # DEBUGGING
				if k in k_ignore: continue
				i, j = ij_from_k(k)
				self.px_decent_delta[i,j]  *= 0 # set to zero
				
				# will a small step improve our RLS measure?
				
				# take a small step
				self.px_decent_delta[i,j] = self.residual[i,j]*small_step_fac
				
				# test what RLS is and record value
				rls_test_points[k] = self.regularised_least_squares(self.px_decent_delta+self.px_decent_dir+self.components)
			
			# take best step for a single pixel
			i,j = ij_from_k(np.argmin(rls_test_poitns))
			self.px_decent_dir[i,j] += self.residual[i,j]*small_step_fac
			
			# add pixel to ignore list
			k_ignore.append(k)
			
		# now we've taken steps in every pixel direction self.px_decent_dir should be what we want
	
	def regularised_least_squares(self, components=None):
		return(
			np.nansum(
				fitscube.deconvolve.helpers.regularised_least_squares(
					self.components if components is None else components, 
					self.obs, 
					self.noise_std, 
					self.alpha, 
					self.psf, 
					self.model, 
					self.regularising_function
				)
			)
		)
	
	def update_plots(self):
		if not (self.show_plots or self.save_plots):
			return
		self.f1.suptitle(f'Iteration {self._i}/{self.n_iter}')
		for k, v in self.h1.items():
			_lgr.INFO(f'Updating plot {k}')
			v.update()
		#mpl.pyplot.pause(0.01)
		mpl.pyplot.pause(3 if self._i==0 else 0.01)
	
	def close_plots(self):
		if self.f1 is not None:
			plt.close(self.f1)


	def init_plots(self):
		self.f1 = None
		if not (self.show_plots or self.save_plots):
			return
		import copy
		cmap = copy.copy(mpl.cm.get_cmap('bwr'))
		cmap.set_over('magenta')
		cmap.set_under('green')
		cmap.set_bad('black')
		mpl.cm.register_cmap(name='user_cmap', cmap=cmap)
		mpl.rcParams['image.cmap'] = 'user_cmap'
		
		(nr, nc, s) = (2,4,6)
		self.f1 = plt.figure(figsize=[x*s for x in (nc,nr)])
		self.a1 = self.f1.subplots(nr, nc, squeeze=False).ravel()
		self.h1 = collections.OrderedDict()
		
		
		# Setting up an iterator so that order of plots can be changed by
		# changing the order they are defined in.
		axes_iter = iter(self.a1)
		
		ax = next(axes_iter)
		ut.plt.remove_axes_ticks_and_labels(ax)
		lims = ut.plt.lim_sym_around_value(self.obs, value=0)
		ax.set_title(f'Observation \n [{lims[0]:07.2E}, {lims[1]:07.2E}]\nsum {np.nansum(self.obs):08.2E}')
		h = ax.imshow(self.obs, origin='lower')
		h.set_clim(lims)
		
		ax = next(axes_iter)
		ut.plt.remove_axes_ticks_and_labels(ax)
		lims = ut.plt.lim_sym_around_value(self.psf, value=0)
		ax.set_title(f'PSF \n [{lims[0]:07.2E}, {lims[1]:07.2E}]\nsum {np.nansum(self.psf):08.2E}')
		h = ax.imshow(self.psf, origin='lower')
		h.set_clim(lims)
	
		@dc.dataclass
		class ImagePlot(PlotInterface):
			def __post_init__(self):
				ut.plt.remove_axes_ticks_and_labels(self.ax)
				return

			def set_title(self):
				cmin, cmax = self.hdl.get_clim()
				self.ax.set_title(f'{self.title}\n[{cmin:08.2E}, {cmax:08.2E}]\nsum {np.nansum(self.data):08.2E}')
				return

			def set_clim(self):
				lims = ut.plt.lim_sym_around_value(self.data, value=0)
				#print(lims)
				self.hdl.set_clim(lims)
				return
				

		ax = next(axes_iter)
		self.h1['residual'] = ImagePlot(hdl=ax.imshow(self.residual, origin='lower'), ax=ax, data=self.residual, title='residual')

		@dc.dataclass
		class OverlayPlot(PlotInterface):
			def update(self):
				self.set_data()
				return
			def set_data(self):
				self.hdl[0].set_data(*self.data.val[:,::-1].T)
				return
			def set_title(self):
				pass

		#self.h1['chosen_idxs'] = OverlayPlot(hdl=ax.plot(*self.chosen_idxs_ptr.val[:,::-1], ls='none', marker='o', mfc='lime', ms=5, mec='none'), ax=ax, data=self.chosen_idxs_ptr)


		ax = next(axes_iter)
		self.h1['components'] = ImagePlot(hdl = ax.imshow(self.components, origin='lower'), ax=ax, data=self.components, title='components')


		ax = next(axes_iter)
		self.h1['px_decent_dir'] = ImagePlot(hdl=ax.imshow(self.px_decent_dir, origin='lower'), ax=ax, data=self.px_decent_dir, title='px_decent_dir')

		@dc.dataclass
		class EachIterPlot(PlotInterface):
			def __post_init__(self):
				if self.title is not None:
					self.ax.set_title(self.title)
				self.ax.set_xlim(0,self.data.size)
			def set_xlim(self):
				pass
			def set_ylim(self):
				ymin, ymax = np.nanmin(self.data), np.nanmax(self.data)
				#print(f'{ymin=} {ymax=}')
				#ymin, ymax = self.ax.get_ylim()
				#ymin1, ymax1 = np.nanmin(self.data), np.nanmax(self.data)
				#ymin = ymin if ymin1>ymin else ymin1
				#ymax = ymax if ymax1<ymax else ymax1
				self.ax.set_ylim(ymin,ymax)
				return
			def set_data(self):
				x, y = (np.arange(self.data.size),self.data)
				#print(x)
				#print(y)
				self.hdl[0].set_data(x,y)
			def set_title(self):
				pass

		ax = next(axes_iter)
		self.h1['regularised_least_squares'] = EachIterPlot(hdl=ax.plot(np.arange(self.iter_stat_record[:,7].size),self.iter_stat_record[:,7], color='tab:blue', ls='-', label='RLS=GLS-RF'), ax=ax, data=self.iter_stat_record[:,7], title='regularised least squares')
		
		
		self.h1['generalised_least_squares'] = EachIterPlot(hdl=ax.plot(np.arange(self.iter_stat_record[:,5].size),self.iter_stat_record[:,5], color='tab:orange', ls='--', label='GLS'), ax=ax, data=self.iter_stat_record[:,5], title=None)
		ax.set_xlabel('Iteration')
		ax.set_ylabel('RLS and GLS value')

		ax2 = ax.twinx()
		self.h1['regularising_function'] = EachIterPlot(hdl=ax.plot(np.arange(self.iter_stat_record[:,7].size),self.iter_stat_record[:,7], color='tab:green', ls=':', label='RF'), ax=ax2, data=self.iter_stat_record[:,7], title=None)
		ax2.set_ylabel('regularising function value')
		ut.plt.set_legend(ax, ax2)


		ax = next(axes_iter)
		self.h1['current_convolved'] = ImagePlot(hdl=ax.imshow(self.current_convolved, origin='lower'), ax=ax, data=self.current_convolved, title='current_convolved')

		# Remove any unused axes
		for ax in axes_iter:
			ax.remove()


		"""
		ax = next(axes_iter)
		ax.set_title('n_steps_taken')
		self.h1['n_steps_taken'] = dict(
			h = ax.plot(tuple(range(0,self.n_iter)), self.n_steps_taken_ptr.val),
			ax=ax,
			v = (tuple(range(0,self.n_iter)), self.n_steps_taken_ptr),
			flags=('distribute_v',),
			transform_func = lambda x: (x[0],x[1].val)
		)
		ax.yaxis.set_visible(True)
		ax.xaxis.set_visible(True)
		"""
		return	
# END class MaximumEntropy


@dc.dataclass
class WienerFilter(BaseAlgorithm):
	def __call__(self, obs, psf, noise_estimate, **kwargs):
		self.nudge = self.nudge_factor*np.max(obs)
		super()._init_algorithm(obs, psf, noise_estimate, **kwargs)
		self._init_algorithm(**kwargs)

		self.components = sp.signal.wiener(obs, self.filter_size, self.noise_estimate)
		self.residual = obs - sp.signal.fftconvolve(self.components, self.psf, mode='same')
		self.iter_stat_record = None
		self.iter_stat_names = None
		self._i = None

		self.close_plots()
		return(	self.components[ut.np.slice_center(self.components.shape, self.obs.shape)] - self.offset,  
				self.residual[ut.np.slice_center(self.residual.shape, self.obs.shape)], 
				self.iter_stat_record, 
				self.iter_stat_names, 
				self._i
				)


	def _init_algorithm(self, obs, psf, noise_estimate, **kwargs):
		brghtest_pixel = np.array(np.unravel_index(np.nanargmax(psf), psf.shape))
		regions = sp.ndimage.find_objects(sp.ndimage.label(psf > 0.05*np.nanmax(psf))[0])
		# assume first region is the one we want
		self.filter_size = tuple(x.stop - x.start for x in regions[0])
		self.noise_estimate = noise_estimate
		return
		
# END class WienerFilter


@dc.dataclass
class PlotInterface:
	hdl : typing.Any
	ax : typing.Any
	data : typing.Any
	title : typing.Optional[str] = None

	def set_data(self):
		self.hdl.set_data(self.data)
		return
	
	def set_title(self):
		if self.title is not None:
			self.ax.set_title(self.title)
		return

	def set_xlim(self):
		self.ax.set_xlim(None,None)
		return

	def set_ylim(self):
		self.ax.set_ylim(None,None)
		return

	def set_clim(self):
		if hasattr(self.hdl, 'set_clim'):
			self.hdl.set_clim(None,None)
		return

	def update(self):
		self.set_data()
		self.set_xlim()
		self.set_ylim()
		self.set_clim()
		self.set_title()
		return

	

if __name__=='__main__':
	import matplotlib.pyplot as plt
	# TESTING BELOW THIS LINE
	print('\n'*8+'########## STARTING OUTPUT ##########')
	#obs, psf = np.ones((100,100)), np.zeros((51,51))
	#psf[tuple([s//2 for s in psf.shape])] = 1


	#real, obs, psf = fitscube.deconvolve.helpers.create_test_images(psf=fitscube.deconvolve.helpers.create_psf_image( ((0,0,12),(3,5,6),(-2,-4,4)) ) )
	obs, psf = fitscube.deconvolve.helpers.get_test_data()

	"""
	ba1 = BaseAlgorithm(obs,psf)
	print(ba1.show_plots)
	print(ba1.__dict__)
	print('------')


	ba2 = BaseAlgorithm(obs, psf,show_plots=False)
	print(ba2.show_plots)
	print(ba2.__dict__)
	print('------')

	print('### exception testing ###')
	try:
		ba3 = BaseAlgorithm(obs, psf,cat='hat', show_plots=False)
		print(ba3.show_plots)
		print(ba3.__dict__)
		print('------')
	except TypeError as err:
		print(err)
	try:
		ba3 = BaseAlgorithm(obs,psf,7, show_plots=False)
		print(ba3.show_plots)
		print(ba3.__dict__)
		print('------')
	except TypeError as err:
		print(err)
	try:
		BaseAlgorithm.args_pos = tuple(['show_plots'])
		ba3 = BaseAlgorithm(obs,psf,show_plots=False)
		print(ba3.show_plots)
		print(ba3.__dict__)
		print('------')
	except SyntaxError as err:
		print(err)
	print('### END exception testing ###')

	print(BaseAlgorithm.kw_args)
	print(LucyRichardson.kw_args)
	print(hasattr(BaseAlgorithm, '_get_kw_args'))
	print('-----------')

	lr1 = LucyRichardson(obs,psf)
	print(lr1.__dict__)
	print('---------')
	"""

	"""
	lr2 = LucyRichardson(obs,psf,verbose=2, nudge=1E-0, n_iter=200)
	print(lr2.__dict__)
	print('---------')
	
	lr2()
	
	print(np.nanargmin(lr2.iter_stat_record[:,-1]))
	"""
	
	deconv = CleanModified(show_plots=True, verbose=2, loop_gain=0.2, threshold=0.6, n_iter=1000, noise_std=1E-1)
	deconv(obs, psf)
	
	# make smaller for testing
	#obs, psf = obs[::10,::10], psf[::10,::10]/np.nansum(psf[::10,::10])
	"""
	deconv = MaximumEntropy(obs, psf, show_plots=True, verbose=2, n_iter=500)
	deconv()
	
	(nr, nc, s) = (2,4,6)
	f1 = plt.figure(figsize=[x*s for x in (nc,nr)])
	a1 = f1.subplots(nr, nc, squeeze=False)
	for i, (data, label) in enumerate(zip(deconv.iter_stat_record.T, deconv.iter_stat_names)):
		a1[i//nc,i%nc].plot(range(0,deconv.n_iter), data)
		a1[i//nc,i%nc].set_title(label)
	
	a1[1,3].plot(range(0,deconv.n_iter), deconv.iter_stat_record[:,5] - deconv.alpha*deconv.iter_stat_record[:,6], color='tab:orange', ls='--')
	plt.show()
	
	(nr, nc, s) = (2,4,6)
	f2 = plt.figure(figsize=[x*s for x in (nc,nr)])
	a2 = f2.subplots(nr, nc, squeeze=False)
	
	a2[0,0].set_title('Observation')
	a2[0,0].imshow(obs, origin='lower')
	
	a2[0,1].set_title('PSF')
	a2[0,1].imshow(psf, origin='lower')
	
	a2[0,2].set_title('Components')
	a2[0,2].imshow(deconv.components, origin='lower')
	
	a2[0,3].set_title('Residual')
	a2[0,3].imshow(deconv.residual, origin='lower')
	
	a2[1,0].set_title('colvolved_image')
	a2[1,0].imshow(sp.signal.convolve(deconv.components, psf, mode='same'), origin='lower')
	
	plt.show()
	"""







