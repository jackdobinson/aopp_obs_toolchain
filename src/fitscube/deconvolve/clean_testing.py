#!/usr/bin/env python3
"""
Various implementations of the CLEAN algorithm
"""

import sys, os
import numpy as np
from astropy.io import fits
import astropy.convolution
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy as sp
import scipy.signal
import fitscube.header
import scipy.ndimage

#-----------------------------------------------------------------------------
# CLEAN algorithms 
#-----------------------------------------------------------------------------

def clean_hogbom(dirty_img, psf_img, window=True, loop_gain=1E-1, rms_frac_threshold=1E-2, fabs_frac_threshold=1E-2, max_iter=int(1E6), norm_psf=False, show_plots=False, quiet=True):
	"""
	Implements the CLEAN algorithm as defined in https://ui.adsabs.harvard.edu/abs/1974A%26AS...15..417H/abstract

	# ARGUMENTS #
		dirty_img
			Array containing the dirty image to be CLEANed
		psf_img
			Array containing the point spread function to be used when CLEANing
		window
			A boolean mask to confine cleaning to a certain region. If True will clean everywhere
		loop_gain
			What fraction of the 'full' point source is removed with each iteration
		rms_frac_threshold
			Stop iterating when the root mean square of the residual reaches this fraction of the original value
		fabs_frac_threshold
			Stop iterating when the absolute brightest pixel reaches this fraction of the original value
		max_iter
			Maximum number of iterations to perform
		norm_psf
			If True, will normalise the PSF so it's sum is 1 before CLEANing
		show_plots
			If True, will show progress plots

	# RETURNS #
		residual
			The components convolved with the PSF subtracted from the dirty image
		components
			The point sources that CLEAN has found
		rms_record
			The root mean square of the residual at each iteration
		fabs_record
			The absolute brightest pixel of the residual at each iteration
		i
			The number of iterations before one of the stopping criteria was reached
	"""
	residual = np.array(dirty_img)
	r_shape = np.array(residual.shape)
	p_shape = np.array(psf_img.shape)
	components = np.zeros_like(residual)
	rms_record = np.full((max_iter,), fill_value=np.nan)
	fabs_record = np.full((max_iter,), fill_value=np.nan)
	fabs_threshold = np.nanmax(np.fabs(residual))*fabs_frac_threshold
	rms_threshold = np.sqrt(np.nansum((residual)**2)/residual.size)*rms_frac_threshold
	

	if norm_psf:
		psf_img /= np.nansum(psf_img)
	psf_img = ensure_centered_psf(psf_img)
	
	if window is True:
		window = np.ones(dirty_img.shape, dtype=bool)
	
	if show_plots:
		plt.ion()
		nc, nr = (4,2)
		s=4
		f1 = plt.figure(figsize=(nc*s, nr*s))
		plt.show()

	residual[~window] = np.nan
	# perform CLEAN algorithm
	for i in range(max_iter):
		my, mx = np.unravel_index(np.nanargmax(np.fabs(residual)), residual.shape)
		mval = residual[my,mx]*loop_gain
		components[my,mx] += mval
		#print(f'my {my} mx {mx} mval {mval}')
		
		# get overlapping rectangles for residual cube and psf_cube
		r_o_min, r_o_max, p_o_min, p_o_max = rect_overlap_lims(r_shape, p_shape, np.array([my, mx])-p_shape//2)
		
		residual[r_o_min[0]:r_o_max[0],r_o_min[1]:r_o_max[1]] -= psf_img[p_o_min[0]:p_o_max[0],p_o_min[1]:p_o_max[1]]*mval
		
		resid_fabs = np.nanmax(np.fabs(residual))
		resid_rms = np.sqrt(np.nansum((residual)**2)/residual.size)
		#print(f'INFO: residual maximum {resid_max} threshold {threshold}')
		if not quiet: print(f'INFO: Iteration {i}/{max_iter} residual rms {resid_rms:07.2E} threshold {rms_threshold:07.2E} fabs {resid_fabs:07.2E} threshold {fabs_threshold:07.2E} {r_o_min} {r_o_max} {p_o_min} {p_o_max}')
		rms_record[i] = resid_rms
		fabs_record[i] = resid_fabs
		
		if show_plots and (i%1000==0):
			clean_hogbom_progress_plots(f1, nr, nc, dirty_img, psf_img, residual, components, rms_record, fabs_record, r_o_min, r_o_max, p_o_min, p_o_max, mx, my)
			f1.suptitle(f'Iteration {i}/{max_iter}')
			plt.draw()
			plt.waitforbuttonpress(0.0001)	

		if (resid_rms < rms_threshold)  or (resid_fabs < fabs_threshold):
			break

	if show_plots:
		plt.close()
		plt.ioff()
	return(residual, components, rms_record, fabs_record, i)

def clean_modified(dirty_img, psf_img, window=True, loop_gain=1E-1, threshold=0.6, rms_frac_threshold=0.01, 
					fabs_frac_threshold=0.01, 
					max_iter=int(1E2),
					norm_psf=False,
					show_plots=True,
					quiet=False,
					n_positive_iter=0,
					restart_tuple=None):
	"""
	A modified version of the CLEAN algorithm that attempts to reduce artifacts when CLEANing non-point sources. See http://adsabs.harvard.edu/full/1984A%26A...137..159S
	"""

	if not quiet: print('INFO: In "clean_modified()"')
	
	if norm_psf:
		psf_img /= np.nansum(psf_img)
	psf_img = ensure_centered_psf(psf_img)

	if window is True:
		window = np.ones(dirty_img.shape, dtype=bool)
	
	if show_plots:
		# DEBUGGING PLOT SETUP
		plt.ion()
		nr,nc = (2, 4)
		s = 4
		f1 = plt.figure(1,figsize=(s*nc,s*nr))
		plt.show()
	
	dirty_img_cpy = np.array(dirty_img)
	
	if restart_tuple is not None:
		residual, components, rms_record_old, fabs_record_old, starting_iteration, accumulator, window = restart_tuple
		fabs_record = np.full((max_iter,), fill_value=np.nan)
		fabs_record[:fabs_record_old.size] = fabs_record_old
		rms_record = np.full((max_iter,), fill_value=np.nan)
		rms_record[:rms_record_old.size] = rms_record_old
	else:
		residual = np.array(dirty_img_cpy)
		components = np.zeros_like(dirty_img_cpy)
		rms_record = np.full((max_iter,), fill_value=np.nan)
		fabs_record = np.full((max_iter,), fill_value=np.nan)
		accumulator = np.zeros_like(dirty_img_cpy, dtype=int)
		starting_iteration = 0
		
	dirty_img_cpy[~window] = np.nan
	selected = np.zeros_like(dirty_img_cpy)
	above_t = np.zeros_like(dirty_img_cpy, dtype=bool)
	current_cleaned = np.zeros_like(dirty_img_cpy)
	
	fabs_threshold = np.nanmax(np.fabs(dirty_img_cpy))*fabs_frac_threshold
	rms_threshold = np.sqrt(np.nansum((dirty_img_cpy)**2)/dirty_img_cpy.size)*rms_frac_threshold
	

	# perform CLEAN algorithm
	for i in range(starting_iteration, max_iter):
		if not quiet: print(f'INFO: Iteration {i}/{max_iter}')
		selected *= 0 # set to all zeros for each step
		if i < n_positive_iter:
			fabs_residual = residual
		else:
			fabs_residual = np.fabs(residual)
		max_fabs_residual = np.nanmax(fabs_residual)
		mod_clean_threshold = threshold*max_fabs_residual
		above_t[:,:] = fabs_residual > mod_clean_threshold
		n_selected = np.sum(above_t)
		selected[above_t] = residual[above_t]
		accumulator[above_t] += 1
		
		if not quiet: print(f'INFO: Number of selected points {n_selected} max_fabs_residual {max_fabs_residual:07.2E} mod_clean_threshold {mod_clean_threshold:07.2E}')
		
		convolved = sp.signal.convolve2d(selected, psf_img, mode='same')
		factor = max_fabs_residual/np.nanmax(np.fabs(convolved))
		components += selected*loop_gain*factor
		#current_cleaned = sp.signal.convolve2d(components, psf_img, mode='same')
		current_cleaned += convolved*loop_gain*factor
		#residual = dirty_img_cpy - current_cleaned
		residual -= convolved*loop_gain*factor
	
		if not quiet: print(np.nanmax(np.fabs(convolved*factor)))
	
		if not quiet: print(f'INFO: factor {factor:07.2E} sum_components {np.nansum(components):07.2E} max_convolved {np.nanmax(np.fabs(convolved)):07.2E} max_residual {np.nanmax(np.fabs(residual)):07.2E}')
		
		resid_fabs = np.nanmax(np.fabs(residual))
		resid_rms = np.sqrt(np.nansum((residual)**2)/residual.size)
		if not quiet: print(f'INFO: residual rms {resid_rms:07.2E} threshold {rms_threshold:07.2E} max {resid_fabs:07.2E} threshold {fabs_threshold:07.2E}')
		rms_record[i] = resid_rms
		fabs_record[i] = resid_fabs
		if (resid_rms < rms_threshold)  or (resid_fabs < fabs_threshold):
			break
		if show_plots:
			clean_modified_progress_plots(f1, nr, nc, dirty_img_cpy, residual, components, current_cleaned, above_t, accumulator, convolved, rms_record, fabs_record, rms_threshold, fabs_threshold)
			f1.suptitle(f'Loop iteration {i}')
			plt.draw()
			plt.waitforbuttonpress(0.001)
			
	if show_plots:
		plt.close()
		plt.ioff()	
	return(residual, components, rms_record, fabs_record, i, accumulator, window)

def clean_multiresolution(dirty_img, 
					psf_img, 
					window=True, 
					loop_gain=1E-1, 
					threshold=0.6, 
					rms_frac_threshold=0.01, 
					fabs_frac_threshold=0.01, 
					max_iter=int(1E4),
					norm_psf='max',
					show_plots=True,
					n_positive_iter=0,
					sigmas=(10, 5, 2.5, 1.25),#, 0.625, 0.3125, 0.15625), #(40,20,10,5,2.5,1.25,),
					calculation_mode='direct',
					base_clean_algorithm='hogbom',
					reject_decomposition_idxs=[4]):#[6]):
	"""
	Runs CLEAN algorithm at multiple resolutions, see http://articles.adsabs.harvard.edu/cgi-bin/nph-iarticle_query?bibcode=1988A%26A...200..312W&db_key=AST&page_ind=3&plate_select=NO&data_type=GIF&type=SCREEN_GIF&classic=YES
	This article specifically uses one gaussian filter, I have expanded the method
	to use many filters, and there is no specific reason they have to be gaussian.
	However, if you use gaussians (or any low-pass filter) you get some nice
	properties. The same is probably true if you use high-pass filters too.

	# DESCRIPTION #
		This works on the principle that any image, I, is made up of a sum of other
		images.
		
			I = \sum_{i=0}^{n} S_i 												(1)
		
		We are free to choose what those other images are, for our purposes we can
		say that S_0, the first image in the sum, is a convolution between the 
		original image I and a gaussian kernel G_0, i.e.
		
			S_0 = I \star G_0													(2)
		
		and we can call the remainder of the sum from (1) the difference
		
			d_0 = I - S_0 = \sum_{i=1}^{n} S_i 									(3)
		
		The value of S_1 we can choose to be a convolution between a different
		gaussian kernel G_1 and d_0, i.e.
		
			S_1 = d_0 \star G_1
			    = (I - S_0) \star G_1
				= (I - I \star G_0) \star G_1
				= I \star (\delta - G_0) \star G_1								(4)
			    = I \star H_1
		
		where
		
			H_1 = (\delta - G_0) \star G_1,
		
		which means that
		
			H_0 = G_0
		
		and 
		
			H_k = \left( \delta - \sum_{i=0}^{k-1} H_{i} \right) \star G_k    (4.5)
		
		so
		
			H_n = \left( \delta - \sum_{i=0}^{n-1} H_i \right)                (4.6)
			
		therefore
		
			I = \sum_{i=0}^{n} S_i \star H_i                                  (4.7)
		
		we can define the k^th element of the sum in (1) in the same way
		
			S_k = d_{k-1} \star G_k												(5)
			    = I \star H_k
		
		where d_k is given by
		
			d_k = I - S_0 - S_1 - ... - S_k										(6)
			    = I - \sum_{i=0}^{k} S_i
		
		and the n^th element of the sum is therefore
		
			I = S_0 + S_1 + S_2 + ... + S_n
			  = \sum_{i=0}^{n-1} S_i + S_n
		
		and
		
			I - \sum_{i=0}^{n-1} S_i = S_n
		
		so,
		
			d_{n-1} = S_n
			S_n     = d_{n-1}.													(7)
		
		
		Now, the CLEAN algorithm works by deconvolving an image into a set of
		delta functions that are convolved with a point spread function, and add a
		residual, i.e.
		
			X = \delta_X \star P + R_X											(8)
		
		where \delta_X are the delta function components, P is the point spread
		function (that integrates to 1) and R_X is the residual of X.
		
		We can use this definition on each of our S_k values to give
		
			S_k = \delta_k \star P_k + R_k										
		
		where \delta_k, P_k, and R_k are the components, psf, and residual of the
		image part S_k. Therefore, we have
		
			I = \sum_{i=0}^{n} \delta_i \star P_i + R_i                         (9)
		
		If we decompose P in the same way as I we get
		
			P = \sum_{i=0}^{n} P \star H_i                                     (10)
		
		we can make the choice that the P_i values in (9) are
		
			P_i = P \star H_i.                                                  (11)
		
		We can do this because for any S_i, we have 
		S_i - R_i = \delta_i \star P_i, we know our S_i values and the choice of
		P_i only changes the values of R_i and \delta_i, which are things we want
		to find anyway. Therefore (9) becomes
		
			I = \sum_{i=0}^{n} \delta_i \star P \star H_i + R_i.               (12)
		
		We can then re-arrange (12) to get a formula that recombines the \delta_i
		components to create I
		
			I = P \star \left( \sum_{i=0}^{n} \delta_i \star H_i \right)
			                                         + \sum_{i=0}^{n} R_i      (13)
		
		where we call 
		
		
			\sum_{i=0}^{n} \delta_i \star H_i                                  (14)
		
		the components,
		
			\sum_{i=0}^{n} R_i                                                 (15)
		
		the residual, and P the point spread function.
		
		
		We use the above method in the following steps
		
		1) choose a set of filters G_k (they are gaussians in our example), it 
		works best if you use filters that pass low frequecy information first.
		I.e. use gaussians with large sigmas first
		
		2) compute S_k values and H_k filters using (5) and (4.5)
		
		3) Compute P_k values using (11)
		
		4) Run the CLEAN algorithm on S_k using a psf of P_k to get \delta_k and 
		\R_k
		
		5) Recombine \delta_k, R_k using (13) to create I. P will be your PSF that
		you have initially. Or you can substitude P for B, where B is the clean 
		beam.
		
		NOTE: For some situations you just need (14), the components. Also,
		depending on your set of G_k's you can strategically remove information you
		do not want or need.
		I.e. For our case of using successively sharper gaussian filters as G_k,
		they function as low-pass filters. Therefore, in this case the higher
		frequency information will be contained in the large i values of \delta_i,
		therefore if we choose our G_{n-1} to be the same approximate size as our
		dirty beam, we can remove structures smaller than our dirty beam by leaving
		out the \delta_n components in (14). This can massively reduce noise and/or
		increase the speed of the CLEAN algorithm.
		
		NOTE: All filters are assumed to sum to 1 (i.e. they conserve flux) if that
		is not true, you will have some mutiplicative factors floating around.
		Just replace 
			P_i = P \star H_i 
		with 
			P_i = f_i P \star H_i
		and you will see that the f_i factors propagate through the equations to 
		give
			I = P \star \left( \sum_{i=0}^{n} f_i \delta_i \star H_i \right)
			                                         + \sum_{i=0}^{n} R_i      
		i.e. there is an extra factor of f_i in the components when compared to
		equation (13)

	# ARGUMENTS #
		dirty_img [nx, ny]
			Array containing the image to be CLEANed. The image will be decomposed
			into n component images which are CLEANed independently.
		psf_img [px,py]
			Array containing the point spread function (PSF) to be used when CLEANing
		window
			A boolean mask of same shape as "dirty_img", confines CLEANing to a certain
			region. If True will clean everywhere.
		loop_gain [n]
			What fraction of the point source is removed with each CLEAN interation,
			can be given as a sequence, a scalar will be repeated for each component image.
		threshold [n]
			In the case that base_clean_algorithm = "modified", is the fraction of the
			maximum pixel value that determines if a pixel is included in a CLEAN iteration
		rms_frac_threshold [n]
			Each compoment image will stop CLEANing when the root-mean-square of the
			residual reaches this fraction of it's original value
		fabs_frac_threshold [n]
			Each component image will stop CLEANing when the absolute brightest pixel
			reaches this fraction of it's original value
		max_iter [n]
			Maximum number of iterations of the CLEAN algorithm to perform for each
			compoment image
		norm_psf
			If True or "max" will normalise the psf_img to have a maximum value of 1,
			if "sum" will normalise psf_img to have a sum of 1, if False will not
			normalise at all. Works best if True or "max"
		show_plots
			<bool> If True will show plots about the result.
		n_positive_iter [n]
			If base_clean_algorithm = "modified", will perform at least n_positive_iter
			CLEAN iterations where only positive pixels are subtracted from the image.
			Need a value for each component image, if just one is present it will be
			used for all.
		sigmas [n-1]
			The standard deivaitons (in pixels) of the gaussian filters used to 
			decompose the images. There will always be one more decomposed image 
			than there are filters, as the last decomposed image is the sum of the 
			previous decomposed images subtracted from the original image, see 
			equation (7)
		calculation_mode
			<str> If "direct" will calculate S_i and P_i by subtracting previous
			results from their original images. If "convolution" will calculate 
			S_i and P_i by colvolving the original images with H_i, see equation
			(4.7)
		base_clean_algorithm
			<str> If "hogbom" will use the CLEAN hogbom algorithm to CLEAN each
			component image. If "modified" will use the modified CLEAN algorithm
			to CLEAN each component image. See the functions "clean_hogbom" and
			"clean_modified" for more information.
		reject_decomposition_idxs
			<array> If you don't want all of the CLEAN results from the component
			images to be combined into the final result [see equation (14)] then
			this list/array holds the indicies of the component images to skip
			over. I.e. If you have sigma=(3, 1.5), then there will be 3 component
			images (S_0, S_1, S_2), one for each sigma (S_0 and S_1) and one for 
			the the difference between the original image and the sum of their 
			filtered versions (S_2). For the case that you don't want the high
			frequency information contained in S_2 to be included in the final
			result, you would set reject_decomposition_idxs = [2].

	# RETURNS #
		residual [nx,ny]
			The sum of the residuals of each compoment image [see equation (15)
		components [nx,ny]
			The sum of the CLEAN components of each component image convolved
			by each H_i. See equation (14)
		rms_record [n]
			The root mean square of the residual of each component image at
			each iteration of it's CLEANing
		fabs_record[n]
			The absolute brightest pixel of the residual of each component
			image at each iteration of it's CLEANing
		n_iters[n]
			The number of iterations of the CLEAN algorithm for each component
			image.
	
	# EXAMPLE #
		```
		(	residual_1, 
			components_1, 
			rms_record_1, 
			fabs_record_1, 
			n_iter_1
		) = clean_multiresolution(	cube_img_2, 
							psf_img, 
							window=(disk_mask | cube_nans), 
							loop_gain=hogbom_loop_gain, 
							rms_frac_threshold=hogbom_rms_frac_threshold, 
							fabs_frac_threshold=hogbom_fabs_frac_threshold,
							max_iter=hogbom_max_iter,
							norm_psf=True,
							show_plots=True
							)
		```	
	"""
	# make sure that we have the correct form and length for our arguments
	sigmas = ensure_subscriptable(sigmas)
	n = len(sigmas) + 1
	n_sigma = len(sigmas)

	loop_gain = ensure_subscriptable(loop_gain, min_size=n)
	threshold = ensure_subscriptable(threshold, min_size=n)
	rms_frac_threshold = ensure_subscriptable(rms_frac_threshold, min_size=n)
	fabs_frac_threshold = ensure_subscriptable(fabs_frac_threshold, min_size=n)
	max_iter = ensure_subscriptable(max_iter, min_size=n)
	n_positive_iter = ensure_subscriptable(n_positive_iter, min_size=n)

	# TESTING DIFFERENT FILTER FUNCTIONS
	filter_function = sp.ndimage.gaussian_filter
	#filter_function = sp.ndimage.gaussian_gradient_magnitude
	#filter_function = lambda x,s: -sp.ndimage.gaussian_laplace(x,s)
	#filter_function = sp.ndimage.median_filter
	#filter_function = sp.ndimage.uniform_filter
	def high_pass_fft_filter(x, s):
		s = np.max(x.shape)*s
		filter_array = np.ones(x.shape)
		grid = np.mgrid[[slice(0,_s) for _s in filter_array.shape]]
		idxs_to_zero = np.sum([(_g - _g.shape[i]//2)**2 for i, _g in enumerate(grid)], axis=0) < s**2
		filter_array[idxs_to_zero] = 0
		x_fft = np.fft.fft2(x)
		return(np.fft.ifft2(x_fft*filter_array).real)
	def low_pass_fft_filter(x, s):
		s= np.max(x.shape)*s
		filter_array = np.ones(x.shape)
		grid = np.mgrid[[slice(0,_s) for _s in filter_array.shape]]
		idxs_to_zero = np.sum([(_g - _g.shape[i]//2)**2 for i, _g in enumerate(grid)], axis=0) > s**2
		filter_array[idxs_to_zero] = 0
		x_fft = np.fft.fft2(x)
		return(np.fft.ifft2(x_fft*filter_array).real)
	#filter_function = low_pass_fft_filter
	#sigmas = [s/np.max(dirty_img.shape) for s in sigmas]


	# normalise PSF if needed (usually is)
	print(f'dirty_img.shape {dirty_img.shape}')

	psf_img = pad_psf_to_minimum_size(psf_img, dirty_img.shape, lambda x: filter_function(x, np.nanmax(sigmas)), np.nanmax(sigmas))
	
	#print('psf sum', np.nansum(psf_img))
	psf_img_norm_fac = 1.0/np.nanmax(psf_img)
	#print(type(norm_psf),norm_psf)
	if ((type(norm_psf) is str) and (norm_psf == 'max')) or ((type(norm_psf) is bool) and (norm_psf is True)):
		psf_img /= np.nanmax(psf_img)
	elif (type(norm_psf) is str) and (norm_psf =='sum'):
		psf_img /= np.nansum(psf_img)
	#print('psf sum after norm', np.nansum(psf_img))

	# create data holder arrays
	h = np.zeros((n, psf_img.shape[0], psf_img.shape[1]))
	s = np.zeros((n, dirty_img.shape[0], dirty_img.shape[1]))
	p = np.zeros((n, psf_img.shape[0], psf_img.shape[1]))
	f = np.ones((n,))
	dirac_delta = dirac_delta_filter(h[0])
	dirty_img2 = np.nan_to_num(dirty_img)
	nan_idxs = np.isnan(dirty_img)

	# calculate the decomposition filters H_i,
	h[0] = filter_function(dirac_delta, sigmas[0])
	#print('H_0 = G_0')
	for i in range(1,n-1):
		#print(f'H_{i} = \left( \delta - \sum_(i=0)^({i-1}) H_i \\right) * G_{i}')
		h[i] = filter_function(dirac_delta - np.nansum(h[:i], axis=0), sigmas[i])
	#print(f'H_{i+1} = \delta - \sum_(i=0)^({i}) H_i')
	h[-1] = dirac_delta - np.nansum(h[:-1], axis=0)

	# calculate the decomposed images S_i, P_i, and f_i if applicable
	if calculation_mode is 'direct':
		s[0] = filter_function(dirty_img2, sigmas[0])
		p[0] = filter_function(psf_img, sigmas[0])
		f[0] = 1.0/np.nanmax(p[0])

		for i in range(1,n-1):
			s[i] = filter_function(dirty_img2 - np.nansum(s[:i], axis=0), sigmas[i])
			p[i] = filter_function(psf_img - np.nansum(p[:i], axis=0), sigmas[i])
			f[i] = 1.0/np.nanmax(p[i])

		s[-1] = dirty_img2 - np.nansum(s[:-1], axis=0)
		p[-1] = psf_img - np.nansum(p[:-1], axis=0)
		f[-1] = 1.0/np.nanmax(p[-1])

	elif calculaiton_mode is 'convolution':
		for i in range(n):
			s[i] = sp.signal.convolve2d(dirty_img2, h[i], mode='same')
			p[i] = sp.signal.convolve2d(psf_img, p[i], mode='same')
			f[i] = 1.0/np.nanmax(p[i])
	else:	
		print(f'ERROR: Argument "calculation_mode" is {calculation_mode}, options are ("direct", "convolution").')
		raise NotImplementedError

	# DEBUGGING
	#plt.imshow(s[0], origin='lower')
	#plt.show()

	# run the CLEAN algorithm for each decomposed image S_i using the psf P_i
	result_parts = []
	for i in range(n):
		s[i][nan_idxs] = np.nan # remember to put back NANs that we removed earlier
		if base_clean_algorithm is 'hogbom':
			result_i =  clean_hogbom(s[i], f[i]*p[i], window=window, 
						loop_gain=loop_gain[i], rms_frac_threshold=rms_frac_threshold[i], 
						fabs_frac_threshold=fabs_frac_threshold[i], max_iter=max_iter[i], 
						norm_psf=False, show_plots=False)
		elif base_clean_algorithm is 'modified':
			result_i =  clean_modified(s[i], f[i]*p[i], window=window, threshold=threshold[i],
						loop_gain=loop_gain[i], rms_frac_threshold=rms_frac_threshold[i], 
						fabs_frac_threshold=fabs_frac_threshold[i], max_iter=max_iter[i], 
						norm_psf=False, show_plots=False, n_positive_iter=n_positive_iter[i])
		else:
			print(f'ERROR: Argument "base_clean_algorithm" is {base_clean_algorithm}, options are ("hogbom", "modified")')
			raise NotImplementedError
		result_parts.append(result_i)

	#print('f_i values',f)

	# create holders for return data
	residual = np.zeros_like(result_parts[0][0])
	components = np.zeros_like(result_parts[0][1])
	rms_record = []
	fabs_record = []
	n_iters = []

	# if we are showing plots, create figure etc.
	if show_plots:
		nc, nr = (6, len(sigmas)+1)
		framesize=6
		f2 = plt.figure(figsize=(nc*framesize, nr*framesize))
		a2 = f2.subplots(nr,nc, squeeze=False) 
		remove_axes = lambda x: (x.xaxis.set_visible(False), x.yaxis.set_visible(False))
		
		f3 = plt.figure(figsize=(4*framesize, framesize))
		a3 = f3.subplots(1,4,squeeze=False)
		
	# combine results of decomposed images together
	for i, result_i in enumerate(result_parts):
		if i in reject_decomposition_idxs: # if we don't want to include some of the decomposed images, skip them.
			continue
		residual += result_i[0]
		component_i = f[i]*sp.signal.convolve2d(result_i[1], h[i], mode='same')
		components += component_i
		rms_record.append(result_i[2])
		fabs_record.append(result_i[3])
		n_iters.append(result_i[4])

		# display data in figure if desired
		if show_plots:
			a2[i,0].imshow(s[i], origin='lower')
			a2[i,0].set_title(f'S[{i}] {s[i].shape}\nsum {np.nansum(s[i]):07.2E}')
			remove_axes(a2[i,0])
			
			a2[i,1].imshow(p[i], origin='lower')
			a2[i,1].set_title(f'P*H[{i}] {p[i].shape} {f[i]:07.2E}\nsum {np.nansum(p[i]):07.2E}')
			remove_axes(a2[i,1])

			a2[i,2].imshow(component_i, origin='lower')
			a2[i,2].set_title(f'component [{i}]\nsum {np.nansum(component_i):07.2E}')
			remove_axes(a2[i,2])

			a2[i,3].imshow(result_i[0], origin='lower')
			a2[i,3].set_title(f'residual [{i}]\nsum {np.nansum(result_i[0]):07.2E}')
			remove_axes(a2[i,3])
			
			a2[i,4].imshow(h[i], origin='lower')
			a2[i,4].set_title(f'h[{i}]\nsum {np.nansum(h[i]):07.2E}')
			remove_axes(a2[i,4])
			"""
			a2[i,4].plot(result_i[2])
			a2[i,4].set_yscale('log')
			a2[i,4].set_title(f'rms_record niter {n_iters[i]}')
			"""
			a2[i,5].plot(result_i[3])
			a2[i,5].set_title('fabs_record')
			a2[i,5].set_yscale('log')

			pos = sigmas[i] if i<len(sigmas) else 0
			w = dirty_img.shape[0] - sigmas[i] if i==0 else sigmas[i-1] - sigmas[i] if i<len(sigmas) else sigmas[i-1]
			a3[0,0].bar(pos, np.nansum(component_i)/w, width=w, align='edge')
			a3[0,0].set_title('Signal density in each spatial scale of components.\nTotal signal is area of bar.')
			a3[0,0].set_xlabel('scale')
			a3[0,0].set_ylabel('signal density')

			a3[0,1].bar(pos, np.nansum(result_i[0])/w, width=w, align='edge')
			a3[0,1].set_title('Signal density in each spatial scale of residual.\nTotal signal is area of bar.')
			a3[0,1].set_xlabel('scale')
			a3[0,1].set_ylabel('signal density')

			a3[0,2].bar(pos, np.nansum(s[i])/w, width=w, align='edge')
			a3[0,2].set_title('Signal density in each spatial scale of image.\nTotal signal is area of bar.')
			a3[0,2].set_xlabel('scale')
			a3[0,2].set_ylabel('signal density')
			
			a3[0,3].bar(pos, np.nansum(p[i])/w, width=w, align='edge')
			a3[0,3].set_title('Signal density in each spatial scale of PSF.\nTotal signal is area of bar.')
			a3[0,3].set_xlabel('scale')
			a3[0,3].set_ylabel('signal density')
			a3[0,3].set_yscale('log')
			

			print(f'{pos:07.2E} {np.nansum(component_i)/w:07.2E} {np.nansum(result_i[0])/w:07.2E} {np.nansum(s[i])/w:07.2E} {np.nansum(p[i])/w:07.2E}')

			#a3[0,0].set_xscale('log')

	if show_plots:	
		plt.show()
	return(residual, psf_img_norm_fac*components, list(zip(*rms_record)), list(zip(*fabs_record)), n_iters)	
	
#-----------------------------------------------------------------------------
# progress plotting routines for CLEAN algorithms
#-----------------------------------------------------------------------------

def clean_modified_progress_plots(f1, nr, nc, dirty_img, residual, components, current_cleaned, above_t, accumulator, convolved, rms_record, fabs_record, rms_threshold, fabs_threshold):
	"""
	Takes a figure and plots progress plots to it.

	# ARGUMENTS #
		f1
			Figure to add plots to
		nr
			Number of rows of axes to create (must be at least 2)
		nc
			Number of columns of axes to create (must be at least 4)
		dirty_img
			Array to plot as the dirty image
		residual
			Array to plot as the residual
		components
			Array to plot as the components
		current_cleaned
			Array to plot as the current CLEANed image
		above_t
			Array to plot that shows all chosen pixels above a certain threshold
		accumulator
			Array to plot that shows the number of times a pixel has been added to the component list
		convolved
			Array that shows the components convolved with the PSF
		rms_record
			Array containing the root-mean-squared for each iteration
		fabs_record
			Array containing the absolute brightest pixel for each iteration
		rms_threshold
			Value that the RMS has to reduce to to stop iteration
		fabs_threshold
			Value that the absolute brightest pixel has to reduce to to stop iteration

	# RETURNS #
		a1
			2D array of axes with size (nr,nc)
	
	"""
	# DEBUGGING PLOTS
	cmap = ('Spectra','twilight','coolwarm', 'bwr')[-3]
	plt.clf()
	a1 = f1.subplots(nr,nc)
	a1[0,0].imshow(dirty_img, origin='lower', cmap=cmap, vmin=-np.nanmax(np.fabs(dirty_img)), vmax=np.nanmax(np.fabs(dirty_img)))
	a1[0,0].set_title(f'dirty_img (max {np.nanmax(np.fabs(dirty_img)):07.2E})')
	a1[0,1].imshow(residual, origin='lower', cmap=cmap, vmin=-np.nanmax(np.fabs(residual)), vmax=np.nanmax(np.fabs(residual)))
	a1[0,1].set_title(f'residual (max {np.nanmax(np.fabs(residual)):07.2E})')
	a1[0,2].imshow(components, origin='lower', cmap=cmap, vmin=-np.nanmax(np.fabs(components)),
					vmax=np.nanmax(np.fabs(components)))
	a1[0,2].set_title('components')
	# put NANs back in
	current_cleaned[np.isnan(dirty_img)] = np.nan
	a1[0,3].imshow(current_cleaned, origin='lower', cmap=cmap, vmin=-np.nanmax(np.fabs(current_cleaned)), 
					vmax=np.nanmax(np.fabs(current_cleaned)))
	a1[0,3].set_title('current CLEANed image')
	
	a1[1,0].imshow(above_t, origin='lower')
	a1[1,0].set_title('above_t')
	a1[1,1].imshow(accumulator, origin='lower')#, cmap=cmap, vmin=-np.nanmax(np.fabs(selected)), vmax=np.nanmax(np.fabs(selected)))
	a1[1,1].set_title('accumulator')
	# put back in the nans
	#convolved[np.isnan(dirty_img)] = np.nan
	a1[1,2].imshow(convolved, origin='lower', cmap=cmap, vmin=-np.nanmax(np.fabs(convolved)), vmax=np.nanmax(np.fabs(convolved)))
	a1[1,2].set_title('convolved')
	
	a1[1,3].plot(rms_record, color='tab:blue', label='rms')
	a1[1,3].set_ylabel('rms')
	a1[1,3].set_ylim([0, np.nanmax(rms_record)])
	a1[1,3].axhline(rms_threshold, color='tab:blue', linestyle='-')
	a1[1,3].set_title('rms and |brightest pixel|')
	a1132 = a1[1,3].twinx()
	a1132.plot(fabs_record, color='tab:orange', label='abp')
	a1132.set_ylabel('abp')
	a1132.set_ylim([0, np.nanmax(fabs_record)])
	a1132.axhline(fabs_threshold, color='tab:orange', linestyle='--')
	#a1132.set_yscale('log')
	h1,l1 = a1[1,3].get_legend_handles_labels()
	h2,l2 = a1132.get_legend_handles_labels()
	#print(h1+h2)
	#print(l1+l2)
	a1[1,3].legend(h1+h2,l1+l2, loc='top right')
	
	return(a1)

def clean_hogbom_progress_plots(f1, nr, nc, dirty_img, psf_img, residual, components, rms_record, fabs_record, r_o_min, r_o_max, p_o_min, p_o_max, mx, my):
	"""
	Plots progress graphs for the 'clean_hogbom()' function
	"""
	plt.clf()
	a1 = f1.subplots(nr, nc, squeeze=False)

	a1[0,0].imshow(dirty_img, origin='lower')
	a1[0,0].set_title('dirty image')

	a1[0,1].imshow(residual, origin='lower')
	a1[0,1].set_title('residual')
	a1[0,1].axvline(r_o_min[1], color='red')
	a1[0,1].axvline(r_o_max[1], color='red')
	a1[0,1].axhline(r_o_min[0], color='red')
	a1[0,1].axhline(r_o_max[0], color='red')
	a1[0,1].scatter(mx, my, facecolors='none', edgecolors='red')
	a1[0,1].set_xlim((0, residual.shape[0]))
	a1[0,1].set_ylim((0, residual.shape[1]))

	a1[0,2].imshow(components, origin='lower')
	a1[0,2].set_title('components')

	a1[0,3].imshow(sp.signal.convolve2d(components, psf_img, mode='same'), origin='lower')
	a1[0,3].set_title('current CLEANed image')

	a1[1,0].plot(rms_record)
	a1[1,0].set_title('rms_record')

	a1[1,1].plot(fabs_record)
	a1[1,1].set_title('fabs_record')

	#a1[1,2].remove()
	a1[1,2].imshow(psf_img, origin='lower')
	a1[1,2].axvline(p_o_min[1], color='red')
	a1[1,2].axvline(p_o_max[1], color='red')
	a1[1,2].axhline(p_o_min[0], color='red')
	a1[1,2].axhline(p_o_max[0], color='red')
	a1[1,2].set_xlim((0, psf_img.shape[0]))
	a1[1,2].set_ylim((0, psf_img.shape[1]))

	a1[1,3].remove()

	return(a1)

#-----------------------------------------------------------------------------
# Helper functions
#-----------------------------------------------------------------------------

def dirac_delta_filter(x):
	"""
	Create a dirac delta filter the same shape as array x
	"""
	cidx = tuple([_x//2 for _x in x.shape])
	dd = np.zeros_like(x)
	dd[cidx] = 1
	return(dd)

def is_subscriptable(x):
	"""
	return true if an object is subscriptable
	"""
	return( hasattr(x, '__getitem__'))

def ensure_subscriptable(x, min_size=1, max_size=None):
	"""
	Check that an object is subscriptable with a maximum and/or minimum size
	if object does not satisfy requirements, extend or truncate it.
	"""
	if is_subscriptable(x):
		if (not (max_size is None)) and (len(x) > max_size):
			x = x[:size]
		if len(x) < min_size:
			x = type(x)(list(x) + [list(x)[-1] for i in range(len(x),size)])
	else:
		x = tuple([x for i in range(min_size)])
	return(x)

def pad_psf_to_minimum_size(psf_img, dirty_img_shape, max_size_filter, pad_step, show_plots=False):
	while True:
		psf_test = max_size_filter(psf_img)
		edges = np.stack([psf_test[0,:], psf_test[-1,:], psf_test[:,0], psf_test[:,-1]]).flatten()
		psf_test_larger_than_2x_dirty_img_shape = all([s >= 2*ds for s, ds in zip(psf_test.shape, dirty_img_shape)])
		if (edges == 0).all() or psf_test_larger_than_2x_dirty_img_shape:
			break
		psf_img = psf_resize_ensure_centered(psf_img, int(pad_step))
		print(f'INFO: psf_img.shape {psf_img.shape}')
		if show_plots:
			plt.imshow(psf_img, origin='lower')
			plt.scatter(*[s//2 for s in psf_img.shape], color='red')
			plt.show()
	return(psf_img)

def psf_resize_ensure_centered(psf_img, n_padding):
	return(ensure_centered_psf(np.pad(psf_img, n_padding)))

def psf_resize_ensure_centered_OLD(psf_img, factor_increase=2):
	"""
	increase the size of an array by a given factor. Padding for the new array should
	be around the edge of the old array (i.e. the old data is centered in the new array)
	"""
	psf_img_temp = np.zeros([factor_increase*_x for _x in psf_img.shape])
	#print(psf_img.shape)
	#print(psf_img_temp.shape)
	offsets = tuple([_x*(factor_increase-1)//2 for _x in psf_img.shape]) 
	#print(offsets)
	slices = tuple([slice(offsets[i]+1-factor_increase%2, psf_img_temp.shape[i]-offsets[i]) for i in range(len(psf_img_temp.shape))])
	#print(slices)
	psf_img_temp[slices] = psf_img
	return(ensure_centered_psf(psf_img_temp))
	
def rect_overlap_lims(sa, sb, x):
	zeros = np.zeros_like(sa)
	return(np.clip(x, zeros, sa), np.clip(sb+x, zeros, sa), np.clip(-x, zeros, sb), np.clip(sa-x, zeros, sb))

def ensure_centered_psf(psf_img, fill_value=0, show_plots=False):
	# pad array to odd number of elements on each axis
	padding = [(0,1-s%2) for s in psf_img.shape]
	#print(psf_img.shape)
	#print(padding)
	psf_adj = np.pad(psf_img, padding, mode='constant', constant_values=fill_value)

	cidx = np.array(psf_adj.shape)//2
	max_idx = np.unravel_index(np.nanargmax(psf_adj), psf_adj.shape)

	for i, (ci, mi) in enumerate(zip(cidx, max_idx)):
		np.roll(psf_adj, ci-mi, axis=i) 
	return(psf_adj)

def ensure_centered_psf_OLD(psf_img, fill_value=0, show_plots=False):
	"""
	To conovlve/deconvolve a PSF we need to make sure the PSF is centered in an image with an odd number of pixels
	
	E.g. this is good
	#-------#
	|0000000|
	|0001000|
	|0012100|
	|0123210|
	|0012100|
	|0001000|
	|0000000|
	#-------#
	[7x7, odd box]

	and this is bad
	#------#
	|000000|
	|000100|
	|001210|
	|012321|
	|001210|
	|000100|
	#------#
	[6x6, even box]

	This routine takes the even box case and creates a box that is larger by one in any even dimension
	"""
	#print('INFO: In "ensure_centered_psf()"')
	
	psf_shape = np.array(psf_img.shape)
	for i in range(psf_shape.size):
		if (psf_shape[i]%2) == 0:
			psf_shape[i] += 1

	psf_adj = np.full(psf_shape, fill_value=fill_value, dtype=psf_img.dtype)
	#print(f'psf_img.shape {psf_img.shape}')
	#print(f'psf_adj.shape {psf_adj.shape}')

	offset = np.array(np.unravel_index(np.nanargmax(psf_img), psf_img.shape)) - np.array(psf_img.shape)//2
	
	#print(f'offset {offset}')

	al, ah, il, ih = rect_overlap_lims(np.array(psf_adj.shape), np.array(psf_img.shape), offset)
	#print('rect lims', al, ah, il, ih)	

	adj_slice = tuple([slice(l,h) for l,h in zip(al, ah)])
	img_slice = tuple([slice(l,h) for l,h in zip(il, ih)])
	
	#print(f'adj_slice {adj_slice}')
	#print(f'img_slice {img_slice}')

	#print(f'type(psf_img) {type(psf_img)}')
	#print(f'type(psf_adj) {type(psf_adj)}')

	#print(f'psf_img.dtype {psf_img.dtype}')
	#print(f'psf_adj.dtype {psf_adj.dtype}')

	psf_adj[adj_slice] = psf_img[img_slice]

	if show_plots:
		nc, nr = (2,1)
		s = 5
		f1 = plt.figure(figsize=(s*nc, s*nr))
		a1 = f1.subplots(nr,nc, squeeze=False)
	
		a1[0,0].imshow(psf_img, origin='lower')
		a1[0,0].scatter(*np.array(psf_img.shape)//2, facecolors='none', edgecolors='red')
		a1[0,0].set_title(f'psf_img {psf_img.shape}')

		a1[0,1].imshow(psf_adj, origin='lower')
		a1[0,1].scatter(*np.array(psf_adj.shape)//2, facecolors='none', edgecolors='red')
		a1[0,1].set_title(f'psf_adj {psf_adj.shape}')

		plt.show()

	return(psf_adj)	

def get_wavrange_img_from_fits(target_cube, wav_range=(1.9,2.0), function=np.nanmedian):
	"""
	Takes a fitscube and returns a stacked slice along the wavelength axis

	# ARGUMENTS #
		target_cube
			<str> fitscube to operate on
		wav_range
			<float,float> Wavelength range (um) to stack cube between
		function
			<callable> Function that does the stacking, should take a (nz,ny,nx) shaped cube and return a (ny,nx) shaped image,
			default is to take the median value of each wavelength
	
	# RETURNS #
		cube_img
			The stacked primary image
		psf_img
			The stacked PSF image
		cube_nans
			A boolean array where True means that a given pixel is best interpreted as a NAN value
		disk_mask
			A boolean array where True means that the given pixel is on the disk of the planet under observation.
	"""
	with fits.open(target_cube) as hdul:
		psf_data = hdul['PSF'].data
		cube_data = hdul['PRIMARY'].data
		wavgrid = fitscube.header.get_wavelength_grid(hdul['PRIMARY'])
		disk_mask = np.array(hdul['DISK_MASK'].data, dtype=bool)
		
	wav_idxs = (np.nanargmin(np.fabs(wavgrid-wav_range[0])), np.nanargmin(np.fabs(wavgrid-wav_range[1])))
			
	psf_img = function(psf_data[wav_idxs[0]:wav_idxs[1]], axis=0)
	psf_img /= np.nansum(psf_img) # normalise psf so integral is one
	
	
	cube_img = np.nan_to_num(function(cube_data[wav_idxs[0]:wav_idxs[1]], axis=0))
	cube_nans = (cube_img==0) | np.isnan(cube_img)
	
	return(cube_img, psf_img, cube_nans, disk_mask)

#-----------------------------------------------------------------------------
# Final plotting routines
#-----------------------------------------------------------------------------

def plot_clean_result(dirty_img, components, residual, psf_img, rms_record, fabs_record, accumulator, gaussian_kernel_std=2,
						fig=None, axes=None, show_plot=True, limits_symmetric_about_zero=True):
	
	reconvolve = sp.ndimage.convolve(components, psf_img)
	
	nr,nc=(2,5)
	s = 4
	if fig is None:
		f1 = plt.figure(figsize=(nc*s,nr*s))
	else:
		f1 = fig
	if axes is None:
		a1 = f1.subplots(nr,nc,squeeze=False)
	else:
		a1 = axes
	
	cmin = np.nanmin(dirty_img)
	cmin = cmin if cmin > 1E-6 else 1E-6
	cmax = np.nanmax(dirty_img)
	
	if limits_symmetric_about_zero:
		clims = {'vmin':-cmax, 'vmax':cmax}
	else:
		clims = {'vmin':0, 'vmax':cmax}
	norm = mpl.colors.SymLogNorm(1E-7, linscale=1, **clims)
	#norm = mpl.colors.LogNorm(**clims)
	#norm = mpl.colors.Normalize(**clims)
	
	
	im100 = a1[0,0].imshow(components, origin='lower', norm=norm)
	a1[0,0].set_title(f'sum of components {np.nansum(components):07.2E}')
	f1.colorbar(im100, ax=a1[0,0])
	
	
	im101 = a1[0,1].imshow(residual, origin='lower', norm=norm)#, norm=mpl.colors.SymLogNorm(1E-11, linscale=1, vmin=np.nanmin(residual), vmax=np.nanmax(residual)))
	a1[0,1].set_title(f'sum of residual {np.nansum(residual):07.2E}')
	f1.colorbar(im101,ax=a1[0,1])
	
	im110 = a1[1,0].imshow(dirty_img, origin='lower', norm=norm)
	a1[1,0].set_title(f'sum of dirty img {np.nansum(dirty_img):07.2E}')
	f1.colorbar(im110, ax=a1[1,0])
	
	im111 = a1[1,1].imshow(psf_img, origin='lower')
	a1[1,1].set_title(f'sum of psf {np.nansum(psf_img):07.2E}')
	f1.colorbar(im111, ax=a1[1,1])
	
	a1[0,2].imshow(accumulator, origin='lower')
	a1[0,2].set_title(f'Sum of accumulator {np.nansum(accumulator)}')
	
	a1[1,2].plot(fabs_record, label='fabs', color='tab:blue')
	a112x = a1[1,2].twinx()
	a112x.plot(rms_record, label='rms', color='tab:orange')
	h1,l1 = a1[1,2].get_legend_handles_labels()
	h2,l2 = a112x.get_legend_handles_labels()
	a1[1,2].set_title('Residual fabs and rms')
	a1[1,2].set_yscale('log')
	a1[1,2].legend(h1+h2,l1+l2)
	a1[1,2].set_xlabel('Iteration')
	
	
	a1[0,3].imshow(reconvolve, origin='lower', norm=norm)
	a1[0,3].set_title('components convolved with psf')
	
	a1[1,3].imshow(reconvolve + residual, origin='lower', norm=norm)
	a1[1,3].set_title('components convolved with psf and residual added')
	
	gaussian_kernel = astropy.convolution.Gaussian2DKernel(gaussian_kernel_std)
	a1[0,4].imshow(astropy.convolution.convolve(components, gaussian_kernel), origin='lower', norm=norm)
	a1[0,4].set_title('components convolved with gaussian kernel')
	
	gk_array = np.zeros_like(psf_img)
	gk_array[psf_img.shape[0]//2, psf_img.shape[1]//2] = 1
	gk_array = astropy.convolution.convolve(gk_array, gaussian_kernel)
	a1[1,4].imshow(gk_array, origin='lower')
	a1[1,4].set_title('gaussian kernel')

	# DEBUGGING 
	dbg = dirty_img - (reconvolve + residual)
	imDEBUG = a1[0,2].imshow(dbg, origin='lower', norm=norm)
	a1[0,2].set_title(f'dbg sum {np.nansum(dbg):07.2E}' )
	f1.colorbar(imDEBUG, ax=a1[0,2])


	if show_plot:
		plt.show()
	return

#-----------------------------------------------------------------------------
# Pre-CLEAN filtering routines to remove speckles/anomalies
#-----------------------------------------------------------------------------



#-----------------------------------------------------------------------------
# script testing routines
#-----------------------------------------------------------------------------
#%%
if __name__=='__main__':
	print(f'INFO: In "clean_testing.py" at start of MAIN')
	#target_cube = os.path.expanduser('~/scratch/reduced_images/SINFO.2018-09-19T02:21:05/MOV_Neptune---H+K_0.025_1_tpl/analysis/obj_NEPTUNE_cal_HD2811_renormed.fits')
	#target_cube = os.path.expanduser('~/scratch/reduced_images/SINFO.2018-09-19T02:21:05/MOV_Neptune---H+K_0.025_1_tpl/analysis/obj_NEPTUNE_cal_HIP105633_renormed.fits')
	target_cube = os.path.expanduser('~/scratch/reduced_images/SINFO.2018-09-30T01:02:39/MOV_Neptune---H+K_0.025_3_tpl/analysis/obj_NEPTUNE_cal_HD216009_renormed.fits')
	#target_cube = os.path.expanduser('~/scratch/reduced_images/SINFO.2018-08-31T05:45:50/MOV_Neptune---H+K_0.1_tpl/analysis/obj_NEPTUNE_cal_HIP001115_renormed.fits')
	
	# set up parameters for CLEAN algorithms
	wav_range = (#(1.5,1.6),
				  #(1.7,1.8),
				  #(1.895,1.9),
				  (1.455, 1.460),
				  #(1.9,2.0),
				  )[-1]

	hogbom_loop_gain = 0.2
	hogbom_rms_frac_threshold = 1E-2
	hogbom_fabs_frac_threshold = 1E-2
	hogbom_max_iter = 10000 # 1000000
	modified_loop_gain=0.2
	modified_threshold=0.6
	modified_rms_frac_threshold=1E-2
	modified_fabs_frac_threshold=1E-2
	modified_max_iter = 2000 # 200


	# wrangle *.fits file into numpy array
	cube_img, psf_img, cube_nans, disk_mask = get_wavrange_img_from_fits(target_cube, wav_range, function=np.nanmedian)
	#psf_img = ensure_centered_psf(psf_img)

	# copy data array so we can modify it before CLEANing if we want to
	cube_img_2 = np.array(cube_img)
	#cube_img_2[~disk_mask] = 0
	cube_img_2[cube_nans] = np.nan
	"""	
	#%%
	cube_img_2_sobel = np.sqrt(scipy.ndimage.sobel(cube_img_2, 0)**2 + scipy.ndimage.sobel(cube_img_2, 1)**2)
	cube_img_2_sobel[cube_img_2_sobel > 7E-7] = np.nan
	plt.imshow(cube_img_2_sobel, origin='lower')
	plt.show()
	
	#%%
	cube_img_2_laplace = np.fabs(scipy.ndimage.laplace(cube_img_2))
	cube_img_2_laplace[cube_img_2_laplace > 3E-7] = np.nan
	plt.imshow(cube_img_2_laplace, origin='lower')
	plt.show()
	
	#%%
	cube_img_2_prewitt = np.sqrt(scipy.ndimage.prewitt(cube_img_2, 0)**2 + scipy.ndimage.prewitt(cube_img_2, 1)**2)
	cube_img_2_prewitt[cube_img_2_prewitt > 7E-7] = np.nan
	plt.imshow(cube_img_2_prewitt, origin='lower')
	plt.show()

	#%%
	#cube_img_2_gl = np.fabs(scipy.ndimage.gaussian_laplace(cube_img_2, 2))
	cube_img_2_gl = scipy.ndimage.gaussian_laplace(cube_img_2, 2)
	#cube_img_2_gl[cube_img_2_gl > 1E-7] = np.nan
	plt.imshow(cube_img_2_gl, origin='lower')
	plt.show()
	
	#%%
	#mask = cube_img_2 < 0.5*np.nanmax(cube_img_2)
	#cube_img_3 = np.array(cube_img_2)
	#cube_img_3[mask] = 0
	cube_img_2_v1 = np.fabs(cube_img_2 - sp.signal.convolve2d(np.nan_to_num(cube_img_2), psf_img, mode='same'))
	mask = cube_img_2_v1 > 3E-7
	cube_img_2_v1[mask] = np.nan
	plt.imshow(cube_img_2_v1, origin='lower')
	#plt.imshow(np.log(cube_img_2_v1), origin='lower')
	plt.show()

	#%%
	cube_img_2_g = np.fabs(cube_img_2 - scipy.ndimage.gaussian_filter(np.nan_to_num(cube_img_2), 1.5))
	mask = cube_img_2_g > 5E-8
	cube_img_2_g[mask] = np.nan
	plt.imshow(cube_img_2_g, origin='lower')
	#plt.imshow(np.log(cube_img_2_g), origin='lower')
	plt.show()

	#%%
	cube_img_adj = np.array(cube_img_2)
	cube_img_adj[mask] = np.nan
	cube_img_adj[cube_img_adj<0] = np.nan
	plt.imshow(cube_img_adj, origin='lower')
	plt.show()

	#%%
	cube_img_adj = np.array(cube_img_2)
	cube_img_adj[np.isnan(cube_img_adj)] = 0
	cube_img_fft = np.fft.fft2(cube_img_adj)
	psf_img_fft = np.fft.fft2(psf_img)
	#plt.imshow(psf_img_fft.real)
	#plt.show()
	
	#cube_img_2 = cube_img_fft
	#psf_img = psf_img_fft
	"""

	#DEBUGGING
	import time
	
	t1 = time.time()
	#%%
	# Perform hogbom CLEAN algorithm
	(	residual_1, 
		components_1, 
		rms_record_1, 
		fabs_record_1, 
		n_iter_1
	) = clean_multiresolution(	cube_img_2, 
						psf_img, 
						window=(disk_mask | cube_nans), 
						loop_gain=hogbom_loop_gain, 
						rms_frac_threshold=hogbom_rms_frac_threshold, 
						fabs_frac_threshold=hogbom_fabs_frac_threshold,
						max_iter=hogbom_max_iter,
						norm_psf=True,
						show_plots=False
						)
	print(f'DEBUG: elapsed time {time.time() - t1}')

	#def get_hot_pixels():
	print(f'INFO: In "get_hot_pixels()"')
	#print(f'INFO: frac {frac}')
	dbg_img = cube_img_2 -(sp.signal.convolve2d(components_1, psf_img, mode='same') + residual_1)
	fabs_flat_dbg_img = np.fabs(dbg_img).flatten()
	ns, edges = np.histogram(dbg_img.flatten(), bins=100, range=(np.nanmin(dbg_img), np.nanmax(dbg_img)), density=True)
	mids = 0.5*(edges[:-1]+edges[1:])

	half_normal = lambda x, sigma: np.sqrt(2)/(sigma*np.sqrt(np.pi))*np.exp(-(x**2)/(2*sigma**2))
	normal = lambda x, sigma, mu: (1.0/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((x-mu)/sigma)**2)

	mle = lambda func, args, xi: (1.0/np.size(xi))*np.nansum(np.log(func(xi,*args)))
	neg_mle = lambda func, args, xi: -mle(func, args, xi)

	def bisection_minima(func, bounds, tol=None, max_iter=100):
		if tol is None:
			tol = 1E-3*(bounds[1]-bounds[0])
		mid = lambda x1, x2: 0.5*(x1+x2)
		test_points = np.array((bounds[0], mid(*bounds), bounds[1]))
		test_at_points = np.vectorize(lambda a: func(a))
		test_vals = test_at_points(test_points)
		tmp = np.zeros((2,))
		i=0
		while test_points[2]-test_points[0] > tol and i < max_iter:
			tmp[0] = mid(*test_points[:2])
			tmp[1] = mid(*test_points[1:])
			print(f'\n{test_points[0]:010.5E} {tmp[0]:010.5E} {test_points[1]:010.5E} {tmp[1]:010.5E} {test_points[2]:010.5E}')
			tmp_val = test_at_points(tmp)	
			print(f'{test_vals[0]:010.5E} {tmp_val[0]:010.5E} {test_vals[1]:010.5E} {tmp_val[1]:010.5E} {test_vals[2]:010.5E}')
			if tmp_val[0] < test_vals[0] and tmp_val[0] < test_vals[1]:
				test_points[2] = test_points[1]
				test_vals[2] = test_vals[1]
				print('>>>')
			elif tmp_val[1] < test_vals[2] and tmp_val[1] < test_vals[1]:
				test_points[0] = test_points[1]
				test_vals[0] = test_vals[1]
				print('<<<')
			elif tmp_val[0] > test_vals[1] and test_vals[1] < tmp_val[1]:
				test_points[0] = tmp[0]
				test_points[2] = tmp[1]
				test_vals[0] = tmp_val[0]
				test_vals[2] = tmp_val[1]
				print('< >')
			else:
				print('ERROR: Detected more than one minima in range, terminating')
				raise NotImplementedError
			test_points[1] = mid(test_points[0], test_points[2])
			test_vals[1] = func(test_points[1])
			i+=1
		return(np.array([test_points[1]]))

	s_grid = np.linspace(0, np.nanmax(dbg_img), 200)
	mle_search = np.array(list(map(lambda sigma: neg_mle(normal, (sigma, 0), dbg_img.flatten()), s_grid)))

	x0 = [np.nanmean(np.fabs(dbg_img.flatten()))]
	#minimise_result = sp.optimize.minimize(lambda sigma, xi: neg_mle(normal, (sigma, 0), xi), x0, args=(dbg_img.flatten()))
	#mle_sigma = minimise_result.x
	mle_sigma = bisection_minima(lambda sigma: neg_mle(normal, (sigma,0), dbg_img.flatten()), [0, np.nanmax(dbg_img)])
	#print(minimise_result)
	cutoff = 5*mle_sigma
	dbg_hot_pixels = (-cutoff > dbg_img) | (dbg_img > cutoff)
	#dbg_hot_pixels = sp.ndimage.binary_closing(dbg_hot_pixels)
	cube_img_2[dbg_hot_pixels] = np.nan
	def plot_dbg_img_stuff():
		nr,nc,s=(1,5,6)
		f1 = plt.figure(figsize=(s*nc,s*nr))
		a1 = f1.subplots(nr,nc,squeeze=False)
		a1[0,0].imshow(dbg_img, origin='lower')
		a1[0,1].imshow(dbg_hot_pixels, origin='lower')
		a1[0,2].imshow(cube_img_2, origin='lower')
		a1[0,3].hist(dbg_img.flatten(), bins=100, density=True)
		a1[0,3].plot(edges, normal(edges, mle_sigma, 0))
		a1[0,3].axvline(cutoff)
		a1[0,3].set_title(f'sigma {mle_sigma[0]:07.2E} x0 {x0[0]:07.2E}')
		a1[0,4].plot(s_grid, mle_search)
		a1[0,4].set_title(f'min at {s_grid[np.nanargmin(mle_search)]:07.2E}')
		plt.show()
	#return()
	#get_hot_pixels(0.5*np.nanmax(np.nanmax([x[-1] for x in fabs_record_1])/np.nanmax(np.fabs(cube_img_2))))
	#get_hot_pixels()
	plot_dbg_img_stuff()	

	plot_clean_result(cube_img_2, components_1, residual_1, psf_img, rms_record_1, fabs_record_1, np.zeros_like(cube_img_2))
	sys.exit('DEBUGGING')

	(	residual_1, 
		components_1, 
		rms_record_1, 
		fabs_record_1, 
		n_iter_1
	) = clean_multiresolution(	cube_img_2, 
						psf_img, 
						window=(disk_mask | cube_nans), 
						loop_gain=hogbom_loop_gain, 
						rms_frac_threshold=hogbom_rms_frac_threshold, 
						fabs_frac_threshold=hogbom_fabs_frac_threshold,
						max_iter=hogbom_max_iter,
						norm_psf=True,
						show_plots=False
						)
	print(f'DEBUG: elapsed time {time.time() - t1}')
	

	plot_clean_result(cube_img_2, components_1, residual_1, psf_img, rms_record_1, fabs_record_1, np.zeros_like(cube_img_2))
	sys.exit(f'DEBUGGING') # DEBUGGING
	#%%
	"""
	# Perform modified CLEAN algorithm
	
	(	residual_2, 
		components_2, 
		rms_record_2,
		fabs_record_2, 
		n_iter_2,
		accumulator_2,
		window_2
	) = clean_modified(	cube_img_2, 
						psf_img, 
						window=(disk_mask | cube_nans), 
						loop_gain=modified_loop_gain, 
						threshold=modified_threshold,
						rms_frac_threshold=modified_rms_frac_threshold, 
						fabs_frac_threshold=modified_fabs_frac_threshold,
						max_iter=modified_max_iter,
						norm_psf=False, 
						show_plots=False
						)

	#plot_clean_result(cube_img_2, components_2, residual_2, psf_img, rms_record_2, fabs_record_2, accumulator_2)
	"""
	
	# START TESTING -----------------------------------------------------------
	# Remove pixels with a high accumulator value (set as NAN and re-run algorithm)	
	#n_max = 40
	#problem_factor = 0.6 # should I use a constant value instead of a factor?
	#n_problem_pixels = np.full((n_max,), fill_value=np.nan)
	#n_iters = np.full((n_max,), fill_value=np.nan)
	
	#for n in range(n_max):
	"""
		(	residual_2, 
			components_2, 
			rms_record_2,
			fabs_record_2, 
			n_iter_2,
			accumulator_2,
			window_2
		)
	"""
	
	n_max = 20
	n = 0
	n_iter_per_chunk = 100
	n_iter_2 = 0
	results_tuple = None
	problem_factor = 0.6
	n_problem_pixels =  np.full((n_max,), fill_value=np.nan)
	n_iters =  np.full((n_max,), fill_value=np.nan)
	while n < n_max:
		max_iter_updated = n_iter_2 + n_iter_per_chunk
		if n_iter_2 == 0:
			results_tuple=None
		results_tuple = clean_modified(	cube_img_2, 
							psf_img, 
							window=(disk_mask | cube_nans), 
							loop_gain=modified_loop_gain, 
							threshold=modified_threshold,
							rms_frac_threshold=modified_rms_frac_threshold, 
							fabs_frac_threshold=modified_fabs_frac_threshold,
							max_iter=max_iter_updated,
							norm_psf=False, 
							show_plots=False,
							n_positive_iter = n_iter_per_chunk//2,
							restart_tuple = results_tuple
							)
		
		n_iter_2 = results_tuple[4]
		accumulator_2 = results_tuple[5]		 
		#plot_clean_result(cube_img_2, components_2, residual_2, psf_img, rms_record_2, fabs_record_2, accumulator_2)
		problem_pixels = accumulator_2 > problem_factor*max_iter_updated # should this compare to the maximum iterations or the iterations performed?
		n_problem_pixels[n] = np.sum(problem_pixels)
		n_iters[n] = n_iter_2
		if n_problem_pixels[n] > 0:
			cube_img_2[problem_pixels] = np.nan
			n_iter_2 = 0
		elif n_iter_2 < (max_iter_updated-1):
			break
		n+=1
	
	#%% PLOT RESULTS
	(	residual_2, 
		components_2, 
		rms_record_2,
		fabs_record_2, 
		n_iter_2,
		accumulator_2,
		window_2
	) = results_tuple
	
	plot_clean_result(cube_img_2, components_2, residual_2, psf_img, rms_record_2, fabs_record_2, accumulator_2)
	
	s, nc, nr = (6, 2, 1)
	f0 = plt.figure(figsize=(nc*s, nr*s))
	a0 = f0.subplots(nr,nc, squeeze=False)
	
	a0[0,0].plot(n_problem_pixels, label='number of problem pixels')
	a0[0,0].plot(np.cumsum(n_problem_pixels), label='cumulative number of problem pixels')
	a0[0,0].legend()
	
	a0[0,1].plot(n_iters, label='Number of iterations')
	a0[0,1].legend()
	
	# END TESTING -------------------------------------------------------------
	
	# convolve the CLEANed component maps with the observation's PSF t
	reconvolve_1 = sp.ndimage.convolve(components_1, psf_img)
	reconvolve_2 = sp.ndimage.convolve(components_2, psf_img)

	#--------------------------------------------------------------------------
	# PLOT AND COMPARE HOGBOM AND MODIFIED CLEAN ALGORITHM RESULTS
	#--------------------------------------------------------------------------
	
	# create a helper function that removes axes from a graph
	remove_axes = lambda ax: (ax.axes.get_xaxis().set_visible(False), ax.axes.get_yaxis().set_visible(False))

	# create a helper function that sets vmin and vmax to be equally spaced around zero
	equal_vmin_vmax_around_zero = lambda a: {'vmin':-np.nanmax(np.fabs(a)), 'vmax':np.nanmax(np.fabs(a))}

	# set the default color map to be used
	default_colormap = ('viridis','twilight','twilight_shifted', 'seismic')[-1]
	mpl.rc('image', cmap= default_colormap)
	
	# create a figure and subplots
	s, nc, nr = (5, 5, 3)
	f1 = plt.figure(figsize=(nc*s, nr*s))
	a1 = f1.subplots(nr,nc, squeeze=False)

	f1.suptitle('Comparing Hogbom CLEAN and Modified CLEAN algorithms')

	a1[0,0].imshow(cube_img_2, origin='lower', **equal_vmin_vmax_around_zero(cube_img_2))
	a1[0,0].set_title(f'dirty image')
	remove_axes(a1[0,0])

	a1[0,1].imshow(components_1, origin='lower', **equal_vmin_vmax_around_zero(components_1))
	a1[0,1].set_title(f'hogbom components\n(max {np.nanmax(np.fabs(components_1)):07.2E})')
	remove_axes(a1[0,1])

	a1[0,2].imshow(residual_1, origin='lower', **equal_vmin_vmax_around_zero(cube_img_2))
	a1[0,2].set_title(f'hogbom residual\n(max {np.nanmax(np.fabs(residual_1)):07.2E})')
	remove_axes(a1[0,2])

	a1[0,3].plot(rms_record_1, color='tab:blue')
	a103_2 = a1[0,3].twinx()
	a103_2.plot(fabs_record_1, color='tab:orange')
	a1[0,3].set_title(f'hogbon rms (blue) fabs (orange)')

	a1[0,4].imshow(reconvolve_1, origin='lower', **equal_vmin_vmax_around_zero(reconvolve_1))
	a1[0,4].set_title(f'hogbom convolve(components,PSF)\n(max {np.nanmax(np.fabs(reconvolve_1)):07.2E})')
	remove_axes(a1[0,4])


	a1[1,0].imshow(cube_img_2, origin='lower', **equal_vmin_vmax_around_zero(cube_img_2))
	a1[1,0].set_title(f'dirty image')
	remove_axes(a1[1,0])

	a1[1,1].imshow(components_2, origin='lower', **equal_vmin_vmax_around_zero(components_2))
	a1[1,1].set_title(f'modified components\n(max {np.nanmax(np.fabs(components_2)):07.2E})')
	remove_axes(a1[1,1])

	a1[1,2].imshow(residual_2, origin='lower', **equal_vmin_vmax_around_zero(cube_img_2))
	a1[1,2].set_title(f'modified residual\n(max {np.nanmax(np.fabs(residual_2)):07.2E})')
	remove_axes(a1[1,2])

	a1[1,3].plot(rms_record_2, color='tab:blue')
	a113_2 = a1[1,3].twinx()
	a113_2.plot(fabs_record_2, color='tab:orange')
	a1[1,3].set_title(f'modified rms(blue) fabs(orange)')

	a1[1,4].imshow(reconvolve_2, origin='lower', **equal_vmin_vmax_around_zero(reconvolve_2))
	a1[1,4].set_title(f'modified convolve(components,PSF)\n(max {np.nanmax(np.fabs(reconvolve_2)):07.2E})')
	remove_axes(a1[1,4])

	components_diff = components_1 - components_2
	residual_diff = residual_1 - residual_2
	reconvolve_diff = reconvolve_1 - reconvolve_2
	a1[2,0].remove()

	a1[2,1].imshow(components_diff, origin='lower', **equal_vmin_vmax_around_zero(components_diff))
	a1[2,1].set_title(f'component difference\n(max {np.nanmax(np.fabs(components_diff)):07.2E})')
	remove_axes(a1[2,1])

	a1[2,2].imshow(residual_diff, origin='lower', **equal_vmin_vmax_around_zero(cube_img_2))
	a1[2,2].set_title(f'residual difference\n(max {np.nanmax(np.fabs(residual_diff)):07.2E})')
	remove_axes(a1[2,2])

	a1[2,3].remove()

	a1[2,4].imshow(reconvolve_diff, origin='lower', **equal_vmin_vmax_around_zero(reconvolve_diff))
	a1[2,4].set_title(f'convolve(component,PSF) difference\n(max {np.nanmax(np.fabs(reconvolve_diff)):07.2E})')
	remove_axes(a1[2,4])

	plt.show()
