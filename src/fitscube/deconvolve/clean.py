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

def clean_hogbom(	dirty_img, 
					psf_img,
					window=True, 
					loop_gain=1E-1, 
					rms_frac_threshold=1E-2, 
					fabs_frac_threshold=1E-2, 
					max_iter=int(1E6), 
					norm_psf=True, 
					show_plots=False, 
					quiet=True, 
					ensure_psf_centered=True, 
					n_positive_iter=0, 
					sum_limit_factor=1.0):
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
			If True, will normalise the PSF so it's sum is 1 before CLEANing. The reason we normalise
			the PSF so it's sum is 1 (rather than it's peak) is so that the flux of the dirty image and
			the flux of the components are the same (if the dirty image is perfectly CLEANed).
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
	residual_windowed = np.array(dirty_img)
	r_shape = np.array(residual.shape)
	p_shape = np.array(psf_img.shape)
	components = np.zeros_like(residual)
	rms_record = np.full((max_iter,), fill_value=np.nan)
	fabs_record = np.full((max_iter,), fill_value=np.nan)
	fabs_threshold = np.nanmax(np.fabs(residual))*fabs_frac_threshold
	rms_threshold = np.sqrt(np.nansum((residual)**2)/residual.size)*rms_frac_threshold
	component_sum_record = np.full((max_iter,), fill_value=np.nan)
	

	if ((type(norm_psf) is str) and (norm_psf=='sum')) or ((type(norm_psf) is bool) and norm_psf):
		psf_img /= np.nansum(psf_img)
	elif (type(norm_psf) is str) and norm_psf=='max':
		psf_img /= np.nanmax(psf_img)

	# if the integral of the PSF != 1 then when we are adding components
	# we need to that the integrated flux is correct
	flux_correction_factor = np.nansum(psf_img)
	#print(f'DEBUG: flux_correction_factor {flux_correction_factor}')

	# if the maximum of the PSF != 1 then we need to adjust the amount
	# of flux minused from the residual to keep the desired loop gain
	loop_gain_correction_factor = 1.0/np.nanmax(psf_img)

	if ensure_psf_centered:
		psf_img = ensure_centered_psf(psf_img)
	
	if window is True:
		window = np.ones(dirty_img.shape, dtype=bool)
	
	if show_plots:
		plt.ion()
		nc, nr = (4,2)
		s=4
		f1 = plt.figure(figsize=(nc*s, nr*s))
		plt.show()

	residual_windowed[~window] = np.nan
	#residual = np.nan_to_num(residual) # FOR DEBUGGING FLUX CONSERVATION, NANs screw up the calculations
	# perform CLEAN algorithm
	fabs_min = 0.5*np.nanmax(np.fabs(residual))
	rms_min = 0.5*np.sqrt(np.nansum((residual)**2)/residual.size)
	component_sum_limit = sum_limit_factor*np.nansum(residual)
	for i in range(max_iter):
		if i < n_positive_iter:
			my, mx = np.unravel_index(np.nanargmax(residual_windowed), residual.shape)
		else:
			my, mx = np.unravel_index(np.nanargmax(np.fabs(residual_windowed)), residual.shape)
		mval = residual[my,mx]*loop_gain*loop_gain_correction_factor
		
		# add flux to component map
		components[my,mx] += mval*flux_correction_factor
		#print(f'my {my} mx {mx} mval {mval}')
		component_sum_record[i] = np.sum(components)
		
	
		# get overlapping rectangles for residual cube and psf_cube
		r_o_min, r_o_max, p_o_min, p_o_max = rect_overlap_lims(r_shape, p_shape, np.array([my, mx])-p_shape//2)
		
		
		residual[r_o_min[0]:r_o_max[0],r_o_min[1]:r_o_max[1]] -= mval*psf_img[p_o_min[0]:p_o_max[0],p_o_min[1]:p_o_max[1]]
		residual_windowed[r_o_min[0]:r_o_max[0],r_o_min[1]:r_o_max[1]] -= mval*psf_img[p_o_min[0]:p_o_max[0],p_o_min[1]:p_o_max[1]]

		resid_fabs = np.nanmax(np.fabs(residual))
		resid_rms = np.sqrt(np.nansum((residual)**2)/residual.size)
		if fabs_min > resid_fabs: fabs_min = resid_fabs
		if rms_min > resid_rms: rms_min = resid_rms
		#print(f'INFO: residual maximum {resid_max} threshold {threshold}')
		if not quiet and (i%1000==0): 
			print('\n'.join((	f'INFO: Iteration {i}/{max_iter}',
								f'----: residual rms {resid_rms:07.2E} threshold {rms_threshold:07.2E}',
								f'----: fabs {resid_fabs:07.2E} threshold {fabs_threshold:07.2E}',
								f'----: residual box {r_o_min} {r_o_max} psf box {p_o_min} {p_o_max}',
								f'----: mval {mval:07.2E} (mx, my) [{my} {mx}]'
								)))

		rms_record[i] = resid_rms
		fabs_record[i] = resid_fabs
	
		if np.fabs(component_sum_record[i]) > component_sum_limit:
			# remove what we just added
			components[my,mx] -= mval*flux_correction_factor 
			residual[r_o_min[0]:r_o_max[0],r_o_min[1]:r_o_max[1]] += mval*psf_img[p_o_min[0]:p_o_max[0],p_o_min[1]:p_o_max[1]]
			print(f'WARNING: Component sum exceeded dirty img sum, terminating...')
			break 

		
		if (resid_rms > 2*rms_min):
			# remove what we just added
			components[my,mx] -= mval*flux_correction_factor 
			residual[r_o_min[0]:r_o_max[0],r_o_min[1]:r_o_max[1]] += mval*psf_img[p_o_min[0]:p_o_max[0],p_o_min[1]:p_o_max[1]]
			print(f'WARNING: rms exceeded initial value, terminating...')
			break

		"""
		if (resid_fabs > 2*fabs_min):
			# remove what we just added
			components[my,mx] -= mval*flux_correction_factor 
			residual[r_o_min[0]:r_o_max[0],r_o_min[1]:r_o_max[1]] += mval*psf_img[p_o_min[0]:p_o_max[0],p_o_min[1]:p_o_max[1]]
			print(f'WARNING: fabs exceeded initial value, terminating...')
			break
		"""

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
	return(residual, components, rms_record, fabs_record, i, component_sum_record)

def clean_modified(dirty_img, psf_img, window=True, loop_gain=1E-1, threshold=0.6, rms_frac_threshold=0.01, 
					fabs_frac_threshold=0.01, 
					max_iter=int(1E2),
					norm_psf=False,
					show_plots=True,
					quiet=False,
					n_positive_iter=0,
					restart_tuple=None,
					jitter_tolerance=1.2,
					flux_conservation_tolerance=1.2):
	"""
	A modified version of the CLEAN algorithm that attempts to reduce artifacts when CLEANing non-point sources. See http://adsabs.harvard.edu/full/1984A%26A...137..159S
	"""

	if not quiet: print('INFO: In "clean_modified()"')
	
	# normalise the psf if desired and ensure it's centered in it's field
	if norm_psf:
		psf_img /= np.nansum(psf_img)
	psf_img = ensure_centered_psf(psf_img)
	
	# if the integral of the PSF != 1 then when we are adding components
	# we need to ensure that the integrated flux is correct
	flux_correction_factor = np.nansum(psf_img)
	print(f'DEBUG: flux_correction_factor {flux_correction_factor}')
	
	# if the maximum of the PSF != 1 then we need to adjust the amount
	# of flux minused from the residual to keep the desired loop gain
	loop_gain_correction_factor = 1.0/np.nanmax(psf_img)
	print(f'DEBUG: loop_gain_correction_factor {loop_gain_correction_factor}')
	
	# create default window if none passed
	if (window is True) or (window is None):
		window = np.ones(dirty_img.shape, dtype=bool)
	
	if show_plots:
		# DEBUGGING PLOT SETUP
		plt.ion()
		nr,nc = (3, 4)
		s = 4
		f1 = plt.figure(1,figsize=(s*nc,s*nr))
		plt.show()
	
	
	if restart_tuple is not None:
		residual, components, rms_record_old, fabs_record_old, starting_iteration, accumulator, window = restart_tuple
		fabs_record = np.full((max_iter,), fill_value=np.nan)
		fabs_record[:fabs_record_old.size] = fabs_record_old
		rms_record = np.full((max_iter,), fill_value=np.nan)
		rms_record[:rms_record_old.size] = rms_record_old
	else:
		residual = np.array(dirty_img)
		components = np.zeros_like(dirty_img)
		rms_record = np.full((max_iter+1,), fill_value=np.nan)
		fabs_record = np.full((max_iter+1,), fill_value=np.nan)
		accumulator = np.zeros_like(dirty_img, dtype=int)
		starting_iteration = 0

	# only use residual_windowed for choosing points, we want to make sure our
	# measures make sense even when we window out large portions of the image
	residual_windowed = np.array(residual)
	residual_windowed[~window] = np.nan
		
	selected = np.zeros_like(dirty_img)
	above_t = np.zeros_like(dirty_img, dtype=bool)
	current_cleaned = np.zeros_like(dirty_img)
	
	fabs_threshold = np.nanmax(np.fabs(residual))*fabs_frac_threshold
	rms_threshold = np.sqrt(np.nansum((residual)**2)/residual.size)*rms_frac_threshold
	
	fabs_record[0] = np.nanmax(np.fabs(residual))
	rms_record[0] = np.sqrt(np.nansum(residual**2)/residual.size)

	min_fabs = (1/jitter_tolerance)*fabs_record[0]
	min_rms = (1/jitter_tolerance)*rms_record[0]

	component_sum_record = np.full((max_iter+1), fill_value=np.nan)
	component_sum_record[0] = 0
	component_sum_limit = flux_conservation_tolerance*np.fabs(np.nansum(residual))

	# perform modified CLEAN algorithm
	i=0 # fixes error if max_iter=0
	for i in range(starting_iteration, max_iter):
		if not quiet: print(f'INFO: Iteration {i}/{max_iter}')
		# reset the selected pixels to zero every step
		selected *= 0

		# only consider +ve pixels if this is a +ve step
		if i < n_positive_iter:
			fabs_residual = residual_windowed # does this ensure +ve pixels are chosen?
		else:
			fabs_residual = np.fabs(residual_windowed)

		# find all pixels that are above our threshold (whether signed or maginitude)
		max_fabs_residual = np.nanmax(fabs_residual)
		mod_clean_threshold = threshold*max_fabs_residual
		above_t[:,:] = fabs_residual > mod_clean_threshold
		n_selected = np.sum(above_t)

		# select the pixels that are above the threshold, only try to explain a "loop_gain" fraction
		# of their flux
		selected[above_t] = residual[above_t]*loop_gain#*loop_gain_correction_factor
		accumulator[above_t] += 1
	
		print(f'DEBUG: {max_fabs_residual:07.2E} {np.nanmax(residual[above_t]):07.2E} {np.nanmax(selected):07.2E}')
	
		# convolve selected pixels with PSF and adjust so that flux is conserved	
		convolved = sp.signal.fftconvolve(selected, psf_img, mode='same')/flux_correction_factor

		# update residual
		residual -= convolved
		residual_windowed -= convolved
	
		# update measures that depend on the residual
		resid_fabs = np.nanmax(np.fabs(residual))
		resid_rms = np.sqrt(np.nansum((residual)**2)/residual.size)
		if min_fabs > resid_fabs: min_fabs = resid_fabs
		if min_rms > resid_rms: min_rms = resid_rms

		# record rms and fabs
		rms_record[i+1] = resid_rms
		fabs_record[i+1] = resid_fabs

		# add the selected fluxes to the component list	
		components += selected
		component_sum_record[i+1] = np.fabs(np.nansum(components))

		# add the convolved data to the "current_cleaned" map
		current_cleaned += convolved
		#residual = dirty_img_cpy - current_cleaned # only use if just updating residual breaks
	

		if not quiet: 
			print('\n'.join((	
				f'INFO: Number of selected points {n_selected}',
				f'----: max_fabs_residual {max_fabs_residual:07.2E}',
				f'----: mod_clean_threshold {mod_clean_threshold:07.2E}',
				f'----: residual rms {resid_rms:07.2E} threshold {rms_threshold:07.2E} limit {2*min_rms:07.2E}',
				f'----: residual fabs {resid_fabs:07.2E} threshold {fabs_threshold:07.2E} limit {2*min_fabs:07.2E}',
				f'----: |component_sum| {component_sum_record[i+1]:07.2E} limit {component_sum_limit:07.2E}',
				f'----: max_convolved {np.nanmax(np.fabs(convolved)):07.2E}',
				f'----: max_residual {np.nanmax(np.fabs(residual)):07.2E}'
			)))
		
		# if the current iteration would screw up the fabs or rms, don't do the step and break out
		if (resid_rms > jitter_tolerance*min_rms):
			residual += convolved # add back in the bit we would remove
			components -= selected
			print(f'WARNING: rms larger than initial or {jitter_tolerance}*minimum level, terminating...')
			break
		"""
		if (resid_fabs > jitter_tolerance*min_fabs):
			residual += convolved # add back in the bit we would remove
			components -= selected
			print(f'WARNING: fabs larger than initial or {jitter_tolerance}*minimum level, terminating...')
			break
		"""
		# check if the sum of the componets is larger than our defined limit
		if component_sum_record[i+1] > component_sum_limit:
			# remove the stuff we just added
			residual += convolved
			components -= selected
			print('WARNING: |Sum of components| > |sum of image|, terminating...')
			break

		if (resid_rms < rms_threshold)  or (resid_fabs < fabs_threshold):
			break

		if show_plots:
			clean_modified_progress_plots(	f1, nr, nc, dirty_img, residual, components, 
											current_cleaned, above_t, accumulator, convolved, 
											rms_record, fabs_record, rms_threshold, fabs_threshold, 
											max_iter)
			f1.suptitle(f'Loop iteration {i}')
			plt.draw()
			plt.waitforbuttonpress(0.001)
			
	if show_plots:
		plt.close()
		plt.ioff()	

	return(residual, components, rms_record, fabs_record, i, component_sum_record, accumulator, window)


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
					quiet=True,
					n_positive_iter=0,
					sigmas=(6,3,1.5),
					calculation_mode='direct',
					base_clean_algorithm='hogbom',
					reject_decomposition_idxs=[3],
					plot_dir=None,
					plot_suffix=''):
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

	# Define filter function here
	# sp.ndimage.gaussian_filter is a 2d gaussian with integral 1
	# TODO: Work out how I should decide on the factor that multiplies the gaussian
	filter_function = lambda *args, **kwargs: 0.9*sp.ndimage.gaussian_filter(*args, **kwargs)
	"""
	def filter_function(*args, **kwargs): 
		a = sp.ndimage.gaussian_filter(*args, **kwargs)
		return(a - (1.0/n)*np.nanmean(a))
	"""

	# normalise PSF, don't usually need to do this as "clean_hogbom" and "clean_modified" should
	# conserve flux wheather PSF is normalised or not
	#print('psf sum', np.nansum(psf_img))
	#print(type(norm_psf),norm_psf)
	if ((type(norm_psf) is str) and (norm_psf == 'max')) or ((type(norm_psf) is bool) and (norm_psf is True)):
		psf_img /= np.nanmax(psf_img)
	elif (type(norm_psf) is str) and (norm_psf =='sum'):
		psf_img /= np.nansum(psf_img)
	#print('psf sum after norm', np.nansum(psf_img))

	psf_img = pad_psf_to_minimum_size(		psf_img, 
											dirty_img.shape, 
											lambda x: filter_function(x, np.nanmax(sigmas)), 
											np.nanmax(sigmas),
											show_plots=False
										)
	
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
	if not quiet: print('H_0 = G_0')
	i=0 # define here in case we only have 1 decomposition
	for i in range(1,n-1):
		if not quiet: print(f'H_{i} = \left( \delta - \sum_(i=0)^({i-1}) H_i \\right) * G_{i}')
		h[i] = filter_function(dirac_delta - np.nansum(h[:i], axis=0), sigmas[i])
	if not quiet: print(f'H_{i+1} = \delta - \sum_(i=0)^({i}) H_i')
	h[-1] = dirac_delta - np.nansum(h[:-1], axis=0)

	# calculate the decomposed images S_i, P_i, and f_i if applicable
	if calculation_mode is 'direct':
		s[0] = filter_function(dirty_img2, sigmas[0])
		p[0] = filter_function(psf_img, sigmas[0])
		f[0] = 1.0/np.nansum(h[0])

		for i in range(1,n-1):
			s[i] = filter_function(dirty_img2 - np.nansum(s[:i], axis=0), sigmas[i])
			p[i] = filter_function(psf_img - np.nansum(p[:i], axis=0), sigmas[i])
			f[i] = 1.0/np.nansum(h[i])

		s[-1] = dirty_img2 - np.nansum(s[:-1], axis=0)
		p[-1] = psf_img - np.nansum(p[:-1], axis=0)
		f[-1] = 1.0/np.nansum(h[-1])

	elif calculation_mode is 'convolution':
		for i in range(n):
			s[i] = sp.signal.convolve2d(dirty_img2, h[i], mode='same')
			p[i] = sp.signal.convolve2d(psf_img, h[i], mode='same')
			f[i] = 1.0/np.nansum(h[i])
	elif calculation_mode is 'convolution_fft':
		for i in range(n):
			s[i] = sp.signal.fftconvolve(dirty_img2, h[i], mode='same')
			p[i] = sp.signal.fftconvolve(psf_img, h[i], mode='same')
			f[i] = 1.0/np.nansum(h[i])
	else:	
		print(f'ERROR: Argument "calculation_mode" is {calculation_mode}, options are ("direct", "convolution").')
		raise NotImplementedError

	# run the CLEAN algorithm for each decomposed image S_i using the psf P_i
	result_parts = []
	for i in range(n):
		if not quiet: print(f'INFO: CLEANing decomposed image {i}')
		if i in reject_decomposition_idxs: # if we don't want to include some of the decomposed images, skip them.
			if not quiet: print(f'INFO: Skipping rejected image')
			continue
		s[i][nan_idxs] = np.nan # remember to put back NANs that we removed earlier
	
		if base_clean_algorithm is 'hogbom':
			result_i =  clean_hogbom(s[i], p[i], window=window, 
						loop_gain=loop_gain[i], rms_frac_threshold=rms_frac_threshold[i], 
						fabs_frac_threshold=fabs_frac_threshold[i], max_iter=max_iter[i], 
						norm_psf=False, show_plots=False, quiet=False, ensure_psf_centered=False,
						n_positive_iter=n_positive_iter[i])
		elif base_clean_algorithm is 'modified':
			result_i =  clean_modified(s[i], p[i], window=window, threshold=threshold[i],
						loop_gain=loop_gain[i], rms_frac_threshold=rms_frac_threshold[i], 
						fabs_frac_threshold=fabs_frac_threshold[i], max_iter=max_iter[i], 
						norm_psf=False, show_plots=False, quiet=False, 
						n_positive_iter=n_positive_iter[i])
		else:
			print(f'ERROR: Argument "base_clean_algorithm" is {base_clean_algorithm}, options are ("hogbom", "modified")')
			raise NotImplementedError
		if not quiet: print(f'INFO: Took {result_i[4]} iterations')
		result_parts.append(result_i)

	#print('f_i values',f)

	# create holders for return data
	residual = np.zeros_like(result_parts[0][0])
	components = np.zeros_like(result_parts[0][1])
	rms_record = []
	fabs_record = []
	n_iters = []
	component_sum_record = []

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
	if not quiet: print(f'INFO: Combining decomposed images')
	for i, result_i in enumerate(result_parts):
		if i in reject_decomposition_idxs: # if we don't want to include some of the decomposed images, skip them.
			if not quiet: print(f'INFO: rejecting image {i}')
			continue
		if not quiet: print(f'INFO: Convolving components_{i} with H_{i}, multiplying by factor {f[i]:07.2E}')
		residual += result_i[0]
		# multiplying by f[i] conserves flux in component_i, may need to use "mode='full'" to see
		# normalisation is correct
		component_i = f[i]*sp.signal.fftconvolve(result_i[1],h[i],mode='same')
		components += component_i
		rms_record.append(result_i[2])
		fabs_record.append(result_i[3])
		n_iters.append(result_i[4])
		component_sum_record.append(result_i[5])

		# display data in figure if desired
		if show_plots:
			s_limits = {'vmin':np.nanmin(s[i]), 'vmax':np.nanmax(s[i])}
			a2[i,0].imshow(s[i], origin='lower', **s_limits)
			a2[i,0].set_title(f'S[{i}] {s[i].shape}\nsum {np.nansum(s[i]):07.2E}')
			remove_axes(a2[i,0])
			
			a2[i,1].imshow(p[i], origin='lower')
			a2[i,1].set_title(f'P*H[{i}] {p[i].shape} {f[i]:07.2E}\nsum {np.nansum(p[i]):07.2E} argmax {np.unravel_index(np.nanargmax(p[i]), p[i].shape)}')
			#a2[i,1].imshow(result_i[1], origin='lower')
			#a2[i,1].set_title(f'Component without convolution with H{i}\nsum {np.nansum(result_i[1]):07.2E}')
			remove_axes(a2[i,1])

			a2[i,2].imshow(component_i, origin='lower', **s_limits)
			a2[i,2].set_title(f'component [{i}]\nsum {np.nansum(component_i):07.2E}')
			# below we use the complete convolution to check flux conservation
			#component_full_i = sp.signal.fftconvolve(result_i[1], h[i], mode='full')*f[i]
			#component_fft = sp.signal.fftconvolve(component_full_i, psf_img, mode='full')/np.nansum(psf_img)
			#a2[i,2].imshow(component_fft, origin='lower', **s_limits)
			#a2[i,2].set_title(f'component [{i}]\nsum {np.nansum(component_fft):07.2E} {np.nansum(component_i):07.2E}')
			remove_axes(a2[i,2])

			a2[i,3].imshow(result_i[0], origin='lower', **s_limits)
			a2[i,3].set_title(f'residual [{i}]\nsum {np.nansum(result_i[0]):07.2E}')
			# below we use the complete convolvution to check flux conservation
			#s_centered = np.zeros_like(component_fft)
			#print('DEBUG: s_centered.shape',s_centered.shape, 'psf_img.shape', psf_img.shape, 's[i].shape', s[i].shape, 'component_i.shape', component_i.shape)
			#x_shape = np.array(psf_img.shape) +np.array(h[i].shape) - np.array([2,2])
			#s_centered[x_shape[0]//2:-(x_shape[0]//2), x_shape[1]//2:-(x_shape[1]//2)] = s[i]
			#residual_fft = s_centered - component_fft
			#a2[i,3].imshow(residual_fft, origin='lower', **s_limits)
			#a2[i,3].set_title(f'residual [{i}]\nsum {np.nansum(residual_fft):07.2E}')
			remove_axes(a2[i,3])
			
			"""
			a2[i,4].imshow(h[i], origin='lower')
			a2[i,4].set_title(f'h[{i}]\nsum {np.nansum(h[i]):07.2E}')
			remove_axes(a2[i,4])
			"""
			a2[i,4].plot(rms_record[i], label='rms', color='tab:blue')
			a2[i,4].axhline(rms_frac_threshold[i]*rms_record[i][0], color='tab:blue', ls='--')
			a2[i,4].axhline(min(2*np.nanmin(rms_record[i]), rms_record[i][0]), color='tab:blue', ls='--', alpha=0.5)
			a2[i,4].set_yscale('log')
			a2_i4_2 = a2[i,4].twinx()
			a2_i4_2.plot(fabs_record[i], label='fabs', color='tab:orange')
			a2_i4_2.axhline(fabs_frac_threshold[i]*fabs_record[i][0], color='tab:orange', ls=':')
			a2_i4_2.axhline(min(2*np.nanmin(fabs_record[i]), fabs_record[i][0]), color='tab:orange', ls=':', alpha=0.5)
			a2_i4_2.set_yscale('log')
			a2[i,4].set_title(f'n_iter {n_iters[i]}')
			combine_handles_and_labels = lambda ax_list: [sum(x, []) for x in zip(*[a.get_legend_handles_labels() for a in ax_list])]
			a2[i,4].legend(*combine_handles_and_labels([a2[i,4],a2_i4_2]), loc='upper right')
			
			a2[i,5].plot(component_sum_record[i], color='tab:blue', label='component sum')
			a2[i,5].axhline(np.nansum(s[i]), color='red', ls='--')
			a2[i,5].set_title('component sum')
			#a2[i,5].set_yscale('log')
			
			
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
			
			
	if show_plots:
		if not (plot_dir is None):
			print(f'INFO: Saving plots to "{plot_dir}"')
			f2.savefig(os.path.join(plot_dir, f'clean_decomposition{plot_suffix}.png'))
			f3.savefig(os.path.join(plot_dir, f'clean_scale_density{plot_suffix}.png'))
		else:
			print('INFO: Showing plots')
			plt.show()
	if not quiet: print(f'INFO: Returning CLEANed data')
	return(residual, components, list(zip(*rms_record)), list(zip(*fabs_record)), n_iters)	
	

#-----------------------------------------------------------------------------
# progress plotting routines for CLEAN algorithms
#-----------------------------------------------------------------------------

def clean_modified_progress_plots(f1, nr, nc, dirty_img, residual, components, current_cleaned, above_t, accumulator, convolved, rms_record, fabs_record, rms_threshold, fabs_threshold, max_iter):
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

	
	a1[0,0].set_title(f'dirty_img\nmax {np.nanmax(np.fabs(dirty_img)):07.2E} sum {np.nansum(dirty_img):07.2E}')
	a1[0,0].imshow(dirty_img, origin='lower', cmap=cmap, 
					vmin=-np.nanmax(np.fabs(dirty_img)), vmax=np.nanmax(np.fabs(dirty_img)))

	a1[0,1].set_title(f'residual\nmax {np.nanmax(np.fabs(residual)):07.2E} sum {np.nansum(residual):07.2E}')
	a1[0,1].imshow(residual, origin='lower', cmap=cmap, 
					vmin=-np.nanmax(np.fabs(residual)), vmax=np.nanmax(np.fabs(residual)))

	a1[0,2].set_title(f'components\nsum {np.nansum(components):07.2E}')
	a1[0,2].imshow(components, origin='lower', cmap=cmap, 
					vmin=-np.nanmax(np.fabs(components)), vmax=np.nanmax(np.fabs(components)))
	# put NANs back in
	current_cleaned[np.isnan(dirty_img)] = np.nan
	
	a1[0,3].set_title(f'current CLEANed image\nsum {np.nansum(current_cleaned):07.2E}')
	a1[0,3].imshow(current_cleaned, origin='lower', cmap=cmap, 
					vmin=-np.nanmax(np.fabs(current_cleaned)), vmax=np.nanmax(np.fabs(current_cleaned)))
	
	
	a1[1,0].set_title(f'above_t\nsum {np.nansum(above_t)}')
	a1[1,0].imshow(above_t, origin='lower')
	
	a1[1,1].set_title(f'accumulator\nsum {np.nansum(accumulator):07.2E}')
	a1[1,1].imshow(accumulator, origin='lower')#, cmap=cmap, vmin=-np.nanmax(np.fabs(selected)), vmax=np.nanmax(np.fabs(selected)))
	# put back in the nans
	#convolved[np.isnan(dirty_img)] = np.nan

	a1[1,2].set_title(f'convolved\nsum {np.nansum(convolved):07.2E}')
	a1[1,2].imshow(convolved, origin='lower', cmap=cmap, vmin=-np.nanmax(np.fabs(convolved)), vmax=np.nanmax(np.fabs(convolved)))
	
	#if axis is None:
	#a1[1,3].clear()
	a1[1,3].set_title('rms and |brightest pixel|')
	a1[1,3].plot(rms_record, color='tab:blue', label='rms')
	a1[1,3].set_ylabel('rms')
	a1[1,3].set_xlim(0, max_iter)
	a1[1,3].set_ylim([0.9*rms_threshold, 1.1*np.nanmax(rms_record)])
	a1[1,3].axhline(rms_threshold, color='tab:blue', linestyle='-')
	a1[1,3].set_yscale('log')
	a1132 = a1[1,3].twinx()
	a1132.plot(fabs_record, color='tab:orange', label='abp')
	a1132.set_ylabel('abp')
	a1132.set_ylim([0.9*fabs_threshold, 1.1*np.nanmax(fabs_record)])
	a1132.axhline(fabs_threshold, color='tab:orange', linestyle='--')
	a1132.set_yscale('log')
	h1,l1 = a1[1,3].get_legend_handles_labels()
	h2,l2 = a1132.get_legend_handles_labels()
	#print(h1+h2)
	#print(l1+l2)
	a1[1,3].legend(h1+h2,l1+l2, loc='upper right')

	#if axis is None:
	#a1[2,0].clear()
	a1[2,0].set_title('dirty_img')
	a1[2,0].hist(dirty_img.ravel(), bins=100)
	a1[2,0].set_yscale('log')

	#if axis is None:
	#a1[2,1].clear()
	a1[2,1].set_title('residual')
	a1[2,1].hist(residual.ravel(), bins=100)
	a1[2,1].set_yscale('log')
	
	#if axis is None:
	#a1[2,2].clear()
	a1[2,2].set_title('components')
	a1[2,2].hist(components.ravel(), bins=100)
	a1[2,2].set_yscale('log')

	#if axis is None:
	#a1[2,3].clear()
	a1[2,3].set_title('accumulator')
	a1[2,3].hist(accumulator.ravel(), bins=100)
	a1[2,3].set_yscale('log')

	return()

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

def decompose_image(img, sigmas, calculation_mode='direct', filter_function=sp.ndimage.gaussian_filter, verbose=0, h_shape=None, h=None):
	"""
	Uses the same scale decomposition algorithm as clean_multiresolution
	"""
	# define constants
	n = len(sigmas)+1
	
	# create data holders
	s = np.zeros((n, img.shape[0], img.shape[1]))
	f = np.ones((n,))
	
	# calculate the decomposition filters H_i,
	if h is None:
		if h_shape is None:
			h_shape = (img.shape[0], img.shape[1])
		h = np.zeros((n, h_shape[0], h_shape[1]))
		# define dirac delta
		dirac_delta = dirac_delta_filter(h[0])
		h[0] = filter_function(dirac_delta, sigmas[0])
		if verbose: print('INFO: H_0 = G_0')
		i=0 # define here in case we only have 1 decomposition
		for i in range(1,n-1):
			if verbose: print(f'INFO: H_{i} = \left( \delta - \sum_(i=0)^({i-1}) H_i \\right) * G_{i}')
			h[i] = filter_function(dirac_delta - np.nansum(h[:i], axis=0), sigmas[i])
		if verbose: print(f'INFO: H_{i+1} = \delta - \sum_(i=0)^({i}) H_i')
		h[-1] = dirac_delta - np.nansum(h[:-1], axis=0)

	# calculate the decomposed images S_i, P_i, and f_i if applicable
	if calculation_mode is 'direct':
		s[0] = filter_function(img, sigmas[0])
		f[0] = 1.0/np.nanmax(s[0])

		for i in range(1,n-1):
			s[i] = filter_function(img - np.nansum(s[:i], axis=0), sigmas[i])
			f[i] = 1.0/np.nanmax(s[i])

		s[-1] = img - np.nansum(s[:-1], axis=0)
		f[-1] = 1.0/np.nanmax(s[-1])

	elif calculation_mode is 'convolution':
		for i in range(n):
			s[i] = sp.signal.convolve2d(img, h[i], mode='same')
			f[i] = 1.0/np.nanmax(s[i])
	elif calculation_mode is 'convolution_fft':
		for i in range(n):
			s[i] = sp.signal.fftconvolve(img, h[i], mode='same')
			f[i] = 1.0/np.nanmax(s[i])
	else:	
		print(f'ERROR: Argument "calculation_mode" is {calculation_mode}, options are ("direct", "convolution").')
		raise NotImplementedError
	return(s, h, f)

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

def pad_psf_to_minimum_size(psf_img, dirty_img_shape, max_size_filter, pad_step, show_plots=False):
	while True:
		psf_test = max_size_filter(psf_img)
		edges = np.concatenate([psf_test[0,:], psf_test[-1,:], psf_test[:,0], psf_test[:,-1]])
		psf_test_larger_than_2x_dirty_img_shape = all([s >= 2*ds for s, ds in zip(psf_test.shape, dirty_img_shape)])
		if (edges == 0).all() or psf_test_larger_than_2x_dirty_img_shape:
			break
		psf_img = psf_resize_ensure_centered(psf_img, int(pad_step))
		#print(f'INFO: psf_img.shape {psf_img.shape}')
		if show_plots:
			plt.imshow(psf_img, origin='lower')
			plt.scatter(*[s//2 for s in psf_img.shape], color='red')
			plt.show()
	return(psf_img)

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

def psf_resize_ensure_centered(psf_img, n_padding):
	return(ensure_centered_psf(np.pad(psf_img, n_padding)))

def rect_overlap_lims(sa, sb, x):
	zeros = np.zeros_like(sa)
	return(np.clip(x, zeros, sa), np.clip(sb+x, zeros, sa), np.clip(-x, zeros, sb), np.clip(sa-x, zeros, sb))

def ensure_centered_psf(psf_img, fill_value=0, show_plots=False):
	""" pad array to odd number of elements on each axis """
	padding = [(0,1-s%2) for s in psf_img.shape]
	#print(psf_img.shape)
	#print(padding)
	psf_adj = np.pad(psf_img, padding, mode='constant', constant_values=fill_value)
	#print(f'INFO: number of NANs {np.sum(np.isnan(psf_adj))} out of {np.prod(psf_adj.shape)}')
	cidx = np.array(psf_adj.shape)//2
	max_idx = np.unravel_index(np.nanargmax(psf_adj), psf_adj.shape)

	for i, (ci, mi) in enumerate(zip(cidx, max_idx)):
		np.roll(psf_adj, ci-mi, axis=i) 
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
	target_cube = os.path.expanduser('~/scratch/reduced_images/SINFO.2018-09-19T02:21:05/MOV_Neptune---H+K_0.025_1_tpl/analysis/obj_NEPTUNE_cal_HD2811_renormed.fits')
	#target_cube = os.path.expanduser('~/scratch/reduced_images/SINFO.2018-08-31T05:45:50/MOV_Neptune---H+K_0.1_tpl/analysis/obj_NEPTUNE_cal_HIP001115_renormed.fits')
	
	# set up parameters for CLEAN algorithms
	wav_range = ((1.5,1.6),
				  (1.7,1.8),
				  (1.9,2.0),
				  )[0]

	hogbom_loop_gain = 0.2
	hogbom_rms_frac_threshold = 1E-2
	hogbom_fabs_frac_threshold = 1E-2
	hogbom_max_iter = 1#100000 # 1000000
	modified_loop_gain=0.2
	modified_threshold=0.6
	modified_rms_frac_threshold=1E-2
	modified_fabs_frac_threshold=1E-2
	modified_max_iter = 2000 # 200

	# wrangle *.fits file into numpy array
	cube_img, psf_img, cube_nans, disk_mask = get_wavrange_img_from_fits(target_cube, wav_range, function=np.nanmedian)
	psf_img = ensure_centered_psf(psf_img)

	# copy data array so we can modify it before CLEANing if we want to
	cube_img_2 = np.array(cube_img)
	cube_img_2[cube_nans] = np.nan
	cube_img_2[cube_img_2<0] = np.nan
	
	
	# Perform hogbom CLEAN algorithm
	(	residual_1, 
		components_1, 
		rms_record_1, 
		fabs_record_1, 
		n_iter_1
	) = clean_hogbom(	cube_img_2, 
						psf_img, 
						window=(disk_mask | cube_nans), 
						loop_gain=hogbom_loop_gain, 
						rms_frac_threshold=hogbom_rms_frac_threshold, 
						fabs_frac_threshold=hogbom_fabs_frac_threshold,
						max_iter=hogbom_max_iter,
						norm_psf=False, 
						show_plots=False
						)

	plot_clean_result(cube_img_2, components_1, residual_1, psf_img, rms_record_1, fabs_record_1, np.zeros_like(cube_img_2))
