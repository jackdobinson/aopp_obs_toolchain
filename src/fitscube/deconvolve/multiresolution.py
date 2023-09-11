#!/usr/bin/env python3

import os
import numpy as np
import scipy as sp
import fitscube.deconvolve.clean
import fitscube.deconvolve.lucy_richardson
import matplotlib.pyplot as plt



def multiresolution(dirty_img, psf_img, base_algorithm, base_algorithm_kwargs,
					sigmas=(6,3,1.5),
					reject_decomposition_idxs=[],
					filter_function = lambda *args, **kwargs: 0.9*sp.ndimage.gaussian_filter(*args, **kwargs),
					calculation_mode='convolution_fft',
					verbose=0,
					show_plots=True,
					plot_dir=None,
					plot_suffix=''):
	"""
	Runs a deconvolution algorithm at multiple resolutions, see 
	<http://articles.adsabs.harvard.edu/cgi-bin/nph-iarticle_query?bibcode=1988A%26A...200..312W&db_key=AST&page_ind=3&plate_select=NO&data_type=GIF&type=SCREEN_GIF&classic=YES>
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
		show_plots
			<bool> If True will show plots about the result.
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
		base_algorithm
			<function> The function used to deconvolve each of the decomposed images.
			Must conform to the prototype
				def base_algorithm(dirty_img, psf, **base_algorithm_kwargs)
		base_algorithm_kwargs
			<dict> or <list,<dict>> A dictionary (or list of dictionaries) of keyword 
			arguments to pass to the base_algorithm. If the argument is a list of
			dictionaries, will use the i^th set of kwargs for the deconvolution of the
			 i^th decomposed image.
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
			The sum of the residuals of each decomposed image [see equation (15)
		components [nx,ny]
			The sum of the components of each decomposed image convolved
			by each H_i. See equation (14)
		statistics_record [n]
			The statistics of each decomposed image at each iteration. The
			exact statistics change between each algorithm. See that algorithm's
			description.
		n_iters[n]
			The number of iterations of the base_algorithm for each component
			image.
	
	# EXAMPLE #
		```
		(	residual_1, 
			components_1, 
			stat_record_1, 
			n_iter_1
		) = multiresolution(cube_img_2, 
							psf_img, 
							base_algorithm          = fitscube.deconvolve.clean.clean_hogbom,
							base_algorithm_kwargs   = dict(	window              = (disk_mask | cube_nans), 
															loop_gain           = hogbom_loop_gain, 
															rms_frac_threshold  = hogbom_rms_frac_threshold, 
															fabs_frac_threshold = hogbom_fabs_frac_threshold,
															max_iter            = hogbom_max_iter,
															norm_psf            = True
															),
							show_plots              = True
							)
		```	
	"""
	# make sure that we have the correct form and length for our arguments
	sigmas = ensure_subscriptable(sigmas)
	n = len(sigmas) + 1
	n_sigma = len(sigmas)

	# if we have been passed a dict for base_algorithm_kwargs, then wrap it in a tuple of length n
	# otherwise, assume that it's in the correct format
	base_algorithm_kwargs = ensure_subscriptable(base_algorithm_kwargs, min_size=n, cast=tuple)


	# no point normalising PSF as each base algorithm can do it for itself

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

	# calculate the decomposition filters H_i, do it here so we can reuse them.
	h[0] = filter_function(dirac_delta, sigmas[0])
	if verbose>0: print('H_0 = G_0')
	i=0 # define here in case we only have 1 decomposition
	for i in range(1,n-1):
		if verbose>0: print(f'H_{i} = \left( \delta - \sum_(i=0)^({i-1}) H_i \\right) * G_{i}')
		h[i] = filter_function(dirac_delta - np.nansum(h[:i], axis=0), sigmas[i])
	if verbose>0: print(f'H_{i+1} = \delta - \sum_(i=0)^({i}) H_i')
	h[-1] = dirac_delta - np.nansum(h[:-1], axis=0)

	s, h, f =  decompose_image(	dirty_img2, sigmas, 
								calculation_mode=calculation_mode, 
								filter_function=filter_function, 
								verbose=verbose-1, 
								h_shape=h[0].shape, 
								h=h
								)
	p, h, f =  decompose_image(	psf_img, sigmas, 
								calculation_mode=calculation_mode, 
								filter_function=filter_function, 
								verbose=verbose-1, 
								h_shape=h[0].shape, 
								h=h
								)
	
	# run the deconvolution algorithm for each decomposed image S_i using the psf P_i
	result_parts = []
	if verbose>0: print(f'INFO: Applying deconvolution algorithm to decomposed images')
	for i in range(n):
		if verbose>1: print(f'INFO: CLEANing decomposed image {i}')
		if i in reject_decomposition_idxs: # if we don't want to include some of the decomposed images, skip them.
			if verbose>1: print(f'INFO: Skipping rejected image')
			continue
		s[i][nan_idxs] = np.nan # remember to put back NANs that we removed earlier
	
		result_i =  base_algorithm(s[i], p[i], **base_algorithm_kwargs[i])
		if show_plots: plt.close('all') # close any images created when running the base_algorithm
		
		if verbose>1: print(f'INFO: Took {result_i[4]} iterations')
		
		result_parts.append(result_i)

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
	if verbose>0: print(f'INFO: Combining decomposed images')
	for i, result_i in enumerate(result_parts):
		if i in reject_decomposition_idxs: # if we don't want to include some of the decomposed images, skip them.
			if verbose>1: print(f'INFO: rejecting image {i}')
			continue
		if verbose>1: print(f'INFO: Convolving components_{i} with H_{i}, multiplying by factor {f[i]:07.2E}')
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
			#a2[i,4].axhline(rms_frac_threshold[i]*rms_record[i][0], color='tab:blue', ls='--')
			#a2[i,4].axhline(min(2*np.nanmin(rms_record[i]), rms_record[i][0]), color='tab:blue', ls='--', alpha=0.5)
			a2[i,4].set_yscale('log')
			a2_i4_2 = a2[i,4].twinx()
			a2_i4_2.plot(fabs_record[i], label='fabs', color='tab:orange')
			#a2_i4_2.axhline(fabs_frac_threshold[i]*fabs_record[i][0], color='tab:orange', ls=':')
			#a2_i4_2.axhline(min(2*np.nanmin(fabs_record[i]), fabs_record[i][0]), color='tab:orange', ls=':', alpha=0.5)
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
		if plot_dir is not None:
			if verbose>0:print(f'INFO: Saving plots to "{plot_dir}"')
			f2.savefig(os.path.join(plot_dir, f'clean_decomposition{plot_suffix}.png'))
			f3.savefig(os.path.join(plot_dir, f'clean_scale_density{plot_suffix}.png'))
		else:
			if verbose>0:print('INFO: Showing plots')
			plt.show()
	if verbose>0: print(f'INFO: Returning CLEANed data')
	return(residual, components, list(zip(*rms_record)), list(zip(*fabs_record)), n_iters)	
	


def decompose_image(img, sigmas, 
					calculation_mode='direct', 
					filter_function=sp.ndimage.gaussian_filter, 
					verbose=0, 
					h_shape=None, 
					h=None):
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
		# make H filters the same size as the image if it hasn't been specified
		if h_shape is None:
			h_shape = (img.shape[0], img.shape[1])
		print('HERE')
		h = np.zeros((n, h_shape[0], h_shape[1]))
		# define dirac delta
		dirac_delta = dirac_delta_filter(h[0])
		h[0] = filter_function(dirac_delta, sigmas[0])
		if verbose>0: print('INFO: H_0 = G_0')
		i=0 # define here in case we only have 1 decomposition
		for i in range(1,n-1):
			if verbose>0: print(f'INFO: H_{i} = \left( \delta - \sum_(i=0)^({i-1}) H_i \\right) * G_{i}')
			h[i] = filter_function(dirac_delta - np.nansum(h[:i], axis=0), sigmas[i])
		if verbose>0: print(f'INFO: H_{i+1} = \delta - \sum_(i=0)^({i}) H_i')
		h[-1] = dirac_delta - np.nansum(h[:-1], axis=0)

	# calculate flux correction factor for each decomposition filter h[i]
	for i in range(n):
		f[i] = 1.0/np.nansum(h[i])

	# calculate the decomposed images S_i, P_i, and f_i if applicable
	if calculation_mode is 'direct':
		s[0] = filter_function(img, sigmas[0])
		for i in range(1,n-1):
			s[i] = filter_function(img - np.nansum(s[:i], axis=0), sigmas[i])
		s[-1] = img - np.nansum(s[:-1], axis=0)

	elif calculation_mode is 'convolution':
		for i in range(n):
			s[i] = sp.signal.convolve2d(img, h[i], mode='same')
	elif calculation_mode is 'convolution_fft':
		for i in range(n):
			s[i] = sp.signal.fftconvolve(img, h[i], mode='same')
	else:	
		print(f'ERROR: Argument "calculation_mode" is {calculation_mode}, options are ("direct", "convolution").')
		raise NotImplementedError
	return(s, h, f)

def ensure_subscriptable(x, min_size=1, max_size=None, cast=None):
	"""
	Check that an object is subscriptable with a maximum and/or minimum size
	if object does not satisfy requirements, extend or truncate it.
	"""
	if is_subscriptable(x) and (type(x) is not dict):
		if cast is None:
			cast = type(x)
		if (not (max_size is None)) and (len(x) > max_size):
			x = x[:size]
		if len(x) < min_size:
			x = cast(list(x) + [x[-1] for i in range(len(x),min_size)])
	else:
		if cast is None:
			cast = tuple
		x = cast(min_size*[x])
	return(x)



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



def psf_resize_ensure_centered(psf_img, n_padding):
	return(ensure_centered_psf(np.pad(psf_img, n_padding)))




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


##############################################################################
############## TESTING ROUTINES GO BELOW THIS LINE ###########################
##############################################################################
if __name__=='__main__':
	import copy
	import plotutils
	import fitscube.deconvolve.lucy_richardson
	import fitscube.deconvolve.clean
	
	# define test psf
	tps=(301,301)
	x,y=(tps[0]//2, tps[1]//2) # center point of PSF
	test_psf = np.zeros(tps)
	test_psf[x,y]=1
	#test_psf[x-4:x+5,y-4:y+5]=0.5
	#test_psf[x-9:x+10,y-9:y+10]=0.2
	#test_psf[x-20,y-20:y+20]=0.4
	test_psf = sp.ndimage.gaussian_filter(test_psf, 20)
	
	# normalise PSF
	test_psf/=np.nansum(test_psf) 
	
	# define test dirty image
	test_image = 'squares'
	if (type(test_image) is np.ndarray) and (a.ndim == 2):
		# we have been given a numpy array directly, therefore use it
		real_img = sp.signal.fftconvolve(test_image, test_psf, mode='same')
	elif test_image is 'squares':
		# create a set of squares
		real_img = np.zeros((400,300))
		real_img[150:170,40:50] += 1
		real_img[270:290,70:250] += 0.5
		real_img[130:180,130:180] += 1
		#real_img = sp.ndimage.gaussian_filter(dirty_img, 10)
	elif test_image is 'points':
		# create a set of points
		real_img = np.zeros((400,300))
		real_img = np.random.random((400,300))
		threshold = 0.9999
		intensity = 1000
		real_img[real_img>threshold] *= intensity
		real_img[real_img<=threshold] = 0
	else:
		print('\n'.join((	f'ERROR: test_image "{test_image}" not recognised,',
							f'-----: should be a 2D numpy array or one of ("squares", "points")'
							)))
	
	# add noise to dirty image
	real_img_min, real_img_max = (np.nanmin(real_img), np.nanmax(real_img))
	dirty_img = sp.signal.fftconvolve(real_img, test_psf, mode='same')/np.nansum(test_psf)
	shape_diff = np.array(dirty_img.shape) - np.array(real_img.shape)
	np.random.seed(1000) # seed generator for repeatability
	noise_power = 0.1
	if True:
		for i, s in enumerate(np.logspace(np.log(np.max(real_img.shape)),0,10,base=np.e)):
			print(f'INFO: Adding noise set {i} at scale {s} to dirty image')
			dirty_img[tuple([slice(diff//2,shape-diff//2) for diff, shape in zip(shape_diff, dirty_img.shape)])] \
				+= sp.ndimage.gaussian_filter(np.random.normal(0,noise_power,real_img.shape), s)
	
	
	# run through multiresolution with each base algorithm

	results = []
	#sigmas = (20,)
	sigmas = (20,10,5)
	#sigmas = (15,10,5,1)
	
	# Lucy-Richardson
	base_algorithm = fitscube.deconvolve.lucy_richardson.lucy_richardson
	kwargs = dict(	n_iter=200,
					nudge=1E-2,
					strength=1E-1,
					correction_factor_negative_fix=True,
					correction_factor_limit=np.inf,
					correction_factor_uclip=np.inf,
					correction_factor_lclip=-np.inf,
					show_plots=True,
					verbose=1
					)
	base_algorithm_kwargs = [	{**kwargs, **dict(n_iter=10)}, 
								{**kwargs, **dict(nudge=1E-1, n_iter=20)}, 
								{**kwargs, **dict(nudge=1E0, n_iter=40)}, 
								{**kwargs, **dict(nudge=1E+1, n_iter=1000, strength=5E-2)}
								]
	result = multiresolution(	dirty_img, 
								test_psf,
								base_algorithm = base_algorithm,
								base_algorithm_kwargs = base_algorithm_kwargs,
								show_plots=False,
								sigmas = sigmas
								)
	results.append(result)
	
	# CLEAN_modified
	base_algorithm = fitscube.deconvolve.clean.clean_modified
	base_algorithm_kwargs = dict(	window=True,
									max_iter=int(1E3),
									n_positive_iter=1E3,
									threshold=0.6,
									loop_gain=0.1,
									rms_frac_threshold=1E-2,
									fabs_frac_threshold=1E-2,
									norm_psf=True,
									show_plots=False
									)
	
	result = multiresolution(	dirty_img, 
								test_psf,
								base_algorithm = base_algorithm,
								base_algorithm_kwargs = base_algorithm_kwargs,
								show_plots=False,
								sigmas = sigmas
								)
	results.append(result)
	
	# CLEAN_hogbom
	base_algorithm = fitscube.deconvolve.clean.clean_hogbom
	base_algorithm_kwargs = dict(	window=True,
									max_iter=int(1E6),
									n_positive_iter=1E3,
									loop_gain=0.1,
									rms_frac_threshold=1E-2,
									fabs_frac_threshold=1E-2,
									norm_psf=True,
									show_plots=False
									)
	
	result = multiresolution(	dirty_img, 
								test_psf,
								base_algorithm = base_algorithm,
								base_algorithm_kwargs = base_algorithm_kwargs,
								show_plots=False,
								sigmas = sigmas
								)
	results.append(result)


	# plot the results
	
	cmap = copy.copy(plt.get_cmap('bwr'))
	cmap.set_over('magenta')
	cmap.set_under('green')
	cmap.set_bad('black')
	f2, a2 = plotutils.create_figure_with_subplots(len(results),4)
	for i, result in enumerate(results):
		a2[i,0].set_title('dirty image')
		im20 = a2[i,0].imshow(dirty_img, origin='lower', cmap=cmap)
		im20_lims = plotutils.lim_sym_around_value(dirty_img)
		im20.set_clim(im20_lims)
		
		a2[i,1].set_title('PSF')
		im21 = a2[i,1].imshow(test_psf, origin='lower', cmap=cmap)
		im21_lims = plotutils.lim_sym_around_value(test_psf)
		im21.set_clim(im21_lims)
		
		a2[i,2].set_title('Components')
		im22 = a2[i,2].imshow(result[1], origin='lower', cmap=cmap)
		im22_lims = plotutils.lim_sym_around_value(dirty_img)
		im22.set_clim(im22_lims)
		
		a2[i,3].set_title('Residual')
		im23 = a2[i,3].imshow(result[0], origin='lower', cmap=cmap)
		im23_lims = plotutils.lim_sym_around_value(dirty_img)
		im23.set_clim(im23_lims)
	
	plt.show()
	

