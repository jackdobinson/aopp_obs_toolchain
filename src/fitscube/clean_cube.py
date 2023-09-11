#!/usr/bin/env python3
"""
Performs the CLEAN algorithm on a fitscube and stores the results in a different fitscube
"""

import sys, os
import numpy as np
from astropy.io import fits
import fitscube.deconvolve.clean
import fitscube.stack
import utils as ut
import plotutils
import astropy.convolution
import scipy as sp
import glob
import matplotlib.pyplot as plt
import geometry


def parse_args(argv):
	"""Parses command line arguments, see https://docs.python.org/3/library/argparse.html"""
	import argparse as ap
	"""
	# =====================
	# FORMATTER INFORMATION
	# ---------------------
	A formatter that inherits from multiple formatter classes has all the attributes of those formatters
	see https://docs.python.org/3/library/argparse.html#formatter-class for more information on what each
	of them do.
	Quick reference:
	ap.RawDescriptionHelpFormatter -> does not alter 'description' or 'epilog' text in any way
	ap.RawTextHelpFormatter -> Maintains whitespace in all help text, except multiple new lines are treated as one
	ap.ArgumentDefaultsHelpFormatter -> Adds a string at the end of argument help detailing the default parameter
	ap.MetavarTypeHelpFormatter -> Uses the type of the argument as the display name in help messages
	# =====================	
	"""
	class RawDefaultTypeFormatter(ap.RawDescriptionHelpFormatter, ap.ArgumentDefaultsHelpFormatter, ap.MetavarTypeHelpFormatter):
		pass
	class RawDefaultFormatter(ap.RawDescriptionHelpFormatter, ap.ArgumentDefaultsHelpFormatter):
		pass
	class TextDefaultTypeFormatter(ap.RawTextHelpFormatter, ap.ArgumentDefaultsHelpFormatter, ap.MetavarTypeHelpFormatter):
		pass
	class TextDefaultFormatter(ap.RawTextHelpFormatter, ap.ArgumentDefaultsHelpFormatter):
		pass

	#parser = ap.ArgumentParser(description=__doc__, formatter_class = ap.TextDefaultTypeFormatter, epilog='END OF USAGE')
	# ====================================
	# UNCOMMENT to enable block formatting
	# ------------------------------------
	parser = ap.ArgumentParser	(	description=ut.str_block_indent_raw(ut.str_rationalise_newline_for_wrap(__doc__), wrapsize=79),
									formatter_class = RawDefaultTypeFormatter,
									epilog=ut.str_block_indent_raw(ut.str_rationalise_newline_for_wrap('END OF USAGE'), wrapsize=79)
								)
	# ====================================

	parser.add_argument('target_cubes', type=str, nargs='+', help='A list of fitscubes to operate on')
	parser.add_argument('--algorithm', type=str, choices=('hogbom', 'modified', 'multiresolution'), help='The CLEAN algorithm to use', default='hogbom')
	
	# Arguments for CLEAN algorithm
	clean_grp = parser.add_argument_group('CLEAN Arguments', description='Arguments that control the behaviour of the CLEAN algorithm.')
	clean_grp.add_argument('--clean.loop_gain', type=float, help='What fraction of a detected "point source" is subtracted from the residual each iteration?', default=0.1)
	clean_grp.add_argument('--clean.rms_frac_threshold', type=float, help="Stopping criterion, when the RMS of the CLEAN residual is reduced to this fraction of it's initial value.", default=1E-2)
	clean_grp.add_argument('--clean.fabs_frac_threshold', type=float, help="Stopping criterion, when the absolute brightest pixel of the CLEAN residual is reduced to this fraction of it's initial value", default=1E-2)
	clean_grp.add_argument('--clean.max_iter', type=int, help="Stopping criterion, when this number of iterations has been reached", default=10000)
	clean_grp.add_argument('--clean.norm_psf', action=plotutils.ActionTf, prefix='clean.', help='Should we normalise the PSF when performing the CLEAN algorithm?')
	clean_grp.add_argument('--clean.show_plots', action=plotutils.ActiontF, prefix='clean.', help='Should we show in-progress plots of the CLEAN algorithm as it is running?')
	clean_grp.add_argument('--clean.quiet', action=plotutils.ActionTf, prefix='clean.', help='Should we print information on the progress of the CLEAN algorithm as it is running?')
	
	# Arguments for modified version of CLEAN algorithm
	clean_modified_grp = parser.add_argument_group('CLEAN Modified Arguments', description='Arguments that control the behviour of the "modified" version of the CLEAN algorithm')
	clean_modified_grp.add_argument('--clean.modified.threshold', type=float, help='When using the "modified" version of CLEAN, what fraction of the peak value is considered to be part of an extended source?', default=0.6)
	clean_modified_grp.add_argument('--clean.modified.n_positive_iter', type=float, help='When using the "modified" versio of CLEAN, number of initial iterations where only positive pixels are considered for CLEANing.', default=0)

	# Arguments for the multiresolution version of the CLEAN algorithm
	clean_multiresolution_grp = parser.add_argument_group('CLEAN Multiresolution Arguments', description='Arguments that control the behaviour of the "multiresolution" version of the CLEAN algorithm')
	clean_multiresolution_grp.add_argument('--clean.multiresolution.sigmas', nargs='+', type=float, help='The standard deviation (in pixels) of successively applied gaussian filters that decompose the dirty image', default=(10, 5, 2.5, 1.25))
	clean_multiresolution_grp.add_argument('--clean.multiresolution.calculation_mode', type=str, help='How the decomposition of the dirty image and PSF will be calculated ("direct","convolution")', default="direct")
	clean_multiresolution_grp.add_argument('--clean.multiresolution.base_clean_algorithm', type=str, help='Which CLEAN algorithm will be used to CLEAN the decomposed images ("hogbom","modified")', default='hogbom')
	clean_multiresolution_grp.add_argument('--clean.multiresolution.reject_decomposition_idxs', nargs='*', type=int, help='Indicies of decomposed images whose CLEAN components should not be included in the final results. Note there are len(sigmas)+1 decompose images, and the indices start from 0.', default=[4])

	# Arguments for stacking fitscube along the wavelength axis
	wavaxis_bin_grp = parser.add_argument_group('Wavelength Binning', description='Arguments that control binning over the wavelength axis')
	wavaxis_bin_grp.add_argument('--wavaxis.bin.min', type=float, help='Value to start bins at', default=1.455)
	wavaxis_bin_grp.add_argument('--wavaxis.bin.max', type=float, help='Value to end bins at', default=2.455)
	wavaxis_bin_grp.add_argument('--wavaxis.bin.n', type=int, help='Number of bins', default=200)
	wavaxis_bin_grp.add_argument('--wavaxis.bin.step', type=float, help='Step between the start of each bin', default=None)
	wavaxis_bin_grp.add_argument('--wavaxis.bin.width', type=float, help='Width of each bin', default=None)
	wavaxis_bin_grp.add_argument('--wavaxis.stack.mode', type=str, choices=('mean','median','mode','sum'), help='How should we combine the data in the wavelength bins', default='median')
	
	# Arguments that control convolving the components with a clean beam
	convolution_grp = parser.add_argument_group('Convolution', description='Arguments that control how the reconvolution with a gaussian will work')
	convolution_grp.add_argument('--convolve.gaussian.std', type=float, help='Standard deviation of a gaussian to convolve the CLEAN components with, in pixel units.', default=0.75)
	
	# Arguments that control how the output behaves
	output_grp = parser.add_argument_group('Output', description='Arguments that control output')
	output_grp.add_argument('--output.fitscube.overwrite', action=plotutils.ActionTf, prefix='output.fitscube.', help='If a file with the same name as the output file already, should we overwrite it?')
	output_grp.add_argument('--output.fitscube.suffix', type=str, help='suffix to append to the name of the operated on file to create the output file name', default='_clean')

	# Arguments that control how the bright-spot removal works
	bright_spot_removal_grp = parser.add_argument_group('Bright Spot Removal', description='Arguments that control bright spot removal')
	bright_spot_removal_grp.add_argument('--bright_spot_removal.enable', 
								action=plotutils.ActiontF, prefix='bright_spot_removal.', 
								help='Should we try to remove bright spots using a version of multiresolution CLEAN'
								)
	bright_spot_removal_grp.add_argument('--bright_spot_removal.cutoff_factor', type=float, help='Controls how aggressive the spot removal is, lower values are more aggressive.', default=5)
	bright_spot_removal_grp.add_argument('--bright_spot_removal.show_plots',
										action=plotutils.ActiontF, prefix='bright_spot_removal.',
										help='Should we display plots when removing bright spots?')
	bright_spot_removal_grp.add_argument('--bright_spot_removal.mode', type=str, choices=('curve_fit', 'maximum_likelihood'), help='How should we fit the spread of residual data?', default='curve_fit')


	parsed_args = vars(parser.parse_args(argv)) # I prefer a dictionary interface
	return(parsed_args)

def args_strip_prefix(args, prefix):
	d = {}
	for k, v in args.items():
		if k.startswith(prefix):
			d[k[len(prefix):]]=v
	return(d)

def remove_bright_spots(dirty_img, psf_img, factor=5, show_plots=True, mode='curve_fit', clean_mr_args={}):
	"""
	Decompose image to remove bright spots
	"""
	print('INFO: Running "remove_bright_spots()"')

	clean_mr_default_args = dict(
					window = True,
					loop_gain = 0.2,
					threshold = 0.6,
					rms_frac_threshold = 0.01,
					fabs_frac_threshold = 0.01,
					max_iter = 10000,
					norm_psf = 'max',
					show_plots = False,
					quiet = True,
					n_positive_iter = 0,
					sigmas = (10, 5, 2.5, 1.25),
					calculation_mode = 'direct',
					base_clean_algorithm = 'hogbom',
					reject_decomposition_idxs = [4]
				)
	# add default arguments to "clear_mr_args" dictionary
	clean_mr_args = {**clean_mr_default_args, **clean_mr_args}

	(	residual,
		components,
		rms_record,
		fabs_record,
		n_iter,
	) = fitscube.deconvolve.clean.clean_multiresolution(
					dirty_img,
					psf_img,
					**clean_mr_args
					)
				
	diff_img = dirty_img - (sp.signal.convolve2d(components, psf_img, mode='same') + residual)
	
	#fitscube.deconvolve.clean.plot_clean_result(dirty_img, components, residual, psf_img, rms_record, fabs_record, diff_img)

	#diff_img_ff = np.fabs(diff_img).flatten()
	diff_img_ff = diff_img.flatten()

	# get histogram of difference between reconstructed and original image
	ns, edges = np.histogram(	diff_img_ff,
								bins=100,
								range=(np.nanmin(diff_img_ff), np.nanmax(diff_img_ff)),
								density=True
							)
	mids = 0.5*(edges[:-1]+edges[1:])
		
	# define half normal distribution and maximum likelihood estimator functions
	half_normal = lambda x, sigma: np.sqrt(2)/(sigma*np.sqrt(np.pi))*np.exp(-(x**2)/(2*sigma**2))
	normal = lambda x, sigma, mu: (1.0/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((x-mu)/sigma)**2)

	mle = lambda func, args, xi: (1.0/np.size(xi))*np.nansum(np.log(func(xi,*args)))
	neg_mle = lambda func, args, xi: -mle(func, args, xi)


	if mode == 'maximum_likelihood':
		print('INFO: Using maximum likelihood method')
		# using maximum likelihood
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
				#print(f'\n{test_points[0]:010.5E} {tmp[0]:010.5E} {test_points[1]:010.5E} {tmp[1]:010.5E} {test_points[2]:010.5E}')
				tmp_val = test_at_points(tmp)	
				#print(f'{test_vals[0]:010.5E} {tmp_val[0]:010.5E} {test_vals[1]:010.5E} {tmp_val[1]:010.5E} {test_vals[2]:010.5E}')
				if tmp_val[0] < test_vals[0] and tmp_val[0] < test_vals[1]:
					test_points[2] = test_points[1]
					test_vals[2] = test_vals[1]
					#print('>>>')
				elif tmp_val[1] < test_vals[2] and tmp_val[1] < test_vals[1]:
					test_points[0] = test_points[1]
					test_vals[0] = test_vals[1]
					#print('<<<')
				elif tmp_val[0] > test_vals[1] and test_vals[1] < tmp_val[1]:
					test_points[0] = tmp[0]
					test_points[2] = tmp[1]
					test_vals[0] = tmp_val[0]
					test_vals[2] = tmp_val[1]
					#print('< >')
				else:
					print('ERROR: Detected more than one minima in range, terminating')
					raise ValueError
				test_points[1] = mid(test_points[0], test_points[2])
				test_vals[1] = func(test_points[1])
				i+=1
			return(np.array([test_points[1]]))
		bounds = (0, np.nanmax(diff_img_ff))
		mle_sigma = bisection_minima(lambda sigma: neg_mle(normal, (sigma, 0), diff_img_ff), bounds)
		"""
		minimise_result = sp.optimize.minimize(lambda sigma, xi: neg_mle(normal, (sigma, 0), xi), 
														[np.nanstd(diff_img_ff)],
														args=(diff_img_ff,), 
														#method='bounded', 
														#bounds=[(0, np.nanmax(diff_img_ff))],
														#tol=1E-2*np.nanstd(diff_img_ff)
														)
		mle_sigma = minimise_result.x
		"""
	elif mode == 'curve_fit':
		print('INFO: Using curve fit')
		# using fitting
		x0 = [np.nanstd(diff_img_ff)]
		minimise_result = sp.optimize.curve_fit(lambda x, sigma: normal(x, sigma, 0) , mids, ns, x0)
		mle_sigma = minimise_result[0][0]
		mle_pcov = minimise_result[1]
	else:
		print('ERROR: Unknown value "{mode}" to argument "mode", accepted values are ("maximum_likelihood","curve_fit")')
		raise NotImplementedError

	print('INFO: Bright spot removal, maximum likelihood of sigma')
	print(mle_sigma)
	
	# define cutoff, i.e. how aggressive we are at removing bright spots
	cutoff = factor*mle_sigma

	#print(f'cutoff {cutoff}')
	
	pixels_to_remove = np.fabs(diff_img) > cutoff
	dirty_img[pixels_to_remove] = np.nan
	
	if show_plots:
		nr,nc,s = (1,4,6)
		f1 = plt.figure(figsize=(s*nc, s*nr))
		a1 = f1.subplots(nr, nc, squeeze=False)

		a1[0,0].imshow(diff_img, origin='lower')
		
		a1[0,1].imshow(pixels_to_remove, origin='lower')
		
		a1[0,2].imshow(dirty_img, origin='lower')
		
		a1[0,3].hist(diff_img_ff, bins=100, density=True, color='tab:blue')
		a1[0,3].plot(edges, normal(edges, mle_sigma, 0), color='tab:orange')
		a1[0,3].axvline(cutoff, color='red')
		plt.show()
	return(pixels_to_remove)

def get_bin_edges(bmin, bmax, bn, bs, bw):
	if bs is None and bw is None:
		# assume they are equal
		bs = (bmax - bmin)/(bn)
		bw = bs
	if bs is None:
		# then bw is not
		bs = (bmax-bmin-bw)/(bn-1)
	if bw  is None:
		# then bs is not
		bw = bmax - bmin - (n-1)*bs
	# ensure our parameters are not contradictory
	if np.fabs((bmax - bmin -bw)/(bs*(bn-1)) - 1) > 1E-6:
		print(f'ERROR: Bin number {bn}, step {bs}, or width {bw} are contradictory. I calculate step {(bmax-bmin-bw)/(bn-1)} width {bmax - bmin - (bn-1)*bs}')
		raise ValueError

	# get lower edges
	bl = np.arange(bmin, bmax-bw, bs)
	# get upper edges
	bu = np.arange(bmax, bmin+bw, -bs)[::-1]

	return(np.stack([bl,bu]))
		

#%%
if __name__=='__main__':
	# Create dummy arguments if running interactively
	if sys.argv[0]=='':
		sys.argv = 	['', 
						*glob.glob('/home/dobinsonl/scratch/reduced_images/*/*_0.025_*/analysis/*_renormed.fits'),
						'--algorithm', 'multiresolution',
						'--clean.rms_frac_threshold', '0.01',
						'--clean.fabs_frac_threshold', '0.01',
						'--output.fitscube.suffix', '_cleanMR_200',
						'--wavaxis.bin.n', '200'
						]
		print(sys.argv)
	args = parse_args(sys.argv[1:])
	print(ut.str_wrap_in_tag(ut.str_dict(args), 'ARGUMENTS'))

	for i, tc in enumerate(args['target_cubes']):
		print(f'INFO: Operating on cube {tc} [{i}/{len(args["target_cubes"])}]')
		
		with fits.open(tc) as hdul:
			# stack cube along wavelength axis 
			#w_bins = np.linspace(args['wavaxis.bin.min'], args['wavaxis.bin.max'], args['wavaxis.bin.n'])
			w_bins, w_bin_params = geometry.get_bin_edges(args['wavaxis.bin.min'], args['wavaxis.bin.max'], args['wavaxis.bin.n'],
									args['wavaxis.bin.step'], args['wavaxis.bin.width'])
			print(f'INFO: Using bins {w_bin_params[0]} -> {w_bin_params[1]} n {w_bin_params[2]} step {w_bin_params[3]} width {w_bin_params[4]}')
			if args['wavaxis.stack.mode'] == 'mean':
				stack_func = np.nanmean
			elif args['wavaxis.stack.mode'] == 'median':
				stack_func = np.nanmedian
			elif args['wavaxis.stack.mode'] == 'mode':
				stack_func = np.nanmode
			elif args['wavaxis.stack.mode'] == 'sum':
				stack_func = np.nansum
			else:
				print(f'ERROR: Unknown value "{args["wavelength.stack.mode"]}" for argument "wavelength.stack.mode"')
				raise NotImplementedError

			stacked_data_hdu = fitscube.stack.stack_hdu_along_spectral_axis(hdul['PRIMARY'], w_bins, stack_func=stack_func)
			stacked_psf_hdu = fitscube.stack.stack_hdu_along_spectral_axis(hdul['PSF'], w_bins, stack_func=stack_func)
			
			#stacked_psf_hdu_norm_factor = np.ones(stacked_psf_hdu.data.shape[0])
			stacked_psf_hdu_norm_factor = 1.0/np.nansum(stacked_psf_hdu.data, axis=(1,2))


			stacked_err_hdu = fitscube.stack.stack_hdu_along_spectral_axis(hdul['ERROR'], w_bins, stack_func=stack_func)
			
			# create holders for CLEAN output and fill with junk
			np.nans_like = lambda x, *a, **k: np.full_like(x, *a, fill_value=np.nan, **k)
			stacked_components = np.nans_like(stacked_data_hdu.data, dtype=stacked_data_hdu.data.dtype)
			stacked_residual = np.nans_like(stacked_data_hdu.data, dtype=stacked_data_hdu.data.dtype)
			stacked_gaussian_convolution = np.nans_like(stacked_data_hdu.data, dtype=stacked_data_hdu.data.dtype)
			
			# perform CLEAN algorithm on each wavelength of stacked hdu
			for j in range(w_bins.shape[1]):
				print(f'INFO: Cleaning stack {j}/{w_bins.shape[1]} wavs {w_bins[0,j]} -> {w_bins[1,j]}')
				cube_nans = np.isnan(stacked_data_hdu.data[j]) | (stacked_data_hdu.data[j]==0)
				disk_mask = (hdul['DISK_MASK'].data == 1)
				
				if np.all(cube_nans):
					print(f'WARNING: All NANs detected for this slice, skipping...')
					continue

				if args['bright_spot_removal.enable']:
					bright_mask = remove_bright_spots(stacked_data_hdu.data[j], 
										stacked_psf_hdu.data[j]*stacked_psf_hdu_norm_factor[j],
										factor = args['bright_spot_removal.cutoff_factor'],
										show_plots = args['bright_spot_removal.show_plots'],
										mode = args['bright_spot_removal.mode'],
										clean_mr_args = dict(	window = (disk_mask | cube_nans),
																loop_gain = args['clean.loop_gain'],
																threshold = args['clean.modified.threshold'],
																rms_frac_threshold = args['clean.rms_frac_threshold'],
																fabs_frac_threshold = args['clean.fabs_frac_threshold'],
																max_iter = args['clean.max_iter'],
																norm_psf = args['clean.norm_psf'],
																show_plots = args['clean.show_plots'],
																quiet = args['clean.quiet'],
																n_positive_iter = args['clean.modified.n_positive_iter'],
																sigmas = args['clean.multiresolution.sigmas'],
																calculation_mode = args['clean.multiresolution.calculation_mode'],
																base_clean_algorithm = args['clean.multiresolution.base_clean_algorithm'],
																reject_decomposition_idxs = args['clean.multiresolution.reject_decomposition_idxs']	
															)
										)
					cube_nans =  cube_nans | bright_mask
					print('INFO: Finished bright spot removal')

				if args['algorithm'] == 'hogbom':
						(	residual,
						components,
						rms_record,
						fabs_record,
						n_iter,
					) = fitscube.deconvolve.clean.clean_hogbom(	stacked_data_hdu.data[j],
																stacked_psf_hdu.data[j]*stacked_psf_hdu_norm_factor[j],
																window = (disk_mask | cube_nans),
																loop_gain = args['clean.loop_gain'],
																rms_frac_threshold = args['clean.rms_frac_threshold'],
																fabs_frac_threshold = args['clean.fabs_frac_threshold'],
																max_iter = args['clean.max_iter'],
																norm_psf = args['clean.norm_psf'],
																show_plots = args['clean.show_plots'],
															)
				elif args['algorithm'] == 'modified':
					(	residual,
						components,
						rms_record,
						fabs_record,
						n_iter,
						accumulator,
						window
					) = fitscube.deconvolve.clean.clean_modified(	
												stacked_data_hdu.data[j],
												stacked_psf_hdu.data[j]*stacked_psf_hdu_norm_factor[j],
												window = (disk_mask | cube_nans),
												loop_gain = args['clean.loop_gain'],
												threshold = args['clean.modified.threshold'],
												rms_frac_threshold = args['clean.rms_frac_threshold'],
												fabs_frac_threshold = args['clean.fabs_frac_threshold'],
												max_iter = args['clean.max_iter'],
												norm_psf = args['clean.norm_psf'],
												show_plots = args['clean.show_plots'],
												quiet = args['clean.quiet'],
												n_positive_iter = args['clean.modified.n_positive_iter']
											)
				elif args['algorithm'] == 'multiresolution':
					# If I'm doing this properly, will need a way to enter lists of arguments,
					# one for each decomposed image, rather than re-using the arguments for the
					# other algorithms. However, I've only used the same CLEAN arguments for each
					# decomposed image up until now so this should suffice.
					(	residual,
						components,
						rms_record,
						fabs_record,
						n_iter,
					) = fitscube.deconvolve.clean.clean_multiresolution(
									stacked_data_hdu.data[j],
									stacked_psf_hdu.data[j]*stacked_psf_hdu_norm_factor[j],
									window = (disk_mask | cube_nans),
									loop_gain = args['clean.loop_gain'],
									threshold = args['clean.modified.threshold'],
									rms_frac_threshold = args['clean.rms_frac_threshold'],
									fabs_frac_threshold = args['clean.fabs_frac_threshold'],
									max_iter = args['clean.max_iter'],
									norm_psf = args['clean.norm_psf'],
									show_plots = args['clean.show_plots'],
									quiet = args['clean.quiet'],
									n_positive_iter = args['clean.modified.n_positive_iter'],
									sigmas = args['clean.multiresolution.sigmas'],
									calculation_mode = args['clean.multiresolution.calculation_mode'],
									base_clean_algorithm = args['clean.multiresolution.base_clean_algorithm'],
									reject_decomposition_idxs = args['clean.multiresolution.reject_decomposition_idxs']
								)
				else:
					print(f'ERROR: Unknown value for argument "algorithm", value is "{args["algorithm"]}"')
					raise NotImplementedError
				
				stacked_components[j] = components
				stacked_residual[j] = residual
				stacked_gaussian_convolution[j] = astropy.convolution.convolve(	
													components, 
													astropy.convolution.Gaussian2DKernel(args['convolve.gaussian.std'])
												)
				# put NANs back into components, residual, and convolution
				data_nans = np.isnan(stacked_data_hdu.data[j])
				stacked_components[j][data_nans] = np.nan
				stacked_residual[j][data_nans] = np.nan
				stacked_gaussian_convolution[j][data_nans] = np.nan
			
			# create new HDUs for CLEANed data
			stacked_components_hdu = fits.ImageHDU(		stacked_components, 
														name='COMPONENTS', 
														header=fitscube.header.get_coord_fields(stacked_data_hdu.header)
													)
			stacked_residual_hdu = fits.ImageHDU(	stacked_residual, 
													name='RESIDUAL', 
													header=fitscube.header.get_coord_fields(stacked_data_hdu.header)
												)
			stacked_gaussian_convolution_hdu = fits.ImageHDU(
													stacked_gaussian_convolution, 
													name='GAUSSIAN_CONVOLUTION', 
													header=fitscube.header.get_coord_fields(stacked_data_hdu.header)
												)
			
			
			# Create HDUList and add PSF and masks to new hdul
			stacked_hdul = fits.HDUList(	[	stacked_data_hdu, 
												stacked_psf_hdu, 
												stacked_components_hdu, 
												stacked_residual_hdu, 
												stacked_gaussian_convolution_hdu, 
												hdul['DISK_MASK'], 
												stacked_err_hdu, 
												hdul['LATITUDE'], 
												hdul['LONGITUDE'], 
												hdul['ZENITH'], 
												hdul['DISK_FRAC']
											]
										)
			
			# Write new hdul to file in the same directory as the source but with 'clean' appeneded to it's name
			outfname = tc.rsplit('.',1)[0] + args['output.fitscube.suffix'] + '.fits'
			stacked_hdul.writeto(outfname, overwrite=args['output.fitscube.overwrite'])
