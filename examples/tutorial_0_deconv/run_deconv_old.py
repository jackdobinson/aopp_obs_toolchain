#!/usr/bin/env python3
"""
This script is intended to provide a bare-bones example of deconvolving a 
*.fits cube with the MODIFIED-CLEAN algorithm.

I've done everything explicity in this file, there are definitely things
you could do to make this semantically simpler, but I thought the 
"straight through" procedural style would be best for the first tutorial.
"""

import sys, os

# Add folder that contains scripts to $PATH environment variable.
# need to do this so directory structure is easily transferrable
# between machines.
SCRIPTS_PATH = "../scripts"
sys.path.insert(0,SCRIPTS_PATH)
print(f'{sys.path=}')


import numpy as np
from astropy.io import fits

#from scripts.algorithms import CleanModified
from fitscube.deconvolve.algorithms import CleanModified

import utilities as ut
import utilities.np

# Global variables hold locations of data, I would not normally do this but for
# illustrative purposes it's good enough.
#EXAMPLE_INPUT_OBS_DATA = "../data/test_rebin.fits"
EXAMPLE_INPUT_OBS_DATA = os.path.expanduser("~/scratch/general_testing/Neptune_NIFS_20090902_H_smooth_renorm.fits")
EXAMPLE_INPUT_OBS_DATA_EXT = 1

#EXAMPLE_INPUT_PSF_DATA = "../data/fit_example_psf_000.fits"
#EXAMPLE_INPUT_PSF_DATA_EXT = 0

#EXAMPLE_INPUT_PSF_DATA = "../data/test_standard_star.fits"
EXAMPLE_INPUT_PSF_DATA = os.path.expanduser("~/scratch/general_testing/Neptune_NIFS_20090902_H_smooth_renorm_standard_star.fits")
EXAMPLE_INPUT_PSF_DATA_EXT = 1


def main(args):
	"""
	All code pertaining to running the algorithm goes here. Need to do the 
	following things:

	1) Pass arguments to CleanModified class in "algorithms.py"
	
	2) Open observation and PSF data files

	3) Ensure that the data is in a known format

	4) Perform the deconvolution

	5) Save the deconvolved data to a new file.
	"""

	# Print out arguments for validation
	print(f'# ARGUMENTS #')
	for k, v in args.items():
		print(f'\t{k} = {v}')

	# get a callable that performs deconvolution
	deconv_func = CleanModified(
		n_iter				= args['CleanModified.n_iter'],
		loop_gain 			= args['CleanModified.loop_gain'],
		threshold 			= args['CleanModified.threshold'],
		n_positive_iter 	= args['CleanModified.n_positive_iter'],
		noise_std 			= args['CleanModified.noise_std'],
		rms_frac_threshold 	= args['CleanModified.rms_frac_threshold'],
		fabs_frac_threshold = args['CleanModified.fabs_frac_threshold'],
		show_plots 			= args['CleanModified.show_plots'],
	)


	with fits.open(args['input.obs.file']) as obs_hdul,	fits.open(args['input.psf.file']) as psf_hdul:
		obs_data_raw = obs_hdul[args['input.obs.ext']].data
		psf_data_raw = psf_hdul[args['input.psf.ext']].data

		# Create arrays that will hold deconvolved data, this will be the
		# same size as the input data. 
		# You could definitely change this to only allocate what you need if
		# you are only deconvolving part of a cube, but this is the simplest 
		# way.
		deconv_components_raw = np.full_like(obs_data_raw, fill_value=np.nan)
		deconv_residual_raw = np.full_like(obs_data_raw, fill_value=np.nan)


		# Remove all NAN values, most deconv algorithms don't like them at all
		# ideally you would interpolate at NANs and/or run one of the filtering
		# routines (see those tutorials for details)
		obs_bp_map = np.isnan(obs_data_raw) | np.isinf(obs_data_raw)
		psf_bp_map = np.isnan(psf_data_raw) | np.isinf(psf_data_raw)
		obs_data_raw[obs_bp_map] = 0.0
		psf_data_raw[psf_bp_map] = 0.0

		# We want to shift the image axes to be the last axes in the cube (it
		# makes iteration easier), so work out the final positions of the image
		# axes and then move them. I've put this into a helper class in the
		# other tutorials.
		obs_image_axes_dest = tuple(obs_data_raw.ndim -1 - i for i, x in enumerate(args['input.image_axes']))
		psf_image_axes_dest = tuple(psf_data_raw.ndim -1 - i for i, x in enumerate(args['input.image_axes']))

		print(f'{args["input.image_axes"]=}')
		print(f'{obs_image_axes_dest=}')
		print(f'{psf_image_axes_dest=}')

		# Move the image axes to the end of the arrays, this doesn't copy any
		# data, just re-arranges it (returns a view of the arrays)
		obs_data = np.moveaxis(obs_data_raw, args['input.image_axes'], obs_image_axes_dest)
		psf_data = np.moveaxis(psf_data_raw, args['input.image_axes'], psf_image_axes_dest)
		deconv_components = np.moveaxis(deconv_components_raw, args['input.image_axes'], obs_image_axes_dest)
		deconv_residual = np.moveaxis(deconv_residual_raw, args['input.image_axes'], obs_image_axes_dest)


		# Want to be able to define a sub-set of the whole dataset to iterate 
		# over in case we don't want to deconvolve the entire cube.
		print(f'{obs_data.shape=}')
		print(f'{psf_data.shape=}')

		# Find the axes that are NOT the image axes.
		obs_non_image_axes = tuple(_i for _i in range(obs_data.ndim) if _i not in obs_image_axes_dest)
		psf_non_image_axes = tuple(_i for _i in range(psf_data.ndim) if _i not in psf_image_axes_dest)

		print(f'{obs_non_image_axes=}')
		# Find how many "images" the cube holds. I.e. how many wavelengths or
		# other things there are in the cube.
		obs_non_image_size = np.prod(tuple(obs_data.shape[_x] for _x in obs_non_image_axes))
		psf_non_image_size = np.prod(tuple(psf_data.shape[_x] for _x in psf_non_image_axes))

		print(f'{obs_non_image_size=}')
	
		# Change to odd psf
		psf_data = psf_data[:,:-1,:-1]

		# Iterate over the desired parts of the dataset and deconvolve each
		# image. I've abstracted the construction of indices into a helper
		# class in other tutorials.
		for i, j in zip(
				np.arange(obs_non_image_size)[args['input.slice']], 
				np.arange(psf_non_image_size)[args['input.slice']]
			):
			# Ensure that the PSF is centered, we should also ensure it's an odd number of pixels
			# so there is an unabiguous center pixel.
			psf_data[j] = ut.np.center_on(psf_data[j], np.array(np.unravel_index(np.nanargmax(psf_data[j]), psf_data[j].shape)))

			# Remove negative bits of the PSF by setting them to zero, a quick and dirty hack but
			# should suffice for now.
			psf_data[psf_data < 0] = 0

			print(f'{i=}/{obs_non_image_size} {j=}/{psf_non_image_size}')
			deconv_func(obs_data[i], psf_data[j])
			deconv_components[i,...] = deconv_func.get_components()
			deconv_residual[i,...] = deconv_func.get_residual()

		# Now we have the deconvolved data, Create Header Data Units for 
		# the deconvolved data

		# Copy the header of the observational data
		components_header = obs_hdul[args['input.obs.ext']].header

	# We can do the following steps after we have closed the original files,
	# all the data we need has been found.

	# Create FITS headers, copy data from the observation (so all of the 
	# coordinate systems etc. are correct, also store the algorithm's name 
	# and the algorithm's parameters
	# I've put this step into helper functions in other tutorials as it's 
	# pretty generic.
	components_header['HIERARCH DECONVOLUTION ALGORITHM'] = 'CleanModified'
	for i, (k,v) in enumerate(deconv_func.get_params().items()):
		components_header[f'HIERARCH DECONVOLUTION ALGORITHM PARAM{i}'] = k
		components_header[f'HIERARCH DECONVOLUTION ALGORITHM VALUE{i}'] = v

	# Add back in the bad pixels, don't what to think we have data where we 
	# actually don't. Just set to NAN for illustrative purposes, but really
	# we should store the NAN, +INF, and -INF separately and re-combine them
	# separately
	deconv_components_raw[obs_bp_map] = np.nan
	deconv_residual_raw[obs_bp_map] = np.nan

	# Combine the headers and data into HDUs, remember to move the data 
	# axes back to their original places
	components_hdu = fits.PrimaryHDU(
		data = deconv_components_raw,
		header = components_header,
	)
	residual_hdu = fits.ImageHDU(
		data = deconv_residual_raw,
		header = components_header,
		name='RESIDUAL',
	)

	# Create Header Data Unit List (HDUL) to hold deconvolved data
	deconv_hdul = fits.HDUList([components_hdu, residual_hdu])

	# Write the HDUL to the desired file
	deconv_hdul.writeto(args['output.deconv.file'], overwrite=args['output.deconv.file.overwrite'])

	print(f'INFO: Written deconvolved cube to')
	print(f'----: \t{args["output.deconv.file"]}')
	print(f'----: Only applied to channels in the following range:')
	print(f'----: \t{args["input.slice"]}')
	
	return


if __name__=='__main__':
	"""
	This code will only execute when the file is run as a script.
	"""

	# Unfortunately the only way to type-hint a dictionary is still
	# to use comments.
	args={
		'input.obs.file' 					: EXAMPLE_INPUT_OBS_DATA, # str
			# Path to a FITS file that contains observation data
		'input.obs.ext'						: EXAMPLE_INPUT_OBS_DATA_EXT, # int | str
			# FITS extension to deconvolve
		
		'input.psf.file' 					: EXAMPLE_INPUT_PSF_DATA, # str
			# Path to a FITS file that contains PSF data
		'input.psf.ext'						: EXAMPLE_INPUT_PSF_DATA_EXT, # int | str
			# FITS extension to use as PSF

		#'input.slice' 						: np.index_exp[229:230], 
		'input.slice' 						: np.index_exp[126:127], 
			# np.ndarray | Slice
			# The indices (0-indexed) of the extension that should be
			# deconvolved. Will be applied as "slice" in 
			# "hdul[extension].data[slice]".
			# Technically you may want to have two different versions, one
			# for the "obs", and another for the "psf". But you would only
			# need that if e.g. the wavelength grids of the "obs" and "psf" 
			# cubes didn't line up for some reason.
			# DEFAULT : 
			# 	(slice(None,None,None),)
			# 		Deconvolves the whole cube.
			# EXAMPLES: 
			# 	(np.array([0,1,2,3,4,5]),)
			#		Only deconvolves the first 6 images.
			# 	np.index_exp[229:230]
			#		Turns into an equivalent slice expression 
			# 		(slice(229,230,None),) that just deconvolves
			#		the 229th image.

		'input.image_axes'					: tuple([1,2]), # tuple(int,int)
			# The axes of the datacube that denote the image-plane, rather than
			# the wavelength axis. NOTE, this is in "astropy" ordering, i.e. if
			# NAXIS1 = 'wav', NAXIS2='ra', NAXIS3='dec', the image axes will be
			# (1,2). I.e. "astropy" ordering is zero-indexed, AND reversed from
			# the numbers in the FITS files (yes I know it's annoying).

		'output.deconv.file'				: './output/deconv.fits', # str
			# File to output deconvolved data to, will put components in
			# PRIMARY extension, and residual in 1st extension (named
			# "RESIDUAL").

		'output.deconv.file.overwrite'		: True, # bool
		
		#'CleanModified.n_iter' 				: 1000, # int
		'CleanModified.n_iter' 				: 5000, # int
													# 0 <= x

		#'CleanModified.loop_gain' 			: 0.02, # float
		'CleanModified.loop_gain' 			: 0.2, # float
													# 0 < x < 1

		'CleanModified.threshold' 			: -0.1,	# float
		#'CleanModified.threshold' 			: 0.5,	# float
													# {-2, -1, -1< x < 1, 0}
		
		'CleanModified.n_positive_iter' 	: 0, 	# int
													# 0 <= x
		
		'CleanModified.noise_std'			: 1E-2, # float
													# 0 < x
		
		'CleanModified.rms_frac_threshold' 	: 1E-2, # float
													# 0 <= x < 1
		
		'CleanModified.fabs_frac_threshold' : 1E-2, # float
													# 0 <= x < 1

		'CleanModified.show_plots' 			: True, # bool
													# True | False
			# Shows debugging and in-progress plots as the algorithm proceeds,
			# useful to work out exactly what is going on at each step.
	}
	main(args)




