#!/usr/bin/env python3

import sys, os
import numpy as np
import scipy as sp
import scipy.optimize
import scipy.ndimage
import scipy.signal
import matplotlib.pyplot as plt
import typing
import glob
#import typing_extensions
#typing.Literal = typing_extensions.Literal
import numbers
import utilities as ut
import utilities.np
import utilities.type
import utilities.fits


"""
Helpers for deconvolution routines.

Working Definitions:
	components
		The pixel values of the underlying "real" image, i.e. our 'hypothetical' image
	model
		A "guess" as to what the underlying "real" image should be. Should be either
		a physically informed model (e.g. a synthetic image of Neptune) or the pixel
		values expected from an empty field (e.g. RMS noise values).
	estimate
		Our current estimate of the dirty_img (observation), is a convolution 
		between the PSF and the components
	dirty_img
		The observation we are trying to deconvolve
	psf
		The (or an estimate of) the point spread function that transforms the
		underlying "real" image into the dirty_img (observation)
	error
		The estimate of the error on the value of each pixel in the dirty_img 
		(observation). Should either be an RMS value, or a pixel map of values.
		Make the error for a pixel large if you want the solution to relax to the
		model values.
		
Useful Concepts:
	entropy
		A measure of how much extra information you have to add to a model.
		More information = more negative value.
"""

def generalised_least_squares(
		components 	: np.ndarray, 
		dirty_img 	: np.ndarray, 
		cov_mat 	: typing.Union[np.ndarray, numbers.Real], 	# variances for the errors on data
		response 	: np.ndarray
		) -> np.ndarray:
		mismatch = sp.signal.fftconvolve(components, response, mode='same') - dirty_img
		return(mismatch*mismatch*(1.0/cov_mat))

def generalised_least_squares_preconv(
		estimate 	: np.ndarray, 
		dirty_img 	: np.ndarray, 
		cov_mat 	: typing.Union[np.ndarray, numbers.Real], 	# variances for the errors on data
		) -> np.ndarray:
		return(((estimate - dirty_img)**2)*(1.0/cov_mat))
	
def generalised_least_squares_mat(
		components 	: np.matrix, 
		dirty_img 	: np.matrix, 
		cov_mat 	: np.matrix, 	# covariance matrix for the errors on data
		response 	: np.matrix
		) -> np.matrix:
		mismatch = response@components - dirty_img
		inv_cov_mat = np.linalg.inv(cov_mat)
		mm_matrix_row = mm_matrix if mm_matrix.shape[0] == 1 else mm_matrix.T
		return(np.einsum('ij,jk,kl->il', mm_matrix_row, inv_cov_mat, mm_matrix_row.T))


def entropy_pos_neg(
		components : np.ndarray,
		error : typing.Union[numbers.Real, np.ndarray]
		) -> np.ndarray:
	psi = np.sqrt(components**2 + 4*error**2)
	return(psi - 2*error - components*np.log((psi + components)/(2*error)))

def entropy_pos(
		components : np.ndarray,
		model : typing.Union[np.ndarray,numbers.Real]
		) -> np.ndarray:
	return(components - model - components*np.log(components/model))

def entropy_rel(
		components : np.ndarray,
		model : typing.Union[np.ndarray,numbers.Real]
		) -> np.ndarray:
	return(-components*np.log(components/model))

def entropy_adj(
		components : np.ndarray,
		model : typing.Union[numbers.Real, np.ndarray],
		error : typing.Union[numbers.Real, np.ndarray] = 1 # this could be absorbed into alpha if we need to only have 2 arguments
		):
	return(entropy_pos_neg(components-model,error))

def regularised_least_squares(
		components 			: np.ndarray,
		dirty_img 			: np.ndarray,
		error 				: typing.Union[np.ndarray, numbers.Real],
		alpha 				: typing.Union[numbers.Real, np.ndarray], # no reason we can't have different values for different pixels
		response			: np.ndarray, # usually an instrumental PSF
		model 				: typing.Union[np.ndarray, numbers.Real, None] 										= None,
		regularising_func 	: typing.Callable[[np.ndarray, typing.Union[np.ndarray,numbers.Real]], np.ndarray] 	= entropy_adj,
		) -> np.ndarray:
	#print(f'{components.shape = }')
	#print(f'{dirty_img.shape = }')
	#print(f'{error = }')
	#print(f'{alpha = }')
	#print(f'{response.shape = }')
	#print(f'{model = }')
	#print(f'{regularising_func = }')
	if model is None:
		model = error
	return(0.5*generalised_least_squares(components, dirty_img, error**2, response) - alpha*regularising_func(components, model))


def create_psf_image(
		psf_component_list 	: typing.Tuple[typing.Tuple[int,int,float],...],				# point sources that make up the PSF, in form (x,y,a) 
																							# where x and y are offset from center of image
		sigma 				: int 											= 20,			# standard deviation of the gaussian blur
		shape 				: typing.Tuple[int,int] 						= (101,101),	# shape of the psf image
		ensure_odd 			: bool 											= True,			# if true, will ensure that the image has an odd shape in each dimension
		dtype 				: type 											= float 		# data type of image
		) -> np.ndarray:
	if ensure_odd:
		shape = tuple([s - s%2 + 1 for s in shape])
		
	psf = np.zeros(shape, dtype=dtype)
	psf_cidxs = ut.np.idx_center(psf)
	cx, cy = psf_cidxs
	#print(psf_component_list)
	for x, y, a in psf_component_list:
		psf[x-cx,y-cy] = a
	psf /= np.nansum(psf)
	return(sp.ndimage.gaussian_filter(psf, sigma))
	

def test_function():
	return('TESTING')

def create_test_images(
		psf 	: typing.Union[np.ndarray, int] 		= 20, # PSF to corrupt real_img with, if an integer will create a gaussian filter with sigma=psf
		shape 	: typing.Tuple[int,int] 				= (400,300), 
		imtype 	: typing.Literal['squares','points'] 	= 'squares',
		dtype 	: type 									= float # should be coerceable to np.dtype
		) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""
	Create a 2D numpy array that can be used to test deconvolve algorithms
	"""
	#assert(type(dtype) is np.dtype, f'ERROR: variable dtype="{dtype}" is not coerceable to type np.dtype')
	ut.type.coerce_to_type(dtype, np.dtype)
	
	
	real_img : np.ndarray = np.zeros(shape,dtype=dtype)
	dirty_img: np.ndarray = np.zeros_like(real_img)
	
	if type(psf) is int:
		psf_sigma = psf
		psf = np.zeros([s - s%2 + 1 for s in shape], dtype=dtype)
		psf[ut.np.idx_center(psf)] = 1
		psf = sp.ndimage.gaussian_filter(psf, psf_sigma)
		psf /= np.nansum(psf)
	
	if imtype == 'squares':
		real_img[150:170,40:50] += 1
		real_img[270:290,70:250] += 0.5
		real_img[130:180,130:180] += 1
	if imtype == 'points':
		threshold = 0.9999
		intensity = 1000
		real_img = np.random.random(dis)
		real_img[real_img>threshold] *= intensity
		real_img[real_img<=threshold] = 0


	noise_power = 0.1
	real_img_min, real_img_max = (np.nanmin(real_img), np.nanmax(real_img))
	dirty_img = sp.signal.fftconvolve(real_img, psf, mode='same')
	shape_diff = np.array(dirty_img.shape) - np.array(real_img.shape)
	np.random.seed(1000) # seed generator for repeatability
	if True:
		for i, s in enumerate(np.logspace(np.log(np.max(real_img.shape)),0,10,base=np.e)):
			#print(f'INFO: Adding noise set {i} at scale {s} to dirty image')
			dirty_img[tuple([slice(diff//2,shape-diff//2) for diff, shape in zip(shape_diff, dirty_img.shape)])] \
				+= sp.ndimage.gaussian_filter(np.random.normal(0,noise_power,real_img.shape), s)
				
	return(real_img, dirty_img, psf)



def get_test_data(
		obs_path 	: str = os.path.expanduser('~/scratch/transfer/GEMINI_OBS/I_BAND/ctfbrgnN20110909S0174.fits'),
		psf_path 	: str = os.path.expanduser('~/scratch/transfer/GEMINI_OBS/I_BAND/ctfbrsnN20110909S0164.fits'),
		obs_ext 	: typing.Union[str, int] = 'SCI',
		psf_ext 	: typing.Union[str, int] = 'SCI',
	) -> typing.Tuple[np.ndarray, np.ndarray]:
	psf_data = ut.fits.get_img_from_fits(psf_path, psf_ext)
	return(ut.fits.get_img_from_fits(obs_path, obs_ext), psf_data/np.nansum(psf_data))

def get_test_data_set_list(filter_function=lambda x: True):
	return(list(filter(filter_function, glob.glob(os.path.expanduser('~/scratch/transfer/**/*.fits'), recursive=True))))

def get_test_data_sci_ext_name(fits_path):
	if 'GEMINI' in fits_path:
		return('SCI')
	if 'MUSE' in fits_path:
		return('DATA')

def get_test_data_set(
		obs_list : typing.Optional[typing.Sequence[str]] = None,
		filter_function : typing.Callable = lambda x: True,
		aslice : slice = slice(None),
		verbose : int = 0
	):
	obs_list = list(filter(filter_function, obs_list if obs_list is not None else get_test_data_set_list()))
	for obs_path in obs_list[aslice]:
		if verbose > 0: print(f'INFO: "get_test_data_set()" yielding data from {obs_path=!r}')
		yield(ut.fits.get_img_from_fits(obs_path, get_test_data_sci_ext_name(obs_path)))

if __name__=='__main__':
	
	real_img, dirty_img, test_psf = create_test_images(psf=create_psf_image( ((0,0,12),(3,5,6),(-2,-4,4)) ) )
	
	guess_img = np.zeros((400,300))
	#guess_img += 1E-2
	
	noise_est = 1E-1
	alpha = 1E+0
	model = 5E-1
	
	n = 40
	
	values = np.zeros((3,n))
	sq_vals = np.linspace(0,2,n)
	
	best_sq_val = np.nan
	best_obj_func_sum = np.inf
	for i, sq_val in enumerate(sq_vals):
		guess_img[130:180,130:180] = sq_val

		chi_sq_img = 0.5*chi_sq(sp.signal.fftconvolve(guess_img, test_psf, mode='same'), dirty_img, noise_est)
		entropy_img = entropy_adj(guess_img, model)
		obj_func_img = obj_func(guess_img, dirty_img, noise_est, alpha, test_psf, model, entropy_func=entropy_adj)
	
		if best_obj_func_sum > np.nansum(obj_func_img):
			best_obj_func_sum = np.nansum(obj_func_img)
			best_sq_val = sq_val
	
		values[:,i] = (np.nansum(chi_sq_img), np.nansum(entropy_img), np.nansum(obj_func_img))
	
	guess_img[130:180,130:180] = best_sq_val
	chi_sq_img = 0.5*chi_sq(sp.signal.fftconvolve(guess_img, test_psf, mode='same'), dirty_img, noise_est)
	entropy_img = entropy_adj(guess_img, model)
	obj_func_img = obj_func(guess_img, dirty_img, noise_est, alpha, test_psf, model, entropy_func=entropy_adj)

	
	print('DEBUG: ',sq_val, np.nansum(chi_sq_img), np.nansum(entropy_img), np.nansum(obj_func_img))
	
	
	
	if True:
		(nr, nc, s) = (1,3,6)
		f2 = plt.figure(figsize=[x*s for x in (nc,nr)])
		a2 = f2.subplots(nr, nc, squeeze=False)
		
		f2.suptitle('Minimising objective function for test square region')
		
		a2[0,0].plot(sq_vals, values[0])
		a2[0,0].set_title('0.5*chi_sq')
		
		a2[0,1].plot(sq_vals, values[1])
		a2[0,1].set_title('entropy_adj')
		
		a2[0,2].plot(sq_vals, values[2])
		a2[0,2].set_title(f'obj_func := 0.5*chi_sq - {alpha}*entropy_adj')
	
	
		(nr, nc, s) = (2,4,6)
		f1 = plt.figure(figsize=[x*s for x in (nc,nr)])
		a1 = f1.subplots(nr, nc, squeeze=False)
		
		f1.suptitle('Inputs and inbetweens with lowest obj_func')
		
		a1[0,0].set_title('test_psf')
		a1[0,0].imshow(test_psf, origin='lower')
		
		a1[0,1].set_title('real_img')
		a1[0,1].imshow(real_img, origin='lower')
		
		a1[0,2].set_title('dirty_img')
		a1[0,2].imshow(dirty_img, origin='lower')
		
		a1[1,0].set_title('chi_sq_img')
		a1[1,0].imshow(chi_sq_img, origin='lower')
		
		a1[1,1].set_title('entropy_img')
		a1[1,1].imshow(entropy_img, origin='lower')
		
		a1[1,2].set_title('obj_func_img')
		a1[1,2].imshow(obj_func_img, origin='lower')
		
		a1[1,3].set_title('guess_img')
		a1[1,3].imshow(guess_img, origin='lower')
		
	plt.show()
	
	
	"""
	results = []
	
	#dirty_img[10:20,:] = np.nan
	
	
	cb = lambda x: print('STEP')
	res = scipy.optimize.minimize(obj_func, np.zeros_like(dirty_img).ravel(), args=(dirty_img.ravel(), 1, 1E-1, test_psf.ravel()), method='Powell',
								options=dict(disp=True), callback=cb)
	
	components = res.x.reshape(dirty_img.shape)
	
	
	plt.imshow(dirty_img)
	plt.show()
	plt.imshow(components)
	plt.show()
	"""
	
	#%%
	import utilities.plt
	
	f3, a3 = ut.plt.create_figure_with_subplots(2,4)
	
	ms = np.array([-10, -1.1, 0.001, 0.5, 1.0, 2.0])
	h_lim = (0-2*np.sign(np.min(ms))*np.min(ms), 0+2*np.sign(np.max(ms))*np.max(ms))
	h = np.linspace(h_lim[0], h_lim[1], 51)
	
	for m in ms:
		a3[0,0].plot(h, entropy_pos(h,m), label=f'model = {m}')
	a3[0,0].set_title('entropy_pos')
	
	for m in ms:
		a3[0,1].plot(h, entropy_pos_neg(h,m))
	a3[0,1].set_title('entropy_pos_neg')
	
	for m in ms:
		a3[0,2].plot(h, entropy_rel(h,m))
	a3[0,2].set_title('entropy_rel')
	
	for m in ms:
		a3[0,3].plot(h, entropy_stat(h,m))
	a3[0,3].set_title('entropy_stat')
	
	for m in ms:
		a3[1,0].plot(h, entropy_jack(h,m))
	a3[1,0].set_title('entropy_jack')
	
	for m in ms:
		a3[1,1].plot(h, -chi_sq(h,m,1))
	a3[1,1].set_title('chi_sq_hack')
	
	for m in ms:
		a3[1,2].plot(h, entropy_exp(h,m))
	a3[1,2].set_title('entropy_exp')
	
	for m in ms:
		a3[1,3].plot(h, entropy_adj(h,m,10))
	a3[1,3].set_title('entropy_adj')

	f3.legend(*a3[0,0].get_legend_handles_labels())
	plt.show()
	
