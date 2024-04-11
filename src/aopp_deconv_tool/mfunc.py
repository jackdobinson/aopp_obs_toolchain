"""
Module containing small mathematical functions that are useful in lots of contexts



## Helpers for deconvolution routines ##

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

import numpy as np
import scipy as sp
import scipy.signal
from typing import Callable

Number = float | int


def logistic_function(
		x : Number | np.ndarray, 
		left_limit : Number = 0, 
		right_limit : Number = 1, 
		transition_scale : Number = 1, 
		center : Number = 0
	) -> Number | np.ndarray:
	return (right_limit-left_limit)/(1+np.exp(-(np.e/transition_scale)*(x-center))) + left_limit



# Different versions of generalised least squares

def generalised_least_squares(
		components 	: np.ndarray, 
		dirty_img 	: np.ndarray, 
		cov_mat 	: np.ndarray | float | int, 	# variances for the errors on data
		response 	: np.ndarray
		) -> np.ndarray:
		mismatch = sp.signal.fftconvolve(components, response, mode='same') - dirty_img
		return(mismatch*mismatch*(1.0/cov_mat))

def generalised_least_squares_preconv(
		estimate 	: np.ndarray, 
		dirty_img 	: np.ndarray, 
		cov_mat 	: np.ndarray | float | int, 	# variances for the errors on data
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


# Different versions of entropy

def entropy_pos_neg(
		components : np.ndarray,
		error : np.ndarray | float | int
		) -> np.ndarray:
	psi = np.sqrt(components**2 + 4*error**2)
	return(psi - 2*error - components*np.log((psi + components)/(2*error)))

def entropy_pos(
		components : np.ndarray,
		model : np.ndarray | float | int
		) -> np.ndarray:
	return(components - model - components*np.log(components/model))

def entropy_rel(
		components : np.ndarray,
		model : np.ndarray | float | int
		) -> np.ndarray:
	return(-components*np.log(components/model))

def entropy_adj(
		components : np.ndarray,
		model : np.ndarray | float | int,
		error : np.ndarray | float | int = 1 # this could be absorbed into alpha if we need to only have 2 arguments
		):
	return(entropy_pos_neg(components-model,error))



def regularised_least_squares(
		components 			: np.ndarray,
		dirty_img 			: np.ndarray,
		error 				: np.ndarray | float | int,
		alpha 				: np.ndarray | float | int, # no reason we can't have different values for different pixels
		response			: np.ndarray, # usually an instrumental PSF
		model 				: np.ndarray | float | int | None										= None,
		regularising_func 	: Callable[[np.ndarray, np.ndarray | float | int], np.ndarray] 	= entropy_adj,
		) -> np.ndarray:
	if model is None:
		model = error
	return(0.5*generalised_least_squares(components, dirty_img, error**2, response) - alpha*regularising_func(components, model))


