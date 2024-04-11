#!/usr/bin/env python3
"""
Implementation of Singular Value Decomposition

Mostly consists of helper functions as numpy does heavy lifting. Run the module as __main__
to see and example of it in action.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import aopp_deconv_tool.cfg.logs
import aopp_deconv_tool.plot_helper as plot_helper

_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'INFO')


def svd(a : np.ndarray, n : None | int = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""
	Perform single value decomposition on array `a`, see https://en.wikipedia.org/wiki/Singular_value_decomposition
	
	# ARGUMENTS #
		a : np.ndarray
			Array to perform SVD on
		n : int | None
			Number of SVD components to report
	
	# RETURNS #
		evecs : np.ndarray
			"Left side" singular vectors, "U" in A = U S V*
		singular_values : np.ndarray
			Singular values of "left side" and "right side" singular vectors, S in A = U S V*
		fvecs : np.ndarray
			"Right side" singular vectors, V in A = U S V*
	"""
	evals, evecs = np.linalg.eig(a @ a.T)
	e_idxs = np.argsort(evals)[::-1][:n]
	evals = evals[e_idxs]
	evecs = evecs.T[e_idxs].T
	fvecs = a.T @ (evecs/np.sqrt(evals))
	return(evecs, np.diag(np.sqrt(evals)), fvecs.T)


def rect_diag(a, shape):
	"""
	Diagonalise vector a into a matrix of non-rectangular shape
	"""
	d = np.zeros(shape)
	for i in range(min(*shape, a.size)):
		d[i,i]=a[i]
	return(d)

def decompose_to_matricies(u : np.ndarray, s : np.ndarray, v_star : np.ndarray, small : bool = True) -> np.ndarray:
	"""
	Using the results of a singular value decomposition (M = U S V*), returns the matrix M_i =  SUM(s_i * u_i * v_i)
	"""
	if small:
		n = min(s.shape) # number of singular vectors to decompose into
		decomp = np.diag(s)[:,None,None]*(u.T[:n,:n,None] @ v_star[:n,None,:n])
	else:
		decomp = np.diag(s)[:,None,None]*(u.T[:,:,None] @ v_star[:,None,:])
		
	return decomp

def plot(u : np.ndarray, s : np.ndarray, v_star : np.ndarray, inmat : np.ndarray, decomp : np.ndarray, recomp_n=None, f1=None, a1=None):
	"""
	Visualisation of the SVD process, where a matrix M is decomposed into components M =  U S V*
	
	# ARGUMENTS #
		u : np.ndarray
			left singular vectors of some matrix M
		s : np.ndarray
			singular values of some matrix M
		v : np.ndarray
			right singular vectors of some matrix M
		inmat : np.ndarray
			The input matrix, M that `u`, `s`, `v` come from
		decom : np.ndarray
			The M_i decomposition of matrix M, where M =  SUM(M_i) = SUM(s_i * u_i * v_i)
		recomp_n: int | None
			The number of decomposed matrices M_i to recombine for visualisation purposes
		f1 : matplotlib.figure.Figure | None
			Figure to put the plot axes in
		a1 : Sequence[matplotlib.axes.Axes] | None
			6 Axes to plot the data in
	"""
	_lgr.debug(f'{u.shape=}')
	_lgr.debug(f'{s.shape=}')
	_lgr.debug(f'{v_star.shape=}')
	if recomp_n  is None:
		recomp_n = decomp.shape[0]
	recomp = np.sum(decomp[:recomp_n], axis=(0))
	reconstruct = u @ s @ v_star

	_lgr.debug(f'Data ready for plotting')
	
	
	# plot SVD of image
	if f1 is None or a1 is None:
		f1, a1 = plot_helper.figure_n_subplots(6, figure=f1)

	_lgr.debug('figures and axes available')

	a1 = a1.ravel()
	[plot_helper.remove_axes_ticks_and_labels(ax) for ax in a1]

	f1.suptitle('Singular value decomposition U S V* of input maxtrix')

	a1[0].set_title('U matrix')
	a1[0].imshow(u)

	a1[1].set_title('Diagonal elements of S (decending order)')
	a1[1].plot(np.diag(s))
	plot_helper.remove_axes_ticks_and_labels(a1[1], state=True)
	a1[1].set_xlabel('sv idx')
	a1[1].set_ylabel('sv')
	a1[1].axvline(recomp_n, color='red', ls='--')

	a1[2].set_title('V* matrix')
	a1[2].imshow(v_star)

	a1[3].set_title(f'Input matrix\nclim [{np.min(inmat):07.2E},{np.max(inmat):07.2E}]')
	a1[3].imshow(inmat)

	a1[4].set_title(f'Reconstruction from M = U S V*\nclim [{np.min(reconstruct):07.2E},{np.max(reconstruct):07.2E}]')
	a1[4].imshow(reconstruct)

	a1[5].set_title(f'Recomposition from sum(X)_0^{recomp_n}\nclim [{np.min(recomp):07.2E},{np.max(recomp):07.2E}]')
	a1[5].imshow(recomp)


	n_decomp_components_to_plot = 8
	_lgr.debug(f'Plotting {n_decomp_components_to_plot} components of image decomposition')
	# Plot components of image decomposition
	n = min(decomp.shape[0], n_decomp_components_to_plot)
	f2, a2 = plot_helper.figure_n_subplots(n)
	_lgr.debug(f'{f2=} {a2=}')
	a2 = a2.ravel()
	f2.suptitle(f'First {n} svd components X_i [M = sum(X_i)] (of {decomp.shape[0]})')
	for i, ax in enumerate(a2):
		plot_helper.remove_axes_ticks_and_labels(ax)
		ax.set_title(f'i = {i}', y=0.9)
		ax.imshow(decomp[i])
	return


if __name__=='__main__':
	# "arguments"
	recomp_n = None

	# get example data
	try:
		import PIL
		_lgr.info('Creating mandelbrot fractal as test case')
		obs = np.asarray(PIL.Image.effect_mandelbrot((60,50),(0,0,1,1),100))
	except ImportError:
		_lgr.info('Creating random numbers as test case')
		obs = np.random.random((60,50))
	#obs, psf = fitscube.deconvolve.helpers.get_test_data()

	_lgr.info('get singular value decomposition')
	u, s, v_star = np.linalg.svd(obs)

	_lgr.info('get svd from numpy into full form')
	s = rect_diag(s, (u.shape[1],v_star.shape[1]))

	_lgr.info('get decomposed matrices from U S V*')
	decomp = decompose_to_matricies(u, s, v_star)

	_lgr.info('plot singular value decomposition')
	plot(u, s, v_star, obs, decomp, recomp_n)
	plt.show()


