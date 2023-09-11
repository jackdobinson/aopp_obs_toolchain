#!/usr/bin/env python3
"""
Implementation of Singular Value Decomposition

Mostly consists of helper functions, as numpy does heavy lifting.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import utilities as ut
import utilities.np 
import utilities.plt


def svd(a, n=None):
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

def decompose_to_matricies(u, s, v_star):
	n = min(s.shape) # number of singular vectors to decompose into
	decomp = np.diag(s)[:,None,None]*(u.T[:,:,None] @ v_star[:,None,:])
	return(decomp)
	"""
	decomp = np.zeros((n,u.shape[0],v_star.shape[0]))
	u_dash, v_dash = np.zeros((u.shape[0],1)), np.zeros((1,v_star.shape[1]))
	for i in range(n):
		u_dash[:,0] = u[:,i]
		v_dash[0,:] = v_star[i,:]
		decomp[i,:,:] = s[i,i] * u_dash @ v_dash
	return(decomp)
	"""

def plot(u, s, v_star, inmat, decomp, recomp_n=None, f1=None, a1=None):
        print(u.shape)
        print(s.shape)
        print(v_star.shape)
        if recomp_n  is None:
                recomp_n = decomp.shape[0]
        recomp = np.sum(decomp[:recomp_n], axis=(0))
        reconstruct = u @ s @ v_star

        # plot SVD of image
        if f1 is None or a1 is None:
                f1, a1 = ut.plt.figure_n_subplots(6, figure=f1)

        a1 = a1.ravel()
        [ut.plt.remove_axes_ticks_and_labels(ax) for ax in a1]

        f1.suptitle('Singular value decomposition U S V* of input maxtrix')

        a1[0].set_title('U matrix')
        a1[0].imshow(u)

        a1[1].set_title('Diagonal elements of S (decending order)')
        a1[1].plot(np.diag(s))
        ut.plt.remove_axes_ticks_and_labels(a1[1], state=True)
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

        # Plot components of image decomposition
        n = min(decomp.shape[0], 2*recomp_n)
        f1, a1 = ut.plt.figure_n_subplots(n)
        a1 = a1.ravel()
        f1.suptitle(f'First {n} svd components X_i [M = sum(X_i)] (of {decomp.shape[0]})')
        for i, ax in enumerate(a1):
                ut.plt.remove_axes_ticks_and_labels(ax)
                ax.set_title(f'i = {i}', y=0.9)
                ax.imshow(decomp[i])
        return

#%%
if __name__=='__main__':
        # "arguments"
        recomp_n = 3

        # get example data
        try:
                import PIL
                print('Creating mandelbrot fractal as test case')
                obs = np.asarray(PIL.Image.effect_mandelbrot((60,50),(0,0,1,1),100))
        except ImportError:
                print('Creating random numbers as test case')
                obs = np.random.random((60,50))
        #obs, psf = fitscube.deconvolve.helpers.get_test_data()

        # get singular value decomposition
        u, s, v_star = np.linalg.svd(obs)

        # get svd from numpy into full form
        s = rect_diag(s, (u.shape[1],v_star.shape[1]))

        decomp = decompose_to_matricies(u, s, v_star)

        # %% plot singular value decomposition

        # setup plotting defaults
        mpl.rc_file(mpl.matplotlib_fname())

        plot(u, s, v_star, obs, decomp, recomp_n)
        plt.show()


