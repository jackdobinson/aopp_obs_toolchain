#!/usr/bin/env python3
"""
Useful routines for curve fitting
"""

import numpy as np
import scipy as sp
import scipy.optimize
import matplotlib.pyplot as plt

def linear_least_squares(y, X, w=None, mode='direct', a0=None, **kwargs):
	"""
	Calculates weighted linear least squares for observations y and independent
	variables X
	"""
	if w is None:
		w = np.ones_like(y)
		
	if mode == 'direct':
		A = np.einsum('ij,ik,i', X, X, w) # A_jk = \sum_i X_ij X_ik w_i
		try:
			A_inv = np.linalg.inv(A)
		except:
			return(np.zeros((0,)),np.zeros((0,)),np.zeros((0,0)))
		a = np.matmul(A_inv, np.einsum('i,ij,i',y, X, w)) # a_k = A_jk^{-1} \sum_i y_i X_ij w_i
	elif mode == 'scipy.optimize.minimize':
		a0 = np.ones((X.shape[1])) if a0 is None else a0
		result = sp.optimize.minimize(lambda a: np.sum(w*(y-np.matmul(X,a))**2), a0, **kwargs)
		#print('\n',result)
		a = result.x
	else:
		print(f'ERROR: Unknown mode "{mode}".')
		raise NotImplementedError

	r = y - np.matmul(X,a)
	
	XtX = np.matmul(X.T, X)
	XtX_inv = np.linalg.inv(XtX)
	cov_mat = np.var(r)*XtX_inv
	return(a, r, cov_mat)

def worst_fits(X, a, cov_mat):
	"""
	Finds the 'worst fit' for independent variables X and coefficients a with 
	a covariance matrix cov_mat
	"""
	wf_max = np.full((x.shape[0],), fill_value=-np.inf)
	wf_min = np.full_like(wf_max, fill_value=np.inf)
	wf_test = np.zeros_like(wf_max)
	
	# build +/- map
	fmt_str = '{'+':0'+f'{len(a)}' + 'b}'
	permutations = np.array([[x if x==1 else -1 for x in map(int,fmt_str.format(y))] for y in range(2**len(a))])
		
	errors = np.sqrt(np.diag(cov_mat))
	for p in permutations:
		wf_test = np.matmul(X, a+p*errors)
		max_idxs = wf_test > wf_max
		min_idxs = wf_test < wf_min
		wf_max[max_idxs] = wf_test[max_idxs]
		wf_min[min_idxs] = wf_test[min_idxs]
	return(np.stack([wf_max, wf_min]))

if __name__ == '__main__':
	x_min, x_max = (-10,10)
	xx = np.linspace(x_min, x_max, 50)
	x = np.stack([xx**0, xx**1, xx**2, xx**3,xx**4]).T
	a = (10,-20,10,-2,0)
	y = np.matmul(x,a)
	
	#w = np.random.normal(300, 100, size=y.shape)
	e_scale = 50
	e = (xx-0.5*(x_min+x_max))**2/(0.5*(x_max-x_min))*e_scale
	
	y_corrupt = y + np.random.normal(0, e, size=None)
	
	w = 1/e**2
	n_fit = 2
	
	p_fit1, r1, c1 = linear_least_squares(y_corrupt, x[:,:n_fit], w=None)
	p_fit_w2, r2, c2 = linear_least_squares(y_corrupt, x[:,:n_fit], w=w)
	p_sp_fit3, r3, c3 = linear_least_squares(y_corrupt, x[:,:n_fit], w=None, mode='scipy.optimize.minimize', a0=(5,8))
	p_sp_fit_w4, r4, c4 = linear_least_squares(y_corrupt, x[:,:n_fit], w=w, mode='scipy.optimize.minimize')
	
	worst_fits1 = worst_fits(x[:,:n_fit], p_fit1, c1)
	worst_fits2 = worst_fits(x[:,:n_fit], p_fit_w2, c2)
	worst_fits3 = worst_fits(x[:,:n_fit], p_sp_fit3, c3)
	worst_fits4 = worst_fits(x[:,:n_fit], p_sp_fit_w4, c4)
	
	(nc,nr,s) = (2,2,6)
	f1 = plt.figure(figsize=(nc*s,nr*s))
	a1 = f1.subplots(nr,nc, squeeze=False)
	
	
	a1[0,0].plot(xx, y, label='real')
	a1[0,0].plot(xx, e, label='error', ls=':')
	a1[0,0].scatter(xx,y_corrupt,color='tab:orange', label='corrupt', marker='.')
	
	a1[0,0].plot(xx, np.matmul(x[:,:n_fit],p_fit1), label='fit', color='tab:red')
	a1[0,0].fill_between(xx, worst_fits1[0], worst_fits1[1], alpha=0.3, color='tab:red')
	
	a1[0,0].plot(xx, np.matmul(x[:,:n_fit],p_fit_w2), label='fit weights', color='tab:green')
	a1[0,0].fill_between(xx, worst_fits2[0], worst_fits2[1], alpha=0.3, color='tab:green')
	a1[0,0].legend()
	
	
	a1[0,1].plot(xx, y, label='real')
	a1[0,1].plot(xx, e, label='error', ls=':')
	a1[0,1].scatter(xx,y_corrupt,color='tab:orange', label='corrupt', marker='.')
	
	a1[0,1].plot(xx, np.matmul(x[:,:n_fit],p_sp_fit3), label='scipy fit', ls='--', color='tab:cyan')
	a1[0,1].fill_between(xx, worst_fits3[0], worst_fits3[1], alpha=0.3, color='tab:cyan')
	
	a1[0,1].plot(xx, np.matmul(x[:,:n_fit],p_sp_fit_w4), label='scipy fit weights', ls='--', color='tab:pink')
	a1[0,1].fill_between(xx, worst_fits4[0], worst_fits4[1], alpha=0.3, color='tab:pink')
	a1[0,1].legend()
	
	
	a1[1,0].scatter(xx, r1, marker='.', color='tab:red', label='fit residual')
	a1[1,0].scatter(xx, r2, marker='.', color='tab:green', label='fit weights residual')
	a1[1,0].legend()
	
	
	a1[1,1].scatter(xx, r3, marker='.', color='tab:cyan', label='scipy fit residual')
	a1[1,1].scatter(xx, r4, marker='.', color='tab:pink', label='scipy fit weights residual')
	a1[1,1].legend()
	
	plt.show()
	
	
	
