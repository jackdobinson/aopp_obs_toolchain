#!/usr/bin/env python3
"""
Contains routines for calculating Minnaert limb darkeing coefficients.

Minnaert limb darkening equation:

			 k  k-1
 I     /I\  u  u
--- = |---|  0                                                              (1)
 F     \F/
		 0

Where
	u = cos(zenith)
	u_0 = cos(solar zenith)
	k, (I/F)_0 are fixed parameters
	I/F is the flux at the zenith angle we are interested in

In our case, the solar zenith angle and the zenith angle are approximately the 
same as the earth and sun are very close when seen from Neptune

Taking the logarithm of (1) gives:

	ln[u(I/F)] = ln[(I/F)_0] + k ln[u_0 u]                                  (2)

-------------------------------------------------------------------------------
"""

import sys, os
import numpy as np
import scipy as sp
import scipy.optimize
import matplotlib.pyplot as plt

def get_coefficients_v_wavelength(u_0, u, IperF):
	"""
	Runs "get_coefficients()" for each wavelength and combines the output into a nice format

	# ARGUMENTS #
		u_0 [n]
			<float, array> cos(solar zenith)
		u [n]
			<float, array> cos(zenith)
		IperF [n]
			<float, array> Radiance (or flux, some brightness measure)

	# RETURNS #
		IperF0s
			<array, float> The values of IperF0 for each wavelength
		ks
			<array, float> The values of k for each wavelength
		log_IperF0s_var
			<array, float> The variance on the IperF0s values
		ks_var
			<array, float> The variance on the ks values
		n_valid_idxs
			<array, float> The number of datapoints used when performing the 
			minnaert fit 
		
	"""
	# get the number of wavelengths (assume wavelengths are 0th axis)
	wavshape = IperF.shape[0]
	
	# create numpy arrays to hold return data (faster than lists)
	IperF0s = np.full(wavshape, fill_value = np.nan)
	ks = np.full(wavshape, fill_value = np.nan)
	log_IperF0s_var = np.full(wavshape, fill_value=np.nan)
	ks_var = np.full(wavshape, fill_value=np.nan)
	n_valid_idxs = np.full(wavshape, fill_value=0)
	
	# loop over the wavelength axis (assume 0th) and get a minnaert fit
	# we only need some of the output so just grab the important parts.
	for i in range(wavshape):
		result = get_coefficients(u_0[i,:], u[i,:], IperF[i,:])
		IperF0s[i] = result[0]
		ks[i] = result[1]
		log_IperF0s_var[i] = result[8][0,0]
		ks_var[i] = result[8][1,1]
		n_valid_idxs[i] = np.nansum(result[7])
	
	return(IperF0s, ks, log_IperF0s_var, ks_var, n_valid_idxs)

def get_coefficients(u_0, u, IperF, weight=None):
	"""
	Uses (2) to find k and (I/F)_0 for a set of u_0, u, (I/F) values

	re-write as a matrix equation of the form y=Ax, for each i^th triple of
	u_0, u, IperF we have:
		y = {... log[u(I/F)]_i ...}
		x = {log[(I/F)_0], k}
		A = {... 1_i ..., ... log[u_0 u]_i ...}
	We want to find the x that best predicts y.

	# ARGUMENTS #
		u_0 [n]
			<float, array> cos(solar zenith)
		u [n]
			<float, array> cos(zenith)
		IperF [n]
			<float, array> Radiance (or flux, some brightness measure)

	# RETURNS #
		IperF0
			<float> The minnaert parameter that tells you the brightness at zenith=0
		k
			<float> The minnaert parameter that determines limb brightening vs limb 
			darkening. Should be between 0 and 1 (0.5 is no brightening or darkening).
		residual [m]
			<float, array> residual = y - Ax for the found values of x
		rank
			<int> Rank of matrix A
		s 
			<int, array> Singular values of A
		log_u0u [n]
			<float, array> The values of log[u_0 u] in (2)
		log_uIperF [n]
			<float, array> The values of log[u(I/F)] in (2)
		valid_idxs [m]
			<bool, array> The indices of log_uIperF that are not NAN or INF 
			(as IperF could be less than zero due to background subtraction)
		cov_mat [2,2]
			<float, array> The covariance matrix for IperF0 and k
	
	# EXAMPLE #
		IperF0, k, residual, rank, s, log_u0u, log_uIperF, valid_idxs, cov_mat = get_coefficients(u0, u, IperF)
	"""
	
	# assume we have numpy arrays for all inputs
	# masked arrays don't play nice with scipy routines sometimes
	if type(IperF) == np.ma.MaskedArray:
		IperF = IperF.filled(fill_value=np.nan)

	# get inputs into the form of (2)
	log_u0u = np.log(u_0*u)
	log_uIperF = np.log(u*IperF)
	
	# sometimes will have -ve I/F values (noise and backgrounding) so find the valid indexes
	valid_idxs = (~np.isnan(log_uIperF)) & (~np.isinf(log_uIperF))
	
	# only operate on the valid indexes to avoid wierdness
	log_u0u_valid = log_u0u[valid_idxs]
	log_uIperF_valid = log_uIperF[valid_idxs]
	
	if np.unique(log_u0u_valid).size < 3:
		# then we can't get meaningful results so just return junk
		return(np.nan, np.nan, np.array([]), np.nan, np.nan, log_u0u, log_uIperF, np.array([]), np.array([[np.nan,np.nan],[np.nan,np.nan]]))
	
	
	# find matrix A, A = {... 1_i ..., ... log[u_0 u]_i ...}
	A = np.vstack([np.ones_like(log_u0u_valid), log_u0u_valid]).T
	
	if A.shape == (0,2):
		# then we didn't have any valid indexes, so the whole thing is NANs or INFs so just return junk
		return(np.nan, np.nan, np.nan, np.nan, np.nan, log_u0u, log_uIperF, [], np.full((2,2), fill_value=np.nan))
	
	# perform least squares fit to determine vector x
	# numpy version cannot constrain to bounds, so use scipy version.
	
	#x, residual, rank, s = np.linalg.lstsq(A, log_uIperF_valid, rcond=None)
	
	# WEIGHTS TESTING
	if weight is None:
		w = np.ones_like(log_uIperF_valid)
	elif weight == 'adjust_for_logscale':
		# adjusts for the logarithmic scale so that fitting in logspace is the same as 
		# fitting in linear space.
		# Have to use 1/err^2 for weighting, in this case we want weighting to be
		# large for more +ve values and low for more -ve values, so err should be
		# 1/(point - smallest point),
		w = (log_uIperF_valid - np.min(log_uIperF_valid))**2
	else:
		print(f'ERROR: Unknown value {weight} passed to argument "weight"')
		raise NotImplementedError

	# bounds in form (IperF0 min, k min), (IperF0 max, k max)
	bounds = ((-np.inf,0),(np.inf,1))
	result = sp.optimize.lsq_linear(A*w[:,None], w*log_uIperF_valid, bounds=bounds)
	x = result.x
	residual = result.fun
	rank = np.nan
	s = np.nan
	
	# get the residuals of the fit
	fit_residuals = np.matmul(A, x) - log_uIperF_valid

	# == THOUGHTS ON COVARIANCE STUFF ==
	# note that cov_mat is the covariance of log(IperF0) and k, so need to take this into account
	# Assuming that the errors on log(IperF0) are distributed normally, then the errors on IperF0 are
	# distributed according to a log-normal distribution. So the variance of those errors is
	# sigma_IperF0 = np.exp(cov_mat[0,0] - 1)*np.exp(cov_mat[0,0])
	# see https://en.wikipedia.org/wiki/Log-normal_distribution
	# and assume that the mean of the errors on log(IperF0) is zero
	# BUT I DONT THINK THAT log(IperF0) is normally distributed, but IperF0 is,
	# Therefore may have to work out the errors by brute force, something like (exp(x+sqrt(var(x))) - exp(x-sqrt(var(x))))^2
	
	# == GETTING VARIANCE OF (I/F)0 FROM log[(I/F)0] ==
	# The error propagation that will give me the variance of IperF0 starting 
	# from log(IperF0) is confusing. Instead I can brute-force the problem
	# by calculating
	# IperF0s_var = (np.exp(np.log(IperF0s)+np.sqrt(log_IperF0s_var)) 
	#               - np.exp(np.log(IperF0s)-np.sqrt(log_IperF0s_var)))**2	
	
	AtA = np.matmul(A.transpose(), A)
	mat_inverted = False
	while not mat_inverted:
		try:
			AtA_inv = np.linalg.inv(AtA)
		except np.linalg.LinAlgError:
			# if we can't invert the matrix, make a small alteration until it's non-singular
			AtA += np.random.rand(2,2)*1E-10
			continue
		mat_inverted=True
		
	cov_mat = np.var(fit_residuals)*AtA_inv # see https://www.stat.purdue.edu/~boli/stat512/lectures/topic3.pdf

	return(np.exp(x[0]), x[1], residual, rank, s, log_u0u, log_uIperF, valid_idxs, cov_mat)
	
def get_coefficients_curvefit(u_0, u, IperF):
	# assume we have numpy arrays for all inputs
	# masked arrays don't play nice with scipy routines sometimes
	if type(IperF) == np.ma.MaskedArray:
		IperF = IperF.filled(fill_value=np.nan)

	# get inputs into the form of (2)
	log_u0u = np.log(u_0*u)
	log_uIperF = np.log(u*IperF)
	
	# sometimes will have -ve I/F values (noise and backgrounding) so find the valid indexes
	valid_idxs = (~np.isnan(log_uIperF)) & (~np.isinf(log_uIperF))
	
	# only operate on the valid indexes to avoid wierdness
	log_u0u_valid = log_u0u[valid_idxs]
	log_uIperF_valid = log_uIperF[valid_idxs]
	
	if np.unique(log_u0u_valid).size < 3:
		# then we can't get meaningful results so just return junk
		return(np.nan, np.nan, np.array([]), np.nan, np.nan, log_u0u, log_uIperF, np.array([]), np.array([[np.nan,np.nan],[np.nan,np.nan]]))
	
	curve_to_fit = lambda log_u0u, IperF0, k: np.exp(np.log(IperF0) + k*log_u0u)
	
	# bounds in form (IperF0 min, k min), (IperF0 max, k max)
	bounds = ((-np.inf,0),(np.inf,1))
	
	# Initial parameter guesses
	p0 = [np.nanmax(IperF), 0.5]
	
	# perform curve fit
	popt, pcov = sp.optimize.curve_fit(curve_to_fit, log_u0u_valid, np.exp(log_uIperF_valid), p0=p0, bounds=bounds)
	
	# change pcov to hold variance of log_IperF0
	pcov[0,0] = np.abs(0.5*(np.log(popt[0]+pcov[0,0])-np.log(popt[0]) + np.log(popt[0]-pcov[0,0])-np.log(popt[0])))
	
	#IperF0, k, residual, rank, s, log_u0u, log_uIperF, valid_idxs, cov_mat = get_coefficients_curvefit(u0, u, IperF)
	return(popt[0], popt[1], np.exp(log_uIperF_valid) - curve_to_fit(log_u0u_valid, *popt), np.nan, np.nan, log_u0u, log_uIperF, valid_idxs, pcov)
	

def minnaert_test_grid(kk_reals, IIperF0_reals, coefficient_fitting_function, fractional_standard_deviation=0.3, n_fit_points=300, suptitle='Minnaert Test Case', seed=676894):
	np.random.seed(seed) # ensure that we are using the same seed every time we call this function
	plot_to_inspect = np.random.randint(0,n_pairs**2-1,1)[0] #5257
	
	k_results = np.zeros((n_pairs, n_pairs))
	k_results_var = np.zeros((n_pairs, n_pairs))
	
	IperF0_results = np.zeros((n_pairs, n_pairs))
	IperF0_results_var = np.zeros((n_pairs, n_pairs))
	
	err_mag_est = np.zeros((n_pairs, n_pairs))
	
	for i, (k_real, IperF0_real) in enumerate(zip(kk_reals.ravel(), IIperF0_reals.ravel())):
		# Create perfect data
		IperF = IperF0_real*(u0**k_real)*(u**(k_real-1))
		
		# Corrupt perfect data with some noise
		IperF += np.random.normal(0, fractional_standard_deviation*IperF0_real, u.shape)
		
		# Find the minnaert coefficients for the dummy data
		IperF0, k, residual, rank, s, log_u0u, log_uIperF, valid_idxs, cov_mat = coefficient_fitting_function(u0, u, IperF)
		
		# extract variances from covariance matrix
		log_IperF0_var = cov_mat[0,0]
		k_var = cov_mat[1,1]
		
		# get a set of points that span the log_u0u range
		log_u0u_grid = np.linspace(np.nanmin(log_u0u), np.nanmax(log_u0u), n_fit_points)
		
		# Calculate the line of best fit
		log_uIperF_fit = np.log(IperF0) + k*log_u0u_grid
		
		# calculate all possible combinations of the line of best fit +/- 1 standard 
		# deviation of the k and IperF0 coefficients
		worst_log_uIperF_fits = np.zeros((n_fit_points, 4))
		k_lims = (k + np.sqrt(k_var), k - np.sqrt(k_var))
		log_IperF0_lims = (np.log(IperF0) + np.sqrt(log_IperF0_var), np.log(IperF0) - np.sqrt(log_IperF0_var))
		_i = 0
		for _k in k_lims:
			for _l in log_IperF0_lims:
				worst_log_uIperF_fits[:,_i] = _l + _k*log_u0u_grid
				_i += 1
		
		# get the minimum and maximum 'worst fit' lines
		wf_min = np.nanmin(worst_log_uIperF_fits, axis=-1)
		wf_max = np.nanmax(worst_log_uIperF_fits, axis=-1)
		
		# get error magnitude estimate
		error_magnitude_estimate = np.sum(wf_max - wf_min)
		err_mag_est[i//n_pairs, i%n_pairs] = error_magnitude_estimate
		
		# store values in output arrays
		IperF0_results[i//n_pairs, i%n_pairs] = IperF0
		k_results[i//n_pairs, i%n_pairs] = k
		
		# store variances in output arrays
		IperF0_results_var[i//n_pairs, i%n_pairs] = log_IperF0_var # DEBUGGING: Do I need to alter this?
		k_results_var[i//n_pairs, i%n_pairs] = k_var
		
		#if i<100:
		#	print(i%n_pairs, i//n_pairs)
		#	print(k_real, IperF0_real)
		#	print(k, IperF0)
		#else:
		#	sys.exit()
		
		if i == plot_to_inspect:
			# calculate real relationship for comparision
			log_uIperF_real = np.log(IperF0_real) + k_real*log_u0u_grid
			# create a plot
			nr, nc, s = (2,2, 6)
			f0 = plt.figure(figsize=(nc*s, nr*s))
			a0 = f0.subplots(nr,nc, squeeze=False)
			
			# fill plot with data
			a0[0,0].plot(log_u0u, log_uIperF, ',')
			a0[0,0].plot(log_u0u_grid, log_uIperF_fit, '-', color='tab:pink', label='Minnaert fitted line')
			a0[0,0].fill_between(log_u0u_grid, wf_min, wf_max, color='tab:pink', alpha=0.5, label='Minnaert fitted line error region')
			a0[0,0].plot(log_u0u_grid, log_uIperF_real, '-', color='black', linewidth=0.7, label='Minnaert real line')
			
			# annotate plot
			f0.suptitle(suptitle, fontsize=12)
			#a0[0,0].set_title(f'Minnaert limb darkening TEST\nIperF0 {IperF0:05.2E} k {k:04.3f}\n IperF0_real {IperF0_real:05.2E} k_real {k_real:04.3f}')
			a0[0,0].set_title(f'Minnaert limb darkening TEST\n        | IperF0   |   k  \n--------------------------\nFitted | {IperF0:05.2E} | {k:04.3f}\nInitial | {IperF0_real:05.2E} | {k_real:04.3f}\nk_var {k_var:05.2E} log_IperF0_var {log_IperF0_var:05.2E} err_mag_est {error_magnitude_estimate:05.2E}')
			a0[0,0].set_xlabel('log[u0 u]')
			a0[0,0].set_ylabel('log[u (I/F)]')
			a0[0,0].legend()
			a0[0,0].grid(True, which='major', axis='both', linestyle='--', color='black', linewidth=0.5, alpha=0.1)
			
			a0[0,1].plot(np.exp(log_u0u), np.exp(log_uIperF), ',')
			a0[0,1].plot(np.exp(log_u0u_grid), np.exp(log_uIperF_fit), '-', color='tab:pink', label='Minnaert fitted line')
			a0[0,1].fill_between(np.exp(log_u0u_grid), np.exp(wf_min), np.exp(wf_max), color='tab:pink', alpha=0.5, label='Minnaert fitted line error region')
			a0[0,1].plot(np.exp(log_u0u_grid), np.exp(log_uIperF_real), '-', color='black', linewidth=0.7, label='Minnaert real line')
			a0[0,1].set_title(f'Minnaert limb darkening TEST on linear scale')
			a0[0,1].set_xlabel('u0*u')
			a0[0,1].set_ylabel('u*(I/F)')
			a0[0,1].legend()
			a0[0,1].grid(True, which='major', axis='both', linestyle='--', color='black', linewidth=0.5, alpha=0.1)
			
			a0[1,0].plot(log_u0u, np.exp(log_uIperF-np.interp(log_u0u, log_u0u_grid, log_uIperF_fit)), ',')
			a0[1,0].set_title('De-trended uIperF value (should be a normal distribution around 1.0)')
			a0[1,0].set_ylabel('exp(ln[u(I/F)] - ln[u(I/F)]_fitted )')
			a0[1,0].set_xlabel('ln[u_0 u]')
			
			a0[1,1].hist(np.exp(log_uIperF-np.interp(log_u0u, log_u0u_grid, log_uIperF_fit)), bins=50, density=True, label='Histogram of de-trended uIperF')
			n_pdf = lambda x, mean, std: 1/(std*np.sqrt(2*np.pi))*np.exp(-0.5*((x - mean)/std)**2)
			x = np.linspace(1-5*fractional_standard_deviation,1+5*fractional_standard_deviation,100)
			a0[1,1].plot(x, n_pdf(x, 1.0, fractional_standard_deviation), label='Test data source distribution')
			a0[1,1].legend()
			a0[1,1].set_ylabel('Probability density')
			a0[1,1].set_xlabel('De-trended uIperF value')
			
			# show plot
			plt.show()
	return(k_results, IperF0_results, k_results_var, IperF0_results_var, err_mag_est)

def plot_fitted_coefficients(kk_reals, IIperF0_reals, k_results, IperF0_results, k_results_var, IperF0_results_var, err_mag_est, suptitle=''):
	nr, nc, s = (2, 6, 6)
	f1 = plt.figure(figsize=(nc*s, nr*s))
	a1 = f1.subplots(nr,nc, squeeze=False)
	f1.suptitle(suptitle)
	
	extent = [k_reals[0],k_reals[-1],IperF0_reals[-1],IperF0_reals[0]]
	#set_xylabels = lambda ax: (ax.set_xlabel('k real'), ax.set_ylabel('IperF0 real'))
	def set_xylabels(axgrid, xlbl='k real', ylbl='IperF0 real'):
		nr, nc = axgrid.shape
		for r in range(nr):
			for c in range(nc):
				if r == nr-1:
					axgrid[r,c].set_xlabel(xlbl)
				if c == 0:
					axgrid[r,c].set_ylabel(ylbl)
					
	######################## PLOT k DATA ######################################
	vlim00 = {'vmin':0, 'vmax':max([np.nanmax(kk_reals),np.nanmax(k_results)])}
	a1[0,0].imshow(kk_reals, extent=extent, aspect='auto', **vlim00, cmap='viridis')
	a1[0,0].set_title(f'real k \nclim:({vlim00["vmin"]:05.2E}, {vlim00["vmax"]:05.2E})')
	#set_xylabels(a1[0,0])
	a1[0,1].imshow(k_results, extent=extent, aspect='auto', **vlim00, cmap='viridis')
	a1[0,1].set_title(f'fitted k \nclim:({vlim00["vmin"]:05.2E}, {vlim00["vmax"]:05.2E})')
	#set_xylabels(a1[0,1])
	
	k_resid = kk_reals - k_results
	absmax02 = np.max(np.abs([np.nanmin(k_resid), np.nanmax(k_resid)]))
	vlim02 = {'vmin':-absmax02, 'vmax':absmax02}
	a1[0,2].imshow(k_resid, extent=extent, aspect='auto', **vlim02, cmap='seismic')
	a1[0,2].set_title(f'residual k \nclim:({vlim02["vmin"]:05.2E}, {vlim02["vmax"]:05.2E})')
	#set_xylabels(a1[0,2])
	
	k_frac_resid = (kk_reals - k_results)/kk_reals
	k_frac_resid[np.isinf(k_frac_resid)] = np.nan
	absmax03 = np.max(np.abs([np.nanmin(k_frac_resid), np.nanmax(k_frac_resid)]))
	vlim03 = {'vmin':-absmax03, 'vmax':absmax03}
	print(absmax03)
	a1[0,3].imshow(k_frac_resid, extent=extent, aspect='auto', **vlim03, cmap='seismic')
	a1[0,3].set_title(f'residual/real k \nclim:({vlim03["vmin"]:04.3f}, {vlim03["vmax"]:04.3f})')
	#set_xylabels(a1[0,3])
	
	vlim04 = {'vmin':np.nanmin(k_results_var), 'vmax':np.nanmax(k_results_var)}
	a1[0,4].imshow(k_results_var, extent=extent, aspect='auto', **vlim04, cmap='viridis')
	a1[0,4].set_title(f'fitted k variance\nclim:({vlim04["vmin"]:05.2E}, {vlim04["vmax"]:05.2E})')
	
	################### PLOT IperF0 DATA ######################################
	vlim10 = {'vmin':0, 'vmax':max([np.nanmax(IIperF0_reals), np.nanmax(IperF0_results)])}
	a1[1,0].imshow(IIperF0_reals, extent=extent, aspect='auto', **vlim10, cmap='viridis')
	a1[1,0].set_title(f'real IperF0 \nclim:({vlim10["vmin"]:05.2E}, {vlim10["vmax"]:05.2E})')
	#set_xylabels(a1[1,0])
	a1[1,1].imshow(IperF0_results, extent=extent, aspect='auto', **vlim10, cmap='viridis')
	a1[1,1].set_title(f'fitted IperF0 \nclim:({vlim10["vmin"]:05.2E}, {vlim10["vmax"]:05.2E})')
	#set_xylabels(a1[1,1])
	
	IperF0_resid = IIperF0_reals - IperF0_results
	absmax12 = np.max(np.abs([np.nanmin(IperF0_resid), np.nanmax(IperF0_resid)]))
	vlim12 = {'vmin':-absmax12, 'vmax':absmax12}
	a1[1,2].imshow(IperF0_resid, extent=extent, aspect='auto', **vlim12, cmap='seismic')
	a1[1,2].set_title(f'residual IperF0 \nclim:({vlim12["vmin"]:05.2E}, {vlim12["vmax"]:05.2E})')
	#set_xylabels(a1[1,2])
	
	IperF0_frac_resid = (IIperF0_reals - IperF0_results)/IIperF0_reals
	absmax13 = np.max(np.abs([np.nanmin(IperF0_frac_resid), np.nanmax(IperF0_frac_resid)]))
	vlim13 = {'vmin':-absmax13, 'vmax':absmax13}
	a1[1,3].imshow(IperF0_frac_resid, extent=extent, aspect='auto', **vlim13, cmap='seismic')
	a1[1,3].set_title(f'residual/real IperF0 \nclim:({vlim13["vmin"]:04.3f}, {vlim13["vmax"]:04.3f})')
	#set_xylabels(a1[1,3])
	
	vlim14 = {'vmin':np.nanmin(IperF0_results_var), 'vmax':np.nanmax(IperF0_results_var)}
	a1[1,4].imshow(IperF0_results_var, extent=extent, aspect='auto', **vlim14, cmap='viridis')
	a1[1,4].set_title(f'fitted IperF0 variance\nclim:({vlim14["vmin"]:05.2E}, {vlim14["vmax"]:05.2E})')
	
	### PLOT OTHER DATA ###
	vlim15 = {'vmin':np.nanmin(err_mag_est), 'vmax':np.nanmax(err_mag_est)}
	a1[1,5].imshow(err_mag_est, extent=extent, aspect='auto', **vlim15, cmap='viridis')
	a1[1,5].set_title(f'error magnitude estimate (contrived units)\nclim:({vlim15["vmin"]:05.2E}, {vlim15["vmax"]:05.2E})')
	
	set_xylabels(a1)
	
	plt.show()
	return
	
if __name__=='__main__':
	# rng seed
	seed = 98465
	
	# parameters for test case
	n_points = 3000
	n_fit_points = 10
	fractional_standard_deviation = 0.4
	
	# create some dummy data
	u0 = np.cos((np.pi/180)*np.linspace(0, 85, n_points))
	u = np.cos((np.pi/180)*np.linspace(0, 85, n_points))
	#k_real = 0.7
	#IperF0_real = 1E-6
	n_pairs = 100
	k_reals = np.linspace(0, 1, n_pairs)
	IperF0_reals = np.linspace(1E-5, 1E-8, n_pairs)
	
	kk_reals, IIperF0_reals = np.meshgrid(k_reals, IperF0_reals)
	
	(k_results_logfit, IperF0_results_logfit,
	k_results_var_logfit, IperF0_results_var_logfit,
	err_mag_est_logfit) = minnaert_test_grid(
		kk_reals, 
		IIperF0_reals, 
		#get_coefficients, 
		lambda a,b,c: get_coefficients(a,b,c,weight='adjust_for_logscale'), 
		fractional_standard_deviation=fractional_standard_deviation, 
		n_fit_points=n_fit_points,
		suptitle='Fit in log space',
		seed=seed
	)
	
	(k_results_linfit, IperF0_results_linfit,
	k_results_var_linfit, IperF0_results_var_linfit,
	err_mag_est_linfit) = minnaert_test_grid(
		kk_reals, 
		IIperF0_reals, 
		get_coefficients_curvefit, 
		fractional_standard_deviation=fractional_standard_deviation, 
		n_fit_points=n_fit_points,
		suptitle='Fit in lin space',
		seed=seed
	)

	#%% plot correspondence between "real" coefficients and fitted coefficients
	
	plot_fitted_coefficients(kk_reals, IIperF0_reals, k_results_logfit, IperF0_results_logfit, k_results_var_logfit, IperF0_results_var_logfit,
							err_mag_est_logfit, suptitle='Non-correspondence between "real" minnaert parameters and fitted values due to assumed normalcy of errors on log[u(I/F)] values')
	
	plot_fitted_coefficients(kk_reals, IIperF0_reals, k_results_linfit, IperF0_results_linfit, k_results_var_linfit, IperF0_results_var_linfit,
							err_mag_est_linfit, suptitle='Correspondence between "real" minnaert parameters and fitted values when fitted in linear space')
	
