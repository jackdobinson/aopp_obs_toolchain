#!/usr/bin/env python3
"""
Contains utility functions to help with using scipy routines
"""
import utilities.logging_setup
logging, _lgr = utilities.logging_setup.getLoggers(__name__, 'WARNING')

import numpy as np
import scipy as sp
import scipy.interpolate
import utilities as ut
import utilities.np
import functools

# DEFINE GLOBALS #
# END DEFINE GLOBALS #

# DEFINE CLASSES #
class ProbDistributionErr:
	"""
	Class to get the error on a fitted probability distribution, uses the
	machinery from scipy.stats.
	
	Use as a drop-in replacement for a frozen distribution in scipy.stats,
	except that instead of returning whatever stats function we ask for,
	it will return the error on that function.
	
	# EXAMPLE #
		import numpy as np
		import scipy as sp
		import scipy.stats
		import matplotlib.pyplot as plt
		import utilities as ut
		import utilities.sp
		
		mean, mean_err = 10, 0.1
		std, std_err = 3, 0.3
		fzn_distrib = sp.stats.norm(mean, std)
		fzn_distrib_err = ut.sp.ProbDistributionErr(fzn_distrib, (mean_err, std_err))
		
		x = np.linspace(0, 20)
		plt.plot(x, fzn_distrib.pdf(x), color='tab:blue')
		plt.fill_between(x, fzn_distrib.pdf(x)-fzn_distrib_err.pdf(x), fzn_distrib.pdf(x)+fzn_distrib_err.pdf(x), alpha=0.3, color='tab:blue')
		
		
		plt.plot(x, fzn_distrib.cdf(x), color='tab:orange')
		plt.fill_between(x, fzn_distrib.cdf(x)-fzn_distrib_err.cdf(x), fzn_distrib.cdf(x)+fzn_distrib_err.cdf(x), alpha=0.3, color='tab:orange')
		
		plt.show()
	
	"""
	def __init__(self, fzn_distrib, err):
		self.distrib = fzn_distrib
		self.p = get_distrib_params(self.distrib)
		self.pn = tuple(self.p.keys())
		self.pv = tuple(self.p.values())
		
		if type(err) is dict:
			self.err = tuple([err.get(k,0) for k in self.pn])
		else:
			self.err = err
		e = np.array(self.err)
		self.e_idx = np.nonzero(e!=0)[0]
		self.e = e[e!=0]
		
		# TODO:
		# * this won't work for more than 8 arguments, make a general solution at some point
		# want to have a (+err, -err) pair for each error on the distribution
		self.e_plusminus_arr = np.unpackbits(np.mgrid[:2**(self.e.size)].astype(np.uint8)[:,None], axis=1, bitorder='litte').astype(int)
		self.e_plusminus_arr[self.e_plusminus_arr==0] = -1
		return

	def _get_distrib_mutated_args(self):
		ma = np.array(self.pv, dtype=float)
		for e_pm in self.e_plusminus_arr:
			for i, idx in enumerate(self.e_idx):
				ma[idx] = self.pv[idx] + self.e[i]*e_pm[i]
			yield ma
	
	def _get_err_of_distrib_method(self, distrib_method):
		def get_err_of_method(*args, **kwargs):
			y = np.array(distrib_method(*args, **kwargs))
			y_dash = np.zeros((3,*y.shape), dtype=y.dtype)
			y_dash[2,...] = np.inf # default value for 'minimum'
			for ma in self._get_distrib_mutated_args():
				self.distrib.args = tuple(ma) # mutate the underlying distribution's arguments
				y_new = np.array(distrib_method(*args, **kwargs))
				y_dash[0] = y-y_new
				y_dash[1] = np.max(y_dash[:2], axis=0)
				y_dash[2] = np.min(y_dash, axis=0)
			#  return underlying distribution's args to previous value, 
			# we didn't copy the distribution so if we don't do this 
			# subsequent calls will be wrong
			self.distrib.args = self.pv
			return(y_dash[1]-y_dash[2])
		return(get_err_of_method) # we want to return the error on the underlying distribution's method
	
	
	def __getattr__(self, name):
		# want to intercept calls to the underlying distribution and numerically
		# calculate the error on whatever method of the underlying distribution
		# the user is trying to get.
		try:
			distrib_method = getattr(self.distrib, name)
		except AttributeError as E:
			raise AttributeError(f'Could not forward call for attribute {name=} to {self.distrib=} and {self} has no attribute {name=}')
		return(self._get_err_of_distrib_method(distrib_method))
	
	def __str__(self):
		return(f'ProbDistributionErr({id(self):#0x}, fzn_distrib={self.distrib}, err={self.err})')


# END DEFINING CLASSES #


# DEFINE FUNCTIONS BELOW THIS LINE #

def construct_cdf_from(data):
	sorted_data = np.sort(data)
	cdf_data = np.cumsum(np.ones((data.shape[0]))*(1/data.shape[0]))
	
	return(lambda x: np.interp(x, sorted_data, cdf_data, left=0, right=1))

def construct_ppf_from(data):
	sorted_data = np.sort(data)
	cdf_data = np.cumsum(np.ones((data.shape[0]))*(1/data.shape[0]))
	return(lambda p: np.interp(p, cdf_data, sorted_data, left=np.nan, right=np.nan))

def construct_pdf_from(data):
	sorted_data = np.sort(data)
	sorted_data_edges = ut.np.get_bin_edges(np.sort(data))
	cdf_data = np.cumsum(np.ones((data.shape[0]))*(1/data.shape[0]))
	cdf_edges = np.interp(sorted_data_edges, sorted_data, cdf_data)
	simple_diff = lambda x0, x1, y0, y1: (y1-y0)/(x1-x0)
	pdf_data = 0.5*(simple_diff(sorted_data_edges[:-1],sorted_data,cdf_edges[:-1],cdf_data)
				    + simple_diff(sorted_data, sorted_data_edges[1:], cdf_data, cdf_edges[1:])
				   )
	print(f'{sorted_data.shape=}')
	print(f'{sorted_data_edges.shape=}')
	print(f'{cdf_data.shape=}')
	print(f'{cdf_edges.shape=}')
	print(f'{pdf_data.shape=}')
	
	return(lambda x: np.interp(x, sorted_data, pdf_data, left=0, right=0))
	

def fit_discrete_distribution_to(data, discrete_distribution, fixed_args=tuple(), fixed_kwargs={}, **curve_fit_kwargs):
	"""
	Fits a discrete distribution (from scipy.stats module) to some data by
	calculating a histogram of the data with integer sized bins, and using
	scipy.optimize.curve_fit to fit the ".pmf()" method of the distribution
	to the histogram.
	
	# ARGUMENTS #
		data : np.ndarray[:]
			A 1 dimensional numpy array of the data we want to fit
			'discrete_distribution' to
		discrete_distribution : scipy.stats.rv_discrete
			Should be something from scipy.stats
		fixed_args : tuple
			The arguments to discrete_distribution that curve_fit should
			not fiddle with. If you want to fix args that are not at the front
			of the	argument list, pass None as it's value in fixed_args
		fixed_kwargs : dict
			Keyword arguments to 'discrete_distribution' that curve_fit should
			not fiddle with.
		curve_fit_kwargs : dict
			Arguments to "scipy.optimize.curve_fit()"
	
	# RETURNS #
		distrib_fit : scipy.stats.rv_frozen
			The fitted distribution
		distrib_params : dict
			A dictionary of the fitted parameters
		distrib_err : utilities.sp.ProbDistributionErr
			A class that numerically calculates the error on a probability
			distribution by intercepting calls to the prob. distribution's methods.
			You can get the error on the parameters by accessing the .err attribute.
			The order of the errors is the same as the order of parameters in
			'distrib_params'
			
	# USAGE #
		distrib_fit, distrib_params, distrib_err = ut.sp.fit_discrete_distribution_to(
			data, 
			sp.stats.normal, 
			fixed_args = (None,None), 
			p0 = (np.mean(data), np.std(data)), 
			bounds = ((-np.inf,0),(np.inf,np.inf),
		)
			
	# EXAMPLE #
		>>> import numpy as np
		>>> import scipy as sp
		>>> import scipy.stats
		>>> import matplotlib.pyplot as plt
		>>> # personal packages
		>>> import utilities as ut
		>>> import utilities.sp
		>>> 
		>>> n, a, b = 36, 0.5, 3
		>>> distrib_bb = sp.stats.betabinom(n, a, b)
		>>> x = np.linspace(0, n, n+1)
		>>> 
		>>> data = distrib_bb.rvs(500)
		>>> 
		>>> bins = ut.np.get_bin_edges(x)
		>>> hv, hb = np.histogram(data, bins=bins, density=True)
		>>> 
		>>> fixed_args_p0_bounds_permutations = (
		>>> 	((None, None, None), (1,1,1), ((0,0,0),(np.inf,np.inf,np.inf))),
		>>> 	((n, None, None), (1,1), ((0,0),(np.inf,np.inf))),
		>>> 	((None, a, None), (1,1), ((0,0),(np.inf,np.inf))),
		>>> 	((None, None, b), (1,1), ((0,0),(np.inf,np.inf))),
		>>> 	((n, a, None), (1,), ((0,),(np.inf,))),
		>>> 	((n, None, b), (1,), ((0,),(np.inf,))),
		>>> 	((None, a, b), (1,), ((0,),(np.inf,))),
		>>> )
		>>> 
		>>> f1, a1 = plt.subplots(7, 2, figsize=[_x*4 for _x in (2,7)], squeeze=False, sharex=True)
		>>> ax_iter = iter(a1.ravel())
		>>> 	
		>>> f1.suptitle('\n'.join((
		>>> 	f'Original distribution {n=:07.2E} {a=:07.2E} {b=:07.2E}',
		>>> )))
		>>> 
		>>> for i, (fa, p0, bounds) in enumerate(fixed_args_p0_bounds_permutations):
		>>> 	bb_fit, bb_params, bb_err= ut.sp.fit_discrete_distribution_to(data, sp.stats.betabinom, fixed_args=fa, p0=p0, bounds=bounds)
		>>> 	
		>>> 	bb_pmf = bb_fit.pmf(x)
		>>> 	bb_pmf_err = bb_err.pmf(x)
		>>> 	bb_cdf = bb_fit.cdf(x)
		>>> 	bb_cdf_err = bb_err.cdf(x)	
		>>> 	
		>>> 	ax=next(ax_iter)
		>>> 	ax.set_title(f'Permutation {i} Fitted Parameters')
		>>> 	
		>>> 	ax.text(0,1,'\n'.join((
		>>> 		f'Parameter | Value     | Fit     | Error  ',
		>>> 		f'----------|-----------|---------|--------',
		>>> 		f'    n     | {n:7}   | {bb_params["n"]:7.2G} | {bb_err.err[0]:7.2G}',
		>>> 		f'    a     | {a:7}   | {bb_params["a"]:7.2G} | {bb_err.err[1]:7.2G}',
		>>> 		f'    b     | {b:7}   | {bb_params["b"]:7.2G} | {bb_err.err[2]:7.2G}',
		>>> 	)), horizontalalignment='left', verticalalignment='top')
		>>> 	
		>>> 	ax.xaxis.set_visible(False)
		>>> 	ax.yaxis.set_visible(False)
		>>> 	[_s.set_visible(False) for _s in ax.spines.values()]
		>>> 	
		>>> 	ax=next(ax_iter)
		>>> 	ax.set_title(f'Permutation {i} plot')
		>>> 	ax.hist(data, bins=bins, label='original data histogram', density=True)
		>>> 	
		>>> 	ax.plot(x, bb_pmf, color='tab:orange', label=f'fitted pmf')
		>>> 	ax.fill_between(x, bb_pmf+bb_pmf_err, bb_pmf-bb_pmf_err, color='tab:orange', alpha=0.3, zorder=99, label=f'pmf_err')
		>>> 	
		>>> 	ax.plot(x, bb_cdf, color='tab:orange', ls='--', label=f'fitted cdf')
		>>> 	ax.fill_between(x, bb_cdf+bb_cdf_err, bb_cdf-bb_cdf_err, color='tab:orange', alpha=0.3, zorder=99, label=f'cdf_err')
		>>> 
		>>> ls, hs = [], []
		>>> for ax in a1.ravel():
		>>> 	l, h = ax.get_legend_handles_labels()
		>>> 	ls += [_l for _l in l if _l not in ls]
		>>> 	hs += [_h for _h in h if _h not in hs]
		>>> f1.legend(ls, hs)
		>>> plt.show()
	"""
	x = ut.np.get_integer_range_from(data)
	hv, hb = np.histogram(data, bins=ut.np.get_bin_edges(x), density=True)
	
	# assume that fixed_args are None in places we want to replace, and the remainder of args should be appended
	
	def combine_args(fixed_args, *args, fixed_kwargs, **kwargs):
		combined_args = []
		i = 0
		for fa in fixed_args:
			if fa is None:
				combined_args.append(args[i])
				i+=1
			else:
				combined_args.append(fa)
		combined_args += list(args[i:])
		combined_kwargs = {**kwargs, **fixed_kwargs}
		return(combined_args, combined_kwargs)
	
	def pmf(x, *args, **kwargs):
		c_args, c_kwargs = combine_args(fixed_args, *args, fixed_kwargs=fixed_kwargs, **kwargs)
		return(discrete_distribution.pmf(x, *c_args, **c_kwargs))
	
	def distrib(*args, **kwargs):
		c_args, c_kwargs = combine_args(fixed_args, *args, fixed_kwargs=fixed_kwargs, **kwargs)
		return(discrete_distribution(*c_args, **c_kwargs))
	
	popt, pcov = sp.optimize.curve_fit(pmf, x, hv, **curve_fit_kwargs)
	
	perr = np.sqrt(np.diag(pcov))
	distrib_param_err = []
	i=0
	for fa in fixed_args:
		if fa is not None: # if fixed_args have a value, then the error should be zero
			distrib_param_err.append(0)
		else:
			distrib_param_err.append(perr[i])
			i+=1
	
	distrib_fit = distrib(*popt)
	distrib_params = get_distrib_params(distrib_fit)
	distrib_err = ProbDistributionErr(distrib_fit, tuple(distrib_param_err))
	
	return(distrib_fit, distrib_params, distrib_err)


def get_distrib_params(fzn_distrib):
	
	params = dict(loc=0)#, scale=1) # fill with defaults
	fzn_distrib_shape_params = fzn_distrib.dist.shapes.split(', ') if fzn_distrib.dist.shapes is not None else []
	fzn_distrib_params = fzn_distrib_shape_params + ['loc']#, 'scale'] # discrete distributions do not have a 'scale' param

	# arguments should be in *shape,loc,scale order
	for key, value in zip(fzn_distrib_params, fzn_distrib.args):
		params[key] = value
	return(params)


def err_from_args(func, args, err, **kwargs):
	"""
	assume that err applies to args in order, if err[i]=0 then there is no error
	on that argument, if we run out of err values, then the remaining arguments have no error
	"""
	y = func(*args, **kwargs)

	a = np.array(args)	
	e_idx = np.nonzero(np.array(err)!=0)
	e = np.array(err)[e_idx]
	e_idx = e_idx[0]
	
	# this will fail when there are more than 8 arguments
	err_adj_arr = np.unpackbits(np.mgrid[:2**(e.size-1)].astype(np.uint8)[:,None], axis=1, bitorder='little').astype(int)
	err_adj_arr[err_adj_arr==0] = -1
	
	a_dash = np.array(a)
	y_dash = np.zeros((2,*y.shape))

	for err_adj in err_adj_arr:
		for i, idx in enumerate(e_idx):
			a_dash[idx] = a[idx] + e[i]*err_adj[i]
		
		y_dash[1] = np.fabs(y-func(*a_dash, **kwargs))
		y_dash[0] = np.max(y_dash, axis=0)
	return(y_dash[0])
	

def interpolate_at_mask(data, mask, edges=None, **kwargs):
	"""
	Interpolates an array 'data' at the True points given in 'mask'
	**kwargs is passed through to "sp.interpolate.griddata()"
	"""
	if edges is None:
		interp_data = np.array(data)
	elif edges == 'convolution':
		# build convolution kernel
		kernel_shape = tuple(3 for s in data.shape)
		kernel_edge_shape = tuple(s//2 for s in kernel_shape)
		kernel = np.ones(kernel_shape)
		kernel /= np.sum(kernel)
		embed_shape = tuple(s+2*kes for kes, s in zip(kernel_edge_shape, data.shape))
		embed_slice = tuple(slice(kes,-kes) for kes in kernel_edge_shape)
		
		# build data and mask with a frame of zeros
		interp_data = np.zeros(embed_shape)
		interp_data[embed_slice] = data
		interp_mask = np.zeros(embed_shape, dtype=mask.dtype)
		interp_mask[embed_slice] = mask
		mask = interp_mask
		
		# interpolate the framed data, this should remove the effects of any masked out points
		interp_data = interpolate_at_mask(interp_data, interp_mask, **kwargs)
		# convolve the interpolated framed data, the frame should now be non-zero but tending towards zero
		interp_data = sp.signal.fftconvolve(interp_data, kernel, mode='same')
		# put the original data back into the frame, we are just using the frame to anchor the interpolation.
		interp_data[embed_slice] = data
	else:
		raise ValueError(f'Unknown edge interpolation strategy "{edges=}"')

	points = np.array(mgrid_from_array(interp_data, gridder=np.mgrid))

	p_known = points[:,~mask].T
	p_unknown = points[:,mask].T
	known_values = interp_data[~mask]

	interp_values = sp.interpolate.griddata(p_known, known_values, p_unknown, **kwargs)

	interp_data[mask] = interp_values
	
	if edges is None:
		return(interp_data)
	elif edges == 'convolution':
		return(interp_data[embed_slice]) # only want the embedded interpolated data, not the frame
	raise ValueError(f'Unknown edge interpolation strategy "{edges=}", this should never happen.')


def otsu_threshold(data, bins, n=1, p=True):
	hvals, hbins = np.histogram(data, bins, density=True)
	
	icd = lambda t: np.sum(hvals[:t])*np.var(data[data<hbins[t]]) + np.sum(hvals[t:])*np.var(data[hbins[t]<data])
	
	icd_array = np.full((len(hbins),), fill_value=np.nan)
	for i in range(len(hbins)):
		icd_array[i] = icd(i)
		
	t_idx = np.nanargmin(icd_array)
	t_value = hbins[t_idx]
	t_p = np.cumsum(hvals)[t_idx]
	t_ret = t_p if p else t_value
	return(t_ret, icd_array)
																					 
																					 

def rotmatrix(rx, ry, rz):
	"""
	Calculates a rotation matrix in terms of lean, latitude, longitude
	rx - the rotation around the x axis (equivalent to leaning the north pole of the planet onto it's side, obliquity)
	ry - the rotation around the y axis 
	rz - the rotation around the z axis
	"""
	rx_m = np.zeros((3,3))
	rx_m[0,0] = 1
	rx_m[1,1] = np.cos(rx)
	rx_m[1,2] = -np.sin(rx)
	rx_m[2,1] = np.sin(rx)
	rx_m[2,2] = np.cos(rx)

	ry_m = np.zeros((3,3))
	ry_m[1,1] = 1
	ry_m[0,0] = np.cos(ry)
	ry_m[0,2] = np.sin(ry)
	ry_m[2,0] = -np.sin(ry)
	ry_m[2,2] = np.cos(ry)

	rz_m = np.zeros((3,3))
	rz_m[2,2] = 1
	rz_m[0,0] = np.cos(rz)
	rz_m[0,1] = -np.sin(rz)
	rz_m[1,0] = np.sin(rz)
	rz_m[1,1] = np.cos(rz)

	rtotal = np.matmul(rz_m, np.matmul(ry_m, rx_m))
	return(rtotal)

def colvec(v):
	# creates a column vector from a 1d array v
	cv = np.array(v)
	cv.shape = (cv.shape[0],1)
	return(cv)

def rowvec(v):
	# creates a row vector from a 1d array v
	rv = np.array(v)
	rv.shape = (1, rv.shape[0])
	return(rv)
	

def ident_mat(i):
	ident_m = np.zeros((i,i))
	for _i in range(i):
		ident_m[_i,_i] = 1
	return(ident_m)

def mgrid_from_array(a, gridder=np.mgrid):
	"""
	Uses np.mgrid to get a grid of indicies for an array, can use 
	gridder=np.ogrid if you want to use more memory efficient version

	INPUTS:
		a
			An ND array
		gridder
			The function to use for computing the grid choices (np.mgrid, np.ogrid)

	RETURNS:
		idx_grid_tuple
			Tuple of indicies of each element for each dimension
	"""
	return(gridder[tuple([slice(0,s) for s in a.shape])])
	








