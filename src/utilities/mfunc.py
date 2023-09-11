#!/usr/bin/env python3
"""
Mathematical functions, domain is the first argument variables are the others
"""
import numpy as np
import scipy as sp
import scipy.special
import scipy.optimize

def poission_pmf(k : np.ndarray, l : int = 1):
	"""
	poission distribution probability mass function
	"""
	return(l**k*np.exp(l)/(sp.special.factorial(k)))

def binomial_pdf(k, p, n):
	q = 1-p
	f = sp.special.factorial(n)/(sp.special.factorial(k)*sp.special.factorial(n-k))
	b_pmf =f * p**k * q**(n-k)
	return(b_pmf / np.sum(b_pmf))

def normal_pdf(x, mean, sigma):
	return (1/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*((x-mean)/sigma)**2))

def folded_normal_pdf(x, mean, sigma, fold=0):
	x_dash = x - fold
	N_folded = normal_pdf(x_dash, mean, sigma) + normal_pdf(x_dash, -mean, sigma)
	return(N_folded)

def gamma_pdf(x, k, theta):
	A = (1/(sp.special.gamma(k)*theta**k))
	B =  x**(k-1)
	C = np.exp(-(x/theta))
	#return((1/(sp.special.gamma(k)*theta**k)) * x**(k-1) * np.exp(-(x/theta)))
	return(A*B*C)

def gamma_cdf(x, k, theta):
	return(sp.special.gammainc(k, x/theta)) # scipy regularises the incomplete gamma function so it is just what we need

def gamma_qf(p, k, theta):
	# quantile function (inverse of cumulative density function)
	return(sp.special.gammaincinv(k, p)*theta)

def cmp_pmf(k, l, nu, j_max=10):
	# conway-maxwell-poission distribution
	j = np.linspace(0,j_max, j_max+1)
	return((l**k)/((sp.special.factorial(k))**nu) * 1/(np.sum((l**j)/(sp.special.factorial(j)**nu))))

def cmp_cdf(k, l, nu, j_max=10):
	# hopefully will treat scalar and array values of k correctly
	#if type(k) in (int, float):
	#	k_dash = np.linspace(0,k,k+1)
	#else:
	k_dash = np.linspace(0,np.ceil(np.nanmax(k)), int(np.ceil(np.nanmax(k))+1))
	cmp = np.cumsum(cmp_pmf(k_dash, l, nu, j_max=j_max))
	return(np.interp(k, k_dash, cmp))

def cmp_mle(data, j_max=10):
	# not sure if this works at all
	s1 = np.nansum(data)
	s2 = np.nansum(np.log(sp.special.factorial(data)))
	j = np.linspace(0,j_max,j_max+1)
	likelihood = lambda l, nu: (l**s1) * np.exp(-nu*s2) * (np.sum((l**j)/(sp.special.factorial(j)**nu)))**-j_max
	mean, var = np.nanmean(data), np.nanvar(data)
	nu_est = var/mean
	l_est = mean/nu_est
	x0 = np.array([l_est,1/nu_est]) # l, nu initial estimates
	print(f'{x0=}')
	x0_bounds = ((0,np.inf),(0,np.inf))
	
	#res = sp.optimize.minimize(lambda x: -likelihood(x[0],x[1]), x0, bounds=x0_bounds, method='L-BFGS-B')
	hvals, hbins = np.histogram(data[data>0], bins=200, density=True)
	hmids = 0.5*(hbins[:-1] + hbins[1:])
	#print(cmp_pmf(hmids, x0[0], x0[1], j_max))
	#print(hvals)
	popt, pcov = sp.optimize.curve_fit(lambda k, l, nu: cmp_pmf(k, l, nu, j_max), hmids, hvals, x0, bounds=tuple(zip(*x0_bounds)), method='trf')
	return(popt)

def cmp_qf(p, l, nu, j_max=10):
	if 0 >= p >= 1: raise ValueError(f'In cmp_qf() probability must be in range [0,1] {p=}') 
	if p == 0: return(0)
	if p == 1: return(np.inf)
	# generate an array of test k values
	k_min = 0
	k_max = 100
	k = np.linspace(k_min,k_max,k_max+1)
	
	# see if largest p is in k range
	pmf = cmp_pmf(k, l, nu, j_max=j_max)
	while np.max(p) > np.nansum(pmf):
		k_min = k_max + 1
		k_max += 100
		k = np.linspace(k_min, k_max, k_max+1)
		pmf = np.concatenate(pmf, cmp_pmf(k, l, nu, j_max=j_max))
	cdf = np.cumsum(pmf)
	k = np.linspace(0,k_max,k_max+1)
	return(np.interp(p, cdf, k))

def cauchy_pdf(x, location, scale):
	# ratio of 2 nomally distributed random variables with mean zero
	return(1/(np.pi*scale*(1+ ((x-location)/(scale))**2 )))

def cauchy_cdf(x, location, scale):
	return( (1/np.pi)*np.arctan((x-location)/scale) + 0.5)

def cauchy_qf(p, location, scale):
	return(location + scale*np.tan(np.pi*(p-0.5))) 