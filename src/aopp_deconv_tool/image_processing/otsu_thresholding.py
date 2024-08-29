#!/usr/bin/env python3
"""
Implements Otsu's method for thresholding images
"""

import aopp_deconv_tool.cfg
_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'WARN')

import sys, os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from astropy.io import fits


def calc(counts, bin_edges):
	"""
	Calculates the inter-class-variance for a histogram with `counts` at all `bin_edges`
	"""
	ns = np.linspace(0, counts.size-1, counts.size)
	inter_class_variance = np.zeros_like(bin_edges, dtype=float)
	for i in range(0,bin_edges.size):
		i1, i2 = ns[:i], ns[i:]
		p1, p2 = counts[:i], counts[i:]
		w1, w2 = np.nansum(p1), np.nansum(p2)
		u1, u2 = np.nansum(p1*i1)/w1, np.nansum(p2*i2)/w2
		inter_class_variance[i] = (w1*w2*(u1-u2)**2)/np.sum(counts)
	return(inter_class_variance)

def threshold(bin_edges, icv):
	"""
	Given the `bin_edges` of a histogram and the inter-class-variance at those edges, returns the edge with the maximum inter-class-variance
	"""
	return(bin_edges[np.nanargmax(icv)] if np.any(~np.isnan(icv)) else np.nan)

def exact(data, max_elements=None) -> float:
	"""
	Returns a threshold that should be used as <= threshold
	"""
	#_lgr.debug(f'{data.size=} {data=}')
	if max_elements is None or max_elements >= data.size:
		pass
	else:
		data = np.random.choice(data.flatten(), size=(max_elements,))
		
	if np.all(np.isnan(data)):
		return 0
	
	sorted_data = np.sort(data[~np.isnan(data)]).flatten() # [1,2,3,4,7,8,9]
	#_lgr.debug(f'{sorted_data.size=} {sorted_data=}')
	
	count = np.cumsum(np.ones_like(sorted_data, dtype=int)) # [1,2,3,4,5,6,7]
	#_lgr.debug(f'{count.size=} {count=}')
	frac = count/count.size # [1/7,2/7,3/7,4/7,5/7,6/8,7/7]
	
	cum_sum = np.nancumsum(sorted_data) # [1,3,6,10,17,25,34]
	
	
	
	cum_mean = cum_sum/count # [1, 1.5, 2, 2+2/8, 3+2/5, 4+1/6, 4+6/7] 
	rev_cum_sum = np.nancumsum(sorted_data[::-1])[::-1]
	rev_cum_mean =  rev_cum_sum/count[::-1]
	
	#_lgr.debug(f'{cum_sum=}')
	#_lgr.debug(f'{cum_mean=}')
	#_lgr.debug(f'{rev_cum_sum=}')
	#_lgr.debug(f'{rev_cum_mean=}')
	
	
	cum_var = np.full_like(sorted_data, fill_value=np.nan, dtype=float)
	rev_cum_var = np.full_like(sorted_data, fill_value=np.nan, dtype=float)
	#_lgr.debug(f'{cum_var=}')
	#_lgr.debug(f'{rev_cum_var=}')
	
	for i in range(1,sorted_data.size-1):
		#_lgr.debug(f'{i=}')
		#_lgr.debug(f'{(sorted_data[:i+1] - cum_mean[i+1])**2=}')
		#_lgr.debug(f'{np.sum((sorted_data[:i+1] - cum_mean[i+1])**2)=}')
		#_lgr.debug(f'{np.sum((sorted_data[:i+1] - cum_mean[i+1])**2)/count[i]=}')
		cum_var[i] = np.sum((sorted_data[:i+1] - cum_mean[i+1])**2)/count[i]
		
		#_lgr.debug(f'{(sorted_data[i+1:] - rev_cum_mean[i+1])**2=}')
		#_lgr.debug(f'{np.sum((sorted_data[i+1:] - rev_cum_mean[i+1])**2)=}')
		#_lgr.debug(f'{np.sum((sorted_data[i+1:] - rev_cum_mean[i+1])**2)/(count.size-count[i])=}')
		rev_cum_var[i] = np.sum((sorted_data[i+1:] - rev_cum_mean[i+1])**2)/(count.size-count[i])
	
	#_lgr.debug(f'{cum_var=}')
	#_lgr.debug(f'{rev_cum_var=}')
	
	#plt.plot(cum_var)
	#plt.plot(rev_cum_var)
	#plt.show()
	
	
	icv = frac*cum_var + frac[::-1]*rev_cum_var
	
	#plt.plot(icv)
	#plt.show()
	
	min_icv_idx = np.nanargmin(icv)
	#_lgr.debug(f'{min_icv_idx=}')
	return sorted_data[min_icv_idx]


def masks_from_thresholds(data, thresholds):
	thresholds = np.sort(thresholds)
	for i in range(0,len(thresholds)+1):
		if i==0:
			mask = data <= thresholds[i]
		elif i == len(thresholds):
			mask = data > thresholds[i-1]
		else:
			mask = (thresholds[i-1] < data) & (data <= thresholds[i])
		yield mask

def n_exact(data, n=1, max_elements=None):
	"""
	n is the number of thresholds we want to get out, to get n+1 thresholds, we must first get n thresholds, then
	find thresholds for each n+1 classes
	"""
	dataset = (data,)
	for i in range(n):
		thresholds = tuple(exact(d, max_elements) for d in dataset)
		dataset = (data[mask] for mask in masks_from_thresholds(data, thresholds))
	return thresholds
	


def n_thresholds(data, n):
	"""
	Calculates `n` otsu-thresholds for `data`. Automatically calculates the bin edges depending on the sqrt of the number of datapoints
	"""
	ots = np.zeros((n,))
	for i in range(n):
		counts, bin_edges = np.histogram(
			data, 
			bins=np.linspace(np.nanmin(data), np.nanmax(data), int(np.sqrt(np.sum(~np.isnan(data)))))
		)
		icv = calc(counts, bin_edges)
		ots[i] = threshold(bin_edges, icv) if icv.size>2 else np.nan
		data[data<ots[i]] = np.nan
	return(ots)


def frac_per_fpix_threshold(data, frac_per_fpix=15, n_max=10, on_fail='return_last'):
	"""
	Calculates the fraction of pixels included in `n_max` otsu thresholds
	"""
	s = np.nansum(data)
	n = np.sum(~np.isnan(data))
	ot=0
	for i in range(n_max):
		counts, bin_edges = np.histogram(
			data, 
			bins=np.linspace(np.nanmin(data), np.nanmax(data), int(np.sqrt(np.sum(~np.isnan(data)))))
		)
		icv = calc(counts, bin_edges)
		if icv.size <=2:
			return(None if on_fail != 'return_last' else ot)
		ot = threshold(bin_edges, icv)
		data[data<ot] = np.nan
		this_frac_per_fpix = (np.nansum(data)*n)/(np.sum(~np.isnan(data))*s)
		_lgr.debug(i, ot, this_frac_per_fpix)
		if  this_frac_per_fpix > frac_per_fpix:
			return(ot)
	return(None if on_fail != 'return_last' else ot)

def max_frac_per_fpix_threshold(data, n_max=10):
	"""
	Calculates the "most selective" otsu threshold out of `n_max` thresholds
	"""
	s = np.nansum(data)
	n = np.sum(~np.isnan(data))
	ot = np.nanmin(data)
	for i in range(n_max):
		counts, bin_edges = np.histogram(
			data, 
			bins=np.linspace(np.nanmin(data), np.nanmax(data), int(np.sqrt(np.sum(~np.isnan(data)))))
		)
		icv = calc(counts, bin_edges)
		if icv.size <= 2:
			return(ot)
		ot = threshold(bin_edges, icv)
		data[data<ot] = np.nan
		this_frac_per_fpix = (np.nansum(data)*n)/(np.sum(~np.isnan(data))*s)
		_lgr.debug(i, ot, this_frac_per_fpix)
	return(ot)


def max_frac_diff_threshold(data, n_max=10):
	"""
	Returns the otsu threshold (of `n_max` thresholds) that selects a group of similarly valued pixels
	"""
	ots = np.zeros((n_max,))
	frac_diffs = np.zeros((n_max,))
	for i in range(n_max):
		n_valid_px = np.sum(~np.isnan(data))
		counts, bin_edges = np.histogram(
			data, 
			bins=np.linspace(np.nanmin(data), np.nanmax(data), int(np.sqrt(n_valid_px)))
		)
		if bin_edges.size==0:
			break
		icv = calc(counts, bin_edges)
		ots[i] = threshold(bin_edges, icv) if icv.size>2 else np.nan
		data[data<ots[i]] = np.nan

		range_frac = (ots[i]-bin_edges[0])/(bin_edges[-1]-bin_edges[0])
		npix_frac = np.sum(~np.isnan(data))/n_valid_px
		frac_diffs[i] = (range_frac-npix_frac)/(range_frac+npix_frac)
		if np.all(np.isnan(frac_diffs)):
			return(np.nan)

	
	return(ots[np.nanargmax(frac_diffs)])



def polynomial(coef, x):
	return(np.nansum(coef[:,None]*np.vstack([x**i for i in range(len(coef)-1,-1,-1)]),axis=0))


def plot_repeats(data, n=4):
	countsl, bin_edgesl, bin_midsl, icvl, otl, rdatal, odatal = [], [], [], [], [], [], []
	polyfits = []

	odatal.append(data)
	for i in range(n):
		counts, bin_edges = np.histogram(
			odatal[i], 
			bins=np.linspace(np.nanmin(odatal[i]), np.nanmax(odatal[i]), int(np.sqrt(np.sum(~np.isnan(odatal[i]))))),
			density=False
		)
		icv = calc(counts, bin_edges)
		if icv.size<=2:
			ot = np.nan
		else:
			ot = threshold(bin_edges, icv)
	
		rdata = np.array(odatal[i])
		rdata[rdata<ot] = np.nan
	
		range_frac = (ot-bin_edges[0])/(bin_edges[-1]-bin_edges[0])
		npix_frac = np.sum(~np.isnan(rdata))/np.sum(~np.isnan(odatal[i]))
		_lgr.debug(range_frac, npix_frac, (range_frac-npix_frac)/(range_frac+npix_frac))

		countsl.append(counts)
		bin_edgesl.append(bin_edges)
		bin_midsl.append(0.5*(bin_edges[1:]+bin_edges[:1]))
		icvl.append(icv)
		otl.append(ot)
		rdatal.append(rdata)
		odatal.append(rdata)

	

	ots = n_thresholds(np.array(data), n)

	f1, axes = ut.plt.create_figure_with_subplots(4, n, sp_kwargs={'gridspec_kw':{'hspace':0.3}})
	axes_iter = iter(axes.T.flatten())

	s = np.nansum(data)
	pn = np.sum(~np.isnan(data))

	for i in range(n):	

		ax = next(axes_iter)
		ax.imshow(odatal[i])
		ax.set_title(f'frac {np.nansum(odatal[i])/s:08.2E}\nfrac/fpx {(np.nansum(odatal[i])*pn)/(s*np.sum(~np.isnan(odatal[i]))):08.2E}\nfrac^2/fpx {(pn*np.nansum(odatal[i])**2)/(np.sum(~np.isnan(odatal[i]))*s**2):08.2E}')
		ut.plt.remove_axes_ticks_and_labels(ax)
		
		ax = next(axes_iter)
		ax.step(bin_edgesl[i][1:], countsl[i])
		ax.axvline(otl[i], color='red')
		ax.axvline(ots[i], color='green', ls='--')


		ax = next(axes_iter)
		ax.plot(bin_edgesl[i], icvl[i])
		ax.axvline(otl[i], color='red')
		ax.axvline(ots[i], color='green', ls='--')

		ax = next(axes_iter)
		ax.imshow(rdatal[i])
		ax.set_title(f'frac {np.nansum(rdatal[i])/s:08.2E}\nfrac/fpx {(np.nansum(odatal[i])*pn)/(s*np.sum(~np.isnan(odatal[i]))):08.2E}')
		ut.plt.remove_axes_ticks_and_labels(ax)

	

	f1.show()
	input("Close figure and press [ENTER] to continue > ")
	plt.close(f1)
	
	


if __name__=='__main__':

	fpath = os.path.expanduser("~/scratch/general_testing/telecope_data/obs10_MUSE.2019-10-18T00:32:18.867/DATACUBE_FINAL_SMOOTH_10.fits")
	fpath = os.path.expanduser("~/scratch/general_testing/test_archive/Neptune/VLT_MUSE/2018/20180619_1_V/Neptune_MUSE_20180619_1_V_ALTERED.fits")


	with fits.open(fpath) as hdul:
		hdul.info()
		#data_idx = 440
		data_idx = 26
		data = np.abs(hdul[1].data[data_idx])

		plot_repeats(data, 10)

		_lgr.DEBUG(f'{frac_per_fpix_threshold(np.array(data), 15)=}')
		_lgr.DEBUG(f'{max_frac_per_fpix_threshold(np.array(data), 20)=}')






