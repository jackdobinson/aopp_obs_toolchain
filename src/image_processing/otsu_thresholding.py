#!/usr/bin/env python3
"""
Implements Otsu's method for thresholding images
"""

import utilities.logging_setup
logging, _lgr = utilities.logging_setup.getLoggers(__name__, 'DEBUG')

import sys, os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from astropy.io import fits

import utilities as ut
import utilities.plt
import fitting

def calc(counts, bin_edges):
	ns = np.linspace(0, counts.size-1, counts.size)
	inter_class_variance = np.zeros_like(bin_edges, dtype=float)
	for i in range(0,bin_edges.size):
		i1, i2 = ns[:i], ns[i:]
		p1, p2 = counts[:i], counts[i:]
		w1, w2 = np.nansum(p1), np.nansum(p2)
		u1, u2 = np.nansum(p1*i1)/w1, np.nansum(p2*i2)/w2
		#_lgr.DEBUG(f'{i1.size=} {i2.size=} {p1.size=} {p2.size=} {w1=:08.2E} {w2=:08.2E} {u1=:08.2E} {u2=:08.2E}')
		inter_class_variance[i] = (w1*w2*(u1-u2)**2)/np.sum(counts)
	return(inter_class_variance)

def threshold(bin_edges, icv):
	#bin_mids = 0.5*(bin_edges[1:]+bin_edges[:1])
	#_lgr.DEBUG(f'{bin_mids=}')
	#_lgr.DEBUG(f'{bin_edges=}')
	#_lgr.DEBUG(f'{icv=}')
	return(bin_edges[np.nanargmax(icv)] if np.any(~np.isnan(icv)) else np.nan)


def n_thresholds(data, n):
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
		print(i, ot, this_frac_per_fpix)
		if  this_frac_per_fpix > frac_per_fpix:
			return(ot)
	return(None if on_fail != 'return_last' else ot)

def max_frac_per_fpix_threshold(data, n_max=10):
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
		print(i, ot, this_frac_per_fpix)
	return(ot)


def max_frac_diff_threshold(data, n_max=10):
	ots = np.zeros((n_max,))
	frac_diffs = np.zeros((n_max,))
	for i in range(n_max):
		n_valid_px = np.sum(~np.isnan(data))
		counts, bin_edges = np.histogram(
			data, 
			bins=np.linspace(np.nanmin(data), np.nanmax(data), int(np.sqrt(n_valid_px)))
		)
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
	
		#icv /= np.nanmax(icv)

		range_frac = (ot-bin_edges[0])/(bin_edges[-1]-bin_edges[0])
		npix_frac = np.sum(~np.isnan(rdata))/np.sum(~np.isnan(odatal[i]))
		print(range_frac, npix_frac, (range_frac-npix_frac)/(range_frac+npix_frac))

		#X = np.vstack([bin_edges**0, bin_edges**1, bin_edges**2]).T
		#polyfit = np.linalg.lstsq(A, np.nan_to_num(icv))
		#polyfit = fitting.linear_least_squares(np.nan_to_num(icv), X)
		#polyfits.append(polyfit)
		#print(polyfit[0])
		#print(np.nansum(polyfit[1]**2)/bin_edges.size)
		#print(polyfit[2])
		#print(np.trace(polyfit[2]))
		#print(np.nansum(polyfit[1]**2)/np.trace(polyfit[2]))

		countsl.append(counts)
		bin_edgesl.append(bin_edges)
		bin_midsl.append(0.5*(bin_edges[1:]+bin_edges[:1]))
		icvl.append(icv)
		otl.append(ot)
		rdatal.append(rdata)
		odatal.append(rdata)

	

	ots = n_thresholds(np.array(data), n)

	#f1, axes = ut.plt.figure_n_subplots(n*3)
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
		#if polyfits[i][0].size > 0:
		#	ax.plot(bin_edgesl[i], polynomial(polyfits[i][0][::-1],bin_edgesl[i]))

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






