#!/usr/bin/env python3

import sys, os
import numpy as np
from astropy.io import fits
import fitscube.header
import copy


def stack_hdu_along_spectral_axis(hdu, bin_edges, w_idx=None, stack_func=np.nanmean, sf_args=[], sf_kwargs={}, stack_coords='world'):
	if w_idx is None:
		w_idx = fitscube.header.get_index_of_ctype(hdu, ctype='WAVE')
	
	return(stack_hdu_along_axis(hdu, bin_edges, w_idx, order='numpy', stack_coords=stack_coords, stack_func=stack_func, sf_args=sf_args, sf_kwargs=sf_kwargs))

def stack_hdu_along_axis(hdu, bin_edges, axis, order='numpy', stack_coords='physical', stack_func=np.nanmean, sf_args=[], sf_kwargs={}):
	"""
	Stacks a header data unit along a specified axis, axis can be deonoted in numpy order or the order in the *.fits file
	
	# ARGUMENTS #
		hdu
			Header data unit to operate on
		bin_edges
			Edges of bins to re-stack data into, fits uses the convention that
			a cell represents it's value at the center. Need N+1 bin edges for
			N resulting cells
		axis
			Index of the axis to stack along, ordering is dependent on the
			'order' argument
		order
			Decides how the 'axis' argument is interpreted, is it the numpy data
			axis or the *.fits file axis. Possible values:
				'numpy'
					N axes have idices from 0 -> N-1, and reversed ordering
					compared to the hdu.header entry
				'fits'
					N axes have indices from 1 -> N, and ordering as per the 
					hdu.header entry
		stack_coords
			The coordinate system to define bin_edges etc. in.
				'physical' - the physical (pixel) coordinates
				'world' - the world (e.g. ra-dec) coordinates
		stack_func
			The function to apply to the hdu.data array that will combine the
			data into a stack
		sf_args
			Any positional arguments to pass to 'stack_func'
		sf_kwargs
			Any keyword arguments to pass to 'stack_func'
	"""
	if order == 'numpy':
		a_idx = axis
		ax_i = hdu.header['NAXIS'] - a_idx
	elif order == 'fits':
		ax_i = axis
		a_idx = hdu.header['NAXIS'] - ax_i
	else:
		sys.exit(f'ERROR: "stack_hdu_along_axis()" requires argument "order" to be one of ("numpy", "fits"), here "order"={order}')
	
	if bin_edges.ndim==1:
		bin_lower_edges = bin_edges[1:]
		bin_upper_edges = bin_edges[:-1]
	elif bin_edges.ndim==2:
		bin_lower_edges = bin_edges[0]
		bin_upper_edges = bin_edges[1]
	else:
		sys.exit(f'ERROR: Cannot deal with "bin_edges" of greater than 2 dimensions')

	#ax_grid = fitscube.header.get_axis_grid(hdu, idx=ax_i)
	phys_grid = np.linspace(1, hdu.header[f'NAXIS{ax_i}'], num=hdu.header[f'NAXIS{ax_i}'], dtype=int)
	if stack_coords == 'world':
		ax_grid = fitscube.header.wcs_physical_to_world(hdu, phys_grid[None,...], axes=(a_idx,), order='numpy')[0]
	elif stack_coords == 'physical':
		ax_grid = phys_grid
	else:
		raise Exception(f'ERROR: "stack_along_axis()" requires argument "stack_coords" be one of ("world","physical"). Currently, stack_coords="{stack_coords}"')
	
	gridpoints_in_bins = np.array([(l<ax_grid)&(ax_grid<h) for l,h in zip(bin_lower_edges,bin_upper_edges)])	

	stacked_data_shape = np.array(hdu.data.shape)
	stacked_data_shape[a_idx] = bin_lower_edges.size
	
	stacked_data = np.full(stacked_data_shape, fill_value=np.nan, dtype=hdu.data.dtype)

	if (stack_func in (np.nanmedian, np.nanmean, np.nansum, np.median, np.mean, np.sum)) and ('axis' not in sf_kwargs):
		sf_kwargs['axis'] = a_idx
	
	for i in range(stacked_data_shape[a_idx]):
		hdu_slice = tuple([gridpoints_in_bins[i] if j==a_idx else slice(None) for j in range(stacked_data_shape.size)])
		stacked_data_slice = tuple([i if j==a_idx else slice(None) for j in range(stacked_data_shape.size)])
		stacked_data[stacked_data_slice] = stack_func(hdu.data[hdu_slice], *sf_args, **sf_kwargs)
	
	# put our stacked data into a header data unit that is just like the original
	# but with the relevant header data changed
	new_hdu = hdu.copy()
	new_hdu.data = stacked_data
	new_hdu.header[f'NAXIS{ax_i}'] = stacked_data.shape[a_idx]
	#new_grid = (bin_edges[:-1]+bin_edges[1:])*0.5
	new_grid = (bin_lower_edges+bin_upper_edges)*0.5
	new_hdu.header[f'CD{ax_i}_{ax_i}'] = np.mean(new_grid[1:]-new_grid[:-1]) if new_grid.size > 1 else 0
	new_hdu.header[f'CRPIX{ax_i}'] = 1
	new_hdu.header[f'CRVAL{ax_i}'] = new_grid[0]
	new_hdu.header[f'CDELT{ax_i}'] = np.mean(new_grid[1:]-new_grid[:-1]) if new_grid.size > 1 else 0
	
	return(new_hdu)

if __name__=='__main__':
	tc = os.path.expanduser('~/scratch/reduced_images/SINFO.2018-08-31T05:45:50/MOV_Neptune---H+K_0.1_tpl/analysis/obj_NEPTUNE_cal_HIP001115_renormed.fits')
	wav_bins = np.linspace(1.455, 2.455, 201)
	
	with fits.open(tc) as hdul:
		hdu_p = hdul['PRIMARY']
		hdu_s = stack_hdu_along_spectral_axis(hdu_p, wav_bins)
		print(hdu_p.data.shape)
		print(hdu_s.data.shape)
		
		
	
	
