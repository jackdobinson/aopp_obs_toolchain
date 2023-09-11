#!/usr/bin/env python3
import sys, os
import numpy as np
from astropy.io import fits


def get_coord_fields(hdr):
	naxis = hdr.get('NAXIS', None)
	wcsaxes = hdr.get('WCSAXES', None)

	cf = fits.Header()
	fl = []
	if wcsaxes is None:
		n = naxis
	else:
		n = wcsaxes
		cf['WCSAXES'] = wcsaxes
	if naxis != None:
		cf['NAXIS'] =naxis
	
	fl.append(f'BITPIX')
	fl.append(f'NAXIS')
	fl.append(f'WCSAXES')
	fl.append(f'DATE-OBS')
	fl.append(f'MJD-OBS')
	for i in range(1,n+1):
		fl.append(f'NAXIS{i}')
		fl.append(f'CRVAL{i}')
		fl.append(f'CRPIX{i}')
		fl.append(f'CDELT{i}')
		fl.append(f'CTYPE{i}')
		fl.append(f'CROTA{i}')
		fl.append(f'CUNIT{i}')
		fl.append(f'')
		for j in range(1,n+1):
			fl.append(f'PC{i}_{j}')
			fl.append(f'CD{i}_{j}')
			fl.append(f'PV{i}_{j}')
			fl.append(f'PS{i}_{j}')
	cf.update(take_if_present(hdr,fl))
	return(cf)


def take_if_present(adict, keys):
	a = {}
	for k in keys:
		try:
			a[k] = adict[k]
		except KeyError:
			pass
	return(a)

def get_axis_of_ctype(hdu, ctype='WAVE', n=1):
	"""
	Gets the axis number of the axis with the matching CTYPE header entry.

	ARGUMENTS:
		hdu
			Header data unit to operate on
		ctype
			<str> CTYPE header entry to match (default='WAVE')
		n
			<int> Number of match to return (i.e. if there are multiple matches, return the n^th one)
	RETURNS:
		ax_i
			<int> Axis number of n^th axis with matching CTYPE or "None" if not found
	"""
	ax_i = None
	naxes = hdu.header['NAXIS']
	for i in range(1, naxes+1):
		if hdu.header[f'CTYPE{i}'].strip() == ctype:
			n -= 1
			if n==0:
				ax_i = i
				break
	return(ax_i)

def get_index_of_ctype(hdu, ctype='WAVE', n=1):
	"""
	Get the index (as numpy uses) of the axis with the matching CTYPE header entry. As numpy uses c-type ordering
	and FITS uses fortran-type ordering this means that the numpy ordering is (NAXIS - ax_i)

	See "get_axis_of_ctype()" for explanation of arguments
	"""
	ax_i = get_axis_of_ctype(hdu, ctype=ctype, n=n)
	if ax_i is None:
		return(None)
	return(hdu.header['NAXIS'] - ax_i)

def get_wavelength_grid(hdu, idx=None, ctype='WAVE'):
	# if we haven't been told which axis index we want, try to find via ctype
	if idx is None:
		idx = get_axis_of_ctype(hdu, ctype)

	# if idx is still None then we couldn't find the wavelength axis with the ctype given
	if idx is None:
		print(f'ERROR: "fitscube.header.get_wavelength_grid()" Could not find an axis of type {ctype}, exiting...')
		sys.exit()

	# if we have gotten here we have an index for the wavelength axis
	naxis_i = hdu.header['NAXIS{}'.format(idx)]
	# get an array of pixel index (p)
	p = np.linspace(1, naxis_i, num=naxis_i, dtype=int) # do FITS files use 0 or 1 indexing? I think 1.
	#print(p)
	#assume 1d
	ltmi_i = float(hdu.header.get('LTM{}_{}'.format(idx,idx),1))
	ltvi = float(hdu.header.get('LTV{}'.format(idx), 0))
	l = ltmi_i*p + ltvi
	#print(l)

	crpix_i = int(hdu.header['CRPIX{}'.format(idx)])
	crval_i = float(hdu.header['CRVAL{}'.format(idx)])
	cdelt_i = float(hdu.header['CDELT{}'.format(idx)])
	cdi_i = float(hdu.header.get('CD{}_{}'.format(idx,idx),cdelt_i))
	#print(crpix_i, crval_i, cdelt_i, cdi_i)
	#w = np.zeros([naxis1])
	w = crval_i + cdi_i*(l-crpix_i)
	return(w)

def get_axis_grid(hdu, idx=None):
	"""
	Get the range of values in the axis grid using the *.fits header index to choose the axis
	
	N.B. ONLY WORKS WITH ORTHOGINAL AXES
	"""
	# we have an index for the  axis
	naxis_i = hdu.header['NAXIS{}'.format(idx)]
	# get an array of pixel index (p)
	# get pixel (physical) coordinates
	p = np.linspace(1, naxis_i, num=naxis_i, dtype=int) # do FITS files use 0 or 1 indexing? I think 1.
	#print(p)
	#assume 1d
	# turn pixel coords into logical coordinates
	ltmi_i = float(hdu.header.get('LTM{}_{}'.format(idx,idx),1))
	ltvi = float(hdu.header.get('LTV{}'.format(idx), 0))
	l = ltmi_i*p + ltvi
	#print(l)

	crpix_i = int(hdu.header['CRPIX{}'.format(idx)])
	crval_i = float(hdu.header['CRVAL{}'.format(idx)])
	cdelt_i = float(hdu.header['CDELT{}'.format(idx)])
	cdi_i = float(hdu.header.get('CD{}_{}'.format(idx,idx),cdelt_i))
	#print(crpix_i, crval_i, cdelt_i, cdi_i)
	#w = np.zeros([naxis1])
	w = crval_i + cdi_i*(l-crpix_i)
	return(w)

def wcs_get_mat(hdu, pref, n, m=None):
	if m is None:
		m = n
	mat = np.eye(n, m, dtype=float)
	for i in range(1,n+1):
		for j in range(1,m+1):
			key = f'{pref}{i}_{j}'
			if key in hdu.header:
				mat[i-1,j-1] = float(hdu.header[key])
	return(mat)

def wcs_get_vec(hdu, pref, n, default=0):
	vec = np.ones((n))*default
	for i in range(1,n+1):
		key = f'{pref}{i}'
		if key in hdu.header:
			vec[i-1] = float(hdu.header[key])
	return(vec)

def as_colvec(vec):
	return(vec[...,None])
def as_rowvec(vec):
	return(vec[None,...])

def wcs_physical_to_logical(hdu, points=None, axes=None):
	"""
	Transforms from physical to 'logical' coordinates
	"""
	N = hdu.header['NAXIS']
	if axes is None:
		ax_range = range(1,N+1)
		m_slice = slice(None)
		v_slice = slice(None)
	else:
		ax_range = tuple([_x+1 for _x in axes])
		m_slice = np.ix_(axes,axes)
		v_slice = np.ix_(axes)
	
	if points is None:
		a = []
		for i in ax_range:
			a.append(np.linspace(1, hdu.header[f'NAXIS{i}'], num=hdu.header[f'NAXIS{i}'], dtype=int))
		b = np.meshgrid(*a)
		points = np.vstack([_x.flatten() for _x in b])
	
	ltm = wcs_get_mat(hdu, 'LTM', N)[m_slice]
	ltv = wcs_get_vec(hdu, 'LTV', N)[v_slice]
	
	l = np.matmul(ltm, points) + as_colvec(ltv)
		
	return(l)			

def wcs_logical_to_world(hdu, l, axes=None):
	"""
	Changes from logical coordinates (a scaled version of pixel or "physical" coordinates) to world coordinates.
	
	TODO: 
		* Doesn't check that we have all the information to uniquely identify a coordinate.
		I.e. currently we can have a coord system that needs axis 1 and 2 to properly identify
		the output (i.e. sm is not an identity matrix), but we can just specify to calculate along
		axis 1 without supplying any axis 2 information.
	"""
	N = hdu.header['NAXIS']
	if axes is None:
		ax_range = range(1,N+1)
		m_slice = slice(None)
		v_slice = slice(None)
	else:
		ax_range = axes
		m_slice = np.ix_(axes,axes)
		v_slice = np.ix_(axes)
	
	
	pc_exists = False
	for i in ax_range:
		for j in ax_range:
			if hdu.header.get(f'PC{i}_{j}', None) is not None:
				pc_exists = True
				break
		if pc_exists:
			break
	
	cd_exists = False
	for i in ax_range:
		for j in ax_range:
			if hdu.header.get(f'CD{i}_{j}', None) is not None:
				cd_exists = True
				break
		if cd_exists:
			break 
	
	if cd_exists and pc_exists:
		raise Exception('ERROR: WCS cannot have PCi_j and CDi_j defined at the same time')
	
	if cd_exists:
		sm = wcs_get_mat(hdu, 'CD', N)[m_slice]
	elif pc_exists:
		sm = wcs_get_mat(hdu, 'PC', N)[m_slice]*as_colvec(wcs_get_vec(hdu, 'CDELT', N, default=1)[v_slice])
	else:
		raise Exception('ERROR: WCS must have one of PCi_J or CDi_j defined')
	
	crpix = wcs_get_vec(hdu, 'CRPIX', N)[v_slice]
	crval = wcs_get_vec(hdu, 'CRVAL', N)[v_slice]
	
	
	w = np.matmul(sm, (l - as_colvec(crpix))) + as_colvec(crval)
		   
	return(w)

def wcs_physical_to_world(hdu, points = None, axes=None, order='numpy'):
	"""
	Changes physical (pixel) coordinates to world (e.g. ra-dec) coordinates
	
	# ARGUMENTS #
		hdu
			Header data unit to grab coordinate system information from
		points
			physical (pixel) coordinates to change to world coordinates, must be
			of shape (NAXIS, M) where NAXIS is the number of axes in the fits file,
			and M is the number of points. I.e. a set of column vectors. If None, will
			try and transform every pixel to it's world coordiante.
		axes
			A tuple (or None to include all) of the axis numbers to perform the calculation for,
			the order of the axes here corresponds to the order of the points and the output order.
			For example, if a cube has 3 axes; 1=ra, 2=dec, and 3=spectral. Then setting 
			axes=(1,2) and order='fits' will result in the input points being in fits file order (1, 2)
			and the output world points being in order (ra, dec).
			Setting the same cube with axes=(0,2) and order='numpy' will result in the 
			input points being (1,3) in fits file order and the output world points
			being (ra, spectral).
		order
			Defines what ordering the axes will be returned and/or input in. If
			'numpy' then the 0th axis will be the Nth axis in the fits file. If
			'fits' then the 0ht axis will be the 1st axis in the fits file.
	
	# RETURNS #
		w
			The world coordinate equivalents of the input points coordinates.
			
	"""
	input_axes_str = f'{axes}'
	
	if order not in ('numpy', 'fits'):
		raise Exception(f'ERROR: "wcs_physical_to_world()" requires "order" argument is one of ("numpy","fits"), order={order}.')
	if order == 'numpy':
		if points is not None:
			points = points[::-1,...]
		if axes is not None:
			axes = tuple([hdu.header['NAXIS']-(_x+1) for _x in axes])
			
	elif order == 'fits':
		if axes is not None:
			axes = tuple([_x-1 for _x in axes])
			
	if (points is not None):
		n_axes = hdu.header['NAXIS'] if axes is None else len(axes)
		if points.shape[0] != n_axes:
			raise Exception(f'ERROR: "wcs_physical_to_world()" requires that the number of dimensions in each point is equal to the number of axes to transform. Currently have {points.shape[0]} dimensions in each point and {n_axes} axes.')
	
	if np.any([_x<0 for _x in axes]) or np.any([_x>(hdu.header['NAXIS']-1) for _x in axes]):
		raise Exception(f'ERROR: "wcs_physical_to_world()" requires axes=None, or axes={{0->(NAXIS-1)}} if order="numpy", or axes={{1->NAXIS}} if order="fits". Currently order="{order}" axes={input_axes_str}.')
	
	l = wcs_physical_to_logical(hdu, points=points, axes=axes)
	w = wcs_logical_to_world(hdu, l, axes=axes)
	
	return(w)

if __name__=='__main__':
	import numpy as np
	from astropy.io import fits
	
	tc = os.path.expanduser('~/scratch/reduced_images/SINFO.2018-08-31T05:45:50/MOV_Neptune---H+K_0.1_tpl/analysis/obj_NEPTUNE_cal_HIP001115_renormed.fits')
	with fits.open(tc) as hdul:
		#print(get_axis_grid(hdul['PRIMARY'], 1))
		
		#a = as_colvec(np.array([2000,68]))
		a = as_rowvec(np.array([2000,68]))
		w = wcs_physical_to_world(hdul['PRIMARY'], points=a, axes=(3,), order='fits')
		print(w)
		print(hdul['PRIMARY'].data.shape)
	
