#!/usr/bin/env python3
"""
Contains utility functions for operating on fits files
"""
import sys
sys.path = sys.path[1:] + sys.path[:1] # move current directory to end of search path.

import utilities.logging_setup
logging, _lgr = utilities.logging_setup.getLoggers(__name__, 'INFO')

import typing
import types
import numpy as np
from astropy.io import fits
import re

import utilities as ut
import utilities.dict
import utilities.np

class AxesOrdering:
	__slots__ = (
		'_idx',
		'_n',
	)
	def __init__(self, 
			idx : int, 
			n : int, 
			ordering : typing.Literal['numpy','fits','fortran']
		) -> None:
		self.n = n
		setattr(self, ordering, idx)
		return

	@classmethod
	def range(cls, start, stop=None, step=1):
		# make interface work like "range()" command
		if stop is None:
			stop = start
			start = 0
		return(cls(x, stop-start, 'numpy') for x in range(start,stop,step))

	@property
	def n(self) -> int:
		return(self._n)
	@n.setter
	def n(self, value: int) -> None:
		self._n = value
		return

	# numpy representation is the internal one, so just return attribute
	@property
	def numpy(self) -> int:
		return(self._idx)
	@numpy.setter
	def numpy(self,value:int)->None:
		self._idx = value
		return

	@property
	def fits(self)->int:
		if isinstance(self._idx, typing.Sequence):
			return(type(self._idx)([self._n - _i for _i in self._idx]))
		return(self._n - self._idx)
	@fits.setter
	def fits(self,value:int)->None:
		# convert to numpy representation and store
		if isinstance(value, typing.Sequence):
			self._idx = type(value)([self._n - _v for _v in value])
		else:
			self._idx = self._n - value
		return

	@property
	def fortran(self)->int:
		if isinstance(self._idx, typing.Sequence):
			return(type(self._idx)([1+_i for _i in self._idx]))
		return(1+self._idx)
	@fortran.setter
	def fortran(self, value:int)->None:
		if isinstance(value, typing.Sequence):
			self._idx = type(value)([_v-1 for _v in value])
		else:
			self._idx = value-1
		return
# END class AxesOrdering


def get_img_from_fits(
		fits_file 			: str,
		fits_ext 			: typing.Union[str, int] = 'PRIMARY',
		off_axes_treatment 	: str = 'center',
		axes 				: typing.Optional[typing.Tuple[int,int]] = (1,2),
		ordering 			: typing.Literal['numpy','fits','fortran'] = 'numpy',
	):
	# get the information we need from the fits file
	with fits.open(fits_file) as hdul:
		data = np.array(hdul[fits_ext].data)
		naxes =  hdul[fits_ext].header['NAXIS']
		saxes = tuple([hdul[fits_ext].header[f'NAXIS{_i+1}'] for _i in range(naxes)][::-1])
	
	# ensure we have numpy axes ordering
	axes = AxesOrdering(axes, naxes, ordering).numpy
	all_axes = tuple(range(naxes))
	not_present_axes = tuple([_i for _i in all_axes if _i not in axes])
	slices_and_idxs = []
	
	# build slices that define desired data from fits file
	for _i in all_axes:
		if off_axes_treatment == 'center':
			#_i_fits = AxesOrdering(_i, naxes, 'numpy').fits
			slices_and_idxs.append(slice(None) if _i in axes else saxes[_i]//2)
		else:
			slices_and_idxs.append(slice(None)) if _i in axes else None
	#print(f'DEBUGGING: slices_and_idxs {slices_and_idxs}')
	if off_axes_treatment == 'mean':
		data = np.mean(data, axis=not_present_axes)
	#print(f'DEBUGGING: data.shape {data.shape}')

	# return desired data slice
	return(data[tuple(slices_and_idxs)])


def hdr_get_iwc_matrix(hdr, wcsaxes_label=''):
	"""
	Get intermediate world coordinate matrix (CDi_j or PCi_j matrix with scaling applied)
	
	IRAF does things differently, see <https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.46.2794&rep=rep1&type=pdf>
	"""
	naxis = hdr['NAXIS']
	wcsaxes = hdr.get(f'WCSAXES{wcsaxes_label}',naxis)
	iwc_mat = np.zeros((wcsaxes,wcsaxes))
	CD_flag = hdr_is_CDi_j_present(hdr, wcsaxes_label)
	for i in (AxesOrdering(_x,wcsaxes,'numpy') for _x in range(0,wcsaxes)):
		for j in (AxesOrdering(_x,wcsaxes,'numpy') for _x in range(0,wcsaxes)):
			#print(i, j)
			#print(i.numpy, j.numpy)
			#print(i.fits, j.fits)
			if CD_flag:
				iwc_mat[i.numpy,j.numpy] = hdr.get(f'CD{i.fits}_{j.fits}{wcsaxes_label}',0)
			else:
				default = 1 if i==j else 0
				iwc_mat[i.numpy,j.numpy] = hdr.get(f'PC{i.fits}_{j.fits}{wcsaxes_label}',default)*hdr.get('CDELT{i.fits}{wcsaxes_label}',1)
	return(iwc_mat)

def hdr_get_wcsaxes(hdr, wcsaxes_label):
	return(hdr.get(f'WCSAXES{wcsaxes_label}',hdr['NAXIS']))

def hdr_set_iwc_matrix(hdr, iwc_matrix, wcsaxes_label='', CD_format=False):
	# for now assume that iwc_matrix is all we are getting, set CDELTi to 1
	wcsaxes = hdr_get_wcsaxes(hdr, wcsaxes_label)
	hdr_mat_str_fmt = ('CD' if CD_format else 'PC')+'{i}_{j}{wcsaxes_label}'
	hdr_CDELTi_fmt = 'CDELT{i}{wcsaxes_label}'
	for i in (AxesOrdering(_x,wcsaxes,'numpy') for _x in range(0,wcsaxes)):
		hdr[hdr_CDELTi_fmt.format(i=i, wcsaxes_label=wcsaxes_label)] = 1.0
		for j in (AxesOrdering(_x,wcsaxes,'numpy') for _x in range(0,wcsaxes)):
			hdr[hdr_mat_str_fmt.format(i=i, j=j, wcsaxes_label=wcsaxes_label)] = iwc_matrix[i,j]
	return

def hdr_get_axis_world_coords(hdr, ax_idx, wcsaxes_label=''):
	"""
	Gets the world coordiates of an axis
	"""
	from astropy.wcs import WCS
	ax_idxs = tuple((x if type(x)==AxesOrdering else AxesOrdering(x, hdr['NAXIS'], 'numpy')) for x in (ax_idx if (type(ax_idx) in (list,tuple)) else (ax_idx,)))


	#ax_idx = ax_idx if type(ax_idx)==AxesOrdering else AxesOrdering(ax_idx,hdr["NAXIS"],"numpy")
	#_lgr.DEBUG(f"{hdr=} {ax_idx=} {wcsaxes_label=} {ax_idx.fits=}")
	wcs = WCS(hdr, key=' ' if wcsaxes_label=='' else wcsaxes_label.upper(), naxis=tuple(x.fits for x in ax_idxs))
	#print(wcs)
	#ns = tuple(int(hdr[f'NAXIS{x.fits}']) for x in ax_idxs)
	#return(wcs.all_pix2world(np.linspace(0, n-1, n)[:,None],0)[:,0], wcs)
	ss = tuple(slice(0,int(hdr[f'NAXIS{x.fits}'])) for x in ax_idxs)
	#print(ss)
	#print(ax_idxs)
	#print(len(ax_idxs))
	coord_array = np.mgrid[ss].reshape(len(ax_idxs),-1).T
	#if len(ax_idx) > 1:
	#	coord_array = np.mgrid[ss].reshape(len(ax_idxs),-1).T
	#else:
	#	coord_array = np.mgrid[ss[0]]
		
	#print(f'{coord_array=}')
	return(wcs.all_pix2world(coord_array, 0))

def hdr_get_unit_si(hdr, ax_idx, wcsaxes_label='', default=1):
	ax_idx = ax_idx if type(ax_idx)==AxesOrdering else AxesOrdering(ax_idx,hdr["NAXIS"],"numpy")
	unit_hdr_str = f'CUNIT{ax_idx.fits}{wcsaxes_label}'
	unit_str = hdr.get(unit_hdr_str, '').strip() # remove whitespace
	if unit_str == 'um':
		return(1E-6)
	if unit_str == 'Angstrom':
		return(1E-10)
	else:
		return(default)


def hdr_axis_is_independent(hdr, ax_idx, wcsaxes_label=''):
	iwc_matrix = hdr_get_iwc_matrix(hdr, wcsaxes_label)
	for i in range(iwc_matrix.shape[0]): # square matrix
		if (i!=ax_idx) and (iwc_matrix[i,ax_idx] != 0):
			return(False)
	return(True)


def hdr_get_spectral_axes(hdr, wcsaxes_label=''):
	fits_spectral_codes = ['FREQ', 'ENER','WAVN','VRAD','WAVE','VOPT','ZOPT','AWAV','VELO','BETA']
	iraf_spectral_codes = ['axtype=wave']
	
	spectral_idxs = []
	for i in AxesOrdering.range(hdr['NAXIS']):
		if any(fits_code==hdr.get(f'CTYPE{i.fits}{wcsaxes_label}', '')[:len(fits_code)] for fits_code in fits_spectral_codes):
			spectral_idxs.append(i.numpy)
		elif any(iraf_code in hdr.get(f'WAT{i.fits}_{"001" if wcsaxes_label=="" else wcsaxes_label}', '') for iraf_code in iraf_spectral_codes):
			spectral_idxs.append(i.numpy)
	return(tuple(spectral_idxs))

def hdr_get_celestial_axes(hdr, wcsaxes_label=''):
	placeholders = ('x','y')
	fits_celestial_codes = ['RA--', 'DEC-', 'xLON','xLAT', 'xyLN', 'xyLT']
	iraf_celestial_codes = ['axtype=xi', 'axtype=eta']

	celestial_idxs = []
	for i in AxesOrdering.range(hdr['NAXIS']):
		if any(fits_code==hdr.get(f'CTYPE{i.fits}{wcsaxes_label}', '')[sum(fits_code.count(p) for p in placeholders):len(fits_code)] 
						for fits_code in fits_celestial_codes) \
				or any(iraf_code in hdr.get(f'WAT{i.fits}_{"001" if wcsaxes_label=="" else wcsaxes_label}', '') 
						for iraf_code in iraf_celestial_codes) \
				:
			celestial_idxs.append(i.numpy)
	return(tuple(celestial_idxs))

def get_hdr_coord_fields(hdr):
	"""
	Get header fields that correspond to data axes and WCS data. If a field is missing, don't grab it.

	THIS FAILS FOR SOME UNKNOWN REASON
	"""
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
	cf.update(dict((f,hdr[f]) for f in fl if f in hdr))
	return(cf)

def hdr_is_CDi_j_present(header, wcsaxes_label=''):
	# matches "CDi_ja" where i,j are digits of indeterminate length, and a is an optional uppercase letter in wcsaxes_label
	cdi_j_pattern = re.compile(r'CD\d+_\d+'+wcsaxes_label)
	for k in header.keys():
		amatch = cdi_j_pattern.match(k)
		if amatch is not None:
			return(True)
	return(False)


def hdu_apply_slice(hdu, slice_tuple):
	hdu.header = header_apply_slice(hdu.header, slice_tuple)
	hdu.data = hdu.data[slice_tuple]
	return(hdu)


HDR_DEFAULT_KEYS = {
	re.compile(r'NAXIS') : 0,
	re.compile(r'NAXIS\d+(_[A-Z])?') : 1,
	re.compile(r'CRPIX\d+(_[A-Z])?') : 1, 
	re.compile(r'CDELT\d+(_[A-Z])?') : 1, 
	re.compile(r'CD(\d+)_(\d+)') : 0,
	re.compile(r'PC(\d+)_(\d+)') : lambda match: 1 if match[1]==match[2] else 0,
}

def hdr_get_default(key):
	for k, v in HDR_DEFAULT_KEYS.items():
		match = k.match(key)
		if match is not None:
			if callable(v):
				return(v(match))
			return(v)
	raise ValueError(f'No match for {key=} in HDR_DEFAULT_KEYS')

def hdr_ensure_keys(hdr, keys):
	"""
	hdr
		Header to ensure has the passed keys
	keys
		Sequence of strings (to be used with "hdr_get_default()") or (string, default) tuples.
		If the key is not present in "hdr", will be assigned the default value found via
		hdr_get_default() if element of "keys" is a string, or the second member of the tuple otherwise.
	"""
	for x in keys:
		if type(x) is str:
			if x not in hdr: hdr[x] = hdr_get_default(x)
		else:
			if x[0] not in hdr: hdr[x[0]] = x[1]
	return


def header_apply_slice(hdr, slice_tuple):
	"""
	It's difficult to correctly slice a fits file as the header data needs
	to change depending on the shape of the slice. Also numpy and fits ordering
	are annoyingly different.

	ASSUMPTION: we're assuming that 'slice_tuple' doesn't change the number of dimensions,
	if that isn't the case I'll have to re-write this to accout for that case.
	"""
	if type(slice_tuple) != tuple:
		raise TypeError(f'In "hdu_apply_slice()", expected argument "slice_tuple" to be of type "tuple", not {type(slice_tuple)=}')

	# assume that "slice_tuple" is full of actual slices, and not indices
	# i.e. assume that NAXIS doesn't change for now.
	axes_idxs = tuple(AxesOrdering(i, hdr['NAXIS'], 'numpy') for i in range(hdr['NAXIS']))
	initial_shape = tuple(hdr[f'NAXIS{ax_j.fits}'] for ax_j in axes_idxs)
	_lgr.DEBUG(f'{tuple(_x.fits for _x in axes_idxs)=}')
	_lgr.DEBUG(f'{initial_shape=}')
	
	pixel_tranformation_matrix_CD_flag = hdr_is_CDi_j_present(hdr)

	for aslice, ax_j in zip(slice_tuple, axes_idxs):
		start, stop, step = aslice.indices(initial_shape[ax_j.numpy])
		NAXISj_str = f'NAXIS{ax_j.fits}'
		hdr[NAXISj_str] = (stop - start)//step # update the number of elements in axis j
		_lgr.DEBUG(f'{start=} {stop=} {step=} {NAXISj_str=} {hdr[NAXISj_str]=}')
		if start != 0:
			CRPIXj_str = f'CRPIX{ax_j.fits}'
			# we have to offset CRPIXj
			hdr[CRPIXj_str] = hdr.get(CRPIXj_str, hdr_get_default(CRPIXj_str)) - start
			_lgr.DEBUG(f'{CRPIXj_str=} {hdr[CRPIXj_str]=}')
		
		if step != 1:
			# we have to scale CRPIXn, and CDELTn CDn_m PCn_m must also scale if they are present
			CRPIXj_str = f'CRPIX{ax_j.fits}'
			CDELTi_str = f'CDELT{ax_j.fits}'
			
			# e.g. if step=2, then what was pixel 100 is now pixel 50, 
			# but the first pixel should stay the same (it's accounted for when we correct start
			scale_1_indexed_value = lambda x, s: (x-1)*s + 1
			hdr[CRPIXj_str] = scale_1_indexed_value(hdr[CRPIXj_str], 1/step)
			_lgr.DEBUG(f'{CRPIXj_str=} {hdr[CRPIXj_str]=}')
		
			# Should always adjust CDELTi values. CDi_j will ignore them, nut PCi_j need them.
			# e.g. if step=2, then instead of moving 0.1" per pixel, we now move 0.2" per pixel
			hdr[CDELTi_str] = hdr.get(CDELTi_str,1.0)*scale # account for defaults
			_lgr.DEBUG(f'{CDELTi_str=} {hdr[CDELTi_str]=}')

			if pixel_tranformation_matrix_CD_flag:
				for other_ax in axes_idxs:
					CDi_j_str = f'CD{other_ax.fits}_{ax_j.fits}'
					# same logic as CDELTn_str
					if CDi_j_str in hdr:
						hdr[CDi_j_str] *= step
						_lgr.DEBUG(f'{CDi_j_str=} {hdr[CDi_j_str]=}')
						CD_present_flag = True
			else:
				for other_ax in axes_idxs:
					default_value = 1 if other_ax.fits==ax_j.fits else 0
					PCi_j_str = f'PC{other_ax.fits}_{ax_j.fits}'
					if (PCi_j_str in hdr) or (other_ax.fits== ax_j.fits):
						hdr[PCi_j_str] = hdr.get(PCi_j_str, default_value)*step
						_lgr.DEBUG(f'{PCi_j_str=} {hdr[PCi_j_str]=}')
	return(hdr)

def header_value_transform(value, escape_quotes=True):
	"""
	Transform values into valid strings for FITS headers
	"""
	_lgr.DEBUG('In "header_value_transform()"')
	_lgr.DEBUG(f'{value=} {type(value)=} {escape_quotes=}')
	if type(value) is str:
		value = value.replace("'", "''") if escape_quotes else value
		return(f"'{value}'")
	elif type(value) is bool:
		return('T' if value else 'F')
	elif type(value) is float:
		if np.isposinf(value): return(header_value_transform("+INF"))
		if np.isneginf(value): return(header_value_transform("-INF"))
		if np.isnan(value): return(header_value_transform("NAN"))
		return(value)
	elif type(value) is int:
		return(value)
	elif type(value) is types.FunctionType:
		return(header_value_transform('.'.join([value.__module__, value.__qualname__])))
	elif value is None:
		return("") # None should be an empty value field as it represents the absence of a value
	else:
		raise TypeError(f'Uknown type {type(value)} when converting to FITS header representation')


def header_add_hierarch_card(hdr, key, value, comment=None, fmt_str='{}', encoding='ascii'):
	"""
	Atropy does not have native support for HIERARCH keywords or the CONTINUE keyword. This function
	enables the ability to write them to a header.
	"""
	hierarch_key = f'HIERARCH {key} = '
	space_left = 80 - len(hierarch_key)
	hierarch_value = fmt_str.format(header_value_transform(value))
	hierarch_comment = ' / ' + comment if comment is not None else ''
	hierarch_field = bytes(''.join([hierarch_key, hierarch_value, hierarch_comment]), encoding)
	_lgr.DEBUG(f'{hierarch_field = }')
	_lgr.DEBUG(f'{len(hierarch_field)=}')
	if len(hierarch_field) <= 80:
		hierarch_card = hierarch_field[:80]
		hdr.append(fits.Card.fromstring(hierarch_card))
	else:
		space_left = 80 - len(hierarch_key)

		# split value in space_left-3, and then 67 character blocks
		hv_raw = hierarch_value[1:-1]
		hv_ss = [hv_raw[:space_left-3]]
		n = space_left -3
		while n < len(hv_raw):
			hv_ss.append(hv_raw[n:n+67])
			n+= 67

		_lgr.DEBUG('TESTING POINT HERE')
		# append '&' characters to each substring except the last one
		hv_ss = [ss+'&' if i != len(hv_ss)-1 else ss for i, ss in enumerate(hv_ss)]


		comment_start_line = len(hv_ss)-1 # zero indexed
		hc_ss = []
		if comment is not None:
			# work out how much space we have left to fit the comment in, single quotes for string take 2 characters, " / " for comment takes 3 characters
			space_left = (70 if len(hv_ss)!=1 else (80 - len(hierarch_key))) - len(hv_ss[-1]) - 2 - 3
			if space_left < len(comment):
				space_left -= 1 # need to add contiuation character to end of substring
				hv_ss[-1] = hv_ss[-1]+'&'
				# and chop up comments into 64 character chunks
				n = space_left
				hc_ss.append(comment[:n])
				while n < len(comment):
					hv_ss.append('&')
					hc_ss.append(comment[n:n+64])
					n+=64
			else:
				# can just append comment as is
				hc_ss.append(comment)

		hd_ss = []
		hk_ss = []
		for i in range(0, len(hc_ss)+comment_start_line+1): # comment_start_line is zero indexed
			hd_ss.append(header_value_transform(hv_ss[i], escape_quotes=False) + ('' if ((comment is None) or (i<comment_start_line)) else ' / '+hc_ss[i-comment_start_line]))
			if i==0:
				hk_ss.append(hierarch_key)
			else:
				hk_ss.append("CONTINUE  ")

		hierarch_cards = []
		for hk, hd in zip(hk_ss, hd_ss):
			# combine header and data parts of header card, remember to pad to 80 total characters.
			hierarch_cards.append(hk+hd+' '*(80-len(hk)-len(hd)))

		# Using hacky back-door access to the fits.Card classes properties, I need to be able to avoid
		# the setters and getters (which enforce verification) to be able to write the HIERARCH... and CONTINUE keywords.
		for hierarch_card in hierarch_cards:
			card = fits.Card()
			card._image = hierarch_card # access protected attribute instead of setting via ```card.image=```
			hdr.append(card)
	return


def hdu_rebin_to(hdu, target_header, ax_idx=0, combine_func=np.sum):
	"""
	Rebins the data along axis 'ax_idx' in an HDU to have the same structure as describe in the WCS of 'target_header'.
	
	ARGUMENTS:
		hdu
			Header data unit to modify
		target_header
			Header that has the WCS structure we want to emulate
		ax_idx
			The axis to change. If a list or tuple, will use ax_idx[0] as the axis to emulate (from target_header)
			and ax_idx[1] as the axis to change (in hdu). If an integer, will use the same index for both.
		combine_func
			How are the old bins to be combined to create the new ones? (default is to sum them)
	"""
	if type(ax_idx) not in (list, tuple):
		ax_idx = (ax_idx, ax_idx)
	_lgr.DEBUG(f"{ax_idx=}")
	target_grid, target_wcs = hdr_get_axis_world_coords(target_header, ax_idx[0])
	original_grid, original_wcs = hdr_get_axis_world_coords(hdu.header, ax_idx[1])
	#_lgr.DEBUG(f"{target_wcs=}")
	#_lgr.DEBUG(f"{original_wcs=}")

	
	_lgr.DEBUG(f"{target_wcs.to_header()=}")
	_lgr.DEBUG(f"{original_wcs.to_header()=}")

	_lgr.DEBUG(f'{target_header}')
	_lgr.DEBUG(f'{hdu.header}')

	_lgr.DEBUG(f'{target_grid}')
	_lgr.DEBUG(f'{original_grid}')

	if np.any(original_grid != target_grid):
		hdu.data = ut.np.rebin_to(
			hdu.data,
			ut.np.get_bin_edges(original_grid),
			ut.np.get_bin_edges(target_grid),
			combine_func = combine_func,
			axis=ax_idx[1]
		)
		wcs_header = wcs_to_header(target_wcs, axes=(ut.fits.AxesOrdering(ax_idx[1],hdu.header['NAXIS'],'numpy'),), CD_format=hdr_is_CDi_j_present(hdu.header), replace_keys = list(hdu.header.keys()))
		hdu.header.update(wcs_header)
	return

wcs_defaults = {
	re.compile(r'CRPIX\d+') : 0,
	re.compile(r'CRVAL\d+') : 0,
	re.compile(r'CDELT\d+') : 1,
	re.compile(r'PC\d+_\d+') : 0,
	re.compile(r'CD\d+_\d+') : 0,
}
def wcs_to_header(wcs, axes : tuple[AxesOrdering] = None, replace_required=True, replace_keys=[], CD_format=False):
	src_axes = tuple(AxesOrdering(x,wcs.world_n_dim,'numpy') for x in range(wcs.world_n_dim))
	if axes is None:
		axes = src_axes[:]
	wcs_header = wcs.to_header()

	if CD_format:
		hdr_PC_to_CD(wcs_header)
	
	output_hdr = fits.Header()
	for k, v in wcs_header.items():
		for sax, ax in zip(src_axes, axes):
			if type(ax) != AxesOrdering:
				raise TypeError("argument 'axes' must be a sequence of type {AxesOrdering}, currently {type(ax)=}")
			k = k.replace(str(sax.fits), str(ax.fits)) # replace all axes numbers we find. may need to fiddle with this for multiple-digit axes
		if k in replace_keys:
			output_hdr[k] = v
		if replace_required:
			for regex, default in wcs_defaults.items():
				if regex.match(k):
					if v != default: output_hdr[k] = v
					break #should only match one regex
		else:
			output_hdr[k] = v
	return(output_hdr)




def hdr_PC_to_CD(hdr):
	PC_regex = re.compile(r'PC(?P<matrix_el>(?P<intermediate_ax_id>\d+)_(?P<pixel_ax_id>\d+)(?P<wcsaxes_label>[A-Z]?))')
	CDELT_regex = re.compile(r'CDELT(?P<ax_id>(?P<pixel_ax_id>\d+)(?P<wcsaxes_label>[A-Z]?))')
	PC_list = []
	CDELT_list = []
	for k, v in hdr.items():
		PC_match = PC_regex.match(k)
		CDELT_match = CDELT_regex.match(k)
		if PC_match is not None:
			PC_list.append((int(PC_match.group('intermediate_ax_id')), int(PC_match.group('pixel_ax_id')), PC_match.group('wcsaxes_label'), v))
			hdr.pop(k)
		if CDELT_match is not None:
			CDELT_list.append((int(CDELT_match.group('pixel_ax_id')), CDELT_match.group('wcsaxes_label'), v))
	
	for PC in PC_list:
		value = None
		for CDELT in CDELT_list:
			if CDELT[:-1] == PC[1:-1]:
				value = CDELT[-1]*PC[-1]
				break
		if value is None:
			value = PC[-1]
		hdr['CD{}_{}{}'.format(*PC[:-1])] = value 
	return
	
if __name__ == '__main__':
	import sys, os, math
	import matplotlib.pyplot as plt
	_lgr.setLevel('DEBUG')

	for afile in sys.argv[1:]:
		if not os.path.exists(afile):
			print(f'File {afile} not found, skipping...')
			continue
		while True:
			fits.info(afile)
			response = input('Print Extension Header (+ve int) | Show Extension Data (-ve int) | Skip (other)? ')
			ext, sign= None, None
			try:
				ext = abs(int(response))
				sign = math.copysign(1, float(response))
			except ValueError:
				break
			if sign == +1:
				print(fits.getheader(afile, ext).tostring(sep='\n'))
			else:
				data = fits.getdata(afile,ext)
				slice_msg = f'Slice specification to show for data of shape {data.shape} (python slice format) | Exit (other) : '
				while True:
					try:
						sliced_data = eval(f"data{input(slice_msg)}")
					except:
						break
					plt.imshow(sliced_data)
					plt.show()
