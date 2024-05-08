"""
Helper functions to operate on FITS headers.
"""
import re

import numpy as np

import astropy as ap
from astropy.wcs import WCS

from aopp_deconv_tool.numpy_helper.axes import AxesOrdering

import aopp_deconv_tool.cfg.logs
_lgr = aopp_deconv_tool.cfg.logs.get_logger_at_level(__name__, 'DEBUG')


class DictReader:
	"""
	Put a dictionary into a format that we can insert into a FITS header file.
	Make all uppercase, replace dots with spaces, turn everything into a string
	and hope it's not over 80 characters long. Will need to use HIERARCH keys
	to have long parameter names, see 
	https://fits.gsfc.nasa.gov/fits_standard.html

	Also we should combine dictionaries so that we have a flat structure, one
	way to do this is to concatenate the keys for member dictionaries with the
	parent dictionary, i.e:
		{"some.key" : {"child_key_1" : X1, "child_key_2": X2}} 
	becomes
		{"some.key.child_key_1" : X1, "some.key.child_key_2": X2}
	and is turned into a FITS header like
		HIERARCH SOME KEY CHILD_KEY_1 = "X1"
		HIERARCH SOME KEY CHILD_KEY_2 = "X2"

	If we need to we can split header entries into a "name" and "value" format,
	e.g. the above example would become:
		PKEY1   = some.key.child_key_1
		PVAL1   = "X1"
		PKEY2   = some.key.child_key_2
		PVAL1   = "X2"
	"""
	
	def __init__(self,
			adict: dict,
			mode : str = 'standard', # 'hierarch' | 'standard',
			prefix : str = None,
			pkey_count_start : int = 0
		):
		self.prefix = prefix
		self.adict = adict
		self.mode = mode
		self._key_count_start = pkey_count_start
		

	def __iter__(self):
		self.fits_dict = {}
		self.key_count = self._key_count_start
		if self.mode == 'hierarch':
			self._to_fits_hierarch_format(
				self.adict, 
				prefix=(' '.join(['HIERARCH',self.prefix])) if self.prefix is not None else 'HIERARCH'
			)
		elif self.mode == 'standard':
			self._to_fits_format(self.adict, prefix=self.prefix)
		else:
			raise RuntimeError(f"Unknown mode \"{self.mode}\" for creating FITS header cards from a dictionary, known modes are ('hierarch', 'standard').")
		return(iter(self.fits_dict.items()))

	@staticmethod
	def find_max_pkey_n(hdr, n_max=1000):
		pkey_fmt = 'PKEY{}'
		pkey_fmt_iter = ((i,) for i in range(n_max))
		pkey_max = get_key_fmt_max(hdr, pkey_fmt, pkey_fmt_iter)
		
		return int(pkey_max[4:]) if pkey_max is not None else 0
	
	@staticmethod
	def remove_pkeys(hdr, n_max=1000):
		pkey_fmt = 'PKEY{}'
		pkey_fmt_iter = ((i,) for i in range(n_max))
		remove_key_fmt(hdr, pkey_fmt, pkey_fmt_iter)

	def _to_fits_hierarch_format(self, bdict, prefix=None):
		for key, value in bdict.items():
			fits_key = ('' if prefix is None else ' ').join([('' if prefix is None else prefix), key.replace('.', ' ').upper()])
			if type(value) is not dict:
				self.fits_dict[fits_key] = str(value)
			else:
				self._to_fits_hierarch_format(value, fits_key)
		return

	def _to_fits_format(self, bdict, prefix=None):
		prefix_str = '' if prefix is None else prefix
		prefix_join_str = '' if prefix is None else '.'

		for k, v in bdict.items():
			_k = prefix_join_str.join([prefix_str, k])
			_v = str(v)
			if type(v) is not dict:
				self._add_kv_to_fits_dict(_k,_v)
			else:
				self._to_fits_format(v, _k)
		return

	def _add_kv_to_fits_dict(self, k, v):
		k_key = f'PKEY{self.key_count}'
		v_key = f'PVAL{self.key_count}'
		key_fmt_str = "{: <8}"
		val_fmt_str = "{}" # astropy can normalise the value string if needed
		self.fits_dict[key_fmt_str.format(k_key)] = val_fmt_str.format(k)
		self.fits_dict[key_fmt_str.format(v_key)] = val_fmt_str.format(v)
		self.key_count += 1
		return

def remove_key_fmt(hdr : ap.io.fits.Header, key_fmt, key_fmt_arg_iter):
	last_valid_key = None
	for items in key_fmt_arg_iter:
		key = key_fmt.format(*items)
		hdr.remove(key, ignore_missing=True, remove_all=True)
	

def get_key_fmt_max(hdr : ap.io.fits.Header, key_fmt, key_fmt_arg_iter) -> str:
	last_valid_key = None
	for items in key_fmt_arg_iter:
		key = key_fmt.format(*items)
		if key in hdr:
			last_valid_key = key
	return last_valid_key

def get_celestial_axes(hdr : ap.io.fits.Header, wcsaxes_label=''):
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


def get_spectral_axes(hdr, wcsaxes_label=''):
	fits_spectral_codes = ['FREQ', 'ENER','WAVN','VRAD','WAVE','VOPT','ZOPT','AWAV','VELO','BETA']
	iraf_spectral_codes = ['axtype=wave']
	
	spectral_idxs = []
	for i in AxesOrdering.range(hdr['NAXIS']):
		if any(fits_code==hdr.get(f'CTYPE{i.fits}{wcsaxes_label}', '')[:len(fits_code)] for fits_code in fits_spectral_codes):
			spectral_idxs.append(i.numpy)
		elif any(iraf_code in hdr.get(f'WAT{i.fits}_{"001" if wcsaxes_label=="" else wcsaxes_label}', '') for iraf_code in iraf_spectral_codes):
			spectral_idxs.append(i.numpy)
	return(tuple(spectral_idxs))

def get_polarisation_axes(hdr, wcsaxes_label=''):
	raise NotImplementedError

def get_time_axes(hdr, wcsaxes_label=''):
	raise NotImplementedError


def get_world_coords_of_axis(hdr, ax_idx, wcsaxes_label='', squeeze=True):
	"""
	Gets the world coordiates of an axis
	"""
	ax_idxs = tuple((x if type(x)==AxesOrdering else AxesOrdering(x, hdr['NAXIS'], 'numpy')) for x in (ax_idx if (type(ax_idx) in (list,tuple)) else (ax_idx,)))

	_lgr.debug(f'{ax_idxs=}')

	wcs = WCS(hdr, key=' ' if wcsaxes_label=='' else wcsaxes_label.upper(), naxis=tuple(x.fits for x in ax_idxs))
	
	ss = tuple(slice(0,int(hdr[f'NAXIS{x.fits}'])) for x in ax_idxs)
	
	coord_array = np.mgrid[ss].reshape(len(ax_idxs),-1).T
	
	return(np.squeeze(wcs.all_pix2world(coord_array, 0)))

def is_CDi_j_present(header, wcsaxes_label=''):
	# matches "CDi_ja" where i,j are digits of indeterminate length, and a is an optional uppercase letter in wcsaxes_label
	cdi_j_pattern = re.compile(r'CD\d+_\d+'+wcsaxes_label)
	for k in header.keys():
		amatch = cdi_j_pattern.match(k)
		if amatch is not None:
			return(True)
	return(False)

def get_axes_ordering(hdr, axes, ordering='numpy', wcsaxes_label=''):
	wcsaxes = hdr.get(f'WCSACES{wcsaxes_label}', hdr['NAXIS'])
	return tuple(AxesOrdering(_x, wcsaxes, ordering) for _x in axes)

def set_axes_transform(hdr, axis=None, unit=None, reference_value=None, delta_value=None, n_values=None, reference_pixel=None):
	new_hdr_keys= {
		f'CD{axis}_{axis}' : delta_value,	# Turn meters into Angstrom
		f'CUNIT{axis}' : unit,				# Tell FITS the units
		f'CRVAL{axis}' : reference_value,	# Set the value of the reference pixel in the spectral direction in the FITS file (center of first bin)
		f'CRPIX{axis}' : reference_pixel,	# Tell FITS the index of the reference pixel in the spectral direction (first pixel, FITS is 1-index based)
		f'NAXIS{axis}' : n_values,			# Tell FITS the number of spectral planes
	}
	
	
	
	hdr.update({(k,v if v is not None else hdr[k]) for k,v in new_hdr_keys.items()}) # Update the old header with the new values (in memory, not on disk) so we can use it to write the altered file.



def get_iwc_matrix(hdr, wcsaxes_label=''):
	"""
	Get intermediate world coordinate matrix (CDi_j or PCi_j matrix with scaling applied)
	
	IRAF does things differently, see <https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.46.2794&rep=rep1&type=pdf>
	"""
	naxis = hdr['NAXIS']
	wcsaxes = hdr.get(f'WCSAXES{wcsaxes_label}',naxis)
	iwc_mat = np.zeros((wcsaxes,wcsaxes))
	CD_flag = is_CDi_j_present(hdr, wcsaxes_label)
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

def set_iwc_matrix(hdr, iwc_matrix, wcsaxes_label='', CD_format=None):
	# for now assume that iwc_matrix is all we are getting, set CDELTi to 1
	wcsaxes = hdr_get_wcsaxes(hdr, wcsaxes_label)
	CD_format = is_CDi_j_present(hdr, wcsaxes_label) if CD_format is None else CD_format
	hdr_mat_str_fmt = ('CD' if CD_format else 'PC')+'{i}_{j}{wcsaxes_label}'
	hdr_CDELTi_fmt = 'CDELT{i}{wcsaxes_label}'
	for i in (AxesOrdering(_x,wcsaxes,'numpy') for _x in range(0,wcsaxes)):
		hdr[hdr_CDELTi_fmt.format(i=i, wcsaxes_label=wcsaxes_label)] = 1.0
		for j in (AxesOrdering(_x,wcsaxes,'numpy') for _x in range(0,wcsaxes)):
			hdr[hdr_mat_str_fmt.format(i=i, j=j, wcsaxes_label=wcsaxes_label)] = iwc_matrix[i,j]
	return




























