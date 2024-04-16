"""
Helper functions to operate on FITS headers.
"""
import astropy as ap

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
			mode : str = 'standard', # 'hierarch' | 'standard'
		):
		self.adict = adict
		self.mode = mode

	def __iter__(self):
		self.fits_dict = {}
		self.key_count = 0
		if self.mode == 'hierarch':
			self._to_fits_hierarch_format(self.adict, prefix='HIERARCH')
		elif self.mode == 'standard':
			self._to_fits_format(self.adict, prefix=None)
		else:
			raise RuntimeError(f"Unknown mode \"{self.mode}\" for creating FITS header cards from a dictionary, known modes are ('hierarch', 'standard').")
		return(iter(self.fits_dict.items()))

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


























